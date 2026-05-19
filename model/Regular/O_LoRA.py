import os
import time
import math
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, get_all_reduce_mean


def _log(log_file, msg):
    """Append a line to the log file."""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

class O_LoRA(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 lamda_1 = 0.5, lamda_2 = 0
                 ):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        '''
        orthological to previous adapters
        '''
        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        
        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)

        # Log file (rank-0 only)
        log_dir = self.args.output_dir if self.args.output_dir else "."
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training.log")

    def _iter_olora_layers(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        for module in model.modules():
            if all(hasattr(module, attr) for attr in ("lora_A", "lora_B", "loranew_A", "loranew_B")):
                yield module

    def _merge_loranew_into_lora(self):
        merged = 0
        with torch.no_grad():
            for module in self._iter_olora_layers():
                adapter_names = (
                    set(module.lora_A.keys())
                    & set(module.lora_B.keys())
                    & set(module.loranew_A.keys())
                    & set(module.loranew_B.keys())
                )
                for adapter_name in adapter_names:
                    lora_a = module.lora_A[adapter_name].weight
                    lora_b = module.lora_B[adapter_name].weight
                    loranew_a = module.loranew_A[adapter_name].weight
                    loranew_b = module.loranew_B[adapter_name].weight

                    lora_a.data = torch.cat(
                        (lora_a.data, loranew_a.data.to(device=lora_a.device, dtype=lora_a.dtype)),
                        dim=0,
                    )
                    lora_b.data = torch.cat(
                        (lora_b.data, loranew_b.data.to(device=lora_b.device, dtype=lora_b.dtype)),
                        dim=1,
                    )
                    merged += 1
        return merged

    def _reset_loranew_parameters(self):
        with torch.no_grad():
            for module in self._iter_olora_layers():
                for adapter_name in module.loranew_A.keys():
                    nn.init.kaiming_uniform_(module.loranew_A[adapter_name].weight, a=math.sqrt(5))
                for adapter_name in module.loranew_B.keys():
                    nn.init.zeros_(module.loranew_B[adapter_name].weight)


    def train_one_task(self, task, i_task, epochs):
        # if i_task > 0:
        #     self.lamda_2 = 0.1
        print_rank_0(f"***** Training on task {task} *****", self.args.global_rank)
        
        num_task = len(self.train_task_list)
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]

        #### TRAIN ####
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        global_step = 0
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                global_step += 1
                del batch['sources']
                batch = to_device(batch, self.device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                ########################### Regularization ##########################
                orthogonal_loss = 0.
                for name, param in self.model.named_parameters():
                    if "lora_A" in name:
                        for name_, param_ in self.model.named_parameters():
                            if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                                orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
                                break 

                # l2-normalization for loranew_A/B
                l2_loss = 0.
                for name, param in self.model.named_parameters():
                    if "loranew_" in name:
                        l2_loss += torch.norm(param, p=2)

                loss = loss + orthogonal_loss * self.lamda_1 + l2_loss * self.lamda_2
                ######################################################################
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                    logging_steps = getattr(self.args, 'logging_steps', 10)
                    if global_step % logging_steps == 0:
                        print_rank_0(f"orthogonal_loss: {orthogonal_loss.item()}; l2_loss: {l2_loss.item()}; accuracy_loss: {outputs.loss.item()}; λ1: {self.lamda_1}; λ2: {self.lamda_2}", self.args.global_rank)

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()

            # Validate on eval split after each epoch
            print_rank_0(
                f"***** Evaluating generation metrics, Epoch {epoch+1}/{epochs} on task {task} *****",
                self.args.global_rank)
            eval_result, eval_predictions = self.task_generation_evaluation(
                task,
                eval_dataloader,
                self.device,
                max_ans_len=int(self.args.max_ans_len[i_task]),
                return_predictions=True,
            )
            print_rank_0(f"[task={task}] validation result: {eval_result}", self.args.global_rank)
            self._save_generation_predictions(f"eval-epoch{epoch+1}", i_task, task, eval_result, eval_predictions)

        self.evaluate_seen_tasks_after_training(task, i_task, self.device)

        #### COMBINE lora with lora_new and INITIALIZE lora_new ####
        merged = self._merge_loranew_into_lora()
        print_rank_0(f"Merged O-LoRA adapters in {merged} modules", self.args.global_rank)

        #### TEST ####
        log_dict = {
            "task_id": i_task,
        }

        trained_task_name = str(task).replace("/", "_").replace(":", "_")
        prediction_dir = os.path.join(self.args.output_dir or ".", "predictions", f"{i_task}-{trained_task_name}")
        if self.args.global_rank == 0:
            os.makedirs(prediction_dir, exist_ok=True)

        # for seen_idx, (eval_task, eval_dataset) in enumerate(list(self.eval_task_list.items())[i_task:i_task+1]):
        #     print_rank_0(f"***** Validating on {eval_task} after task training: {task} *****", self.args.global_rank)
        #     test_result, prediction_rows = self.task_generation_evaluation(
        #         eval_task,
        #         eval_dataset,
        #         self.device,
        #         max_ans_len=int(self.args.max_ans_len[seen_idx]),
        #         return_predictions=True,
        #     )
        #     print_rank_0(f"[task={eval_task}] validation result: {test_result}", self.args.global_rank)
        #     log_dict[f"eval_task/seen_task_{eval_task}/exact_match"] = test_result["exact_match"]
        #     log_dict[f"eval_task/seen_task_{eval_task}/bleu"] = test_result["bleu"]
        #     log_dict[f"eval_task/seen_task_{eval_task}/codebleu"] = test_result["codebleu"]

        #     if self.args.global_rank == 0:
        #         eval_task_name = str(eval_task).replace("/", "_").replace(":", "_")
        #         prediction_file = os.path.join(prediction_dir, f"{seen_idx}_{eval_task_name}.json")
        #         with open(prediction_file, "w", encoding="utf-8") as f:
        #             json.dump(prediction_rows, f, ensure_ascii=False, indent=2)
        #         print_rank_0(f"Saved predictions to {prediction_file}", self.args.global_rank)


        #### RESET ####
        self._reset_loranew_parameters()

        
        for name, param in self.model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = True
            elif name.find("lora_") != -1:
                param.requires_grad = False

        #### SAVE ####
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(i_task))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)
            self.model.save_pretrained(peft_model_id)  
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Sucessfully saving the final model to {peft_model_id}', self.args.global_rank)


    def save_model(self, i_task):
        pass

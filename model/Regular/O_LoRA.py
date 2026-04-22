import os
import time
import math
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, get_all_reduce_mean
from utils.model.model_utils import get_transformer_layers


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

                print_rank_0(f"orthogonal_loss: {orthogonal_loss.item()}; l2_loss: {l2_loss.item()}; accuracy_loss: {loss.item()}; λ1: {self.lamda_1}; λ2: {self.lamda_2}", self.args.global_rank)
                loss = loss + orthogonal_loss * self.lamda_1 + l2_loss * self.lamda_2
                ######################################################################
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                    _log(self.log_file,
                        f"[train] task={task} epoch={epoch+1} step={global_step} "
                        f"loss={loss.item():.6f} accuracy_loss={outputs.loss.item():.6f} "
                        f"orthogonal_loss={orthogonal_loss.item():.6f} l2_loss={l2_loss.item():.6f}"
                    )

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()

            # Validate on eval split after each epoch
            print_rank_0(
                f"***** Evaluating generation metrics, Epoch {epoch+1}/{epochs} on task {task} *****",
                self.args.global_rank)
            eval_result = self.task_generation_evaluation(
                task,
                eval_dataloader,
                self.device,
                max_ans_len=int(self.args.max_ans_len[i_task]),
            )
            print_rank_0(f"[task={task}] validation result: {eval_result}", self.args.global_rank)
            if self.args.global_rank == 0:
                _log(self.log_file,
                    f"[eval_epoch] task={task} epoch={epoch+1} "
                    f"exact_match={eval_result['exact_match']:.4f} "
                    f"bleu={eval_result['bleu']:.4f} "
                    f"codebleu={eval_result['codebleu']:.4f}"
                )

        #### COMBINE lora with lora_new and INITIALIZE lora_new ####
        flag = 0
        layer_id = 0
        ## Different models may have different naming of modules.
        ## 'setattr' is not work. So We have to hard code the name temporarily.
        # layer_list = self.model.base_model.model.model.decoder.layers   # opt
        # layer_list = self.model.base_model.model.model.layers           # llama
        layer_list = get_transformer_layers(self.model)
        state_dict = self.model.state_dict()    
        for k in state_dict:
            # # e.g. opt-1.3b
            # self.model.base_model.model.model.decoder.layers[layer_id].\
            # self_attn.v_proj.lora_A.default.weight.data\
            # = state_dict[k]
            if "v_proj.lora_A" in k:   
                for k_ in state_dict:
                    if ("v_proj.loranew_A" in k_) and (k.split("v_proj.lora_A")[0] == k_.split("v_proj.loranew_A")[0]):
                        state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=0) # [r_sum + r, dim]
                        layer_list[layer_id].self_attn.v_proj.lora_A.default.weight.data = state_dict[k]
                        break 
                flag += 1
            elif "q_proj.lora_A" in k:   
                for k_ in state_dict:
                    if ("q_proj.loranew_A" in k_) and (k.split("q_proj.lora_A")[0] == k_.split("q_proj.loranew_A")[0]):
                        state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=0) # [r_sum + r, dim]
                        layer_list[layer_id].self_attn.q_proj.lora_A.default.weight.data = state_dict[k]
                        break 
                flag += 1
            elif "v_proj.lora_B" in k:
                for k_ in state_dict:
                    if ("v_proj.loranew_B" in k_) and (k.split("v_proj.lora_B")[0] == k_.split("v_proj.loranew_B")[0]):
                        state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=1) # [dim, r_sum + r]
                        layer_list[layer_id].self_attn.v_proj.lora_B.default.weight.data = state_dict[k]
                        break 
                flag += 1
            elif "q_proj.lora_B" in k:
                for k_ in state_dict:
                    if ("q_proj.loranew_B" in k_) and (k.split("q_proj.lora_B")[0] == k_.split("q_proj.loranew_B")[0]):
                        state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=1) # [dim, r_sum + r]
                        layer_list[layer_id].self_attn.q_proj.lora_B.default.weight.data = state_dict[k]
                        break 
                flag += 1
            if flag == 4:
                layer_id += 1
                flag = 0

        #### TEST ####
        log_dict = {
            "task_id": i_task,
        }

        trained_task_name = str(task).replace("/", "_").replace(":", "_")
        prediction_dir = os.path.join(self.args.output_dir or ".", "predictions", f"{i_task}-{trained_task_name}")
        if self.args.global_rank == 0:
            os.makedirs(prediction_dir, exist_ok=True)

        for seen_idx, (eval_task, eval_dataset) in enumerate(list(self.eval_task_list.items())[:i_task+1]):
            print_rank_0(f"***** Validating on {eval_task} after task training: {task} *****", self.args.global_rank)
            test_result, prediction_rows = self.task_generation_evaluation(
                eval_task,
                eval_dataset,
                self.device,
                max_ans_len=int(self.args.max_ans_len[seen_idx]),
                return_predictions=True,
            )
            print_rank_0(f"[task={eval_task}] validation result: {test_result}", self.args.global_rank)
            log_dict[f"eval_task/seen_task_{eval_task}/exact_match"] = test_result["exact_match"]
            log_dict[f"eval_task/seen_task_{eval_task}/bleu"] = test_result["bleu"]
            log_dict[f"eval_task/seen_task_{eval_task}/codebleu"] = test_result["codebleu"]

            if self.args.global_rank == 0:
                eval_task_name = str(eval_task).replace("/", "_").replace(":", "_")
                prediction_file = os.path.join(prediction_dir, f"{seen_idx}_{eval_task_name}.json")
                with open(prediction_file, "w", encoding="utf-8") as f:
                    json.dump(prediction_rows, f, ensure_ascii=False, indent=2)
                print_rank_0(f"Saved predictions to {prediction_file}", self.args.global_rank)
                _log(self.log_file,
                    f"[eval_task] trained_on={task} eval_task={eval_task} "
                    f"exact_match={test_result['exact_match']:.4f} "
                    f"bleu={test_result['bleu']:.4f} "
                    f"codebleu={test_result['codebleu']:.4f}"
                )


        #### RESET ####
        for k in state_dict:
            if "loranew_A" in k:
                nn.init.kaiming_uniform_(state_dict[k], a=math.sqrt(5))
            elif "loranew_B" in k:
                nn.init.zeros_(state_dict[k])
        self.model.load_state_dict(state_dict)

        
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

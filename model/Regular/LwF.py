import numpy as np
import torch
import quadprog
import random
import os
from tqdm.auto import tqdm
import copy
import json
import torch.nn.functional as F
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0

class LwF(CL_Base_Model):
    def __init__(self,model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args
                 ):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)

        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)

            
    def train_step(self,
                    batch):

        lm_labels = batch["labels"]
        outputs = self.model(input_ids=batch['input_ids'], labels=lm_labels, attention_mask=batch['attention_mask'],use_cache=False)
        return outputs
    
    
    def KD_loss(self, new_logits, prev_logits, T):
        prev_logits = torch.from_numpy(prev_logits).to(torch.bfloat16)
        prev_logits = prev_logits.to(self.device)
        # prev_logits = F.log_softmax(prev_logits/T, dim=1)
        # new_logits = F.softmax(new_logits/T, dim=1)
        # kd_loss = torch.sum(prev_logits * new_logits, dim=1, keepdim=False)
        # kd_loss = -torch.mean(kd_loss, dim=0, keepdim=False)
        kd_loss = F.kl_div(F.log_softmax(prev_logits / T, dim=-1),
                        F.softmax(new_logits / T, dim=-1),
                        reduction='batchmean')
        
        return kd_loss
    
    def new_input_old_model_logits(self, i_task):
        task_name = list(self.train_task_list.keys())[i_task+1]
        train_dataloader = self.train_task_list[task_name]
        self.new_task_logits = {}
        for step, batch in enumerate(tqdm(train_dataloader)):
            del batch['sources']
            batch = {k: batch[k].to(self.device) for k in batch}
            outputs = self.train_step(batch)
            logits = outputs.logits.to(torch.float32).detach().cpu().numpy()
            self.new_task_logits[str(step)] = logits
            del logits
            del outputs
            
        # new_task_logits = json.dumps(new_task_logits)
        # with open(self.args.output_dir+"/LwF/{}.json".format(i_task+1),"w") as w:
        #     w.write(new_task_logits)
            

        
    
    def train_one_task(self,
                       task,
                       i_task,
                       epochs=40
                       ):
        # if i_task!=0:
        #     with open(self.args.output_dir+"/LwF/{}.json") as f:
        #         prev_logits = json.load(f)

        def resolve_max_ans_len(task_idx):
            max_ans_len = self.args.max_ans_len
            if isinstance(max_ans_len, (list, tuple)):
                if len(max_ans_len) == 0:
                    return 256
                if len(max_ans_len) == 1:
                    return int(max_ans_len[0])
                return int(max_ans_len[task_idx])
            return int(max_ans_len)

        dataloader_train = self.train_task_list[task]
        dataloader_eval = self.eval_task_list[task]
        for epoch in range(epochs):
            print(epoch)
            self.model.train()
            total_steps = epochs * len(dataloader_train)
            progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

            for step, batch in enumerate(tqdm(dataloader_train)):
                del batch['sources']
                batch = {k: batch[k].to(self.device) for k in batch}
                outputs = self.train_step(batch)
                loss = outputs.loss
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                    
                if i_task!=0:
                    loss += self.KD_loss(outputs.logits, self.new_task_logits[str(step)], 2)
                
                self.model.backward(loss)
                self.model.step()
            # Validate on eval split after each epoch
            print_rank_0(
                f"***** Evaluating generation metrics, Epoch {epoch+1}/{epochs} on task {task} *****",
                self.args.global_rank)
            eval_result = self.task_generation_evaluation(
                task,
                dataloader_eval,
                self.device,
                max_ans_len=resolve_max_ans_len(i_task),
            )
            print_rank_0(f"[task={task}] eval result: {eval_result}", self.args.global_rank)

        #### TEST on all seen tasks ####
        trained_task_name = str(task).replace("/", "_").replace(":", "_")
        prediction_dir = os.path.join("predictions_LwF", f"{i_task}-{trained_task_name}")
        if self.args.global_rank == 0:
            os.makedirs(prediction_dir, exist_ok=True)

        for seen_idx, (eval_task, eval_dataset) in enumerate(list(self.eval_task_list.items())[:i_task+1]):
            print_rank_0(
                f"***** Validating on {eval_task} after task training: {task} *****",
                self.args.global_rank,
            )
            test_result, prediction_rows = self.task_generation_evaluation(
                eval_task,
                eval_dataset,
                self.device,
                max_ans_len=resolve_max_ans_len(seen_idx),
                return_predictions=True,
            )
            print_rank_0(f"[task={eval_task}] validation result: {test_result}", self.args.global_rank)

            if self.args.global_rank == 0:
                eval_task_name = str(eval_task).replace("/", "_").replace(":", "_")
                prediction_file = os.path.join(prediction_dir, f"{seen_idx}_{eval_task_name}.json")
                with open(prediction_file, "w", encoding="utf-8") as f:
                    json.dump(prediction_rows, f, ensure_ascii=False, indent=2)
                print_rank_0(f"Saved predictions to {prediction_file}", self.args.global_rank)
                
        if i_task+1 < len(self.train_task_list):
            self.new_input_old_model_logits(i_task)

            

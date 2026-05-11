import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse
import json
from copy import deepcopy
from transformers import AdamW
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device


def _base_module(model):
    return model.module if hasattr(model, "module") else model


def get_prompt_container(model):
    base = _base_module(model)
    candidates = [
        model,
        base,
        getattr(model, "model", None),
        getattr(base, "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "prompt"):
            return candidate
    if hasattr(base, "model"):
        return base.model
    return base


class ResMLP(torch.nn.Module):
    def __init__(self, 
                 bottleneck_size,
                 module_type='MLP2',
                 emb_dimension=512,
                 residual=True,
                 ):
        """MLP class for soft prompt re-parameterization. MLP can have a Residual connection.
        Args:
            bottleneck_size (int): Dimension of the MLP bottlenack.
            module_type (str, optional): Type of MLP to be used. 
                Currently supports 1-layer and 2-layer MLPs, and simple transformer layer ('MLP1'/'MLP2'/'transformer'). 
                Defaults to 'MLP1'.
            emb_dimension (int, optional): Dimension of T5 model embeddings. Defaults to 512 (T5-small embedding dimension).
            residual (bool, optional): Whether to use residual connection in MLP. Defaults to True.
        """
        super().__init__()
        if module_type=='MLP1':
            # if layer_norm:
            self.module = nn.Sequential(
                nn.Linear(emb_dimension, bottleneck_size, dtype=torch.bfloat16),
                nn.ReLU(),
                nn.Linear(bottleneck_size, emb_dimension, dtype=torch.bfloat16),
                nn.LayerNorm(emb_dimension, dtype=torch.bfloat16),
            )
            # else:
            #     self.module = nn.Sequential(
            #         nn.Linear(emb_dimension, bottleneck_size),
            #         nn.Tanh(),
            #         nn.Linear(bottleneck_size, emb_dimension),
            #     )

        elif module_type=='MLP2':
            self.module = nn.Sequential(
                nn.Linear(emb_dimension, bottleneck_size),
                nn.ReLU(),
                nn.Linear(bottleneck_size, bottleneck_size // 2),
                nn.Tanh(),
                nn.Linear(bottleneck_size // 2, emb_dimension),
            )

        elif module_type=='transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dimension, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward(self, inputs):
        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)



class PP(CL_Base_Model):
    def __init__(self,
                 model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 prefix_len=20,
                 prefix_path=None, # path to the pre-trained progressive prompt
                 freeze_weights=True,
                 seq_len=512,
                 early_stopping=True,
                 prefix_MLP="MLP2",
                 bottleneck_size=800, # bottleneck size in case of using MLP reparametrization
                 ):
        
        """Class for CL & prompt tuning experiments with T5 model.
        Args:
            model_name (str): T5 model type to use (e.g. base/small/large etc.)
            task_list (List[str]): list of downstream tasks to be trained on. In case of 1 task - regular training.
            batch_size (int, optional): Batch size used. Defaults to 8.
            select_k_per_class (int, optional): Limit data to k samples/class. Defaults to -1 (keep original dataset size).
            prefix_len (int, optional): Prompt length to use. Defaults to 0 (i.e. no prompt).
            prefix_path (str, optional): Path to the pre-trained progressive prompt. Defaults to None.
            freeze_weights (bool, optional): Whether to freeze model weights. Defaults to True (prompt tuning setup).
            freeze_except (str, optional): Freeze all weights except parameters matching this condition. 
                Defaults to 'shared' (freeze all weights except word embeddings).
            lr (float, optional): Learning rate. Defaults to 0.3.
            weight_decay (float, optional): Weight decay coefficient. Defaults to 1e-5.
            seq_len (int, optional): Input text lengths in tokens. Defaults to 512.
            early_stopping (bool, optional): Use early stopping to select best prompt/model. Defaults to True.
            prefix_MLP (str, optional): what MLP to use for prompt re-parameterization. Defaults to 'MLP-1'.
            bottleneck_size (int, optional): Bottleneck size in case of using MLP reparametrization. Defaults to 800.
            mlp_lr (float, optional): MLP learning rate to use. Defaults to None (lr value will be used).
            weight_decay_mlp (float, optional): Wight decay coefficient in MLP. Defaults to None.
            get_test_subset (bool, optional): Whether to create a test subset. Defaults to True.
            memory_perc (float, optional): Percentage of data saved for memory replay in CL settings. Defaults to 0.0.
                 
                 
            prefix_len (int, optional): Soft prompt length (only needed if virtual tokens are added to the vocab). Defaults to 0.
            freeze_weights (bool, optional): Whether to freeze base model weights. 
                Model weights need to be frozen for prompt tuning (i.e. True)! Defaults to False.
            freeze_except (str, optional): If freeze_weights, do not freeze weights that contain this text. 
                Defaults to 'shared' (will avoid freezing word embeddings layer in T5).
            lr (float, optional): Prompt (model) learning rate. Defaults to 0.1.
            weight_decay (float, optional): Prompt (model) weight decay coefficient. Defaults to 0.00.
            prompt_name (str, optional): Shared name for prompt virtual tokens (when added to the vocab). 
                Not used in the latest implementation. Defaults to 'PRE'.
            
            prefix_MLP (str, optional): . Defaults to 'None'.
            mlp_bottleneck (int, optional): MLP bottleneck dimension. Defaults to 1000.
            weight_decay_mlp (float, optional): MLP weight decay coefficient. Defaults to 0.01.
            mlp_lr (float, optional): MLP learning rate. Defaults to 1e-4.
            mlp_layer_norm (bool, optional): Whether to use LN in MLP. Defaults to True.
            
            early_stopping (bool, optional): Whether to select best paramteres via early stopping. Defaults to True.
            opt (str, optional): Optimizer to use. Curretnly AdamW and LAMB are supported. Defaults to 'AdamW'.
        
            bottleneck_size (int): Dimension of the MLP bottlenack.
            module_type (str, optional): Type of MLP to be used. 
                Currently supports 1-layer and 2-layer MLPs ('MLP1' and 'MLP2'). Defaults to 'MLP1'.
            emb_dimension (int, optional): . Defaults to 512.
            layer_norm (bool, optional): . Defaults to True.
        
        
        
        """
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        self.freeze_weights = freeze_weights
        self.seq_len = seq_len
        self.early_stopping = early_stopping
        self.embed_tokens_dim = self.args.embed_tokens_dim
        self.embed_tokens_length = self.args.embed_tokens_length
        self.embed_tokens = self.args.embed_tokens
        if self.is_encoder_decoder and prefix_MLP == "MLP2":
            # T5 prompt tuning is more stable as direct encoder soft prompts.
            # The decoder-only PP path keeps the historical MLP reparameterizer.
            prefix_MLP = None
        self.prefix_MLP = prefix_MLP


        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)
        # Freezing model weights for prompt tuning
        if freeze_weights:
            print('Freezing weights')
            self.do_freeze_weights()
           
        self.prefix_len = prefix_len
        # Creating a trainable soft prompt
        if prefix_len>0:
            if prefix_path==None:
                prompt = self._prompt_container().prompt
                self.previous_prompts = torch.zeros(
                    [0, prompt.shape[1]],
                    requires_grad=False,
                    dtype=prompt.dtype,
                    device=self.device,
                )
            else: # initializing previous prompts from the path
                print('Using pre-trained progressive prompt - ' + prefix_path)
                self.previous_prompts = torch.tensor(
                    np.load(prefix_path),
                    requires_grad=False,
                    dtype=self.embed_tokens.weight.dtype,
                    device=self.device,
                )

    def _prompt_container(self):
        return get_prompt_container(self.model)

    def _init_prompt_from_random_tokens(self):
        weights = []
        for _ in range(self.prefix_len):
            with torch.no_grad():
                token_idx = np.random.randint(self.embed_tokens_length)
                weight = self.embed_tokens.weight[token_idx].detach().clone()
                if not self.is_encoder_decoder:
                    weight = weight / 100
                weights.append(weight)
        return torch.stack(weights, dim=0).to(self.device)

    def _reset_optimizer_state_for_prompt(self):
        prompt = self._prompt_container().prompt
        optimizers = [
            self.optimizer,
            getattr(self.model, "optimizer", None),
            getattr(self.optimizer, "optimizer", None),
        ]
        for optimizer in optimizers:
            state = getattr(optimizer, "state", None)
            if isinstance(state, dict) and prompt in state:
                state[prompt].clear()

    def _prepend_prompt(self, inputs_embeds, attention_mask, prompt):
        batch_size = inputs_embeds.shape[0]
        prefix_len = prompt.shape[0]
        prompt_emb = prompt.unsqueeze(0).expand(batch_size, -1, -1).to(inputs_embeds.dtype)

        is_left_padded = bool((attention_mask[:, 0] == 0).any().item())
        if self.is_encoder_decoder or not is_left_padded:
            prefix_mask = torch.ones(
                batch_size,
                prefix_len,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            return (
                torch.cat([prompt_emb, inputs_embeds], dim=1),
                torch.cat([prefix_mask, attention_mask], dim=1),
            )

        pad_counts = (attention_mask == 0).sum(dim=1)
        new_seq_len = inputs_embeds.shape[1] + prefix_len
        new_inputs_embeds = torch.zeros(
            batch_size,
            new_seq_len,
            inputs_embeds.shape[2],
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        new_attention_mask = torch.zeros(
            batch_size,
            new_seq_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        for b in range(batch_size):
            n_pad = int(pad_counts[b].item())
            new_inputs_embeds[b, :n_pad] = inputs_embeds[b, :n_pad]
            new_inputs_embeds[b, n_pad:n_pad + prefix_len] = prompt_emb[b]
            new_inputs_embeds[b, n_pad + prefix_len:] = inputs_embeds[b, n_pad:]
            new_attention_mask[b, n_pad:n_pad + prefix_len] = 1
            new_attention_mask[b, n_pad + prefix_len:] = attention_mask[b, n_pad:]
        return new_inputs_embeds, new_attention_mask
    

    # Concatenate newly learned prompt to the joint "Progressive Prompts"
    def progress_previous_prompts(self, task_num=None):


        if task_num!=None and self.prefix_MLP!=None:
            with torch.no_grad():
                prompt_container = self._prompt_container()
                new_prompt = prompt_container.mlps[task_num](prompt_container.prompt)
        else:
            new_prompt = self._prompt_container().prompt

        # Detach so previous_prompts is always a frozen leaf tensor with no
        # gradient path back to any MLP or prompt parameter.  Without this,
        # DeepSpeed ZeRO tries to reduce gradients for already-frozen MLPs
        # (from earlier tasks) and crashes because their gradients are None.
        self.previous_prompts = torch.concat(
            [new_prompt.clone().detach(), self.previous_prompts], axis=0
        )
        with torch.no_grad():
            self._prompt_container().prompt.copy_(self._init_prompt_from_random_tokens())
        self._reset_optimizer_state_for_prompt()
        print('Updated progressive prompts ', self.previous_prompts.shape)

    def task_generation_evaluation(self,
                task,
                test_dataloader,
                device,
                max_ans_len=None,
                return_predictions=False,
                prompt=None):
        """
        Example validate function that calculates BLEU for T5 code generation.
        """
        model = self.model
        tokenizer = self.tokenizer
        model.eval()

        if max_ans_len is None:
            max_ans_len = getattr(self.args, "max_ans_len", 256)

        # We'll accumulate references & predictions across the whole dataset
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []

        progress_bar = tqdm(total=len(test_dataloader), leave=True, disable=(self.args.global_rank != 0))
        for i, batch in enumerate(test_dataloader):
            sources_sequences += batch['sources']
            if 'gts' in batch:
                ground_truths += batch['gts']
                del batch['gts']
            elif 'labels' in batch:
                label_tensor = batch['labels']
                for row in label_tensor:
                    valid_ids = row[row != -100].detach().cpu().tolist()
                    ground_truths.append(
                        self.tokenizer.decode(valid_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    )
                del batch['labels']
            else:
                ground_truths += [''] * len(batch['sources'])

            del batch['sources']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]

            inputs_embeds = self.embed_tokens(batch["input_ids"]).to(self.device)

            if prompt is not None:
                inputs_embeds, source_mask_updated = self._prepend_prompt(
                    inputs_embeds,
                    batch["attention_mask"],
                    prompt,
                )
            else:
                source_mask_updated = batch["attention_mask"]

            # total input length (pad + prompt + real tokens) used to slice generated output
            total_input_len = inputs_embeds.shape[1]

            # encoder_outputs = model.encoder(
            #     attention_mask=source_mask_updated,
            #     inputs_embeds=inputs_embeds,
            #     head_mask=None,
            #     output_attentions=None,
            #     output_hidden_states=None,
            #     return_dict=None,
            # )
            with torch.no_grad():
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.tokenizer.eos_token_id
                
                outs = model.generate(
                    input_ids=None,
                    #input_ids=batch["source_ids"],
                    attention_mask=source_mask_updated,
                    # encoder_outputs=encoder_outputs,
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=max_ans_len,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=self.generation_config,
                    use_cache=True,
                )

            # generate() with inputs_embeds (no input_ids) returns ONLY the newly
            # generated token IDs — there is nothing to skip.
            sequences = self.tokenizer.batch_decode(
                outs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            predicted_sequences += sequences

            if self.args.global_rank == 0:
                progress_bar.update(1)
                description = f"Test step {i}"
                progress_bar.set_description(description, refresh=False)

        metrics = self._task_eval_from_predictions(task, sources_sequences, predicted_sequences, ground_truths)
        if return_predictions:
            prediction_rows = [
                {
                    "source": source,
                    "ground-truth": gt,
                    "prediction": pred,
                }
                for source, gt, pred in zip(sources_sequences, ground_truths, predicted_sequences)
            ]
            return metrics, prediction_rows
        return metrics  

    # Perform one train step for prompt tuning (following Lester et al.)
    def train_step_lester(self,
                          batch,
                          task=None,
                          task_num=None,
                          progressive=True):
        embed_prompt = self.prefix_MLP!=None
        prompt_container = self._prompt_container()
        if embed_prompt:
            assert task!=None
            mlp = prompt_container.mlps[task_num]

        batch = {k: batch[k].to(self.device) for k in batch}
        lm_labels = batch["labels"]
        # lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

        # inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
        inputs_embeds = self.embed_tokens(batch["input_ids"])  #向量，【batch * embedding_size】
        k = inputs_embeds.shape[0]
        if embed_prompt:
            prompt = mlp(prompt_container.prompt)
        else:
            prompt = prompt_container.prompt
        
        
        '''
        增量添加prompt代码处
        '''
        full_prompt = torch.concat([prompt, self.previous_prompts], axis=0) if progressive else prompt
        inputs_embeds, source_mask_updated = self._prepend_prompt(
            inputs_embeds,
            batch["attention_mask"],
            full_prompt,
        )

        # Prefix positions must be ignored by the loss — fill with -100.
        if not self.is_encoder_decoder:
            prefix_ignore = torch.full(
                (k, inputs_embeds.shape[1] - lm_labels.shape[1]),
                fill_value=-100, dtype=lm_labels.dtype, device=lm_labels.device
            )
            lm_labels = torch.concat([prefix_ignore, lm_labels], axis=1)

        
        '''
        encoder_outputs = model.encoder(
                                attention_mask=source_mask_updated,
                                inputs_embeds=inputs_embeds,
                                head_mask=None,  
                                output_attentions=None,  
                                output_hidden_states=None, 
                                return_dict=None,  
                            )

        outputs = model(
            input_ids=batch["source_ids"],
            attention_mask=source_mask_updated, 
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            encoder_outputs=encoder_outputs,
        )
        
        outputs = self.model.generate(inputs.input_ids, max_length=1024)
        '''
        outputs = self.model(inputs_embeds=inputs_embeds, labels=lm_labels, attention_mask=source_mask_updated)
        loss = outputs[0]

        return loss




    # Freeze model weights
    def do_freeze_weights(self):
        model = self.model
        for name, param in model.named_parameters():
            param.requires_grad = False


    # Freeze / unfreeze MLPs for given tasks (when requires_grad==False then freezing)
    def freeze_unfreeze_mlps(self, tasks, requires_grad=False):
        assert self.prefix_MLP != None

        for task_num in tasks:
            #for name, param in self.prefix_MLPs[t].named_parameters():
            for name, param in self._prompt_container().mlps[task_num].named_parameters():
                if param.requires_grad != requires_grad:
                    param.requires_grad = requires_grad
                    param.grad = None # remove old gradient

    
    # Perform training on a single task
    def train_one_task(self,
                       task,
                       task_num,
                       epochs,
                       progressive=True,
                       last_task=False):
        
        #将新的prompt加入优化器
        # old_prompt = deepcopy(self.model.model.prompt)
        print('task = ', task)
        if progressive:
            assert self.prefix_len>0 # can only do progressive prompts when prompt tuning
            print('progressive prompts')
        print_rank_0(
            f"[task={task}] prompt shape: {self._prompt_container().prompt.shape}, "
            f"previous_prompts shape: {self.previous_prompts.shape}, "
            f"total prefix len: {self._prompt_container().prompt.shape[0] + self.previous_prompts.shape[0]}",
            self.args.global_rank,
        )
        if self.early_stopping:
            self.best_acc = 0.0 # re-setting best acc

        if self.prefix_MLP!=None:
            print('Freezing all MLPs except for ', task)
            mlp = self._prompt_container().mlps[task_num]
            self.freeze_unfreeze_mlps([x for x in range(len(self.train_task_list)) if x!=task_num], requires_grad=False)
            self.freeze_unfreeze_mlps([task_num], requires_grad=True) # unfreezing current task

        model = self.model
        self._prompt_container().prompt.requires_grad=True
        # with torch.no_grad():
        #     # prompts = nn.Parameter(torch.tensor(self.init_new_prompt(self.prefix_len),
        #     #                             requires_grad=True))
        #     model.model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(self.prefix_len),
        #                                 requires_grad=True))
        # model.model.prompt.requires_grad = True
        # model.model.to(self.device)

        dataloader_train = self.train_task_list[task]
        total_steps = epochs * len(dataloader_train)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

        val_acc = []

        for epoch in range(epochs):
            print(epoch)
            model.train()
            if self.prefix_MLP!=None:
                mlp.train()

            for step, batch in enumerate(tqdm(dataloader_train)):
                del batch['sources']
                batch = {k:batch[k].to('cuda') for k in batch}
                if self.prefix_len>0: # prompt tuning
                    loss = self.train_step_lester(batch,
                                                  task=task if self.prefix_MLP!=None else None,
                                                  task_num=task_num if self.prefix_MLP!=None else None,
                                                  progressive=progressive)
                    
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                model.backward(loss)
                model.step()
            
                

            # evaluate accuracy after each epoch
            if self.prefix_MLP!=None:
                #生成prompt并拼接到previous_prompts
                mlp.eval()
                prompt = mlp(self._prompt_container().prompt)
            else:
                if self.prefix_len>0:
                    prompt = self._prompt_container().prompt
                    print(prompt.shape)
                else:
                    prompt = None
            if progressive:
                prompt = torch.concat([prompt, self.previous_prompts], axis=0)

                        # Validate on eval split after each epoch
            print_rank_0(
                f"***** Evaluating generation metrics, Epoch {epoch+1}/{epochs} on task {task} *****",
                self.args.global_rank)
            eval_result, eval_predictions = self.task_generation_evaluation(
                task,
                self.eval_task_list[task],
                self.device,
                max_ans_len=self._resolve_max_ans_len(task_num),
                return_predictions=True,
                prompt=prompt
            )
            print_rank_0(f"[task={task}] validation result: {eval_result}", self.args.global_rank)
            self._save_generation_predictions(f"eval-epoch{epoch+1}", task_num, task, eval_result, eval_predictions)

        if progressive:
            self.progress_previous_prompts(task_num=task_num)
            # model.model.prompt.data = deepcopy(old_prompt.data)

        # Validate on val split for all seen tasks after training on the whole task
        if progressive:
            curr_prompt = self.previous_prompts.clone().detach().to(self.device)
        else:
            if self.prefix_len>0:
                curr_prompt = self._prompt_container().prompt
            else:
                curr_prompt = None
        split_name = f"test-after-task-{task_num}"
        for seen_idx, (seen_task, test_dataloader) in enumerate(list(self.test_task_list.items())[:task_num + 1]):
            print_rank_0(
                f"***** Testing on seen task {seen_task} after training task {task} *****",
                self.args.global_rank)
            test_result, test_predictions = self.task_generation_evaluation(
                seen_task,
                test_dataloader,
                self.device,
                max_ans_len=self._resolve_max_ans_len(seen_idx),
                return_predictions=True,
                prompt=curr_prompt
            )
            print_rank_0(
                f"[seen-task={seen_task} after-task={task}] test result: {test_result}",
                self.args.global_rank)
            self._save_generation_predictions(split_name, seen_idx, seen_task, test_result, test_predictions)
        # log_dict = {
        #     "task_id": task_num,
        # }
        # trained_task_name = str(task).replace("/", "_").replace(":", "_")
        # prediction_dir = os.path.join(self.args.output_dir or ".", "predictions", f"{task_num}-{trained_task_name}")
        # if self.args.global_rank == 0:
        #     os.makedirs(prediction_dir, exist_ok=True)
        # for seen_idx, (eval_task, eval_dataset) in enumerate(list(self.eval_task_list.items())[task_num:task_num+1]):
        #     print_rank_0(f"***** Validating on {eval_task} after task training: {task} *****", self.args.global_rank)
        #     test_result, prediction_rows = self.task_generation_evaluation(
        #         eval_task,
        #         eval_dataset,
        #         self.device,
        #         max_ans_len=int(self.args.max_ans_len[seen_idx]),
        #         return_predictions=True,
        #         prompt=curr_prompt
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

        # TEST ON ALL TASKS FOR THE LAST TRAINED TASK
        if last_task:
            self.test_all_tasks_and_save_predictions(prompt=curr_prompt)

    def test_all_tasks_and_save_predictions(self, prompt):
        prediction_root = os.path.join(self.args.output_dir or ".", "predictions", f"final-{self.__class__.__name__}")
        if self.args.global_rank == 0:
            os.makedirs(prediction_root, exist_ok=True)

        final_metrics = {}
        for task_idx, (task_name, test_dataloader) in enumerate(self.test_task_list.items()):
            print_rank_0(
                f"***** Final testing on task {task_name} after continual training *****",
                self.args.global_rank,
            )
            test_result, prediction_rows = self.task_generation_evaluation(
                task_name,
                test_dataloader,
                self.device,
                max_ans_len=self._resolve_max_ans_len(task_idx),
                return_predictions=True,
                prompt=prompt
            )
            final_metrics[task_name] = test_result
            print_rank_0(f"[final-test task={task_name}] result: {test_result}", self.args.global_rank)

            if self.args.global_rank == 0:
                safe_task_name = str(task_name).replace("/", "_").replace(":", "_")
                prediction_file = os.path.join(prediction_root, f"{task_idx}_{safe_task_name}.json")
                with open(prediction_file, "w", encoding="utf-8") as f:
                    json.dump(prediction_rows, f, ensure_ascii=False, indent=2)
                print_rank_0(f"Saved final-test predictions to {prediction_file}", self.args.global_rank)

        if self.args.global_rank == 0:
            metrics_file = os.path.join(prediction_root, "metrics_summary.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(final_metrics, f, ensure_ascii=False, indent=2)
            print_rank_0(f"Saved final-test metrics to {metrics_file}", self.args.global_rank)
    
    def train_continual(self):
        for i_task, task in enumerate(self.train_task_list):
            last_task = (i_task == len(self.train_task_list)-1)
            # num_train_epochs may be a single-element list when the same epoch
            # count applies to all tasks; fall back to the last entry if needed.
            epoch_idx = min(i_task, len(self.args.num_train_epochs) - 1)
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[epoch_idx]), True, last_task)
            self.save_model(i_task)

def convert_PP_model(model, args):
    def init_new_prompt(prompt_len):
        N = args.embed_tokens_length
        prompt_weigths = []

        for i in range(prompt_len):
            with torch.no_grad():
                j = np.random.randint(N) # random token
                w = deepcopy(args.embed_tokens.weight[j].detach().cpu().numpy())
                if getattr(args, "is_encoder_decoder", False):
                    prompt_weigths.append(w)
                else:
                    prompt_weigths.append(w/100)
                # prompt_weigths.append(w)

        prompt_weigths = np.array(prompt_weigths)
        return prompt_weigths
    prompt_container = get_prompt_container(model)
    prompt_container.prompt = nn.Parameter(
        args.embed_tokens.weight.detach().new_tensor(
            init_new_prompt(args.prefix_len),
            requires_grad=True,
        )
    )
    if getattr(args, "is_encoder_decoder", False):
        if hasattr(prompt_container, "mlps"):
            delattr(prompt_container, "mlps")
    else:
        prompt_container.mlps = nn.ModuleList([
            ResMLP(bottleneck_size=2*args.embed_tokens_dim, emb_dimension=args.embed_tokens_dim)
            for _ in range(args.task_length)
        ])
    
    return model

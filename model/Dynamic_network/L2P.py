from copy import deepcopy

import torch
import torch.utils.data
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch import nn
from model.base_model import CL_Base_Model
import numpy as np
from deepspeed.utils import safe_get_full_grad
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds # to be continued
from transformers import GenerationConfig
import json
import os
generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=False,
    num_return_sequences=1
)


def convert_L2P_model(model, args):
    def init_new_prompt(pool_size,prompt_len):
        N = args.embed_tokens_length
        prompt_weigths = []
        for t in range(pool_size):
            prompt_weight = []
            for i in range(prompt_len):
                with torch.no_grad():
                    j = np.random.randint(N) # random token
                    w = deepcopy(args.embed_tokens.weight[j].detach().cpu().numpy())
                    prompt_weight.append(w)
                    # prompt_weigths.append(w)
            prompt_weigths.append(prompt_weight)

        prompt_weigths = np.array(prompt_weigths)
        return prompt_weigths
    # prompt_pool_shape = (args.pool_size, args.prompt_length, args.embed_tokens_dim)
    # if args.prompt_init == 'zero':
    #     model.model.prompt = nn.Parameter(torch.zeros(prompt_pool_shape, dtype=torch.bfloat16,requires_grad=True)).to("cuda")
    if args.prompt_init == 'uniform':
        model.model.prompt = nn.Parameter(torch.tensor(init_new_prompt(args.pool_size , args.prompt_length),requires_grad=True))
            
    return model



class L2P(CL_Base_Model):

    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,
                 embedding_key='mean',
                 top_k=3,
                 batchwise_prompt=False,
                 pull_constraint_coeff=0.5
                 ):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)

        # Device (match Regular methods like `O_LoRA`)
        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)

        self.embed_dim = self.args.embed_tokens_dim
        self.embed_tokens = self.args.embed_tokens
        self.embedding_key = embedding_key
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.pull_constraint_coeff = pull_constraint_coeff

        # use mean of prompt as key
        # only compatible with prompt, not prefix
        prompt_mean = torch.mean(self.model.model.prompt, dim=1)
        self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def train_step(self, batch, prompt_mask=None, cls_features=None):
        input_ids = batch['input_ids']
        attn_masks = batch['attention_mask']
        labels = batch['labels']
        inputs_embeds = self.embed_tokens(input_ids)
        
        out = dict()
        if self.embedding_key == 'mean':
            inputs_embeds_mean = torch.mean(inputs_embeds, dim=1)
        elif self.embedding_key == 'max':
            inputs_embeds_mean = torch.max(inputs_embeds, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            inputs_embeds_mean = torch.max(inputs_embeds, dim=1)[0] + 2 * torch.mean(inputs_embeds, dim=1)
        elif self.embedding_key == 'cls':
            if cls_features is None:
                inputs_embeds_mean = torch.max(inputs_embeds, dim=1)[0] # B, C
            else:
                inputs_embeds_mean = cls_features
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")

        prompt_norm = self.l2_normalize(self.prompt_key, dim=1).to("cuda") # Pool_size, C
        inputs_embeds_norm = self.l2_normalize(inputs_embeds_mean, dim=1) # B, C

        similarity = torch.matmul(inputs_embeds_norm, prompt_norm.t()) # B, Pool_size
        
        if prompt_mask is None:
            _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                major_prompt_id = prompt_id[major_idx] # top_k
                # expand to batch
                idx = major_prompt_id.expand(inputs_embeds.shape[0], -1) # B, top_k
        else:
            idx = prompt_mask # B, top_k

        batched_prompt_raw = self.model.model.prompt[idx] # B, top_k, length, C
        batch_size, top_k, length, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C
        inputs_embeds = torch.cat([batched_prompt, inputs_embeds],axis=1)
        
        prefix_length = batched_prompt.shape[1]
        attn_masks = torch.concat((torch.tensor(1).to("cuda").repeat(batch_size,prefix_length),attn_masks), axis=1)
        labels = torch.concat((labels[0][0].repeat(batch_size,inputs_embeds.shape[1]-labels.shape[1]),labels),axis=1)
        outputs = self.model(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attn_masks, use_cache=False)
        loss = outputs[0]
        
        batched_key_norm = prompt_norm[idx]
        inputs_embeds_norm = inputs_embeds_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * inputs_embeds_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / inputs_embeds.shape[0] # Scalar

        loss -= reduce_sim * self.pull_constraint_coeff
        return loss
    
    
    
    
    def train_one_task(self, task, i_task, epochs):
        print_rank_0(f"***** Training on task {task} *****", self.args.global_rank)

        train_dataloader = self.train_task_list[task]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank,
            )

            self.model.train()
            for step, batch in enumerate(tqdm(train_dataloader)):
                del batch['sources']
                batch = {k: batch[k].to(self.device) for k in batch}
                loss = self.train_step(batch)

                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self.model.backward(loss, retain_graph=True)
                self.model.step()

            # Validate (match O_LoRA style)
            print_rank_0(
                f"***** Evaluating generation metrics, Epoch {epoch+1}/{epochs} on task {task} *****",
                self.args.global_rank,
            )
            self.evaluate_one_task(round=i_task, infer_task_id=i_task, task=task, infer_dataloader=self.eval_task_list[task])

    def test_all_tasks_and_save_predictions(self):
        self.args.output_dir = './L2P_final_results'
        for task_idx, (task_name, test_dataloader) in enumerate(self.test_task_list.items()):
            print_rank_0(
                f"***** Final testing on task {task_name} after continual training *****",
                self.args.global_rank,
            )
            self.evaluate_one_task(round=task_idx, infer_task_id=task_idx, task=task_name, infer_dataloader=test_dataloader, save_results=True)

    def evaluate(self, round, infer_task_id, task, infer_dataloader):
        self.evaluate_one_task(round,infer_task_id, task, infer_dataloader)

    def dist_results_gather(self, generate_ids, pad_token=-1):
        if not dist.is_available() or not dist.is_initialized():
            return generate_ids, generate_ids.size(1)

        result = generate_ids
        local_batch_size = torch.tensor([result.size(0)], dtype=torch.int, device=result.device)
        local_seq_len = torch.tensor([result.size(1)], dtype=torch.int, device=result.device)

        world_size = dist.get_world_size()
        global_batch_sizes = [torch.zeros_like(local_batch_size) for _ in range(world_size)]
        global_seq_len = [torch.zeros_like(local_seq_len) for _ in range(world_size)]
        dist.all_gather(global_batch_sizes, local_batch_size)
        dist.all_gather(global_seq_len, local_seq_len)

        max_seq_len = max(int(seq_len.item()) for seq_len in global_seq_len)

        if result.size(1) < max_seq_len:
            pad_seq_len = (max_seq_len - result.size(1), 0)
            result = F.pad(result, pad_seq_len, "constant", pad_token)

        total_results = [
            torch.zeros((int(bs.item()), max_seq_len), dtype=result.dtype, device=result.device)
            for bs in global_batch_sizes
        ]
        dist.all_gather(total_results, result)
        flat_results = torch.cat(total_results, dim=0)

        return flat_results, max_seq_len
        
    def evaluate_one_task(self, round, infer_task_id, task, infer_dataloader, save_results=False):
        device = self.device

        progress_bar = tqdm(total=len(infer_dataloader), leave=True, disable=(self.args.global_rank != 0))
        
        def prediction(model, infer_dataloader):
            predicted_sequences = []
            sources_sequences = []
            label_sequences = []
            model.eval()

            for step, batch in enumerate(infer_dataloader):
                sources_sequences += batch.get('sources', [])
                if 'gts' in batch:
                    label_sequences += batch['gts']
                    ground_truths_ids = self.tokenizer(
                        batch['gts'],
                        truncation=True,
                        max_length=int(self.args.max_ans_len[infer_task_id]),
                        add_special_tokens=False,
                        padding='max_length',
                        return_tensors='pt',
                    )['input_ids'].to(device)
                    del batch['gts']
                elif 'labels' in batch:
                    label_tensor = batch['labels']
                    for row in label_tensor:
                        valid_ids = row[row != -100].detach().cpu().tolist()
                        label_sequences.append(
                            self.tokenizer.decode(
                                valid_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                        )
                    ground_truths_ids = batch['labels']
                    del batch['labels']
                else:
                    label_sequences += [''] * len(batch.get('sources', []))
                    ground_truths_ids = torch.empty(0, dtype=torch.long, device=device)

                del batch['sources']
                batch = to_device(batch, device)
                progress_bar.update(1)

                # update progress bar
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    description = f"Step {step}"
                    progress_bar.set_description(description, refresh=False)

                with torch.no_grad():
                    # sft config
                    input_ids = batch['input_ids']
                    attn_masks = batch['attention_mask']
                    inputs_embeds = self.embed_tokens(input_ids)

                    if self.embedding_key == 'mean':
                        inputs_embeds_mean = torch.mean(inputs_embeds, dim=1)

                    prompt_norm = self.l2_normalize(self.prompt_key, dim=1).to("cuda") # Pool_size, C
                    inputs_embeds_norm = self.l2_normalize(inputs_embeds_mean, dim=1) # B, C

                    similarity = torch.matmul(inputs_embeds_norm, prompt_norm.t()) # B, Pool_size

                    _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                        

                    batched_prompt_raw = self.model.model.prompt[idx] # B, top_k, length, C
                    batch_size, top_k, length, c = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

                    prefix_length = batched_prompt.shape[1]
                    pad_counts = (attn_masks == 0).sum(dim=1)
                    new_seq_len = inputs_embeds.shape[1] + prefix_length
                    new_inputs_embeds = torch.zeros(
                        batch_size,
                        new_seq_len,
                        inputs_embeds.shape[2],
                        dtype=inputs_embeds.dtype,
                        device=inputs_embeds.device,
                    )
                    new_attn_mask = torch.zeros(
                        batch_size,
                        new_seq_len,
                        dtype=attn_masks.dtype,
                        device=attn_masks.device,
                    )

                    for b in range(batch_size):
                        n_pad = int(pad_counts[b].item())
                        # Layout: [pad_tokens | prompt_tokens | real_tokens]
                        new_inputs_embeds[b, :n_pad] = inputs_embeds[b, :n_pad]
                        new_inputs_embeds[b, n_pad:n_pad + prefix_length] = batched_prompt[b]
                        new_inputs_embeds[b, n_pad + prefix_length:] = inputs_embeds[b, n_pad:]
                        new_attn_mask[b, n_pad:n_pad + prefix_length] = 1
                        new_attn_mask[b, n_pad + prefix_length:] = attn_masks[b, n_pad:]

                    inputs_embeds = new_inputs_embeds
                    attn_masks = new_attn_mask
                    
                    generate_ids = model.generate(inputs_embeds=inputs_embeds,
                                                  attention_mask=attn_masks,
                                                  max_new_tokens=int(self.args.max_ans_len[infer_task_id]),
                                                  bos_token_id=self.tokenizer.bos_token_id,
                                                  eos_token_id=self.tokenizer.eos_token_id,
                                                  pad_token_id=self.tokenizer.unk_token_id,
                                                  generation_config=generation_config,
                                                use_cache=False

                                                  )
                pre_sequences = self.tokenizer.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                predicted_sequences += pre_sequences

            return sources_sequences, predicted_sequences, label_sequences


        def save_inference_results(evaluation_result: dict, sources_sequences: list, predicted_sequences: list,
                                   ground_truths: list, round: int, i_task: int, task: str):
            # save as a json file
            df = {"eval": evaluation_result}
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)
            prediction_rows = [
                {
                    "source": source,
                    "ground-truth": gt,
                    "prediction": pred,
                }
                for source, gt, pred in zip(sources_sequences, ground_truths, predicted_sequences)
            ]
            df["predictions"] = prediction_rows
            with open(self.args.output_dir + "/results-" + str(round) + "-" + str(i_task) + "-" + task + ".json", "w+", encoding='utf-8') as file:
                json.dump(df, file, ensure_ascii=False)


        # Inference !
        print_rank_0("***** Start inference *****", self.args.global_rank)
        sources_sequences, predicted_sequences, ground_truths = prediction(self.model, infer_dataloader)

        # Get Accuracy/ROUGE/BLEU/CodeBLEU/...
        # Prefer the shared evaluator in `CL_Base_Model` (handles CodeBLEU/SmoothBLEU/etc. for code tasks).
        if self.args.global_rank <= 0:
            for source, gt, pred in list(zip(sources_sequences, ground_truths, predicted_sequences))[:3]:
                print_rank_0("***** Sample inference results *****", self.args.global_rank)
                print_rank_0(f"Source: {source}", self.args.global_rank)
                print_rank_0(f"Ground Truth: {gt}", self.args.global_rank)
                print_rank_0(f"Prediction: {pred}", self.args.global_rank)
                print_rank_0("-----", self.args.global_rank)

            evaluation_result = self._task_eval_from_predictions(
                task,
                sources_sequences=sources_sequences,
                predicted_sequences=predicted_sequences,
                ground_truths=ground_truths,
            )
            print_rank_0(f"***** Evaluation results on task {task} *****", self.args.global_rank)
            print_rank_0(evaluation_result, self.args.global_rank)

            if save_results:
                print_rank_0("***** Saving inference results *****", self.args.global_rank)
                save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, round, infer_task_id, task)
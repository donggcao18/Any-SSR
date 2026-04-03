import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers.models.llama import LlamaForCausalLM
import torch
from torch.nn import CrossEntropyLoss
from typing import Optional, List, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging
from transformers.models.llama import LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaSdpaAttention, LlamaConfig, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaAttention, LlamaFlashAttention2, LlamaMLP, repeat_kv, apply_rotary_pos_emb
import torch.nn as nn

import torch.nn.functional as F

import logging
import transformers

from peft import LoraConfig
import torch

import json
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from peft import get_peft_model

from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import torch.nn.functional as F

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}

feature_layers = 4
gamma = 10000

# ---- configurable paths (works on Windows/Linux) ----
router_weights_path = os.environ.get(
    "ANYSSR_ROUTER_WEIGHTS_PATH",
    os.path.join("output_models", "router_weights"),
)
dataset_path = os.environ.get(
    "ANYSSR_DATASET_PATH",
    os.path.join("dataset", "TRACE-Benchmark", "LLM-CL-Benchmark_5000"),
)
dataset_cache_path = os.environ.get(
    "ANYSSR_DATASET_CACHE_PATH",
    os.path.join("output_models", "outputs_router_dataset_cache"),
)

paths = [router_weights_path, dataset_cache_path]
for path in paths:
    os.makedirs(path, exist_ok=True)

class NewLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config, task_number, gamma):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.model.layers = torch.nn.ModuleList(self.model.layers[:feature_layers])  # 仅取前4层进行分类

        # self.cls_head = torch.nn.Linear(in_features=4096, out_features=task_number)
        self.fe = torch.nn.Linear(in_features=4096, out_features=gamma)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        # only takes output of the first 4 layers 

        hidden_mean = outputs[0].mean(dim=1)

        out = self.fe(hidden_mean)

        return out
    
    def MoeClassifier():
        pass

def load_model_and_tokenizer():
    model = NewLlamaForCausalLM.from_pretrained(
                'meta-llama/Llama-2-7b-chat-hf',
                device_map="auto",
                torch_dtype="auto",
                task_number=8,
                trust_remote_code=True,
                gamma=10000
            )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True
    )

    return model, tokenizer

def train():
    num_epochs = 1
    max_length = 150

    model, tokenizer = load_model_and_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token

    for name, param in model.named_parameters():
        if "cls_head" not in name:
            param.requires_grad = False
        else:
            print(1)

    # Task order (must match your downstream inference order)
    inference_tasks = [
        "hf:CONCODE",
        "hf:CodeTrans",
        "hf:CodeSearchNet",
        "hf:BFP",
        "hf:TheVault_Csharp",
        "hf:KodCode",
        "hf:RunBugRun",
        "hf:CoST",
    ]
    import numpy as np

    def train_initial_router(model, infer_dataloader, step):
        """Train first tasks"""
        with torch.no_grad():
            # print(f'---start training from {inference_tasks[0]} to {inference_tasks[step+1]}')
            print(f'---start training from {inference_tasks[0]} to {inference_tasks[step]}')
            count = 0
            print('-----------------------start training-------------------')
            
            for steps, batch in enumerate(infer_dataloader):
                labels = batch['gts']
                input_ids = batch['input_ids'].to('cuda')
                

                new_activation = model(input_ids).to(torch.float32)
                labels = torch.tensor(labels)
                # label_onehot = F.one_hot(labels, step + 2).float().to('cuda')
                label_onehot = F.one_hot(labels, step + 1).float().to('cuda')
                

                if count == 0:
                    auto_cor = torch.t(new_activation) @ new_activation
                    crs_cor = torch.t(new_activation) @ (label_onehot)
                else:
                    auto_cor += torch.t(new_activation) @ new_activation
                    crs_cor += torch.t(new_activation) @ (label_onehot)
                
                count += 1
            
            print('Calculating Reverse')

            R = np.mat(auto_cor.cpu().numpy() + 100 * np.eye(gamma)).I
            R_tensor = torch.tensor(R).float().cuda(non_blocking=True).cpu()
            Delta = R_tensor @ crs_cor.cpu()
            

            torch.save(Delta, f'{router_weights_path}/step{step}_router_weight.pth')
            torch.save(model.fe.weight, f'{router_weights_path}/step{step}_fe_weight.pth')
            torch.save(R_tensor, f'{router_weights_path}/step{step}_R_matrix.pth')
            
            print(f'Finished initial training from {inference_tasks[0]} to {inference_tasks[step]}')
            return R_tensor, Delta  

    def train_subsequent_router(model, infer_dataloader, step, prev_R, prev_Delta):
        """Recursive Train"""
        # router dimension expansion
        prev_Delta = F.pad(prev_Delta, (0, 1), mode='constant', value=0).to('cuda')
        prev_R = prev_R.to('cuda')
        with torch.no_grad():
            print(f'Start training from {inference_tasks[0]} to {inference_tasks[step]}')
            print(f'Use step{step-1} as initial R')
            count = 0
            print('-----------------------Start Recursive Train-------------------')
            
            for steps, batch in enumerate(infer_dataloader):
                labels = batch['gts']
                input_ids = batch['input_ids'].to('cuda')
                
                new_activation = model(input_ids).to(torch.float32)
                labels = torch.tensor(labels)
                # label_onehot = F.one_hot(labels, step + 2).float().to('cuda')
                label_onehot = F.one_hot(labels, step + 1).float().to('cuda')

                prev_R = prev_R - prev_R @ new_activation.t() @ torch.pinverse(torch.eye(new_activation.shape[0]).to('cuda') +
                                                                    new_activation @ prev_R @ new_activation.t()) @ new_activation @ prev_R
                prev_Delta = prev_Delta + prev_R @ new_activation.t() @ (label_onehot - new_activation @ prev_Delta)
            
            print('Calculate new R')
            new_R = prev_R
            

            new_Delta = prev_Delta
            
            torch.save(new_Delta, f'{router_weights_path}/step{step}_router_weight.pth')
            torch.save(model.fe.weight, f'{router_weights_path}/step{step}_fe_weight.pth')
            torch.save(new_R, f'{router_weights_path}/step{step}_R_matrix.pth')
            
            # print(f'Finished training from {inference_tasks[0]} to {inference_tasks[step+1]}')
            print(f'Finished training from {inference_tasks[0]} to {inference_tasks[step]}')
            return new_R, new_Delta  # 返回新的R矩阵


    # for i in range(0, len(inference_tasks)-1):
    for i in range(0, len(inference_tasks)):
        # cur_inference_tasks = inference_tasks[0:i+2]
        cur_inference_tasks = inference_tasks[0:i+1]
        all_datasets = []
        
        if i == 0:
            for inference_task_id in range(len(cur_inference_tasks)):
                inference_task = inference_tasks[inference_task_id]
                # hf:* datasets are dataset identifiers, not filesystem paths
                if isinstance(inference_task, str) and inference_task.startswith("hf:"):
                    cur_dataset_path = inference_task
                else:
                    cur_dataset_path = os.path.join(dataset_path, inference_task)

                train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
                    -1,
                    cur_dataset_path,
                    dataset_cache_path,
                    42,
                    distributed=False
                )
                # set the label for router training
                train_dataset.answer_dataset = [inference_task_id for _ in train_dataset.answer_dataset]
                all_datasets.append(train_dataset)
        else:
            inference_task = inference_tasks[i]
            inference_task_id = i
            # hf:* datasets are dataset identifiers, not filesystem paths
            if isinstance(inference_task, str) and inference_task.startswith("hf:"):
                cur_dataset_path = inference_task
            else:
                cur_dataset_path = os.path.join(dataset_path, inference_task)

            # data preparation
            train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
                -1,
                cur_dataset_path,
                dataset_cache_path,
                42,
                distributed=False
            )
            # set the label for router training
            train_dataset.answer_dataset = [inference_task_id for _ in train_dataset.answer_dataset]
            all_datasets.append(train_dataset)
        
        try:
            infer_dataset = torch.utils.data.ConcatDataset(all_datasets)
        except:
            infer_dataset = all_datasets[0]

        inf_data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=512,
            max_ans_len=256,
            pad_to_multiple_of=8,
            inference=True
        )
        infer_sampler = SequentialSampler(infer_dataset)
        infer_dataloader = DataLoader(
            infer_dataset,
            collate_fn=inf_data_collator,
            sampler=infer_sampler,
            batch_size=1
        )

        print("***** Start Training *****")
        if i == 0:
            # Obtain original R and router
            current_R, Delta = train_initial_router(model, infer_dataloader, i)
        else:
            # Recursive update
            current_R, Delta = train_subsequent_router(model, infer_dataloader, i, current_R, Delta)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    print(
        "-----------------------------------start training---------------------------------------"
    )
    train()
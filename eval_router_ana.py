import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Model
from transformers.models.llama import LlamaForCausalLM, LlamaModel
import torch
from torch.nn import CrossEntropyLoss
from typing import Optional, List, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

import logging

from transformers.cache_utils import Cache, DynamicCache

from transformers.utils import logging

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

from utils.data.data_utils import create_codetask_dataset, PromptDataset
from utils.data.data_collator import DataCollator
from utils.data.hf_task_specs import TASK_LIST
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import torch.nn.functional as F

feature_layers = 4
gamma = 10000
router_weights_path = './output_models/router_weights'
dataset_cache_path = './output_models/outputs_router_dataset_cache'

class NewQwen2ForCausalLM(Qwen2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config, task_number, gamma):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.model.layers = torch.nn.ModuleList(self.model.layers[:feature_layers])  # 仅取前4层进行分类

        self.fe = torch.nn.Linear(in_features=config.hidden_size, out_features=gamma)
        self.cls_head = torch.nn.Linear(in_features=gamma, out_features=task_number)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        hidden_mean = outputs[0].mean(dim=1)
        out = self.fe(hidden_mean)
        return self.cls_head(out)

    def MoeClassifier():
        pass


class NewLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config, task_number, gamma):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.model.layers = torch.nn.ModuleList(self.model.layers[:feature_layers])  # 仅取前4层进行分类

        self.fe = torch.nn.Linear(in_features=config.hidden_size, out_features=gamma)
        self.cls_head = torch.nn.Linear(in_features=gamma, out_features=task_number)
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
        # Only takes the output of first 4 layers of the model 

        hidden_mean = outputs[0].mean(dim=1)

        out = self.fe(hidden_mean)

        out = self.cls_head(out)
        return out
    
    def MoeClassifier():
        pass

def load_model_and_tokenizer(step, model_name_or_path='Qwen/Qwen2.5-Coder-1.5B'):
    if 'qwen' in model_name_or_path.lower():
        ModelClass = NewQwen2ForCausalLM
    else:
        ModelClass = NewLlamaForCausalLM

    model = ModelClass.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype="auto",
                # task_number=step+2,
                task_number=step+1,
                trust_remote_code=True,
                gamma=gamma
            )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    return model, tokenizer

def load_tokenizer(model_name_or_path='Qwen/Qwen2.5-Coder-1.5B'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    return tokenizer

def train():
    inference_tasks = TASK_LIST
    import numpy as np

    def eval_router(model, infer_dataloader, step):
        fe_weight = torch.load(f'{router_weights_path}/step{step}_fe_weight.pth', map_location=model.device).to(torch.float16)
        classifier_weight = torch.load(f'{router_weights_path}/step{step}_router_weight.pth', map_location=model.device).transpose(0, 1).to(torch.float16)
        model.cls_head.weight = torch.nn.Parameter(classifier_weight)
        model.fe.weight = torch.nn.Parameter(fe_weight)
        with torch.no_grad():
            count = 0
            correct = 0
            print(f'-----------------------start evaluation of step {step}-------------------')
            for steps, batch in enumerate(infer_dataloader):
                labels = batch['gts']
                input_ids = batch['input_ids']
                input_ids = input_ids.to('cuda')
                prediction = model(input_ids).to(torch.float32)

                if labels == [prediction.argmax().item()]:
                    correct += 1
                else:
                    print(f'prediction: {prediction.argmax().item()}, labels: {labels}')
                
                count += 1
                
            acc = correct / count
            print(f'step{step} has an acc of：{acc}')

    # for i in range(0, len(inference_tasks) - 1):
    for i in range(0, len(inference_tasks)):
        model, tokenizer = load_model_and_tokenizer(i)
        tokenizer.pad_token = tokenizer.eos_token

        # cur_inference_tasks = inference_tasks[0:i+2]
        cur_inference_tasks = inference_tasks[0:i+1]
        all_datasets = []
        for inference_task_id in range(len(cur_inference_tasks)):    # evaluation for previous tasks in a single round
            inference_task = inference_tasks[inference_task_id]

            # Prepare the data
            _, _, hf_test = create_codetask_dataset(inference_task, 42, -1, -1, -1)
            infer_dataset = PromptDataset(
                list(hf_test['prompt']),
                [inference_task_id for _ in hf_test]
            )

            all_datasets.append(infer_dataset)
        
        # continue
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
        infer_dataloader = DataLoader(infer_dataset,
                                        collate_fn=inf_data_collator,
                                        # sampler=infer_sampler,
                                        shuffle=True,
                                        batch_size=16)

        # Inference !
        # print("***** Start evaluation *****")
        eval_router(model, infer_dataloader, i)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    print(
        "-----------------------------------start router evaluation---------------------------------------"
    )
    train()
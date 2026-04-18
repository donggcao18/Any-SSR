# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers import LlamaForCausalLM, LlamaConfig

def get_transformer_layers(model):
    """
    Return the nn.ModuleList of decoder blocks/layers for common HF causal LMs,
    including OPT, LLaMA, and Qwen2 / Qwen2.5.

    Works with plain models and most PEFT-wrapped models.
    """
    m = model

    # Unwrap common wrappers gradually
    for attr in ["module", "base_model", "model"]:
        while hasattr(m, attr):
            nxt = getattr(m, attr)
            if nxt is m:
                break
            m = nxt

            # Stop early if we already found a known layout
            if hasattr(m, "decoder") and hasattr(m.decoder, "layers"):
                return m.decoder.layers          # OPT
            if hasattr(m, "layers"):
                return m.layers                  # LLaMA / Qwen base model
            if hasattr(m, "model") and hasattr(m.model, "layers"):
                return m.model.layers            # LlamaForCausalLM / Qwen2ForCausalLM

    # Final fallback checks
    if hasattr(m, "decoder") and hasattr(m.decoder, "layers"):
        return m.decoder.layers                  # OPT

    if hasattr(m, "layers"):
        return m.layers                          # direct base model

    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers                    # Qwen2ForCausalLM / LlamaForCausalLM

    raise AttributeError(f"Could not find transformer layers in object of type {type(model)}")

def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
        trust_remote_code=True)

    # llama use eos_token_id but not end_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    # compatible with OPT and llama2
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model

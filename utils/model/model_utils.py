# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from huggingface_hub import snapshot_download
from transformers.integrations.deepspeed import HfDeepSpeedConfig

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


def unwrap_model(model):
    """Return the underlying HF model from DeepSpeed/PEFT-style wrappers."""
    m = model
    seen = set()
    for attr in ["module", "base_model", "model"]:
        while hasattr(m, attr):
            if id(m) in seen:
                return m
            seen.add(id(m))
            nxt = getattr(m, attr)
            if nxt is m:
                break
            m = nxt
    return m


def is_encoder_decoder_config(config):
    return bool(getattr(config, "is_encoder_decoder", False))


def is_encoder_decoder_model(model):
    m = model
    seen = set()
    while m is not None and id(m) not in seen:
        seen.add(id(m))
        config = getattr(m, "config", None)
        if config is not None and is_encoder_decoder_config(config):
            return True
        next_model = None
        for attr in ["module", "base_model", "model"]:
            if hasattr(m, attr):
                candidate = getattr(m, attr)
                if candidate is not m:
                    next_model = candidate
                    break
        m = next_model
    return False


def get_hf_model_class(model_name_or_path, config=None):
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if is_encoder_decoder_config(config):
        return AutoModelForSeq2SeqLM
    return AutoModelForCausalLM


def get_input_embeddings(model):
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return embeddings

    m = unwrap_model(model)
    if hasattr(m, "get_input_embeddings"):
        embeddings = m.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    if hasattr(m, "model") and hasattr(m.model, "embed_tokens"):
        return m.model.embed_tokens
    if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "embed_tokens"):
        return m.model.decoder.embed_tokens
    if hasattr(m, "encoder") and hasattr(m.encoder, "embed_tokens"):
        return m.encoder.embed_tokens
    if hasattr(m, "shared"):
        return m.shared
    raise AttributeError(f"Could not find input embeddings in object of type {type(model)}")


def decode_generated_sequences(tokenizer, generate_ids, prompt_len=None, is_encoder_decoder=False):
    if not is_encoder_decoder and prompt_len is not None:
        generate_ids = generate_ids[:, prompt_len:]
    return tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def get_lora_target_modules(model_name_or_path, is_encoder_decoder=False):
    if is_encoder_decoder:
        return ["q", "v"]
    model_name = (model_name_or_path or "").lower()
    if "qwen" in model_name or "llama" in model_name:
        return ["q_proj", "v_proj"]
    return None


def get_peft_task_type(TaskType, is_encoder_decoder=False):
    return TaskType.SEQ_2_SEQ_LM if is_encoder_decoder else TaskType.CAUSAL_LM

def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if model_class is None or (
        is_encoder_decoder_config(model_config) and model_class is AutoModelForCausalLM
    ):
        model_class = get_hf_model_class(model_name_or_path, model_config)

    if disable_dropout:
        for dropout_attr in ["dropout", "dropout_rate", "attention_dropout", "activation_dropout"]:
            if hasattr(model_config, dropout_attr):
                setattr(model_config, dropout_attr, 0.0)
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

    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
        # llama use eos_token_id but not end_token_id
        model.config.end_token_id = tokenizer.eos_token_id
    if is_encoder_decoder_config(model.config):
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    else:
        # compatible with OPT, LLaMA, and Qwen decoder-only generation
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        elif model.config.eos_token_id is not None:
            model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model

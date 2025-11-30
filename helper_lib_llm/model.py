# helper_lib_llm/model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path

BASE_MODEL_NAME = "openai-community/gpt2"


def load_base_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model():
    tokenizer = load_base_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def load_finetuned_model(ckpt_dir: str):
    ckpt_path = Path(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(ckpt_path)
    model.eval()
    return model, tokenizer

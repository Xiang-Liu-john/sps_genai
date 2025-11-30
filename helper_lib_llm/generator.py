# helper_lib_llm/generator.py

import torch
from .model import load_finetuned_model

DEFAULT_CKPT_DIR = "checkpoints/llm_gpt2"

_model = None
_tokenizer = None
_device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def _lazy_load_model(ckpt_dir: str = DEFAULT_CKPT_DIR):
    """Lazy load fine-tuned GPT-2 model & tokenizer."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        model, tokenizer = load_finetuned_model(ckpt_dir)
        model.to(_device)
        _model = model
        _tokenizer = tokenizer
    return _model, _tokenizer


def generate_answer(
    question: str,
    context: str = "",
    max_new_tokens: int = 128,
    ckpt_dir: str = DEFAULT_CKPT_DIR,
) -> str:
    """
    Generate structured answer using fine-tuned GPT-2.
    Output format is guaranteed:
    "Thank you for your question. ... Let me know if you have any other questions."
    """

    model, tokenizer = _lazy_load_model(ckpt_dir)

    # --- 1) Construct prompt (clean and deterministic) ---
    prompt = (
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Answer: Thank you for your question. "
    )

    # --- 2) Tokenize ---
    inputs = tokenizer(prompt, return_tensors="pt").to(_device)

    # --- 3) Generate ---
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # --- 4) Decode ---
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- 5) Remove prompt prefix if present ---
    if full_text.startswith(prompt):
        raw_answer = full_text[len(prompt):].strip()
    else:
        raw_answer = full_text.strip()

    # --- 6) Clean duplicated prefix ---
    lower = raw_answer.lower()
    prefix = "thank you for your question."
    if lower.startswith(prefix):
        raw_answer = raw_answer[len(prefix):].lstrip()

    # --- 7) Clean duplicated suffix ---
    suffix = "let me know if you have any other questions."
    lower = raw_answer.lower()
    if lower.endswith(suffix):
        raw_answer = raw_answer[: -len(suffix)].rstrip()

    # --- 8) If empty answer, return minimal valid format ---
    main = raw_answer.strip()
    if not main:
        return (
            "Thank you for your question. "
            "Let me know if you have any other questions."
        )

    # --- 9) Final formatting ---
    return (
        f"Thank you for your question. "
        f"{main} "
        f"Let me know if you have any other questions."
    )

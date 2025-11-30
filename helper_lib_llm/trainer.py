# helper_lib_llm/trainer.py
import os
from pathlib import Path
from transformers import Trainer, TrainingArguments
import torch

from .model import load_base_model
from .data_loader import prepare_squad


# ===================================================================
# Toggle this for fast debugging vs full training
# True  → use small subset (quick training, for development)
# False → use full SQuAD dataset (for final submission)
# ===================================================================
DEBUG_MODE = True


def train_llm(
    output_dir: str = "checkpoints/llm_gpt2",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    learning_rate: float = 5e-5,
    max_length: int = 512,
):
    """
    Fine-tune GPT-2 on the SQuAD dataset.
    Supports both full training and lightweight debugging mode.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load pretrained GPT-2 + tokenizer
    model, tokenizer = load_base_model()

    # Load preprocessed SQuAD dataset
    train_dataset, eval_dataset = prepare_squad(tokenizer, max_length=max_length)

    # ================================================================
    # FAST MODE: Use only small subset of SQuAD for quick iterations
    # ================================================================
    if DEBUG_MODE:
        train_dataset = train_dataset.select(range(min(2000, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(200, len(eval_dataset))))

    print(f"Training samples: {len(train_dataset)}")
    print(f"Eval samples:     {len(eval_dataset)}")

    # ================================================================
    # Device selection: CUDA / MPS / CPU
    # ================================================================
    fp16 = False
    bf16 = False

    if torch.backends.cuda.is_built() and torch.cuda.is_available():
        device_type = "cuda"
        fp16 = True
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    print(f"Using device: {device_type}")

    # ================================================================
    # Training configuration
    # ================================================================
    training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    logging_steps=50,
    learning_rate=learning_rate,
    warmup_steps=50,
    weight_decay=0.01,
    fp16=fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    trainer.train()

    # Save model + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to: {Path(output_dir).absolute()}")


if __name__ == "__main__":
    train_llm()

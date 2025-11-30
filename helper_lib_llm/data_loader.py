# helper_lib_llm/data_loader.py
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

START_PREFIX = "Thank you for your question. "
END_SUFFIX = " Let me know if you have any other questions."

def format_example(example):
    prompt = (
        f"Question: {example['question']}\n"
        f"Context: {example['context']}\n"
        f"Answer: "
    )
    target = example["answer"]

    full_text = prompt + target

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=256,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def prepare_squad(tokenizer: PreTrainedTokenizerBase,
                  max_length: int = 512):
                  
    dataset = load_dataset("rajpurkar/squad")

    train_dataset = dataset["train"].map(format_example)
    val_dataset = dataset["validation"].map(format_example)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized_train = train_dataset.map(
        tokenize_fn, batched=True, remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        tokenize_fn, batched=True, remove_columns=val_dataset.column_names
    )

    # causal LM: labels = input_ids
    def add_labels(batch):
        batch["labels"] = batch["input_ids"].copy()
        return batch

    tokenized_train = tokenized_train.map(add_labels, batched=True)
    tokenized_val = tokenized_val.map(add_labels, batched=True)

    tokenized_train.set_format(type="torch")
    tokenized_val.set_format(type="torch")

    return tokenized_train, tokenized_val

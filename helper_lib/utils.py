import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(1) == targets).float().mean().item() * 100.0

import os
import torch


def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir: str):
    # Create directory if needed
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "loss": float(loss),
        "accuracy": float(accuracy),
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
    torch.save(checkpoint, checkpoint_path)

    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    torch.save(checkpoint, latest_path)

    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path: str, device="cpu", strict: bool = True):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore states
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)
    accuracy = checkpoint.get("accuracy", None)

    return epoch, loss, accuracy
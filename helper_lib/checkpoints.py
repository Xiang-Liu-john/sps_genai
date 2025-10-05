import os
import torch


def save_checkpoint(model, optimizer, epoch: int, loss: float, accuracy: float,
                    checkpoint_dir: str = 'checkpoints', is_best: bool = False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    path = os.path.join(checkpoint_dir, f"model_epoch_{epoch:03d}.pth")
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(state, best_path)


def load_checkpoint(model, optimizer, checkpoint_path: str, device: str = 'cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optim_state' in ckpt:
        optimizer.load_state_dict(ckpt['optim_state'])
    return ckpt.get('epoch', 0), ckpt.get('loss', None), ckpt.get('accuracy', None)

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .checkpoints import save_checkpoint


def train_model(model, train_loader: DataLoader, val_loader: DataLoader, criterion, optimizer,
                device: str = 'cpu', epochs: int = 10, checkpoint_dir: str = 'checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        val_loss, val_acc = 0.0, 0.0
        if val_loader is not None:
            model.eval()
            vloss, vcorrect, vtotal = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    vloss += loss.item() * x.size(0)
                    vcorrect += (logits.argmax(1) == y).sum().item()
                    vtotal += y.size(0)
            val_loss = vloss / vtotal
            val_acc = 100.0 * vcorrect / vtotal

        save_checkpoint(model, optimizer, epoch, val_loss or train_loss, val_acc or train_acc, checkpoint_dir)
        if (val_acc or train_acc) > best_acc:
            best_acc = val_acc or train_acc
            save_checkpoint(model, optimizer, epoch, val_loss or train_loss, val_acc or train_acc, checkpoint_dir, is_best=True)
    return model


def _vae_loss(recon, x, mu, logvar):
    bce = torch.nn.functional.binary_cross_entropy(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0)


def train_vae_model(model, data_loader: DataLoader, criterion=None, optimizer=None, device: str = 'cpu', epochs: int = 10):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, n = 0.0, 0
        for x, _ in tqdm(data_loader, desc=f"VAE Epoch {epoch}/{epochs}"):
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = (_vae_loss if criterion is None else criterion)(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        _ = total_loss / max(1, n)
    return model

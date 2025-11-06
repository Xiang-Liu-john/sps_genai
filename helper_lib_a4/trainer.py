import os, torch
from torch import optim
from tqdm import tqdm
from .data_loader import get_cifar10_loaders
from .model import EnergyCNN, energy_nce_loss, UNetSmall, DDPM

def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

class EnergyTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = EnergyCNN().to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        self.device = device

    def train(self, epochs=5, batch_size=128, ckpt='checkpoints/energy.pt'):
        train_loader, _, _ = get_cifar10_loaders(batch_size)
        for e in range(epochs):
            self.model.train()
            for x, _ in tqdm(train_loader, desc=f'Epoch {e+1}'):
                x = x.to(self.device)
                x_neg = x[torch.randperm(x.size(0))]
                e_pos = self.model(x)
                e_neg = self.model(x_neg)
                loss = energy_nce_loss(e_pos, e_neg)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        save_ckpt(self.model, ckpt)

class DiffusionTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.unet = UNetSmall().to(device)
        self.ddpm = DDPM(self.unet, device)
        self.opt = optim.AdamW(self.unet.parameters(), lr=2e-4)
        self.device = device

    def train(self, epochs=5, batch_size=128, ckpt='checkpoints/diffusion.pt'):
        train_loader, _, _ = get_cifar10_loaders(batch_size)
        for e in range(epochs):
            for x, _ in tqdm(train_loader, desc=f'Diffusion {e+1}'):
                x = (x.to(self.device) * 2 - 1)
                loss = self.ddpm.loss(x)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        save_ckpt(self.unet, ckpt)

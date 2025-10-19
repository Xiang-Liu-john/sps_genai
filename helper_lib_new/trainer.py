import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional


@torch.no_grad()
def sample_fixed_noise(generator, device, n=64, z_dim=100):
    z = torch.randn(n, z_dim, device=device)
    fake = generator(z)                        # (n,1,28,28)
    imgs = (fake + 1.0) / 2.0
    return imgs.clamp(0, 1)


def train_gan_mnist(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    z_dim: int = 100,
    lr: float = 2e-4,
    betas=(0.5, 0.999),
    ckpt_path: Optional[str] = None,
):
    G, D = generator.to(device), discriminator.to(device)
    G.train(); D.train()

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    bce   = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        for i, (real, _) in enumerate(dataloader):
            real = real.to(device)                     # (B,1,28,28)
            B = real.size(0)

            # ===== Train D =====
            D.zero_grad()
            label_real = torch.ones(B, 1, device=device)
            label_fake = torch.zeros(B, 1, device=device)

            out_real = D(real)
            loss_real = bce(out_real, label_real)

            z = torch.randn(B, z_dim, device=device)
            fake = G(z).detach()
            out_fake = D(fake)
            loss_fake = bce(out_fake, label_fake)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # ===== Train G =====
            G.zero_grad()
            z = torch.randn(B, z_dim, device=device)
            fake = G(z)
            out = D(fake)
            loss_G = bce(out, label_real)
            loss_G.backward()
            opt_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                      f"D: {loss_D.item():.4f}  G: {loss_G.item():.4f}")

        _ = sample_fixed_noise(G, device, n=64, z_dim=z_dim)

        if ckpt_path:
            torch.save({"G": G.state_dict(), "D": D.state_dict()}, ckpt_path)
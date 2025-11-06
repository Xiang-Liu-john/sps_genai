import torch
import torch.nn as nn
import torch.nn.functional as F

class EnergyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc(h)

def energy_nce_loss(e_pos, e_neg, temp=1.0):
    logits = torch.cat([-e_pos / temp, -e_neg / temp], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

# --- Diffusion simplified ---
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()
        )
    def forward(self, x): return self.seq(x)

class UNetSmall(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_channels + 1, base)
        self.down2 = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.mid = DoubleConv(base * 2, base * 2)
        self.up = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.final = nn.Conv2d(base, 3, 1)

    def add_t(self, x, t):
        b, _, h, w = x.shape
        t = t.view(b,1,1,1).float() / 1000
        return torch.cat([x, t.expand(b,1,h,w)], 1)

    def forward(self, x, t):
        x = self.add_t(x, t)
        d = self.down1(x)
        d2 = self.down2(d)
        m = self.mid(d2)
        u = self.up(m)
        return self.final(u)

class DDPM:
    def __init__(self, model, device='cpu', timesteps=1000):
        self.model = model
        self.device = device
        self.T = timesteps
        beta = torch.linspace(1e-4, 0.02, timesteps, device=device)
        self.a = torch.cumprod(1 - beta, dim=0)
        self.sa = torch.sqrt(self.a)
        self.so = torch.sqrt(1 - self.a)

    def loss(self, x0):
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=self.device)
        eps = torch.randn_like(x0)
        xt = self.sa[t].view(b,1,1,1)*x0 + self.so[t].view(b,1,1,1)*eps
        pred = self.model(xt, t)
        return F.mse_loss(pred, eps)

    @torch.no_grad()
    def sample(self, n=4, shape=(3,32,32)):
        x = torch.randn((n,*shape), device=self.device)
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps = self.model(x, tt)
            a_t = self.a[t]
            a_prev = self.a[t-1] if t>0 else torch.tensor(1.0, device=self.device)
            beta_t = 1 - a_t/a_prev
            x = (1/torch.sqrt(1-beta_t))*(x - beta_t/torch.sqrt(1-a_t)*eps)
            if t>0: x += torch.sqrt(beta_t)*torch.randn_like(x)
        return x.clamp(-1,1)

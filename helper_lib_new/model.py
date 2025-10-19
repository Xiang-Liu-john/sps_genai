import torch
import torch.nn as nn


class GeneratorMNIST(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, (nn.BatchNorm2d,)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        x = self.fc(z)                         # (B, 7*7*128)
        x = x.view(-1, 128, 7, 7)              # (B,128,7,7)
        x = self.net(x)                        # (B,1,28,28)
        return x


class DiscriminatorMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, (nn.BatchNorm2d,)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
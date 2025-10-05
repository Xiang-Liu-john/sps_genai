import torch
import matplotlib.pyplot as plt


def generate_samples(model, device: str = 'cpu', num_samples: int = 10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.fc_mu.out_features, device=device)
        imgs = model.decode(z).cpu()
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        plt.imshow(imgs[i].permute(1, 2, 0).numpy())
    plt.tight_layout()
    plt.show()

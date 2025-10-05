import torch
from torchvision import datasets, transforms


def get_data_loader(data_dir: str, batch_size: int = 32, train: bool = True):
    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = datasets.CIFAR10(
        root=data_dir, train=train, transform=tfm, download=True
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)
    return loader

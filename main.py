# main.py
import os
import torch
from helper_lib_new.data_loader import get_data_loader
from helper_lib_new.model import GeneratorMNIST, DiscriminatorMNIST
from helper_lib_new.trainer import train_gan_mnist

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    dataloader = get_data_loader(
        data_dir="./data",
        batch_size=128,
        train=True,
        dataset="MNIST",
        transform=None
    )

    generator = GeneratorMNIST()
    discriminator = DiscriminatorMNIST()

    train_gan_mnist(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        epochs=20,
        ckpt_path="./checkpoints/gan_mnist.pth"
    )

    print("GAN training is done")

if __name__ == "__main__":
    main()

import argparse
import torch
from helper_lib_a4.trainer import EnergyTrainer, DiffusionTrainer

def main():
    parser = argparse.ArgumentParser(description="Train Energy or Diffusion Model on CIFAR-10")
    parser.add_argument('--model', type=str, choices=['energy', 'diffusion'], required=True,
                        help="Select which model to train: 'energy' or 'diffusion'")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'energy':
        print("Training Energy Model on CIFAR-10 ...")
        trainer = EnergyTrainer(device=device)
        trainer.train(epochs=args.epochs, batch_size=args.bs,
                      ckpt=f"{args.ckpt_dir}/energy.pt")
        print("Energy Model training complete! Saved to checkpoints/energy.pt")

    elif args.model == 'diffusion':
        print("Training Diffusion Model on CIFAR-10 ...")
        trainer = DiffusionTrainer(device=device)
        trainer.train(epochs=args.epochs, batch_size=args.bs,
                      ckpt=f"{args.ckpt_dir}/diffusion.pt")
        print("Diffusion Model training complete! Saved to checkpoints/diffusion.pt")

if __name__ == "__main__":
    main()

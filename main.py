from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.checkpoints import load_checkpoint
import torch.nn as nn
import torch.optim as optim

def main():
    # Load data
    train_loader = get_data_loader('data/train', batch_size=64)
    val_loader = get_data_loader('data/val', batch_size=64, train=False)
    test_loader = get_data_loader('data/test', batch_size=64, train=False)

    # Initialize model and training components
    model = get_model("CNN")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train from scratch with checkpoint saving
    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        epochs=10, checkpoint_dir='checkpoints'
    )

    # Evaluate final model
    avg_loss, accuracy = evaluate_model(trained_model, test_loader, criterion)
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

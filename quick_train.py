import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--bs', type=int, default=64)
args = parser.parse_args()

train_loader = get_data_loader(args.data, batch_size=args.bs, train=True)
val_loader = get_data_loader(args.data, batch_size=args.bs, train=False)

model = get_model('CNN')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

_ = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=args.epochs, checkpoint_dir='checkpoints')
print('Training complete. Best model saved at checkpoints/best.pth')

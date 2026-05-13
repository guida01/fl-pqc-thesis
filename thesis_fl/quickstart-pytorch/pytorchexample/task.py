"""fl_pqc: A Flower / PyTorch app."""

import io
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST — small by design, thesis focus is signature overhead."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc   = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load a partition of MNIST for a given client."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset   = datasets.MNIST("./data", train=True, download=True, transform=transform)
    split     = len(dataset) // num_partitions
    subset    = Subset(dataset, range(partition_id * split, (partition_id + 1) * split))
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def train(net, trainloader, epochs, lr, device):
    """Train the model for a given number of epochs."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()
    return {}


def weights_to_bytes(state_dict: dict) -> bytes:
    """Serialize model state dict to bytes — payload to be signed and verified."""
    buf    = io.BytesIO()
    arrays = [v.cpu().numpy() for v in state_dict.values()]
    np.save(buf, np.array(arrays, dtype=object), allow_pickle=True)
    return buf.getvalue()
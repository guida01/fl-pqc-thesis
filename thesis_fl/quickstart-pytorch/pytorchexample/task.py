"""fl_pqc: A Flower / PyTorch app."""

import io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2023, 0.1994, 0.2010)


class CIFAR10CNN(nn.Module):
    """~620K-param CNN for CIFAR-10 — larger payload stresses PQC signing realistically."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


def load_data(partition_id: int, num_partitions: int, batch_size: int, run_number: int = 1) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    split   = len(dataset) // num_partitions
    subset  = Subset(dataset, range(partition_id * split, (partition_id + 1) * split))
    g = torch.Generator()
    g.manual_seed(42 + run_number)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, generator=g)


def train(net, trainloader, epochs, lr, device) -> float:
    """Train for `epochs` local epochs; returns avg loss of the last epoch."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    total_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        total_loss, n_batches = 0.0, 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
    return total_loss / n_batches if n_batches else 0.0


def weights_to_bytes(state_dict: dict) -> bytes:
    """Serialize model state dict to bytes — payload signed and verified."""
    buf    = io.BytesIO()
    arrays = [v.cpu().numpy() for v in state_dict.values()]
    np.save(buf, np.array(arrays, dtype=object), allow_pickle=True)
    return buf.getvalue()

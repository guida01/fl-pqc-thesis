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
_NUM_CLASSES = 10

# Fixed regardless of run_number/partition_id: partitions must be identical
# across runs and across signature schemes, so that only the seeds that are
# SUPPOSED to vary (model init, DataLoader shuffle — both keyed on
# run_number) can affect training. A contiguous index slice (the previous
# approach) does not guarantee balanced classes; a fixed-seed shuffle does,
# in expectation, since CIFAR-10's 50 000 training images are already
# class-balanced (5 000 per class).
_PARTITION_SEED = 42


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


def _partition_indices(num_examples: int, num_partitions: int) -> list:
    """Deterministic shuffle-then-split: the same `num_partitions` index
    groups every time, regardless of run_number or partition_id, so
    partitions are identical across runs and across signature schemes."""
    rng = np.random.default_rng(_PARTITION_SEED)
    shuffled = rng.permutation(num_examples)
    split = num_examples // num_partitions
    return [shuffled[i * split:(i + 1) * split] for i in range(num_partitions)]


def load_data(partition_id: int, num_partitions: int, batch_size: int, run_number: int = 1) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    partition_idx = _partition_indices(len(dataset), num_partitions)[partition_id]
    subset = Subset(dataset, partition_idx.tolist())
    g = torch.Generator()
    g.manual_seed(42 + run_number)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, generator=g)


def report_class_distribution(num_partitions: int) -> str:
    """Print and return the per-partition class-count table for the
    CIFAR-10 training set, using the exact same deterministic partitioning
    as load_data(). Intended to be called once at startup so the printed
    numbers can be cited directly."""
    targets = np.asarray(datasets.CIFAR10("./data", train=True, download=True).targets)
    partitions = _partition_indices(len(targets), num_partitions)

    lines = [
        f"Class distribution per partition ({num_partitions} partitions, "
        f"seed={_PARTITION_SEED}, {len(targets)} training images):",
        "partition  " + "  ".join(f"class{c}" for c in range(_NUM_CLASSES)) + "    total",
    ]
    for pid, idx in enumerate(partitions):
        counts = np.bincount(targets[idx], minlength=_NUM_CLASSES)
        row = f"{pid:^9d}  " + "  ".join(f"{c:6d}" for c in counts) + f"  {counts.sum():6d}"
        lines.append(row)
    text = "\n".join(lines)
    print(text)
    return text


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

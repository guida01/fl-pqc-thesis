import flwr as fl
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from fl_client import SignedFlowerClient
from fl_server import SignedFedAvg

SCHEMES      = ["RSA-2048", "ECDSA-256", "Dilithium2", "Dilithium3", "Falcon-512"]
NUM_CLIENTS  = 5
NUM_ROUNDS   = 3
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_mnist_splits(num_clients):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    split_size = len(dataset) // num_clients
    loaders = []
    for i in range(num_clients):
        subset = Subset(dataset, range(i * split_size, (i + 1) * split_size))
        loaders.append(DataLoader(subset, batch_size=32, shuffle=True))
    return loaders

def run_scheme(scheme):
    print(f"\n{'='*60}")
    print(f"  Running scheme: {scheme}")
    print(f"{'='*60}")

    results_path = os.path.join(RESULTS_DIR, f"{scheme.replace('/', '_')}.csv")
    train_loaders = load_mnist_splits(NUM_CLIENTS)

    def client_fn(cid):
        return SignedFlowerClient(
            client_id=int(cid),
            train_loader=train_loaders[int(cid)],
            scheme=scheme
        )

    strategy = SignedFedAvg(
        scheme=scheme,
        results_path=results_path,
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

def plot_results():
    all_data = []
    for scheme in SCHEMES:
        path = os.path.join(RESULTS_DIR, f"{scheme.replace('/', '_')}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["scheme"] = scheme
            all_data.append(df)

    if not all_data:
        print("No results to plot yet.")
        return

    df = pd.concat(all_data)
    summary = df.groupby("scheme").agg(
        mean_sign_time=("sign_time", "mean"),
        mean_verify_time=("verify_time", "mean"),
        mean_sig_size=("sig_size", "mean"),
        mean_pubkey_size=("pubkey_size", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("PQC vs Classical Digital Signatures in Federated Learning", fontsize=13)

    axes[0].bar(summary["scheme"], summary["mean_sign_time"] * 1000)
    axes[0].set_title("Mean signing time (ms)")
    axes[0].set_ylabel("milliseconds")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(summary["scheme"], summary["mean_verify_time"] * 1000, color="orange")
    axes[1].set_title("Mean verification time (ms)")
    axes[1].set_ylabel("milliseconds")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(summary["scheme"], summary["mean_sig_size"], color="green")
    axes[2].set_title("Mean signature size (bytes)")
    axes[2].set_ylabel("bytes")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    for scheme in SCHEMES:
        run_scheme(scheme)
    plot_results()
    print("\nDone! Check the results/ folder.")

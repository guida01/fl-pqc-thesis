import os
os.environ["RAY_DEDUP_LOGS"] = "0"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)

import flwr as fl
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.common import Context

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fl_client import SignedFlowerClient, load_mnist_splits
from fl_server import SignedFedAvg
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

## para testar rapidamente
SCHEMES     = ["ML-DSA-65"]  
NUM_CLIENTS = 5
NUM_ROUNDS  = 3

RESULTS_DIR = "results_test"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_scheme(scheme):
    print(f"\n{'='*60}")
    print(f"  [TESTE] Running scheme: {scheme}")
    print(f"{'='*60}")

    results_path = os.path.join(RESULTS_DIR, f"{scheme.replace('/', '_')}.csv")
    train_loaders = load_mnist_splits(NUM_CLIENTS)

    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        return SignedFlowerClient(
            client_id=cid,
            train_loader=train_loaders[cid],
            scheme=scheme
        ).to_client()

    def server_fn(context: Context):
        strategy = SignedFedAvg(
            scheme=scheme,
            results_path=results_path,
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
        )
        config = ServerConfig(num_rounds=NUM_ROUNDS)
        return ServerAppComponents(strategy=strategy, config=config)

    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )

if __name__ == "__main__":
    print("TESTE RÁPIDO: 1 scheme, 5 clients, 3 rounds")
    run_scheme("ML-DSA-65")
    print("\n Teste completo! Verifica results_test/")
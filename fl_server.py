import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays
import numpy as np
import io
import time
import csv
import os
from signature_manager import SignatureManager

def weights_to_bytes(parameters):
    arrays = parameters_to_ndarrays(parameters)
    buf = io.BytesIO()
    np.save(buf, np.array(arrays, dtype=object), allow_pickle=True)
    return buf.getvalue()


class SignedFedAvg(FedAvg):

    def __init__(self, scheme, results_path, **kwargs):
        super().__init__(**kwargs)
        self.scheme = scheme
        self.results_path = results_path

        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "client_id", "scheme",
                "sign_time", "verify_time",
                "sig_size", "pubkey_size",
                "verified"
            ])

    def aggregate_fit(self, server_round, results, failures):
        valid_results = []

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            signature  = bytes.fromhex(metrics["signature"])
            public_key = bytes.fromhex(metrics["public_key"])
            payload    = weights_to_bytes(fit_res.parameters)

            verifier = SignatureManager(self.scheme)
            is_valid, verify_time = verifier.verify(payload, signature, public_key)

            client_id = metrics.get("client_id", "?")
            print(f"[Server round {server_round}] Client {client_id}: verified={is_valid} in {verify_time:.4f}s")

            with open(self.results_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    server_round, client_id, self.scheme,
                    metrics["sign_time"], verify_time,
                    metrics["sig_size"], metrics["pubkey_size"],
                    is_valid
                ])

            if is_valid:
                valid_results.append((client_proxy, fit_res))
            else:
                print(f"[Server round {server_round}] REJECTED update from client {client_id}!")

        return super().aggregate_fit(server_round, valid_results, failures)

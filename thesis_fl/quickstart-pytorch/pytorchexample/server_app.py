"""fl_pqc: A Flower / PyTorch app."""

import csv
import os
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import CIFAR10CNN, weights_to_bytes
from pytorchexample.signature_manager import SignatureManager
app = ServerApp()

_CSV_HEADER = [
    "run", "round", "node_id", "scheme",
    "keygen_time", "sign_time", "verify_time",
    "train_time", "train_loss",
    "payload_size", "sig_size", "pubkey_size", "verified",
]


class SignedFedAvg(FedAvg):

    def __init__(self, scheme: str, results_path: str, run_number: int, **kwargs):
        super().__init__(**kwargs)
        self.scheme      = scheme
        self.results_path = results_path
        self.run_number  = run_number

        # run 1 always starts fresh; subsequent runs append
        mode = "w" if run_number == 1 else "a"
        with open(results_path, mode, newline="") as f:
            if run_number == 1:
                csv.writer(f).writerow(_CSV_HEADER)

    def aggregate_train(self, server_round, train_replies):
        valid_replies = []

        for reply in train_replies:
            node_id = reply.metadata.src_node_id
            if not reply.has_content():
                print(f"  Node {node_id}: empty reply — skipping")
                continue

            sig_record = reply.content["signature"]
            signature  = bytes.fromhex(sig_record["signature"])
            public_key = bytes.fromhex(sig_record["public_key"])

            # Always extract state_dict — needed for NaN check and payload reconstruction
            state_dict = reply.content["arrays"].to_torch_state_dict()

            if self.scheme == "no_signature":
                is_valid    = True
                verify_time = 0.0
            else:
                payload = (
                    weights_to_bytes(state_dict)
                    + server_round.to_bytes(4, "big")
                    + node_id.to_bytes(8, "big")
                )
                is_valid, verify_time = SignatureManager(self.scheme).verify(
                    payload, signature, public_key
                )

            # NaN weight check — prevent FedAvg poisoning by a diverged client
            has_nan = any(torch.isnan(v).any() for v in state_dict.values())

            m = reply.content["metrics"]
            aggregate = is_valid and not has_nan
            print(f"  Node {node_id}: verified={is_valid} nan={has_nan} ({verify_time:.4f}s)")

            with open(self.results_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.run_number, server_round, node_id, self.scheme,
                    m["keygen_time"], m["sign_time"], verify_time,
                    m["train_time"], m["train_loss"],
                    m["payload_size"], m["sig_size"], m["pubkey_size"],
                    aggregate,
                ])

            if aggregate:
                valid_replies.append(reply)
            elif has_nan:
                print(f"  Node {node_id}: REJECTED — NaN weights detected!")
            else:
                print(f"  Node {node_id}: REJECTED — invalid signature!")

        return super().aggregate_train(server_round, valid_replies)


@app.main()
def main(grid: Grid, context: Context) -> None:
    scheme      = context.run_config["scheme"]
    num_rounds  = context.run_config["num-server-rounds"]
    run_number  = int(context.run_config.get("run-number", 1))
    results_dir = str(context.run_config["results-dir"])
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{scheme.replace('/', '_')}.csv")

    num_supernodes = int(context.run_config.get("num-supernodes", 5))
    strategy = SignedFedAvg(
        scheme=scheme,
        results_path=results_path,
        run_number=run_number,
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=num_supernodes,
        min_available_nodes=num_supernodes,
    )

    torch.manual_seed(run_number * 42)
    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(CIFAR10CNN().state_dict()),
        train_config=ConfigRecord({"lr": context.run_config["learning-rate"]}),
        num_rounds=num_rounds,
    )

    print(f"\nDone. Results saved to {results_path}")

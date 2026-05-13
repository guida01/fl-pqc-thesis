"""fl_pqc: A Flower / PyTorch app."""

import csv
import os
from flwr.app import ArrayRecord, ConfigRecord, Context, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import SimpleCNN, weights_to_bytes
from pytorchexample.signature_manager import SignatureManager

RESULTS_DIR = "results"
app = ServerApp()


class SignedFedAvg(FedAvg):
    """FedAvg with signature verification before aggregation."""

    def __init__(self, scheme: str, results_path: str, **kwargs):
        super().__init__(**kwargs)
        self.scheme = scheme
        self.results_path = results_path

        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(results_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "round", "node_id", "scheme",
                "sign_time", "verify_time",
                "sig_size", "pubkey_size", "verified"
            ])

    def aggregate_train(self, server_round, train_replies):
        valid_replies = []

        for reply in train_replies:
            node_id    = reply.metadata.src_node_id
            sig_record = reply.content["signature"]
            signature  = bytes.fromhex(sig_record["signature"])
            public_key = bytes.fromhex(sig_record["public_key"])
            state_dict = reply.content["arrays"].to_torch_state_dict()
            payload    = weights_to_bytes(state_dict)

            is_valid, verify_time = SignatureManager(self.scheme).verify(
                payload, signature, public_key
            )

            metrics = reply.content["metrics"]
            print(f"  Node {node_id}: verified={is_valid} ({verify_time:.4f}s)")

            with open(self.results_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    server_round, node_id, self.scheme,
                    metrics["sign_time"], verify_time,
                    metrics["sig_size"], metrics["pubkey_size"],
                    is_valid
                ])

            if is_valid:
                valid_replies.append(reply)
            else:
                print(f"  Node {node_id}: REJECTED — invalid signature!")

        return super().aggregate_train(server_round, valid_replies)


@app.main()
def main(grid: Grid, context: Context) -> None:
    scheme     = context.run_config["scheme"]
    num_rounds = context.run_config["num-server-rounds"]
    results_path = os.path.join(RESULTS_DIR, f"{scheme.replace('/', '_')}.csv")

    strategy = SignedFedAvg(
        scheme=scheme,
        results_path=results_path,
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=2,
        min_available_nodes=2,
    )

    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(SimpleCNN().state_dict()),
        train_config=ConfigRecord({"lr": context.run_config["learning-rate"]}),
        num_rounds=num_rounds,
    )

    print(f"\nDone. Results saved to {results_path}")
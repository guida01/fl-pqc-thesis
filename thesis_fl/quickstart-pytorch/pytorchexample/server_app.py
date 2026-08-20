"""fl_pqc: A Flower / PyTorch app."""

import csv
import os
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.signature_manager import SignatureManager
from pytorchexample.task import (
    CIFAR10CNN,
    DEVICE,
    evaluate,
    load_test_data,
    report_class_distribution,
    weights_to_bytes,
)


app = ServerApp()


_CSV_HEADER = [
    "run",
    "round",
    "node_id",
    "scheme",
    "keygen_time",
    "client_serialize_time",
    "sign_time",
    "server_serialize_time",
    "verify_time",
    "server_verify_total_time",
    "train_time",
    "train_loss",
    "signed_payload_size",
    "sig_size",
    "pubkey_size",
    "num_examples",
    "sig_valid",
    "has_nan",
]

_EVAL_CSV_HEADER = [
    "run",
    "round",
    "accuracy",
    "loss",
]


class SignedFedAvg(FedAvg):

    def __init__(
        self,
        scheme: str,
        results_path: str,
        run_number: int,
        eval_results_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.scheme = scheme
        self.results_path = results_path
        self.run_number = run_number
        self.eval_results_path = eval_results_path

        # Run 1 always starts fresh; subsequent runs append.
        mode = "w" if run_number == 1 else "a"

        with open(results_path, mode, newline="") as f:
            if run_number == 1:
                csv.writer(f).writerow(_CSV_HEADER)

        if eval_results_path:
            with open(eval_results_path, mode, newline="") as f:
                if run_number == 1:
                    csv.writer(f).writerow(_EVAL_CSV_HEADER)

            # Loaded once per run and reused across all round evaluations.
            self._testloader = load_test_data()

    def aggregate_train(self, server_round, train_replies):
        valid_replies = []

        for reply in train_replies:
            node_id = reply.metadata.src_node_id

            if not reply.has_content():
                print(f"  Node {node_id}: empty reply — skipping")
                continue

            sig_record = reply.content["signature"]
            signature = sig_record["signature"]
            public_key = sig_record["public_key"]

            # Always extract the state_dict: it is needed for the NaN check
            # and, for signed schemes, to reconstruct the signed payload.
            state_dict = reply.content["arrays"].to_torch_state_dict()

            if self.scheme == "no_signature":
                is_valid = True
                server_serialize_time = 0.0
                verify_time = 0.0
                server_verify_total_time = 0.0
            else:
                # Reconstruct exactly the payload produced by the client.
                t0 = time.perf_counter()
                payload = (
                    weights_to_bytes(state_dict)
                    + server_round.to_bytes(4, "big")
                    + node_id.to_bytes(8, "big")
                )
                server_serialize_time = time.perf_counter() - t0

                # Total server-side verification processing includes creation
                # of the verifier object plus SignatureManager.verify().
                # verify_time itself remains the timing reported internally by
                # SignatureManager for the backend verification operation.
                t0 = time.perf_counter()
                is_valid, verify_time = SignatureManager(
                    self.scheme,
                    generate_keypair=False,
                ).verify(
                    payload,
                    signature,
                    public_key,
                )
                server_verify_total_time = time.perf_counter() - t0

            # Prevent FedAvg aggregation of client updates containing NaNs.
            has_nan = any(
                torch.isnan(value).any()
                for value in state_dict.values()
            )

            metrics = reply.content["metrics"]
            aggregate = is_valid and not has_nan

            print(
                f"  Node {node_id}: "
                f"sig_valid={is_valid} "
                f"has_nan={has_nan} "
                f"(verify={verify_time:.4f}s, "
                f"total={server_verify_total_time:.4f}s)"
            )

            with open(self.results_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        self.run_number,
                        server_round,
                        node_id,
                        self.scheme,
                        metrics["keygen_time"],
                        metrics["client_serialize_time"],
                        metrics["sign_time"],
                        server_serialize_time,
                        verify_time,
                        server_verify_total_time,
                        metrics["train_time"],
                        metrics["train_loss"],
                        metrics["signed_payload_size"],
                        metrics["sig_size"],
                        metrics["pubkey_size"],
                        metrics["num-examples"],
                        is_valid,
                        has_nan,
                    ]
                )

            if aggregate:
                valid_replies.append(reply)
            elif has_nan:
                print(
                    f"  Node {node_id}: "
                    "REJECTED — NaN weights detected!"
                )
            else:
                print(
                    f"  Node {node_id}: "
                    "REJECTED — invalid signature!"
                )

        return super().aggregate_train(
            server_round,
            valid_replies,
        )

    def central_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
    ) -> MetricRecord | None:
        """Evaluate the global model centrally on the CIFAR-10 test set.

        This hook runs server-side only. It is called once before round 1
        (server_round=0) and once after each completed training round.
        """
        model = CIFAR10CNN()
        model.load_state_dict(arrays.to_torch_state_dict())

        loss, accuracy = evaluate(
            model,
            self._testloader,
            DEVICE,
        )

        with open(self.eval_results_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.run_number,
                    server_round,
                    accuracy,
                    loss,
                ]
            )

        return MetricRecord(
            {
                "central_accuracy": accuracy,
                "central_loss": loss,
            }
        )


@app.main()
def main(grid: Grid, context: Context) -> None:
    scheme = context.run_config["scheme"]
    num_rounds = context.run_config["num-server-rounds"]
    run_number = int(context.run_config.get("run-number", 1))
    results_dir = str(context.run_config["results-dir"])

    os.makedirs(results_dir, exist_ok=True)

    safe_scheme_name = scheme.replace("/", "_")
    results_path = os.path.join(
        results_dir,
        f"{safe_scheme_name}.csv",
    )

    eval_central = bool(
        context.run_config.get("eval-central", False)
    )

    eval_results_path = (
        os.path.join(
            results_dir,
            f"{safe_scheme_name}_eval.csv",
        )
        if eval_central
        else None
    )

    num_supernodes = int(
        context.run_config.get("num-supernodes", 5)
    )

    report_class_distribution(num_supernodes)

    strategy = SignedFedAvg(
        scheme=scheme,
        results_path=results_path,
        run_number=run_number,
        eval_results_path=eval_results_path,
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=num_supernodes,
        min_available_nodes=num_supernodes,
    )

    torch.manual_seed(run_number * 42)

    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(
            CIFAR10CNN().state_dict()
        ),
        train_config=ConfigRecord(
            {
                "lr": context.run_config["learning-rate"],
            }
        ),
        num_rounds=num_rounds,
        evaluate_fn=(
            strategy.central_evaluate
            if eval_central
            else None
        ),
    )

    print(f"\nDone. Results saved to {results_path}")
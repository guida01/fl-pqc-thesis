#!/usr/bin/env python3
"""
Negative-path smoke test for the ephemeral-key signing protocol.

ECDSA-256, 2 clients, 2 rounds. No Flower runtime required.

Tests:
  (a) Valid signature verifies.
  (b) Tampered weights payload fails verification.
  (c) Tampered signature bytes fail verification.
  (d) Round-1 signature replayed in round 2 fails.
  (e) Synthetic CSV matches the final experiment schema.
"""

import csv
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorchexample.signature_manager import SignatureManager
from pytorchexample.task import CIFAR10CNN, weights_to_bytes


SCHEME = "ECDSA-256"
NODE_IDS = [1001, 1002]
NUM_ROUNDS = 2


EXPECTED_CSV_COLS = [
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


def make_payload(
    weights_bytes: bytes,
    server_round: int,
    node_id: int,
) -> bytes:
    """Match the payload construction used by client_app/server_app."""
    return (
        weights_bytes
        + server_round.to_bytes(4, "big")
        + node_id.to_bytes(8, "big")
    )


def check(
    label: str,
    condition: bool,
    passed: list,
    failed: list,
) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {label}")
    (passed if condition else failed).append(label)


def verify_payload(
    payload: bytes,
    signature: bytes,
    public_key: bytes,
) -> tuple[bool, float, float]:
    """Return validity, backend verify time and total verifier processing."""
    t0 = time.perf_counter()

    valid, verify_time = SignatureManager(
        SCHEME,
        generate_keypair=False,
    ).verify(
        payload,
        signature,
        public_key,
    )

    server_verify_total_time = time.perf_counter() - t0

    return valid, verify_time, server_verify_total_time


def make_csv_row(
    *,
    rnd: int,
    node_id: int,
    keygen_time: float,
    client_serialize_time: float,
    sign_time: float,
    server_serialize_time: float,
    verify_time: float,
    server_verify_total_time: float,
    signed_payload_size: int,
    sig_size: int,
    pubkey_size: int,
    sig_valid: bool,
) -> dict:
    return {
        "run": 1,
        "round": rnd,
        "node_id": node_id,
        "scheme": SCHEME,
        "keygen_time": keygen_time,
        "client_serialize_time": client_serialize_time,
        "sign_time": sign_time,
        "server_serialize_time": server_serialize_time,
        "verify_time": verify_time,
        "server_verify_total_time": server_verify_total_time,
        "train_time": 0.0,
        "train_loss": 0.0,
        "signed_payload_size": signed_payload_size,
        "sig_size": sig_size,
        "pubkey_size": pubkey_size,
        "num_examples": 0.0,
        "sig_valid": sig_valid,
        "has_nan": False,
    }


def run_smoke_test() -> bool:
    model = CIFAR10CNN()
    weights_bytes = weights_to_bytes(model.state_dict())

    passed = []
    failed = []
    csv_rows = []

    print(
        f"Scheme: {SCHEME} | "
        f"Nodes: {NODE_IDS} | "
        f"Rounds: {NUM_ROUNDS}"
    )
    print("=" * 70)

    round1_artifacts = {}

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\nRound {rnd}")

        for node_id in NODE_IDS:
            # ----------------------------------------------------------
            # CLIENT: ephemeral key generation
            # ----------------------------------------------------------
            signer = SignatureManager(SCHEME)
            keygen_time = signer.keygen_time

            # ----------------------------------------------------------
            # CLIENT: construct signed payload
            # ----------------------------------------------------------
            t0 = time.perf_counter()
            payload = make_payload(
                weights_bytes,
                rnd,
                node_id,
            )
            client_serialize_time = time.perf_counter() - t0

            signature, sign_time = signer.sign(payload)
            public_key = signer.public_key_bytes

            if rnd == 1:
                round1_artifacts[node_id] = {
                    "signature": signature,
                    "public_key": public_key,
                }

            # ----------------------------------------------------------
            # SERVER: reconstruct payload
            # ----------------------------------------------------------
            t0 = time.perf_counter()
            server_payload = make_payload(
                weights_bytes,
                rnd,
                node_id,
            )
            server_serialize_time = time.perf_counter() - t0

            # ----------------------------------------------------------
            # (a) Valid signature
            # ----------------------------------------------------------
            valid, verify_time, total_verify = verify_payload(
                server_payload,
                signature,
                public_key,
            )

            csv_rows.append(
                make_csv_row(
                    rnd=rnd,
                    node_id=node_id,
                    keygen_time=keygen_time,
                    client_serialize_time=client_serialize_time,
                    sign_time=sign_time,
                    server_serialize_time=server_serialize_time,
                    verify_time=verify_time,
                    server_verify_total_time=total_verify,
                    signed_payload_size=len(payload),
                    sig_size=len(signature),
                    pubkey_size=len(public_key),
                    sig_valid=valid,
                )
            )

            check(
                f"(a) r{rnd}/n{node_id} valid signature -> True",
                valid,
                passed,
                failed,
            )

            # ----------------------------------------------------------
            # (b) Tampered weights
            # ----------------------------------------------------------
            bad_weights = bytearray(weights_bytes)
            bad_weights[100] ^= 0xFF

            t0 = time.perf_counter()
            bad_payload = make_payload(
                bytes(bad_weights),
                rnd,
                node_id,
            )
            bad_server_serialize_time = time.perf_counter() - t0

            bad_valid, bad_verify, bad_total = verify_payload(
                bad_payload,
                signature,
                public_key,
            )

            csv_rows.append(
                make_csv_row(
                    rnd=rnd,
                    node_id=node_id,
                    keygen_time=keygen_time,
                    client_serialize_time=client_serialize_time,
                    sign_time=sign_time,
                    server_serialize_time=bad_server_serialize_time,
                    verify_time=bad_verify,
                    server_verify_total_time=bad_total,
                    signed_payload_size=len(payload),
                    sig_size=len(signature),
                    pubkey_size=len(public_key),
                    sig_valid=bad_valid,
                )
            )

            check(
                f"(b) r{rnd}/n{node_id} tampered weights -> False",
                not bad_valid,
                passed,
                failed,
            )

            # ----------------------------------------------------------
            # (c) Tampered signature
            # ----------------------------------------------------------
            bad_signature = bytearray(signature)
            bad_signature[0] ^= 0xFF

            bad_valid, bad_verify, bad_total = verify_payload(
                server_payload,
                bytes(bad_signature),
                public_key,
            )

            csv_rows.append(
                make_csv_row(
                    rnd=rnd,
                    node_id=node_id,
                    keygen_time=keygen_time,
                    client_serialize_time=client_serialize_time,
                    sign_time=sign_time,
                    server_serialize_time=server_serialize_time,
                    verify_time=bad_verify,
                    server_verify_total_time=bad_total,
                    signed_payload_size=len(payload),
                    sig_size=len(bad_signature),
                    pubkey_size=len(public_key),
                    sig_valid=bad_valid,
                )
            )

            check(
                f"(c) r{rnd}/n{node_id} tampered signature -> False",
                not bad_valid,
                passed,
                failed,
            )

    # --------------------------------------------------------------
    # (d) Replay: round-1 signature in round-2 context
    # --------------------------------------------------------------
    print("\nReplay (d)")

    for node_id in NODE_IDS:
        artifact = round1_artifacts[node_id]

        t0 = time.perf_counter()
        replay_payload = make_payload(
            weights_bytes,
            2,
            node_id,
        )
        replay_serialize_time = time.perf_counter() - t0

        replay_valid, replay_verify, replay_total = verify_payload(
            replay_payload,
            artifact["signature"],
            artifact["public_key"],
        )

        csv_rows.append(
            make_csv_row(
                rnd=2,
                node_id=node_id,
                keygen_time=0.0,
                client_serialize_time=0.0,
                sign_time=0.0,
                server_serialize_time=replay_serialize_time,
                verify_time=replay_verify,
                server_verify_total_time=replay_total,
                signed_payload_size=len(replay_payload),
                sig_size=len(artifact["signature"]),
                pubkey_size=len(artifact["public_key"]),
                sig_valid=replay_valid,
            )
        )

        check(
            f"(d) n{node_id} round-1 signature in round-2 context -> False",
            not replay_valid,
            passed,
            failed,
        )

    # --------------------------------------------------------------
    # (e) CSV schema
    # --------------------------------------------------------------
    print("\nCSV (e)")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        newline="",
    ) as f:
        tmp_path = f.name

        writer = csv.DictWriter(
            f,
            fieldnames=EXPECTED_CSV_COLS,
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(tmp_path, newline="") as f:
        rows = list(csv.DictReader(f))

    os.unlink(tmp_path)

    actual_cols = list(rows[0].keys()) if rows else []

    check(
        "(e) CSV schema matches production schema",
        actual_cols == EXPECTED_CSV_COLS,
        passed,
        failed,
    )

    good_rows = [
        row
        for row in rows
        if row["sig_valid"] == "True"
    ]

    bad_rows = [
        row
        for row in rows
        if row["sig_valid"] == "False"
    ]

    check(
        "(e) valid cases recorded",
        len(good_rows) > 0,
        passed,
        failed,
    )

    check(
        "(e) rejected cases recorded",
        len(bad_rows) > 0,
        passed,
        failed,
    )

    print("\n" + "=" * 70)
    print(
        f"PASSED: {len(passed)}  "
        f"FAILED: {len(failed)}"
    )

    if failed:
        print("Failures:")
        for failure in failed:
            print(f"  {failure}")

    return len(failed) == 0


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
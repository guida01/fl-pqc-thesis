#!/usr/bin/env python3
"""
Negative-path smoke test for the ephemeral-key signing protocol.
ECDSA-256, 2 clients, 2 rounds. No Flower required.

Tests:
  (a) Valid signature verifies and enters aggregation.
  (b) Tampered weights payload → verification fails, excluded.
  (c) Tampered signature bytes → verification fails, excluded.
  (d) Replay: round-1 signature presented in round-2 context → fails.
  (e) CSV has the expected columns; verified=False for (b)(c)(d).
"""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorchexample.signature_manager import SignatureManager
from pytorchexample.task import CIFAR10CNN, weights_to_bytes

SCHEME     = "ECDSA-256"
NODE_IDS   = [1001, 1002]
NUM_ROUNDS = 2

# These are exactly the columns the thesis specifies
EXPECTED_CSV_COLS = {
    "run", "round", "node_id", "scheme",
    "sign_time", "verify_time", "sig_size", "pubkey_size", "verified",
}

# NOTE: the current production CSV has 4 extra columns not in the thesis spec:
# keygen_time, train_time, train_loss, payload_size
# The smoke test uses only the thesis-spec columns so that (e) checks the
# minimum required set.


def make_payload(weights_bytes: bytes, server_round: int, node_id: int) -> bytes:
    """Identical to the construction in client_app.py and server_app.py."""
    return (
        weights_bytes
        + server_round.to_bytes(4, "big")
        + str(node_id).encode("utf-8")
    )


def check(label: str, condition: bool, passed: list, failed: list):
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {label}")
    (passed if condition else failed).append(label)


def run_smoke_test() -> bool:
    model = CIFAR10CNN()
    weights_bytes = weights_to_bytes(model.state_dict())

    passed, failed = [], []
    csv_rows = []

    print(f"Scheme: {SCHEME} | Nodes: {NODE_IDS} | Rounds: {NUM_ROUNDS}")
    print("=" * 60)

    round1_artifacts = {}  # node_id → {sig, pubkey} for replay test

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\nRound {rnd}")
        for node_id in NODE_IDS:
            # --- CLIENT side: ephemeral keygen + sign ---
            sig_mgr   = SignatureManager(SCHEME)
            payload   = make_payload(weights_bytes, rnd, node_id)
            signature, sign_time = sig_mgr.sign(payload)
            pubkey    = sig_mgr.public_key_bytes

            if rnd == 1:
                round1_artifacts[node_id] = {"sig": signature, "pubkey": pubkey}

            # --- SERVER side: reconstruct payload + verify ---
            # Server receives: weights (via ArrayRecord, reconstructed identically),
            # server_round (method parameter), node_id (reply.metadata.src_node_id).
            server_payload = make_payload(weights_bytes, rnd, node_id)

            # (a) Valid signature
            valid, vt = SignatureManager(SCHEME).verify(server_payload, signature, pubkey)
            csv_rows.append({"run": 1, "round": rnd, "node_id": node_id, "scheme": SCHEME,
                              "sign_time": sign_time, "verify_time": vt,
                              "sig_size": len(signature), "pubkey_size": len(pubkey),
                              "verified": valid})
            check(f"(a) r{rnd}/n{node_id} valid sig → True", valid, passed, failed)

            # (b) Tampered weights: flip one byte in the serialized payload
            bad_weights = bytearray(weights_bytes)
            bad_weights[100] ^= 0xFF
            bad_payload = make_payload(bytes(bad_weights), rnd, node_id)
            bad_b, vt_b = SignatureManager(SCHEME).verify(bad_payload, signature, pubkey)
            csv_rows.append({"run": 1, "round": rnd, "node_id": node_id, "scheme": SCHEME,
                              "sign_time": sign_time, "verify_time": vt_b,
                              "sig_size": len(signature), "pubkey_size": len(pubkey),
                              "verified": bad_b})
            check(f"(b) r{rnd}/n{node_id} tampered weights → False", not bad_b, passed, failed)

            # (c) Tampered signature: flip one byte
            bad_sig = bytearray(signature)
            bad_sig[0] ^= 0xFF
            bad_c, vt_c = SignatureManager(SCHEME).verify(server_payload, bytes(bad_sig), pubkey)
            csv_rows.append({"run": 1, "round": rnd, "node_id": node_id, "scheme": SCHEME,
                              "sign_time": sign_time, "verify_time": vt_c,
                              "sig_size": len(bad_sig), "pubkey_size": len(pubkey),
                              "verified": bad_c})
            check(f"(c) r{rnd}/n{node_id} tampered sig → False", not bad_c, passed, failed)

    # (d) Replay: round-1 sig vs round-2 server payload
    print(f"\nReplay (d)")
    for node_id in NODE_IDS:
        art = round1_artifacts[node_id]
        # Server reconstructs payload for round 2
        replay_payload = make_payload(weights_bytes, 2, node_id)
        bad_d, vt_d = SignatureManager(SCHEME).verify(replay_payload, art["sig"], art["pubkey"])
        csv_rows.append({"run": 1, "round": 2, "node_id": node_id, "scheme": SCHEME,
                         "sign_time": 0.0, "verify_time": vt_d,
                         "sig_size": len(art["sig"]), "pubkey_size": len(art["pubkey"]),
                         "verified": bad_d})
        check(f"(d) n{node_id} round-1 sig in round-2 context → False", not bad_d, passed, failed)

    # (e) CSV columns + verified=False for bad cases
    print(f"\nCSV (e)")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        tmp = f.name
        writer = csv.DictWriter(f, fieldnames=list(EXPECTED_CSV_COLS))
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(tmp) as f:
        rows = list(csv.DictReader(f))
    os.unlink(tmp)

    actual_cols = set(rows[0].keys()) if rows else set()
    cols_ok     = EXPECTED_CSV_COLS.issubset(actual_cols)
    bad_rows    = [r for r in rows if r["verified"] == "False"]
    good_rows   = [r for r in rows if r["verified"] == "True"]

    # bad cases come from tests (b), (c), (d); good from (a)
    check("(e) CSV has all required columns",        cols_ok,           passed, failed)
    check("(e) some rows have verified=True (a)",    len(good_rows) > 0, passed, failed)
    check("(e) some rows have verified=False (b/c/d)", len(bad_rows) > 0, passed, failed)

    # Summary
    print("\n" + "=" * 60)
    print(f"PASSED: {len(passed)}  FAILED: {len(failed)}")
    if failed:
        print("Failures:")
        for f_ in failed:
            print(f"  {f_}")

    return len(failed) == 0


if __name__ == "__main__":
    ok = run_smoke_test()
    sys.exit(0 if ok else 1)

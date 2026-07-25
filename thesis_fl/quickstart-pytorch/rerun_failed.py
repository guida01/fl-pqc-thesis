#!/usr/bin/env python3
"""
Re-run specific (scheme, run_number) pairs without touching completed data.

Still needed after run_all_schemes.py moved to a randomized per-run scheme
order (see its module docstring): a rerun of run_number=1 for some scheme
still hits SignedFedAvg's `mode = "w" if run_number == 1 else "a"` in
server_app.py, which would wipe that scheme's CSV and any later runs
already recorded in it. This script backs those up first regardless of
which schemes/runs are being redone.

Fill in RERUNS below with the (scheme, run_number) pairs that failed in
the CURRENT campaign before using this — the list intentionally starts
empty. (The v1-experimental-baseline campaign's specific failures —
ML-DSA-44 runs 1-2, Falcon-padded-512 run 1 — no longer apply: that data
predates tasks 1-4's pipeline changes and lives only in
results_v1_frozen/.)

Every rerun this script performs is appended, timestamped, to the same
results/execution_order.log that run_all_schemes.py writes to, so the
effective execution order stays fully auditable regardless of which
script triggered a given run.

Strategy:
  1. Back up good rows from each affected CSV (runs > max_failed_run).
  2. Delete the affected CSV so server_app recreates it cleanly on run_number=1.
  3. Re-run the failed (scheme, run_number) pairs in order.
  4. Append backed-up rows to restore the full dataset.
"""

import csv
import os
import sys

from run_all_schemes import NUM_ROUNDS, NUM_SUPERNODES, RESULTS_DIR, log_execution, run_scheme

# (scheme, run_number) pairs to re-run, in order — fill in for the current
# campaign; empty by default (see docstring).
RERUNS = []


def csv_path(scheme: str) -> str:
    return os.path.join(RESULTS_DIR, f"{scheme.replace('/', '_')}.csv")


def backup_good_rows(scheme: str, max_failed_run: int):
    """Return (header, good_rows) where good_rows have run_number > max_failed_run."""
    path = csv_path(scheme)
    if not os.path.exists(path):
        return None, []
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None, []
    header    = rows[0]
    good_rows = [r for r in rows[1:] if int(r[0]) > max_failed_run]
    print(f"  Backed up {len(good_rows)} good rows from {scheme} (runs > {max_failed_run})")
    return header, good_rows


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Step 1: determine max failed run per scheme and back up good rows ---
    from collections import defaultdict
    max_failed = defaultdict(int)
    for scheme, run_num in RERUNS:
        max_failed[scheme] = max(max_failed[scheme], run_num)

    backups = {}
    for scheme, mfr in max_failed.items():
        header, good_rows = backup_good_rows(scheme, mfr)
        backups[scheme] = (header, good_rows)
        log_execution(f"rerun_failed: backed up {len(good_rows)} good rows from {scheme} (runs > {mfr})")
        # Delete the CSV so run_number=1 creates it fresh
        path = csv_path(scheme)
        if os.path.exists(path):
            os.remove(path)
            log_execution(f"rerun_failed: deleted {path} (will be recreated)")

    # --- Step 2: re-run failed (scheme, run_number) pairs ---
    # run_scheme() (imported from run_all_schemes) does ray stop, config
    # update, `flwr run`, wait, and CSV row-count reporting — identical to
    # what a normal campaign run does — and already logs to
    # results/execution_order.log itself, so every rerun is auditable
    # alongside the runs run_all_schemes.py performed.
    results = {}
    total = len(RERUNS)
    for idx, (scheme, run_num) in enumerate(RERUNS, 1):
        log_execution(f"rerun_failed: [{idx}/{total}] rerunning {scheme} run {run_num}")
        results[(scheme, run_num)] = run_scheme(scheme, run_num)

    # --- Step 3: append backed-up good rows ---
    print("\n--- Reconstructing CSVs ---")
    for scheme, (header, good_rows) in backups.items():
        if not good_rows:
            print(f"  {scheme}: no saved rows to append")
            continue
        path = csv_path(scheme)
        with open(path, "a", newline="") as f:
            csv.writer(f).writerows(good_rows)
        log_execution(f"rerun_failed: restored {len(good_rows)} backed-up rows to {path}")

    # --- Final report ---
    print("\n=== Final row counts ===")
    all_ok = True
    for scheme in max_failed:
        path = csv_path(scheme)
        if os.path.exists(path):
            with open(path) as f:
                n = len(f.readlines()) - 1
            expected = 5 * NUM_ROUNDS * NUM_SUPERNODES
            ok = "✓" if n == expected else f"✗ (expected {expected})"
            print(f"  {scheme}: {n} rows  {ok}")
            if n != expected:
                all_ok = False
        else:
            print(f"  {scheme}: CSV missing!")
            all_ok = False

    print("\n=== Re-run outcomes ===")
    for (scheme, run_num), success in results.items():
        print(f"  {scheme} run {run_num}: {'✓' if success else '✗ FAILED'}")

    sys.exit(0 if all_ok else 1)

#!/usr/bin/env python3
"""
Re-run only the failed runs without touching completed data.

Failed:
  ML-DSA-44    runs 1 and 2  (run 1: missing round 30; run 2: cut at round 17)
  Falcon-padded-512  run 1   (cut at round 26)

Strategy:
  1. Back up good rows from each affected CSV (runs > max_failed_run).
  2. Delete the affected CSV so server_app recreates it cleanly on run_number=1.
  3. Re-run the failed (scheme, run_number) pairs in order.
  4. Append backed-up rows to restore the full dataset.
"""

import csv
import os
import re
import subprocess
import sys
import time

RESULTS_DIR    = "results"
PYPROJECT      = "pyproject.toml"
NUM_SUPERNODES = 5
NUM_ROUNDS     = 30
MAX_WAIT_SEC   = 1800

# (scheme, run_number) pairs to re-run, in order
RERUNS = [
    ("ML-DSA-44",         1),
    ("ML-DSA-44",         2),
    ("Falcon-padded-512", 1),
]


def update_config(scheme: str, run_num: int):
    with open(PYPROJECT) as f:
        content = f.read()
    content = re.sub(r'scheme\s*=\s*"[^"]*"',    f'scheme = "{scheme}"',          content)
    content = re.sub(r'run-number\s*=\s*\d+',     f'run-number = {run_num}',       content)
    content = re.sub(r'num-supernodes\s*=\s*\d+', f'num-supernodes = {NUM_SUPERNODES}', content)
    with open(PYPROJECT, "w") as f:
        f.write(content)


def wait_for_run(run_id: str) -> bool:
    start = time.time()
    while time.time() - start < MAX_WAIT_SEC:
        time.sleep(10)
        env = {**os.environ, "COLUMNS": "300"}
        ls  = subprocess.run(["flwr", "ls"], capture_output=True, text=True, env=env)
        for line in ls.stdout.splitlines():
            if run_id in line:
                if "finished:completed" in line:
                    return True
                elif "finished:failed" in line:
                    return False
                else:
                    elapsed = int(time.time() - start)
                    print(f"  [{elapsed}s] still running...")
                break
    print("  Timeout!")
    return False


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
        # Delete the CSV so run_number=1 creates it fresh
        path = csv_path(scheme)
        if os.path.exists(path):
            os.remove(path)
            print(f"  Deleted {path} (will be recreated)")

    # --- Step 2: re-run failed (scheme, run_number) pairs ---
    results = {}
    total   = len(RERUNS)
    for idx, (scheme, run_num) in enumerate(RERUNS, 1):
        print(f"\n[{idx}/{total}] {'='*50}")
        print(f"  {scheme}  —  run {run_num}")
        print(f"  {'='*50}")

        subprocess.run(["ray", "stop", "--force"], capture_output=True)
        time.sleep(5)

        update_config(scheme, run_num)

        result = subprocess.run(
            ["flwr", "run", ".", "--federation-config", f"num-supernodes={NUM_SUPERNODES}"],
            capture_output=True, text=True,
        )
        output = result.stdout + result.stderr

        match = re.search(r"run (\d+)", output)
        if not match:
            print("  Could not get run ID — skipping.")
            print("  stdout:", result.stdout[:300])
            print("  stderr:", result.stderr[:300])
            results[(scheme, run_num)] = False
            continue

        run_id = match.group(1)
        print(f"  Run ID: {run_id}")

        success = wait_for_run(run_id)
        results[(scheme, run_num)] = success
        status  = "✓ completed" if success else "✗ FAILED"
        print(f"  {status}")

        path = csv_path(scheme)
        if os.path.exists(path):
            with open(path) as f:
                n_rows = len(f.readlines()) - 1
            expected = run_num * NUM_ROUNDS * NUM_SUPERNODES
            print(f"  CSV rows: {n_rows}  (expected ≤ {expected})")

        time.sleep(15)

    # --- Step 3: append backed-up good rows ---
    print("\n--- Reconstructing CSVs ---")
    for scheme, (header, good_rows) in backups.items():
        if not good_rows:
            print(f"  {scheme}: no saved rows to append")
            continue
        path = csv_path(scheme)
        with open(path, "a", newline="") as f:
            csv.writer(f).writerows(good_rows)
        print(f"  Appended {len(good_rows)} saved rows → {path}")

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

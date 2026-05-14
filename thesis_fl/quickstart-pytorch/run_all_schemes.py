#!/usr/bin/env python3
"""Run all 7 signature schemes × N_RUNS times sequentially."""

import subprocess
import time
import os
import re

SCHEMES = [
    "ML-DSA-44",
    "ML-DSA-65",
    "ML-DSA-87",
    "Falcon-padded-512",
    "SLH_DSA_PURE_SHA2_128S",
    "RSA-2048",
    "ECDSA-256",
]

N_RUNS         = 5
NUM_SUPERNODES = 5
NUM_ROUNDS     = 50   # must match num-server-rounds in pyproject.toml

PYPROJECT    = "pyproject.toml"
RESULTS_DIR  = "results"
MAX_WAIT_SEC = 1800   # 30 min per run (10 clients × 50 rounds)


def update_config(scheme: str, run_num: int):
    """Update scheme and run-number in pyproject.toml."""
    with open(PYPROJECT) as f:
        content = f.read()
    content = re.sub(r'scheme\s*=\s*"[^"]*"', f'scheme = "{scheme}"', content)
    content = re.sub(r'run-number\s*=\s*\d+', f'run-number = {run_num}', content)
    with open(PYPROJECT, "w") as f:
        f.write(content)


def wait_for_run(run_id: str) -> bool:
    """Poll flwr ls until the run finishes. Returns True if completed."""
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


def run_scheme(scheme: str, run_num: int):
    print(f"\n{'='*60}\n  {scheme}  —  run {run_num}/{N_RUNS}\n{'='*60}")

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
        return

    run_id = match.group(1)
    print(f"  Run ID: {run_id}")

    success = wait_for_run(run_id)
    status  = "✓ completed" if success else "✗ FAILED"
    print(f"  {status}")

    csv_path = os.path.join(RESULTS_DIR, f"{scheme.replace('/', '_')}.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = len(f.readlines()) - 1
        expected = run_num * NUM_ROUNDS * NUM_SUPERNODES
        print(f"  CSV rows: {rows}  (expected ≤ {expected})")

    time.sleep(15)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total = len(SCHEMES) * N_RUNS
    done  = 0
    for scheme in SCHEMES:
        for run_num in range(1, N_RUNS + 1):
            done += 1
            print(f"\n[{done}/{total}]", end="")
            run_scheme(scheme, run_num)

    print("\n" + "=" * 60)
    print("All schemes done! Results:")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith(".csv"):
            path = os.path.join(RESULTS_DIR, fname)
            rows = len(open(path).readlines()) - 1
            expected = N_RUNS * NUM_ROUNDS * NUM_SUPERNODES
            print(f"  {fname}: {rows} rows  (expected {expected})")

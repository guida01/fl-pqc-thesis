#!/usr/bin/env python3
"""Run all 7 signature schemes sequentially."""

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

PYPROJECT    = "pyproject.toml"
RESULTS_DIR  = "results"
MAX_WAIT_SEC = 600  # 10 min per scheme


def update_scheme(scheme: str):
    """Replace the scheme value in pyproject.toml."""
    with open(PYPROJECT) as f:
        content = f.read()
    content = re.sub(r'scheme\s*=\s*"[^"]*"', f'scheme = "{scheme}"', content)
    with open(PYPROJECT, "w") as f:
        f.write(content)


def wait_for_run(run_id: str) -> bool:
    """Poll flwr ls until the run finishes. Returns True if completed."""
    start = time.time()
    while time.time() - start < MAX_WAIT_SEC:
        time.sleep(10)
        env = {**os.environ, "COLUMNS": "300"}
        ls = subprocess.run(["flwr", "ls"], capture_output=True, text=True, env=env)
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


def run_scheme(scheme: str):
    print(f"\n{'='*55}\n  {scheme}\n{'='*55}")

     # kill any leftover Ray processes before each run
    subprocess.run(["ray", "stop", "--force"], capture_output=True)
    time.sleep(5)

    update_scheme(scheme)

    result = subprocess.run(["flwr", "run", "."], capture_output=True, text=True)
    output = result.stdout + result.stderr

    match = re.search(r"run (\d+)", output)
    if not match:
        print("  Could not get run ID — skipping.")
        return

    run_id = match.group(1)
    print(f"  Run ID: {run_id}")

    success = wait_for_run(run_id)
    status  = "✓ completed" if success else "✗ FAILED"
    print(f"  {status}")

    # show CSV row count
    csv_path = os.path.join(RESULTS_DIR, f"{scheme.replace('/', '_')}.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = len(f.readlines()) - 1  # minus header
        print(f"  CSV rows: {rows}")

    time.sleep(15)  # brief pause between schemes


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for scheme in SCHEMES:
        run_scheme(scheme)

    print("\n" + "="*55)
    print("All schemes done! Results:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith(".csv"):
            path = os.path.join(RESULTS_DIR, f)
            rows = len(open(path).readlines()) - 1
            print(f"  {f}: {rows} rows")
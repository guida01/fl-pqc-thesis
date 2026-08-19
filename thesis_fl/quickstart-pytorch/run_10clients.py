#!/usr/bin/env python3
"""10-supernode variant of run_all_schemes.py — same structure (randomized
per-run scheme order, execution_order.log, CPU governor check), different
RESULTS_DIR/NUM_SUPERNODES. Not launched yet; kept coherent with the
current pipeline so it's ready when the 10-client experiment is approved.
"""

import subprocess
import time
import os
import re
import random
import datetime

SCHEMES = [
    "no_signature",
    "ML-DSA-44",
    "ML-DSA-65",
    "ML-DSA-87",
    "Falcon-padded-512",
    "SLH_DSA_PURE_SHA2_128S",
    "RSA-2048",
    "ECDSA-256",
]

N_RUNS         = 5
NUM_SUPERNODES = 10
NUM_ROUNDS     = 30   # must match num-server-rounds in pyproject.toml

PYPROJECT      = "pyproject.toml"
RESULTS_DIR    = os.path.abspath("results_10clients")
EXECUTION_LOG  = os.path.join(RESULTS_DIR, "execution_order.log")
MAX_WAIT_SEC   = 1800   # 30 min per run (10 clients × 30 rounds)

CPU_GOVERNOR_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"


def log_execution(message: str):
    """Print and append a timestamped line to
    results_10clients/execution_order.log — same convention as
    run_all_schemes.py's log_execution(), kept separate since this driver
    writes to a different results directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    line = f"{timestamp}  {message}"
    print(line)
    with open(EXECUTION_LOG, "a") as f:
        f.write(line + "\n")


def check_cpu_governor() -> str:
    """Read-only: report the CPU governor (cpu0) and warn if it isn't
    'performance'. Never changes it — see README.md for the manual fix."""
    try:
        with open(CPU_GOVERNOR_PATH) as f:
            governor = f.read().strip()
    except OSError as e:
        governor = f"unknown (could not read {CPU_GOVERNOR_PATH}: {e})"

    log_execution(f"CPU governor (cpu0): {governor}")
    if governor != "performance":
        print(
            f"  WARNING: CPU governor is '{governor}', not 'performance'. "
            "keygen/sign/verify/train timings may be affected by frequency "
            "scaling during the campaign. This script will NOT change it "
            "automatically — see README.md for the manual command to fix "
            "it, if that's what you want."
        )
    return governor


def randomized_scheme_order(run_num: int) -> list:
    """Shuffle SCHEMES for this run. Seeded off run_num so the order is
    reproducible per run but differs across runs. Uses a different base
    offset (2000 vs run_all_schemes.py's 1000) so the two drivers don't
    coincidentally produce the same permutation for the same run_num."""
    order = SCHEMES.copy()
    random.Random(2000 + run_num).shuffle(order)
    return order


def update_config(scheme: str, run_num: int):
    """Update scheme, run-number, num-supernodes, AND results-dir in
    pyproject.toml. results-dir must be rewritten too — server_app.py
    reads it straight from context.run_config, not from this script's own
    RESULTS_DIR constant, so without this the campaign would silently
    write into whatever results-dir the file already had (e.g. the
    5-client results/ directory), clobbering unrelated data."""
    with open(PYPROJECT) as f:
        content = f.read()
    content = re.sub(r'scheme\s*=\s*"[^"]*"', f'scheme = "{scheme}"', content)
    content = re.sub(r'run-number\s*=\s*\d+', f'run-number = {run_num}', content)
    content = re.sub(r'num-supernodes\s*=\s*\d+', f'num-supernodes = {NUM_SUPERNODES}', content)
    content = re.sub(r'results-dir\s*=\s*"[^"]*"', f'results-dir = "{RESULTS_DIR}"', content)
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


def run_scheme(scheme: str, run_num: int) -> bool:
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
        log_execution(f"{scheme} run {run_num}: could not get run ID — skipping")
        print("  stdout:", result.stdout[:300])
        print("  stderr:", result.stderr[:300])
        return False

    run_id = match.group(1)

    success = wait_for_run(run_id)
    status  = "completed" if success else "FAILED"
    log_execution(f"{scheme} run {run_num}: run_id={run_id} -> {status}")

    csv_path = os.path.join(RESULTS_DIR, f"{scheme.replace('/', '_')}.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = len(f.readlines()) - 1
        expected = run_num * NUM_ROUNDS * NUM_SUPERNODES
        print(f"  CSV rows: {rows}  (expected ≤ {expected})")

    time.sleep(15)
    return success


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    check_cpu_governor()

    total = len(SCHEMES) * N_RUNS
    done  = 0
    for run_num in range(1, N_RUNS + 1):
        order = randomized_scheme_order(run_num)
        log_execution(f"run {run_num}: scheme order = {order}")
        for scheme in order:
            done += 1
            log_execution(f"[{done}/{total}] starting {scheme} run {run_num}")
            run_scheme(scheme, run_num)

    print("\n" + "=" * 60)
    print("All schemes done! Results:")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith(".csv"):
            path = os.path.join(RESULTS_DIR, fname)
            rows = len(open(path).readlines()) - 1
            expected = N_RUNS * NUM_ROUNDS * NUM_SUPERNODES
            print(f"  {fname}: {rows} rows  (expected {expected})")

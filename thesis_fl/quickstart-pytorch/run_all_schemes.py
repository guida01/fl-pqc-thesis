#!/usr/bin/env python3
"""Run all 8 signature-scheme configs × N_RUNS times, scheme order
randomized independently per run so per-scheme block position can't be
confounded with thermal/frequency drift over the campaign's wall-clock
duration. See results/execution_order.log for the exact order and
timestamps of every run this script executes."""

import subprocess
import time
import os
import re
import random
import datetime

# NOTE: this process only orchestrates `flwr run .` as subprocesses (see
# run_scheme() below) — it never trains or signs anything itself. Seeding
# random/numpy/torch here has no effect on those subprocesses' RNG state;
# the seeds that actually matter (model init, DataLoader order, PQC/RSA/
# ECDSA keygen) are set inside server_app.py:111 and task.py:45/_PARTITION_SEED,
# which run in the subprocess. A previous version of this script seeded
# random/numpy/torch at module level here — removed as dead code.

SCHEMES = [
    "no_signature",
    "ML-DSA-44",
    "ML-DSA-65",
    "ML-DSA-87",
    "Falcon-padded-512",
    "SPHINCS+-SHA2-128s-simple",
    "RSA-2048",
    "ECDSA-256",
]

N_RUNS         = 5
NUM_SUPERNODES = 5
NUM_ROUNDS     = 30   # must match num-server-rounds in pyproject.toml

PYPROJECT      = "pyproject.toml"
RESULTS_DIR    = "results"
EXECUTION_LOG  = os.path.join(RESULTS_DIR, "execution_order.log")
MAX_WAIT_SEC   = 1800   # 30 min per run (10 clients × 50 rounds)

CPU_GOVERNOR_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"


def log_execution(message: str):
    """Print and append a timestamped line to results/execution_order.log
    — the audit trail of the actual order runs executed in, independent
    of any (scheme, run) pairing convention."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    line = f"{timestamp}  {message}"
    print(line)
    with open(EXECUTION_LOG, "a") as f:
        f.write(line + "\n")


def check_cpu_governor() -> str:
    """Read-only: report the CPU governor (cpu0) and warn if it isn't
    'performance'. Never changes it — frequency scaling affects timing
    measurements (keygen/sign/verify/train), so the campaign should be run
    with a fixed governor, but that's an operator decision, not something
    this script should do silently."""
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
    """Shuffle SCHEMES for this run. Seeded off run_num (not the global
    SEED) so the order is reproducible per run but differs across runs,
    and doesn't depend on module-level RNG state."""
    order = SCHEMES.copy()
    random.Random(1000 + run_num).shuffle(order)
    return order


def update_config(scheme: str, run_num: int):
    """Update scheme, run-number, and num-supernodes in pyproject.toml."""
    with open(PYPROJECT) as f:
        content = f.read()
    content = re.sub(r'scheme\s*=\s*"[^"]*"', f'scheme = "{scheme}"', content)
    content = re.sub(r'run-number\s*=\s*\d+', f'run-number = {run_num}', content)
    content = re.sub(r'num-supernodes\s*=\s*\d+', f'num-supernodes = {NUM_SUPERNODES}', content)
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

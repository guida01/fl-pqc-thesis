#!/usr/bin/env python3
"""Final 5-client experimental campaign.

Runs all 8 configurations for 10 runs and 30 federated rounds.
Scheme order is randomized independently for each run.

The runner:
- verifies the pinned OQS 0.16.0 environment;
- refuses to overwrite an existing final-results directory;
- configures pyproject.toml for each Flower execution;
- restores pyproject.toml when the campaign finishes or is interrupted;
- stops immediately if a Flower execution fails;
- validates the expected CSV row counts.
"""

import datetime
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time


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

PQ_SCHEMES = [
    "ML-DSA-44",
    "ML-DSA-65",
    "ML-DSA-87",
    "Falcon-padded-512",
    "SLH_DSA_PURE_SHA2_128S",
]

EXPECTED_OQS_VERSION = "0.16.0"

N_RUNS = 10
NUM_SUPERNODES = 5
NUM_ROUNDS = 30

PROJECT_DIR = Path(__file__).resolve().parent
PYPROJECT = PROJECT_DIR / "pyproject.toml"
RESULTS_DIR = PROJECT_DIR / "results_final_5clients"
EXECUTION_LOG = RESULTS_DIR / "execution_order.log"

MAX_WAIT_SEC = 1800

CPU_GOVERNOR_PATH = Path(
    "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
)


def log_execution(message: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().isoformat(
        timespec="seconds"
    )
    line = f"{timestamp}  {message}"

    print(line)

    with EXECUTION_LOG.open("a") as f:
        f.write(line + "\n")


def check_environment() -> None:
    """Fail fast unless the final pinned OQS environment is active."""
    try:
        import oqs
    except ImportError as exc:
        raise SystemExit(
            "ERROR: liboqs-python is not available.\n"
            "Run:\n"
            "  source scripts/activate_final_env.sh"
        ) from exc

    native = oqs.oqs_version()
    wrapper = oqs.oqs_python_version()

    print(f"Python:        {sys.executable}")
    print(f"liboqs:        {native}")
    print(f"liboqs-python: {wrapper}")

    if native != EXPECTED_OQS_VERSION:
        raise SystemExit(
            f"ERROR: expected native liboqs "
            f"{EXPECTED_OQS_VERSION}, got {native}.\n"
            "Run:\n"
            "  source scripts/activate_final_env.sh"
        )

    if wrapper != EXPECTED_OQS_VERSION:
        raise SystemExit(
            f"ERROR: expected liboqs-python "
            f"{EXPECTED_OQS_VERSION}, got {wrapper}."
        )

    enabled = set(oqs.get_enabled_sig_mechanisms())

    missing = [
        scheme
        for scheme in PQ_SCHEMES
        if scheme not in enabled
    ]

    if missing:
        raise SystemExit(
            "ERROR: required liboqs mechanisms are unavailable: "
            + ", ".join(missing)
        )

    print("OQS environment: OK")


def ensure_fresh_results_dir() -> None:
    """Never silently mix a new campaign with existing final results."""
    if RESULTS_DIR.exists():
        existing = list(RESULTS_DIR.iterdir())

        if existing:
            print(
                f"ERROR: final results directory is not empty:\n"
                f"  {RESULTS_DIR}\n\n"
                "Move/rename the existing directory before starting "
                "a new final campaign."
            )
            raise SystemExit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def check_cpu_governor() -> str:
    try:
        governor = CPU_GOVERNOR_PATH.read_text().strip()
    except OSError as exc:
        governor = (
            "unknown "
            f"(could not read {CPU_GOVERNOR_PATH}: {exc})"
        )

    log_execution(
        f"CPU governor (cpu0): {governor}"
    )

    if governor != "performance":
        print(
            f"WARNING: CPU governor is '{governor}', "
            "not 'performance'. Timing measurements may be "
            "affected by frequency scaling."
        )

    return governor


def randomized_scheme_order(run_num: int) -> list[str]:
    order = SCHEMES.copy()
    random.Random(1000 + run_num).shuffle(order)
    return order


def replace_one(
    content: str,
    pattern: str,
    replacement: str,
    label: str,
) -> str:
    updated, count = re.subn(
        pattern,
        replacement,
        content,
    )

    if count != 1:
        raise RuntimeError(
            f"Expected exactly one '{label}' entry in "
            f"{PYPROJECT}, found {count}."
        )

    return updated


def update_config(
    scheme: str,
    run_num: int,
) -> None:
    """Set all campaign-critical Flower run configuration."""
    content = PYPROJECT.read_text()

    content = replace_one(
        content,
        r'scheme\s*=\s*"[^"]*"',
        f'scheme = "{scheme}"',
        "scheme",
    )

    content = replace_one(
        content,
        r"run-number\s*=\s*\d+",
        f"run-number = {run_num}",
        "run-number",
    )

    content = replace_one(
        content,
        r"num-supernodes\s*=\s*\d+",
        f"num-supernodes = {NUM_SUPERNODES}",
        "num-supernodes",
    )

    content = replace_one(
        content,
        r"num-server-rounds\s*=\s*\d+",
        f"num-server-rounds = {NUM_ROUNDS}",
        "num-server-rounds",
    )

    content = replace_one(
        content,
        r'results-dir\s*=\s*"[^"]*"',
        f'results-dir = "{RESULTS_DIR}"',
        "results-dir",
    )

    content = replace_one(
        content,
        r"eval-central\s*=\s*(?:true|false)",
        "eval-central = true",
        "eval-central",
    )

    PYPROJECT.write_text(content)


def wait_for_run(run_id: str) -> bool:
    start = time.time()

    while time.time() - start < MAX_WAIT_SEC:
        time.sleep(10)

        env = {
            **os.environ,
            "COLUMNS": "300",
        }

        result = subprocess.run(
            ["flwr", "ls"],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_DIR,
        )

        for line in result.stdout.splitlines():
            if run_id not in line:
                continue

            if "finished:completed" in line:
                return True

            if "finished:failed" in line:
                return False

            elapsed = int(time.time() - start)
            print(
                f"  [{elapsed}s] still running..."
            )
            break

    print(
        f"  Timeout after {MAX_WAIT_SEC}s!"
    )
    return False


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return -1

    with path.open() as f:
        return max(
            sum(1 for _ in f) - 1,
            0,
        )


def validate_result_counts(
    scheme: str,
    run_num: int,
) -> bool:
    safe_scheme = scheme.replace("/", "_")

    update_csv = (
        RESULTS_DIR / f"{safe_scheme}.csv"
    )

    eval_csv = (
        RESULTS_DIR / f"{safe_scheme}_eval.csv"
    )

    expected_updates = (
        run_num
        * NUM_ROUNDS
        * NUM_SUPERNODES
    )

    expected_eval = (
        run_num
        * (NUM_ROUNDS + 1)
    )

    update_rows = count_csv_rows(update_csv)
    eval_rows = count_csv_rows(eval_csv)

    print(
        f"  update CSV: {update_rows} rows "
        f"(expected {expected_updates})"
    )

    print(
        f"  eval CSV:   {eval_rows} rows "
        f"(expected {expected_eval})"
    )

    return (
        update_rows == expected_updates
        and eval_rows == expected_eval
    )


def run_scheme(
    scheme: str,
    run_num: int,
) -> bool:
    print(
        "\n"
        + "=" * 70
        + f"\n  {scheme} — run {run_num}/{N_RUNS}"
        + "\n"
        + "=" * 70
    )

    subprocess.run(
        ["ray", "stop", "--force"],
        capture_output=True,
        cwd=PROJECT_DIR,
    )

    time.sleep(5)

    update_config(
        scheme,
        run_num,
    )

    result = subprocess.run(
        [
            "flwr",
            "run",
            ".",
            "--federation-config",
            f"num-supernodes={NUM_SUPERNODES}",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
    )

    output = result.stdout + result.stderr

    if result.returncode != 0:
        log_execution(
            f"{scheme} run {run_num}: "
            "flwr run command FAILED"
        )

        print(result.stdout[-1000:])
        print(result.stderr[-1000:])
        return False

    match = re.search(
        r"\brun (\d+)\b",
        output,
    )

    if not match:
        log_execution(
            f"{scheme} run {run_num}: "
            "could not obtain Flower run ID"
        )

        print(result.stdout[-1000:])
        print(result.stderr[-1000:])
        return False

    run_id = match.group(1)

    success = wait_for_run(run_id)

    status = (
        "completed"
        if success
        else "FAILED"
    )

    log_execution(
        f"{scheme} run {run_num}: "
        f"run_id={run_id} -> {status}"
    )

    if not success:
        return False

    counts_ok = validate_result_counts(
        scheme,
        run_num,
    )

    if not counts_ok:
        log_execution(
            f"{scheme} run {run_num}: "
            "unexpected CSV row count"
        )
        return False

    time.sleep(15)
    return True


def print_final_summary() -> None:
    print(
        "\n"
        + "=" * 70
        + "\nFinal campaign summary"
        + "\n"
        + "=" * 70
    )

    expected_updates = (
        N_RUNS
        * NUM_ROUNDS
        * NUM_SUPERNODES
    )

    expected_eval = (
        N_RUNS
        * (NUM_ROUNDS + 1)
    )

    for scheme in SCHEMES:
        safe_scheme = scheme.replace("/", "_")

        update_csv = (
            RESULTS_DIR / f"{safe_scheme}.csv"
        )

        eval_csv = (
            RESULTS_DIR / f"{safe_scheme}_eval.csv"
        )

        print(
            f"{scheme:30s} "
            f"updates={count_csv_rows(update_csv):4d}"
            f"/{expected_updates}  "
            f"eval={count_csv_rows(eval_csv):3d}"
            f"/{expected_eval}"
        )


def main() -> None:
    ensure_fresh_results_dir()
    check_environment()
    check_cpu_governor()

    original_pyproject = PYPROJECT.read_text()

    total = len(SCHEMES) * N_RUNS
    completed = 0

    try:
        for run_num in range(
            1,
            N_RUNS + 1,
        ):
            order = randomized_scheme_order(
                run_num
            )

            log_execution(
                f"run {run_num}: "
                f"scheme order = {order}"
            )

            for scheme in order:
                completed += 1

                log_execution(
                    f"[{completed}/{total}] "
                    f"starting {scheme} "
                    f"run {run_num}"
                )

                success = run_scheme(
                    scheme,
                    run_num,
                )

                if not success:
                    raise RuntimeError(
                        "Campaign stopped after failed "
                        f"execution: {scheme}, "
                        f"run {run_num}."
                    )

        print_final_summary()

    finally:
        PYPROJECT.write_text(
            original_pyproject
        )

        subprocess.run(
            ["ray", "stop", "--force"],
            capture_output=True,
            cwd=PROJECT_DIR,
        )


if __name__ == "__main__":
    main()
"""Isolated microbenchmark of the configured signature schemes.

Single process, no Flower, no Ray, no client contention. The FL campaign
timings are collected while simulated clients share host resources; this
benchmark provides a contention-free reference.

The benchmark reuses the timing windows implemented by SignatureManager.
Key generation is measured separately. sign_time measures the backend
signing call over the full serialized payload. verify_time measures the
backend verification call over the full serialized payload; verifier and
public-key setup remain outside that internal timing window.

The fixed benchmark payload mirrors the message signed in the FL pipeline:

    weights_to_bytes(state_dict)
    + server_round.to_bytes(4, "big")
    + node_id.to_bytes(8, "big")

A fixed model seed, round, and node ID make the payload identical across
repetitions and schemes.
"""

import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from pytorchexample.signature_manager import SignatureManager  # noqa: E402
from pytorchexample.task import CIFAR10CNN, weights_to_bytes  # noqa: E402

ALL_SCHEMES = SignatureManager.PQC_SCHEMES + SignatureManager.CLASSICAL_SCHEMES

WARMUP = 10
DEFAULT_REPS = 1000
BENCH_ROUND = 1
BENCH_NODE_ID = 1001
# RSA-2048 keygen and SLH_DSA_PURE_SHA2_128S signing are slow enough that
# 1000 reps would blow the ~15 min budget; both use fewer reps for ALL
# three operations, per the task spec.
SLOW_SCHEMES = {"SLH_DSA_PURE_SHA2_128S", "RSA-2048"}
SLOW_REPS = 100

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "microbench.csv")
CSV_FIELDS = [
    "scheme", "operation", "n_reps",
    "mean_s", "median_s", "std_s", "p95_s", "min_s", "max_s",
    "cpu_governor",
]

CPU_GOVERNOR_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"


def n_reps_for(scheme: str) -> int:
    return SLOW_REPS if scheme in SLOW_SCHEMES else DEFAULT_REPS


def read_cpu_governor() -> str:
    try:
        with open(CPU_GOVERNOR_PATH) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def summarize(samples: list) -> dict:
    n = len(samples)
    ordered = sorted(samples)
    p95_idx = min(n - 1, max(0, round(0.95 * (n - 1))))
    return {
        "n_reps": n,
        "mean_s": statistics.fmean(samples),
        "median_s": statistics.median(samples),
        "std_s": statistics.pstdev(samples) if n > 1 else 0.0,
        "p95_s": ordered[p95_idx],
        "min_s": ordered[0],
        "max_s": ordered[-1],
    }


def bench_keygen(scheme: str, n_reps: int):
    for _ in range(WARMUP):
        SignatureManager(scheme)
    samples = []
    mgr = None
    for _ in range(n_reps):
        mgr = SignatureManager(scheme)
        samples.append(mgr.keygen_time)
    return samples, mgr  # last instance reused for the sign benchmark


def bench_sign(mgr: SignatureManager, data: bytes, n_reps: int):
    for _ in range(WARMUP):
        mgr.sign(data)
    samples = []
    for _ in range(n_reps):
        _, sign_time = mgr.sign(data)
        samples.append(sign_time)
    return samples


def bench_verify(scheme: str, data: bytes, signature: bytes, public_key: bytes, n_reps: int):
    for _ in range(WARMUP):
        SignatureManager(
            scheme, generate_keypair=False
        ).verify(data, signature, public_key)

    samples = []
    for _ in range(n_reps):
        _, verify_time = SignatureManager(
            scheme, generate_keypair=False
        ).verify(data, signature, public_key)
        samples.append(verify_time)
    return samples


def main():
    torch.manual_seed(42)
    state_dict = CIFAR10CNN().state_dict()
    data = (
        weights_to_bytes(state_dict)
        + BENCH_ROUND.to_bytes(4, "big")
        + BENCH_NODE_ID.to_bytes(8, "big")
        )
    governor = read_cpu_governor()

    print(
        f"Fixed serialized update payload: {len(data)} bytes "
        f"(weights + round + node_id)"
    )
    print(f"CPU governor: {governor}")
    print(f"Schemes: {ALL_SCHEMES}")

    rows = []
    t_start_all = time.perf_counter()

    for scheme in ALL_SCHEMES:
        n_reps = n_reps_for(scheme)
        print(f"\n=== {scheme}  (n_reps={n_reps}, warmup={WARMUP}) ===")

        t0 = time.perf_counter()
        keygen_samples, mgr = bench_keygen(scheme, n_reps)
        print(f"  keygen done in {time.perf_counter() - t0:6.1f}s wall")

        t0 = time.perf_counter()
        sign_samples = bench_sign(mgr, data, n_reps)
        print(f"  sign   done in {time.perf_counter() - t0:6.1f}s wall")

        signature, _ = mgr.sign(data)
        public_key = mgr.public_key_bytes

        t0 = time.perf_counter()
        verify_samples = bench_verify(scheme, data, signature, public_key, n_reps)
        print(f"  verify done in {time.perf_counter() - t0:6.1f}s wall")

        for operation, samples in (
            ("keygen", keygen_samples),
            ("sign", sign_samples),
            ("verify", verify_samples),
        ):
            s = summarize(samples)
            rows.append({"scheme": scheme, "operation": operation, "cpu_governor": governor, **s})
            print(
                f"    {operation:7s} mean={s['mean_s']*1e3:10.4f} ms  "
                f"median={s['median_s']*1e3:10.4f} ms  std={s['std_s']*1e3:9.4f} ms  "
                f"p95={s['p95_s']*1e3:10.4f} ms  min={s['min_s']*1e3:10.4f} ms  "
                f"max={s['max_s']*1e3:10.4f} ms"
            )

    total_wall = time.perf_counter() - t_start_all
    print(f"\nTotal wall time: {total_wall:.1f}s ({total_wall/60:.1f} min)")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {CSV_PATH}")


if __name__ == "__main__":
    main()

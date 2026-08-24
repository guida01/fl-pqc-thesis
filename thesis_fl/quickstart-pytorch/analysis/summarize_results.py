#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare


SCHEMES = [
    "no_signature",
    "RSA-2048",
    "ECDSA-256",
    "ML-DSA-44",
    "ML-DSA-65",
    "ML-DSA-87",
    "Falcon-padded-512",
    "SLH_DSA_PURE_SHA2_128S",
]

DISPLAY_NAMES = {
    "no_signature": "no_signature",
    "RSA-2048": "RSA-2048",
    "ECDSA-256": "ECDSA-256",
    "ML-DSA-44": "ML-DSA-44",
    "ML-DSA-65": "ML-DSA-65",
    "ML-DSA-87": "ML-DSA-87",
    "Falcon-padded-512": "Falcon-padded-512",
    "SLH_DSA_PURE_SHA2_128S": "SLH-DSA-SHA2-128s",
}

RUNS = 5
ROUNDS = 30

UPDATE_COLUMNS = [
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

EVAL_COLUMNS = [
    "run",
    "round",
    "accuracy",
    "loss",
]

TIMING_COLUMNS = [
    "keygen_time",
    "client_serialize_time",
    "sign_time",
    "server_serialize_time",
    "verify_time",
    "server_verify_total_time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise final FL signature experiment results."
    )
    parser.add_argument(
        "--clients",
        type=int,
        choices=[5, 10],
        required=True,
        help="Number of clients in the campaign.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override the default results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the default analysis output directory.",
    )
    return parser.parse_args()


def load_and_validate(
    results_dir: Path,
    clients: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    expected_updates = RUNS * ROUNDS * clients
    expected_eval = RUNS * (ROUNDS + 1)

    updates: dict[str, pd.DataFrame] = {}
    evals: dict[str, pd.DataFrame] = {}

    print("=" * 78)
    print(f"VALIDATING {clients}-CLIENT CAMPAIGN")
    print("=" * 78)

    for scheme in SCHEMES:
        update_path = results_dir / f"{scheme}.csv"
        eval_path = results_dir / f"{scheme}_eval.csv"

        if not update_path.exists():
            raise FileNotFoundError(f"Missing update CSV: {update_path}")
        if not eval_path.exists():
            raise FileNotFoundError(f"Missing evaluation CSV: {eval_path}")

        u = pd.read_csv(update_path)
        e = pd.read_csv(eval_path)

        if list(u.columns) != UPDATE_COLUMNS:
            raise ValueError(
                f"{scheme}: unexpected update schema:\n{list(u.columns)}"
            )

        if list(e.columns) != EVAL_COLUMNS:
            raise ValueError(
                f"{scheme}: unexpected evaluation schema:\n{list(e.columns)}"
            )

        if len(u) != expected_updates:
            raise ValueError(
                f"{scheme}: {len(u)} update rows, expected {expected_updates}"
            )

        if len(e) != expected_eval:
            raise ValueError(
                f"{scheme}: {len(e)} evaluation rows, expected {expected_eval}"
            )

        if set(u["run"]) != set(range(1, RUNS + 1)):
            raise ValueError(f"{scheme}: incorrect update run IDs")

        if set(e["run"]) != set(range(1, RUNS + 1)):
            raise ValueError(f"{scheme}: incorrect evaluation run IDs")

        if set(u["scheme"]) != {scheme}:
            raise ValueError(f"{scheme}: incorrect scheme column")

        if u.duplicated(["run", "round", "node_id"]).any():
            raise ValueError(f"{scheme}: duplicate logical update rows")

        if e.duplicated(["run", "round"]).any():
            raise ValueError(f"{scheme}: duplicate evaluation rows")

        if u["has_nan"].fillna(True).astype(bool).any():
            raise ValueError(f"{scheme}: has_nan=True found")

        if not np.isfinite(u["train_time"]).all():
            raise ValueError(f"{scheme}: non-finite train_time")

        if not np.isfinite(u["train_loss"]).all():
            raise ValueError(f"{scheme}: non-finite train_loss")

        if not np.isfinite(e["accuracy"]).all():
            raise ValueError(f"{scheme}: non-finite evaluation accuracy")

        if not np.isfinite(e["loss"]).all():
            raise ValueError(f"{scheme}: non-finite evaluation loss")

        for run in range(1, RUNS + 1):
            ur = u[u["run"] == run]
            er = e[e["run"] == run]

            if len(ur) != ROUNDS * clients:
                raise ValueError(
                    f"{scheme} run {run}: wrong update count {len(ur)}"
                )

            if set(ur["round"]) != set(range(1, ROUNDS + 1)):
                raise ValueError(
                    f"{scheme} run {run}: incorrect training rounds"
                )

            per_round = ur.groupby("round").size()
            if not (per_round == clients).all():
                raise ValueError(
                    f"{scheme} run {run}: incorrect updates per round"
                )

            unique_nodes = ur.groupby("round")["node_id"].nunique()
            if not (unique_nodes == clients).all():
                raise ValueError(
                    f"{scheme} run {run}: duplicate/missing node within round"
                )

            if len(er) != ROUNDS + 1:
                raise ValueError(
                    f"{scheme} run {run}: wrong evaluation count"
                )

            if set(er["round"]) != set(range(0, ROUNDS + 1)):
                raise ValueError(
                    f"{scheme} run {run}: incorrect evaluation rounds"
                )

        if scheme == "no_signature":
            zero_columns = TIMING_COLUMNS + [
                "signed_payload_size",
                "sig_size",
                "pubkey_size",
            ]

            for col in zero_columns:
                if not (u[col].fillna(0) == 0).all():
                    raise ValueError(
                        f"no_signature: {col} should contain only zero"
                    )
        else:
            if not u["sig_valid"].astype(bool).all():
                raise ValueError(f"{scheme}: invalid signature found")

            for col in [
                "signed_payload_size",
                "sig_size",
                "pubkey_size",
            ]:
                if not (u[col] > 0).all():
                    raise ValueError(
                        f"{scheme}: non-positive value in {col}"
                    )

            for col in TIMING_COLUMNS:
                if not (u[col] >= 0).all():
                    raise ValueError(
                        f"{scheme}: negative timing in {col}"
                    )

        updates[scheme] = u
        evals[scheme] = e

        print(
            f"{DISPLAY_NAMES[scheme]:23s} "
            f"updates={len(u):4d}/{expected_updates}  "
            f"eval={len(e):3d}/{expected_eval}  PASS"
        )

    # Matched runs should start from the same global model for every scheme.
    for run in range(1, RUNS + 1):
        round0_accuracy = []
        round0_loss = []

        for scheme in SCHEMES:
            row = evals[scheme][
                (evals[scheme]["run"] == run)
                & (evals[scheme]["round"] == 0)
            ].iloc[0]

            round0_accuracy.append(float(row["accuracy"]))
            round0_loss.append(float(row["loss"]))

        if not np.allclose(
            round0_accuracy,
            round0_accuracy[0],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Run {run}: round-0 accuracy differs across schemes"
            )

        if not np.allclose(
            round0_loss,
            round0_loss[0],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Run {run}: round-0 loss differs across schemes"
            )

    print("\nDATASET VALIDATION: PASS")
    print("MATCHED ROUND-0 MODELS: PASS")

    return updates, evals


def build_timing_run_summary(
    updates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for scheme in SCHEMES:
        u = updates[scheme]

        grouped = u.groupby("run")[
            TIMING_COLUMNS + ["train_time", "train_loss"]
        ].mean()

        for run, row in grouped.iterrows():
            out = {
                "scheme": DISPLAY_NAMES[scheme],
                "scheme_id": scheme,
                "run": int(run),
            }

            for col in TIMING_COLUMNS:
                out[f"{col}_ms"] = float(row[col]) * 1000

            out["train_time_s"] = float(row["train_time"])
            out["train_loss"] = float(row["train_loss"])

            rows.append(out)

    return pd.DataFrame(rows)


def build_derived_costs(
    updates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for scheme in SCHEMES:
        u = updates[scheme].copy()

        u["crypto_operation_cost"] = (
            u["keygen_time"]
            + u["sign_time"]
            + u["verify_time"]
        )

        u["security_wrapper_cost"] = (
            u["client_serialize_time"]
            + u["keygen_time"]
            + u["sign_time"]
            + u["server_serialize_time"]
            + u["server_verify_total_time"]
        )

        grouped = u.groupby("run")[
            ["crypto_operation_cost", "security_wrapper_cost"]
        ].mean()

        for run, row in grouped.iterrows():
            rows.append(
                {
                    "scheme": DISPLAY_NAMES[scheme],
                    "scheme_id": scheme,
                    "run": int(run),
                    "crypto_operation_cost_ms":
                        float(row["crypto_operation_cost"]) * 1000,
                    "security_wrapper_cost_ms":
                        float(row["security_wrapper_cost"]) * 1000,
                }
            )

    return pd.DataFrame(rows)


def build_metadata_summary(
    updates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for scheme in SCHEMES:
        u = updates[scheme]

        sig_mean = float(u["sig_size"].mean())
        pk_mean = float(u["pubkey_size"].mean())
        payload_mean = float(u["signed_payload_size"].mean())
        metadata_mean = sig_mean + pk_mean

        relative = (
            metadata_mean / payload_mean * 100
            if payload_mean > 0
            else 0.0
        )

        rows.append(
            {
                "scheme": DISPLAY_NAMES[scheme],
                "scheme_id": scheme,
                "signed_payload_size_B": payload_mean,
                "sig_size_mean_B": sig_mean,
                "sig_size_min_B": float(u["sig_size"].min()),
                "sig_size_max_B": float(u["sig_size"].max()),
                "pubkey_size_mean_B": pk_mean,
                "pubkey_size_min_B": float(u["pubkey_size"].min()),
                "pubkey_size_max_B": float(u["pubkey_size"].max()),
                "metadata_mean_B": metadata_mean,
                "relative_metadata_pct": relative,
            }
        )

    return pd.DataFrame(rows)


def build_break_even(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in metadata.iterrows():
        if row["scheme_id"] == "no_signature":
            continue

        m = float(row["metadata_mean_B"])

        rows.append(
            {
                "scheme": row["scheme"],
                "scheme_id": row["scheme_id"],
                "metadata_mean_B": m,
                "payload_at_1pct_B": m / 0.01,
                "payload_at_1pct_KB": m / 0.01 / 1000,
                "payload_at_5pct_B": m / 0.05,
                "payload_at_5pct_KB": m / 0.05 / 1000,
                "payload_at_10pct_B": m / 0.10,
                "payload_at_10pct_KB": m / 0.10 / 1000,
            }
        )

    return pd.DataFrame(rows)


def build_final_accuracy_by_run(
    evals: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for scheme in SCHEMES:
        e = evals[scheme]
        final = e[e["round"] == ROUNDS].sort_values("run")

        for _, row in final.iterrows():
            rows.append(
                {
                    "scheme": DISPLAY_NAMES[scheme],
                    "scheme_id": scheme,
                    "run": int(row["run"]),
                    "final_accuracy": float(row["accuracy"]),
                    "final_loss": float(row["loss"]),
                }
            )

    return pd.DataFrame(rows)


def build_final_accuracy_summary(
    final_accuracy: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for scheme in SCHEMES:
        name = DISPLAY_NAMES[scheme]
        x = final_accuracy[
            final_accuracy["scheme_id"] == scheme
        ]["final_accuracy"]

        rows.append(
            {
                "scheme": name,
                "scheme_id": scheme,
                "mean_accuracy": float(x.mean()),
                "std_accuracy": float(x.std(ddof=1)),
                "min_accuracy": float(x.min()),
                "max_accuracy": float(x.max()),
            }
        )

    return pd.DataFrame(rows)


def build_accuracy_by_round(
    evals: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for scheme in SCHEMES:
        e = evals[scheme]

        grouped = e.groupby("round").agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_loss=("loss", "mean"),
            std_loss=("loss", "std"),
        )

        for round_number, row in grouped.iterrows():
            rows.append(
                {
                    "scheme": DISPLAY_NAMES[scheme],
                    "scheme_id": scheme,
                    "round": int(round_number),
                    "mean_accuracy": float(row["mean_accuracy"]),
                    "std_accuracy": float(row["std_accuracy"]),
                    "mean_loss": float(row["mean_loss"]),
                    "std_loss": float(row["std_loss"]),
                }
            )

    return pd.DataFrame(rows)


def friedman_final_accuracy(
    final_accuracy: pd.DataFrame,
) -> tuple[float, int, float, float]:
    groups = []

    for scheme in SCHEMES:
        x = (
            final_accuracy[
                final_accuracy["scheme_id"] == scheme
            ]
            .sort_values("run")["final_accuracy"]
            .to_numpy()
        )
        groups.append(x)

    stat, p = friedmanchisquare(*groups)

    k = len(SCHEMES)
    n = RUNS
    kendall_w = stat / (n * (k - 1))

    return float(stat), k - 1, float(p), float(kendall_w)


def write_friedman_report(
    path: Path,
    stat: float,
    df: int,
    p: float,
    kendall_w: float,
) -> None:
    if p < 0.05:
        decision = (
            "Global Friedman test is significant. "
            "Corrected paired post-hoc comparisons may be justified."
        )
    else:
        decision = (
            "Global Friedman test is not significant. "
            "No post-hoc pairwise comparisons are performed."
        )

    text = (
        "Final central accuracy: matched-run Friedman test\n"
        "================================================\n"
        f"Runs (n):          {RUNS}\n"
        f"Configurations:   {len(SCHEMES)}\n"
        f"Degrees freedom:  {df}\n"
        f"Chi-square:       {stat:.8f}\n"
        f"p-value:          {p:.8f}\n"
        f"Kendall's W:      {kendall_w:.8f}\n"
        "\n"
        f"{decision}\n"
        "\n"
        "The inferential unit is the run. Client-update rows are not "
        "treated as independent replicates.\n"
    )

    path.write_text(text, encoding="utf-8")


def print_summary(
    clients: int,
    output_dir: Path,
    derived: pd.DataFrame,
    metadata: pd.DataFrame,
    accuracy_summary: pd.DataFrame,
    stat: float,
    df: int,
    p: float,
    kendall_w: float,
) -> None:
    print("\n" + "=" * 78)
    print(f"{clients}-CLIENT FINAL SUMMARY")
    print("=" * 78)

    print("\nSecurity-wrapper cost, mean ± SD across run-level means:")
    for scheme in SCHEMES:
        x = derived[
            derived["scheme_id"] == scheme
        ]["security_wrapper_cost_ms"]

        print(
            f"  {DISPLAY_NAMES[scheme]:23s} "
            f"{x.mean():10.4f} ± {x.std(ddof=1):8.4f} ms"
        )

    print("\nCryptographic metadata:")
    for scheme in SCHEMES:
        if scheme == "no_signature":
            continue

        row = metadata[metadata["scheme_id"] == scheme].iloc[0]

        print(
            f"  {DISPLAY_NAMES[scheme]:23s} "
            f"{row['metadata_mean_B']:9.3f} B  "
            f"{row['relative_metadata_pct']:.6f}%"
        )

    print("\nFinal accuracy, mean ± SD:")
    for scheme in SCHEMES:
        row = accuracy_summary[
            accuracy_summary["scheme_id"] == scheme
        ].iloc[0]

        print(
            f"  {DISPLAY_NAMES[scheme]:23s} "
            f"{row['mean_accuracy']:.6f} "
            f"± {row['std_accuracy']:.6f}"
        )

    print("\nFriedman final accuracy:")
    print(f"  chi-square = {stat:.8f}")
    print(f"  df         = {df}")
    print(f"  p          = {p:.8f}")
    print(f"  Kendall W  = {kendall_w:.8f}")

    print(f"\nOutputs written to: {output_dir}")
    print("\nANALYSIS COMPLETE")


def main() -> None:
    args = parse_args()

    results_dir = (
        args.results_dir
        if args.results_dir is not None
        else Path(f"results_final_{args.clients}clients")
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("analysis") / "results" / f"{args.clients}clients"
    )

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory does not exist: {results_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    updates, evals = load_and_validate(
        results_dir=results_dir,
        clients=args.clients,
    )

    timing = build_timing_run_summary(updates)
    derived = build_derived_costs(updates)
    metadata = build_metadata_summary(updates)
    break_even = build_break_even(metadata)
    final_accuracy = build_final_accuracy_by_run(evals)
    accuracy_summary = build_final_accuracy_summary(final_accuracy)
    accuracy_by_round = build_accuracy_by_round(evals)

    stat, df, p, kendall_w = friedman_final_accuracy(final_accuracy)

    timing.to_csv(
        output_dir / "timing_run_summary.csv",
        index=False,
    )

    derived.to_csv(
        output_dir / "derived_costs.csv",
        index=False,
    )

    metadata.to_csv(
        output_dir / "metadata_summary.csv",
        index=False,
    )

    break_even.to_csv(
        output_dir / "break_even_sizes.csv",
        index=False,
    )

    final_accuracy.to_csv(
        output_dir / "final_accuracy_by_run.csv",
        index=False,
    )

    accuracy_summary.to_csv(
        output_dir / "final_accuracy_summary.csv",
        index=False,
    )

    accuracy_by_round.to_csv(
        output_dir / "accuracy_by_round.csv",
        index=False,
    )

    write_friedman_report(
        output_dir / "friedman_accuracy.txt",
        stat,
        df,
        p,
        kendall_w,
    )

    print_summary(
        clients=args.clients,
        output_dir=output_dir,
        derived=derived,
        metadata=metadata,
        accuracy_summary=accuracy_summary,
        stat=stat,
        df=df,
        p=p,
        kendall_w=kendall_w,
    )


if __name__ == "__main__":
    main()

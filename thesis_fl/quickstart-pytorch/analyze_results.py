"""
FL-PQC thesis — results analysis
Data: 8 schemes × 5 runs × 30 rounds × 5 clients = 750 rows/scheme
Columns: run, round, node_id, scheme, keygen_time, sign_time, verify_time,
         train_time, train_loss, payload_size, sig_size, pubkey_size,
         sig_valid, has_nan
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

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

LABELS = {
    "no_signature":              "No Sig.",
    "RSA-2048":                  "RSA-2048",
    "ECDSA-256":                 "ECDSA-256",
    "ML-DSA-44":                 "ML-DSA-44",
    "ML-DSA-65":                 "ML-DSA-65",
    "ML-DSA-87":                 "ML-DSA-87",
    "Falcon-padded-512":         "Falcon-512",
    "SLH_DSA_PURE_SHA2_128S": "SLH-DSA",
}

COLORS = {
    "no_signature":              "#222222",
    "RSA-2048":                  "#888780",
    "ECDSA-256":                 "#5F5E5A",
    "ML-DSA-44":                 "#1D9E75",
    "ML-DSA-65":                 "#0F6E56",
    "ML-DSA-87":                 "#085041",
    "Falcon-padded-512":         "#185FA5",
    "SLH_DSA_PURE_SHA2_128S": "#BA7517",
}

CLASSICAL = {"RSA-2048", "ECDSA-256"}

LEGEND_PATCHES = [
    mpatches.Patch(color="#222222", label="Baseline (no signature)"),
    mpatches.Patch(color="#888780", label="Classical"),
    mpatches.Patch(color="#1D9E75", label="PQC (NIST)"),
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all():
    frames = []
    for scheme in SCHEMES:
        path = os.path.join(RESULTS_DIR, f"{scheme}.csv")
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            continue
        df = pd.read_csv(path)
        df["scheme"] = scheme
        frames.append(df)
        print(f"  Loaded {scheme}: {len(df)} rows, "
              f"{df['run'].nunique()} runs, {df['round'].nunique()} rounds, "
              f"{df['node_id'].nunique()} nodes, "
              f"sig_valid={df['sig_valid'].all()}, has_nan={df['has_nan'].any()}")
    return pd.concat(frames, ignore_index=True)


# ── Aggregate stats ────────────────────────────────────────────────────────────

def aggregate(df):
    """Per-scheme summary: mean ± std of all time/size columns."""
    g = df.groupby("scheme")
    stats = g.agg(
        keygen_mean  =("keygen_time",  "mean"),
        keygen_std   =("keygen_time",  "std"),
        sign_mean    =("sign_time",    "mean"),
        sign_std     =("sign_time",    "std"),
        verify_mean  =("verify_time",  "mean"),
        verify_std   =("verify_time",  "std"),
        train_mean   =("train_time",   "mean"),
        train_std    =("train_time",   "std"),
        sig_size     =("sig_size",     "mean"),    # mean handles ECDSA variable-length DER
        pubkey_size  =("pubkey_size",  "mean"),
        payload_size =("payload_size", "mean"),
        n            =("sign_time",    "count"),
    ).reset_index()

    # ms conversion for time columns
    for col in ["keygen_mean", "keygen_std",
                "sign_mean",   "sign_std",
                "verify_mean", "verify_std",
                "train_mean",  "train_std"]:
        stats[col] *= 1000

    # overhead: (sign + verify) as % of (train + sign + verify)
    stats["overhead_pct"] = (
        (stats["sign_mean"] + stats["verify_mean"])
        / (stats["train_mean"] + stats["sign_mean"] + stats["verify_mean"])
        * 100
    )

    # total sig bytes per round (nodes_per_round × sig_size)
    nodes_per_round = df.groupby(["scheme", "round"])["node_id"].nunique().groupby("scheme").mean()
    stats = stats.merge(nodes_per_round.rename("nodes_per_round"), on="scheme")
    stats["comm_kb_per_round"] = (stats["sig_size"] * stats["nodes_per_round"]) / 1024

    order = {s: i for i, s in enumerate(SCHEMES)}
    stats["_o"] = stats["scheme"].map(order)
    return stats.sort_values("_o").drop(columns="_o").reset_index(drop=True)


# ── Generic bar chart ──────────────────────────────────────────────────────────

def bar_chart(stats, y, err=None, title="", ylabel="", filename="",
              log_scale=False, value_fmt=".2f"):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [LABELS[s] for s in stats["scheme"]]
    colors = [COLORS[s] for s in stats["scheme"]]
    yerr   = stats[err] if err else None
    bars   = ax.bar(labels, stats[y], yerr=yerr,
                    color=colors, capsize=4, edgecolor="white", linewidth=0.5)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Signature scheme")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(handles=LEGEND_PATCHES, loc="upper left")

    for bar, val in zip(bars, stats[y]):
        offset = bar.get_height() * (1.3 if log_scale else 1.02)
        ax.text(bar.get_x() + bar.get_width() / 2, offset,
                f"{val:{value_fmt}}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    _save(filename)


# ── Overhead breakdown (stacked bar) ──────────────────────────────────────────

def overhead_breakdown(stats):
    """Stacked bar: train vs sign vs verify time (ms), per scheme."""
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [LABELS[s] for s in stats["scheme"]]
    x = np.arange(len(labels))
    w = 0.55

    b1 = ax.bar(x, stats["train_mean"],  w, label="Training",     color="#AECDE8")
    b2 = ax.bar(x, stats["sign_mean"],   w, label="Signing",      color="#E8875A",
                bottom=stats["train_mean"])
    b3 = ax.bar(x, stats["verify_mean"], w, label="Verification", color="#D64545",
                bottom=stats["train_mean"] + stats["sign_mean"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("FL round time breakdown: training vs signature operations")
    ax.legend()

    # annotate overhead % on top of each bar
    for i, row in stats.iterrows():
        total = row["train_mean"] + row["sign_mean"] + row["verify_mean"]
        ax.text(x[i], total * 1.01,
                f"{row['overhead_pct']:.2f}%", ha="center", va="bottom",
                fontsize=8, color="#333333")

    plt.tight_layout()
    _save("overhead_breakdown.png")


# ── Trade-off scatter ──────────────────────────────────────────────────────────

def tradeoff_scatter(stats):
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in stats.iterrows():
        s = row["scheme"]
        ax.scatter(row["sign_mean"], row["sig_size"],
                   color=COLORS[s], s=130, zorder=3,
                   marker="o" if s not in CLASSICAL else "s")
        ax.annotate(LABELS[s], (row["sign_mean"], row["sig_size"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Mean signing time (ms)")
    ax.set_ylabel("Signature size (bytes)")
    ax.set_title("Trade-off: signing time vs signature size")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=LEGEND_PATCHES)
    plt.tight_layout()
    _save("tradeoff_sign_vs_size.png")


# ── Training-loss convergence ──────────────────────────────────────────────────

def loss_convergence(df):
    """Mean train_loss per round (averaged across nodes and runs, client nodes only)."""
    clients = df.dropna(subset=["train_loss"])
    per_round = (
        clients.groupby(["scheme", "run", "round"])["train_loss"].mean()
               .groupby(["scheme", "round"]).mean()
               .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for scheme in SCHEMES:
        sub = per_round[per_round["scheme"] == scheme]
        if sub.empty:
            continue
        ax.plot(sub["round"], sub["train_loss"],
                color=COLORS[scheme], label=LABELS[scheme],
                linewidth=1.5,
                linestyle="--" if scheme in CLASSICAL else "-")

    ax.set_xlabel("Round")
    ax.set_ylabel("Mean training loss")
    ax.set_title("Training loss convergence per scheme")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save("loss_convergence.png")


# ── Sign time stability across rounds ─────────────────────────────────────────

def sign_time_per_round(df):
    """Mean signing time per round per scheme — checks timing stability."""
    per_round = (
        df.groupby(["scheme", "round"])["sign_time"]
          .mean()
          .reset_index()
    )
    per_round["sign_ms"] = per_round["sign_time"] * 1000

    fig, ax = plt.subplots(figsize=(9, 5))
    for scheme in SCHEMES:
        sub = per_round[per_round["scheme"] == scheme]
        ax.plot(sub["round"], sub["sign_ms"],
                color=COLORS[scheme], label=LABELS[scheme],
                linewidth=1.2,
                linestyle="--" if scheme in CLASSICAL else "-")

    ax.set_xlabel("Round")
    ax.set_ylabel("Mean signing time (ms)")
    ax.set_title("Signing time stability across FL rounds")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save("signing_time_per_round.png")


# ── LaTeX table ────────────────────────────────────────────────────────────────

def latex_table(stats):
    n = int(stats["n"].iloc[0])
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Performance of digital signature schemes in FL "
        f"(mean~$\\pm$~std over {n} measurements)" + r"}",
        r"\label{tab:signature-performance}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Scheme & Keygen (ms) & Sign (ms) & Verify (ms) "
        r"& Sig (B) & PK (B) & Overhead (\%) & Comm./round (KB) \\",
        r"\midrule",
    ]

    for _, row in stats.iterrows():
        name   = LABELS[row["scheme"]].replace("-", r"\nobreakdash-")
        keygen = f"{row['keygen_mean']:.3f} $\\pm$ {row['keygen_std']:.3f}"
        sign   = f"{row['sign_mean']:.3f} $\\pm$ {row['sign_std']:.3f}"
        vfy    = f"{row['verify_mean']:.3f} $\\pm$ {row['verify_std']:.3f}"
        sig    = f"{row['sig_size']:.0f}"
        pk     = f"{row['pubkey_size']:.0f}"
        ovh    = f"{row['overhead_pct']:.2f}"
        comm   = f"{row['comm_kb_per_round']:.1f}"
        lines.append(f"{name} & {keygen} & {sign} & {vfy} & {sig} & {pk} & {ovh} & {comm} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table = "\n".join(lines)

    path = os.path.join(RESULTS_DIR, "latex_table.tex")
    with open(path, "w") as f:
        f.write(table)
    print(f"  Saved: {path}")
    return table


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(stats):
    w = 90
    print("\n" + "=" * w)
    print(f"{'Scheme':<16} {'Keygen(ms)':>11} {'Sign(ms)':>10} {'Verify(ms)':>11} "
          f"{'Sig(B)':>7} {'PK(B)':>6} {'Ovhd%':>7} {'Comm/round':>11}")
    print("-" * w)
    for _, row in stats.iterrows():
        print(f"{LABELS[row['scheme']]:<16} "
              f"{row['keygen_mean']:>8.3f}±{row['keygen_std']:.3f}  "
              f"{row['sign_mean']:>7.3f}±{row['sign_std']:.3f}  "
              f"{row['verify_mean']:>7.3f}±{row['verify_std']:.3f}  "
              f"{row['sig_size']:>7.1f}  "
              f"{row['pubkey_size']:>5.1f}  "
              f"{row['overhead_pct']:>6.2f}%  "
              f"{row['comm_kb_per_round']:>8.1f} KB")
    print("=" * w)

    # Exclude no_signature from comparisons that don't apply to it
    signed = stats[stats["scheme"] != "no_signature"]
    fastest  = signed.loc[signed["sign_mean"].idxmin()]
    slowest  = signed.loc[signed["sign_mean"].idxmax()]
    smallest = signed.loc[signed["sig_size"].idxmin()]
    largest  = signed.loc[signed["sig_size"].idxmax()]
    ratio    = slowest["sign_mean"] / fastest["sign_mean"]
    best_ovh = signed.loc[signed["overhead_pct"].idxmin()]

    print(f"\nFastest sign:       {LABELS[fastest['scheme']]}  ({fastest['sign_mean']:.3f} ms)")
    print(f"Slowest sign:       {LABELS[slowest['scheme']]}  ({slowest['sign_mean']:.3f} ms, {ratio:.0f}× slower)")
    print(f"Smallest sig:       {LABELS[smallest['scheme']]}  ({smallest['sig_size']:.1f} B)")
    print(f"Largest sig:        {LABELS[largest['scheme']]}  ({largest['sig_size']:.1f} B)")
    print(f"Lowest FL overhead: {LABELS[best_ovh['scheme']]}  ({best_ovh['overhead_pct']:.2f}%)")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(filename):
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading CSVs...")
    df = load_all()
    print(f"\nTotal rows: {len(df)}  |  Schemes: {df['scheme'].nunique()}  |  "
          f"All sig_valid: {df['sig_valid'].all()}  |  Any has_nan: {df['has_nan'].any()}\n")

    stats = aggregate(df)

    print("Generating plots...")
    bar_chart(stats, "keygen_mean", "keygen_std",
              "Mean key generation time per scheme",
              "Time (ms)", "keygen_time.png", log_scale=True, value_fmt=".3f")

    bar_chart(stats, "sign_mean", "sign_std",
              "Mean signing time per scheme",
              "Time (ms)", "signing_time.png", log_scale=True, value_fmt=".3f")

    bar_chart(stats, "verify_mean", "verify_std",
              "Mean verification time per scheme",
              "Time (ms)", "verification_time.png", value_fmt=".3f")

    bar_chart(stats, "sig_size", None,
              "Signature size per scheme",
              "Size (bytes)", "signature_size.png", value_fmt=".0f")

    bar_chart(stats, "overhead_pct", None,
              "Signature overhead as % of FL round time",
              "Overhead (%)", "overhead_pct.png", value_fmt=".2f")

    bar_chart(stats, "comm_kb_per_round", None,
              "Signature communication overhead per FL round",
              "Size (KB)", "communication_overhead.png", value_fmt=".1f")

    overhead_breakdown(stats)
    tradeoff_scatter(stats)
    loss_convergence(df)
    sign_time_per_round(df)

    print("\nGenerating LaTeX table...")
    print(latex_table(stats))

    print_summary(stats)
    print(f"\nAll outputs saved to: {PLOTS_DIR}/")

"""
Thesis analysis script — compare 7 signature schemes in FL.
Generates plots and LaTeX table for the thesis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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
    "RSA-2048",
    "ECDSA-256",
    "ML-DSA-44",
    "ML-DSA-65",
    "ML-DSA-87",
    "Falcon-padded-512",
    "SLH_DSA_PURE_SHA2_128S",
]

SCHEME_LABELS = {
    "RSA-2048":             "RSA-2048",
    "ECDSA-256":            "ECDSA-256",
    "ML-DSA-44":            "ML-DSA-44",
    "ML-DSA-65":            "ML-DSA-65",
    "ML-DSA-87":            "ML-DSA-87",
    "Falcon-padded-512":    "Falcon-512",
    "SLH_DSA_PURE_SHA2_128S": "SLH-DSA",
}

COLORS = {
    "RSA-2048":             "#888780",  # gray — classical
    "ECDSA-256":            "#5F5E5A",  # dark gray — classical
    "ML-DSA-44":            "#1D9E75",  # teal — PQC
    "ML-DSA-65":            "#0F6E56",  # dark teal — PQC
    "ML-DSA-87":            "#085041",  # darkest teal — PQC
    "Falcon-padded-512":    "#185FA5",  # blue — PQC
    "SLH_DSA_PURE_SHA2_128S": "#BA7517", # amber — PQC
}


# ── Load data ──────────────────────────────────────────────────────────────────

def load_results():
    frames = []
    for scheme in SCHEMES:
        path = os.path.join(RESULTS_DIR, f"{scheme}.csv")
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
        df = pd.read_csv(path)
        df["scheme"] = scheme
        frames.append(df)
        print(f"  Loaded {scheme}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True)


def compute_stats(df):
    stats = df.groupby("scheme").agg(
        sign_mean   =("sign_time",   "mean"),
        sign_std    =("sign_time",   "std"),
        verify_mean =("verify_time", "mean"),
        verify_std  =("verify_time", "std"),
        sig_mean    =("sig_size",    "mean"),
        sig_std     =("sig_size",    "std"),
        pk_mean     =("pubkey_size", "mean"),
        n_samples   =("sign_time",   "count"),
    ).reset_index()
    # convert times to ms
    for col in ["sign_mean", "sign_std", "verify_mean", "verify_std"]:
        stats[col] = stats[col] * 1000
    return stats


# ── Plots ──────────────────────────────────────────────────────────────────────

def bar_chart(stats, y_col, err_col, title, ylabel, filename, log_scale=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [SCHEME_LABELS[s] for s in stats["scheme"]]
    colors = [COLORS[s] for s in stats["scheme"]]
    bars = ax.bar(labels, stats[y_col], yerr=stats[err_col],
                  color=colors, capsize=4, edgecolor="white", linewidth=0.5)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Signature scheme")
    ax.tick_params(axis="x", rotation=30)

    # value labels on bars
    for bar, val in zip(bars, stats[y_col]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * (1.05 if not log_scale else 1.3),
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # legend: classical vs PQC
    from matplotlib.patches import Patch
    legend = [Patch(color="#888780", label="Classical"),
              Patch(color="#1D9E75", label="PQC (NIST)")]
    ax.legend(handles=legend, loc="upper left")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def tradeoff_plot(stats):
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in stats.iterrows():
        s = row["scheme"]
        ax.scatter(row["sign_mean"], row["sig_mean"],
                   color=COLORS[s], s=120, zorder=3)
        ax.annotate(SCHEME_LABELS[s],
                    (row["sign_mean"], row["sig_mean"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Mean signing time (ms)")
    ax.set_ylabel("Signature size (bytes)")
    ax.set_title("Trade-off: signing time vs signature size")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    from matplotlib.patches import Patch
    legend = [Patch(color="#888780", label="Classical"),
              Patch(color="#1D9E75", label="PQC (NIST)")]
    ax.legend(handles=legend)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "tradeoff_sign_vs_size.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def overhead_plot(stats, num_clients=2, num_rounds=10):
    total_sigs = num_clients * num_rounds
    stats["overhead_kb"] = (stats["sig_mean"] * total_sigs) / 1024
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [SCHEME_LABELS[s] for s in stats["scheme"]]
    colors = [COLORS[s] for s in stats["scheme"]]
    bars = ax.bar(labels, stats["overhead_kb"], color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_title(f"Total signature overhead ({num_clients} clients × {num_rounds} rounds = {total_sigs} signatures)")
    ax.set_ylabel("Total size (KB)")
    ax.set_xlabel("Signature scheme")
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, stats["overhead_kb"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.1f} KB", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "communication_overhead.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── LaTeX table ───────────────────────────────────────────────────────────────

def latex_table(stats):
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance of digital signature schemes in FL (mean ± std, " +
                 f"{int(stats['n_samples'].iloc[0])} measurements){'}'}")
    lines.append(r"\label{tab:signature-performance}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Scheme & Sign (ms) & Verify (ms) & Sig (B) & PK (B) \\")
    lines.append(r"\midrule")

    for _, row in stats.iterrows():
        name = SCHEME_LABELS[row["scheme"]].replace("-", r"\nobreakdash-")
        sign = f"{row['sign_mean']:.3f} $\\pm$ {row['sign_std']:.3f}"
        vfy  = f"{row['verify_mean']:.3f} $\\pm$ {row['verify_std']:.3f}"
        sig  = f"{row['sig_mean']:.0f}"
        pk   = f"{row['pk_mean']:.0f}"
        lines.append(f"{name} & {sign} & {vfy} & {sig} & {pk} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_str = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, "latex_table.tex")
    with open(path, "w") as f:
        f.write(table_str)
    print(f"  Saved: {path}")
    print("\n" + table_str)
    return table_str


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(stats):
    print("\n" + "="*70)
    print(f"{'Scheme':<22} {'Sign(ms)':>10} {'Verify(ms)':>11} {'Sig(B)':>8} {'PK(B)':>7}")
    print("-"*70)
    for _, row in stats.iterrows():
        print(f"{SCHEME_LABELS[row['scheme']]:<22} "
              f"{row['sign_mean']:>7.3f}±{row['sign_std']:.3f}  "
              f"{row['verify_mean']:>7.3f}±{row['verify_std']:.3f}  "
              f"{row['sig_mean']:>7.0f}  "
              f"{row['pk_mean']:>6.0f}")
    print("="*70)

    fastest = stats.loc[stats["sign_mean"].idxmin(), "scheme"]
    slowest = stats.loc[stats["sign_mean"].idxmax(), "scheme"]
    smallest_sig = stats.loc[stats["sig_mean"].idxmin(), "scheme"]
    largest_sig  = stats.loc[stats["sig_mean"].idxmax(), "scheme"]
    ratio = stats.loc[stats["sign_mean"].idxmax(), "sign_mean"] / \
            stats.loc[stats["sign_mean"].idxmin(), "sign_mean"]

    print(f"\nFastest sign:    {SCHEME_LABELS[fastest]}")
    print(f"Slowest sign:    {SCHEME_LABELS[slowest]} ({ratio:.0f}× slower than fastest)")
    print(f"Smallest sig:    {SCHEME_LABELS[smallest_sig]}")
    print(f"Largest sig:     {SCHEME_LABELS[largest_sig]}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading results...")
    df = load_results()
    print(f"Total rows: {len(df)}\n")

    stats = compute_stats(df)
    # preserve SCHEMES order
    stats["_order"] = stats["scheme"].map({s: i for i, s in enumerate(SCHEMES)})
    stats = stats.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    print("\nGenerating plots...")
    bar_chart(stats, "sign_mean", "sign_std",
              "Mean signing time per scheme",
              "Time (ms)", "signing_time.png", log_scale=True)

    bar_chart(stats, "verify_mean", "verify_std",
              "Mean verification time per scheme",
              "Time (ms)", "verification_time.png")

    bar_chart(stats, "sig_mean", "sig_std",
              "Signature size per scheme",
              "Size (bytes)", "signature_size.png")

    tradeoff_plot(stats)
    overhead_plot(stats)

    print("\nGenerating LaTeX table...")
    latex_table(stats)

    print_summary(stats)
    print(f"\nAll outputs in: {PLOTS_DIR}/")
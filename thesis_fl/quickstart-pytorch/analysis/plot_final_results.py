from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_5 = ANALYSIS_DIR / "results" / "5clients"
RESULTS_10 = ANALYSIS_DIR / "results" / "10clients"
FIGURE_DIR = ANALYSIS_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCHEME_ORDER = [
    "no_signature",
    "RSA-2048",
    "ECDSA-256",
    "ML-DSA-44",
    "ML-DSA-65",
    "ML-DSA-87",
    "Falcon-padded-512",
    "SLH-DSA-SHA2-128s",
]

SIGNED_SCHEMES = [
    "RSA-2048",
    "ECDSA-256",
    "ML-DSA-44",
    "ML-DSA-65",
    "ML-DSA-87",
    "Falcon-padded-512",
    "SLH-DSA-SHA2-128s",
]

DISPLAY_NAMES = {
    "no_signature": "No signature",
    "RSA-2048": "RSA-2048",
    "ECDSA-256": "ECDSA-256",
    "ML-DSA-44": "ML-DSA-44",
    "ML-DSA-65": "ML-DSA-65",
    "ML-DSA-87": "ML-DSA-87",
    "Falcon-padded-512": "Falcon-padded-512",
    "SLH-DSA-SHA2-128s": "SLH-DSA-SHA2-128s",
}

# Keep the same visual identity for a scheme in every figure.
COLORS = {
    "no_signature": "#4d4d4d",
    "RSA-2048": "#1f77b4",
    "ECDSA-256": "#ff7f0e",
    "ML-DSA-44": "#2ca02c",
    "ML-DSA-65": "#d62728",
    "ML-DSA-87": "#9467bd",
    "Falcon-padded-512": "#8c564b",
    "SLH-DSA-SHA2-128s": "#e377c2",
}

LINESTYLES = {
    "no_signature": "--",
    "RSA-2048": "-",
    "ECDSA-256": "-.",
    "ML-DSA-44": ":",
    "ML-DSA-65": (0, (5, 1)),
    "ML-DSA-87": (0, (3, 1, 1, 1)),
    "Falcon-padded-512": (0, (5, 2, 1, 2)),
    "SLH-DSA-SHA2-128s": (0, (1, 1)),
}


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"File is empty: {path}")

    return df


def check_schemes(df: pd.DataFrame, filename: str) -> None:
    observed = set(df["scheme"].unique())
    expected = set(SCHEME_ORDER)

    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)

        raise ValueError(
            f"{filename}: unexpected scheme set.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )


def save_figure(fig, stem: str) -> None:
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"  created {pdf_path.relative_to(ANALYSIS_DIR.parent)}")
    print(f"  created {png_path.relative_to(ANALYSIS_DIR.parent)}")


def wrapper_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["scheme"].isin(SIGNED_SCHEMES)]
        .groupby("scheme")["security_wrapper_cost_ms"]
        .agg(["mean", "std"])
        .reindex(SIGNED_SCHEMES)
    )


# ---------------------------------------------------------------------
# Load final analysis outputs
# ---------------------------------------------------------------------

timing_5 = load_csv(RESULTS_5 / "timing_run_summary.csv")
metadata_5 = load_csv(RESULTS_5 / "metadata_summary.csv")
accuracy_5 = load_csv(RESULTS_5 / "accuracy_by_round.csv")
derived_5 = load_csv(RESULTS_5 / "derived_costs.csv")

timing_10 = load_csv(RESULTS_10 / "timing_run_summary.csv")
metadata_10 = load_csv(RESULTS_10 / "metadata_summary.csv")
accuracy_10 = load_csv(RESULTS_10 / "accuracy_by_round.csv")
derived_10 = load_csv(RESULTS_10 / "derived_costs.csv")


for name, df in [
    ("5clients/timing_run_summary.csv", timing_5),
    ("5clients/metadata_summary.csv", metadata_5),
    ("5clients/accuracy_by_round.csv", accuracy_5),
    ("5clients/derived_costs.csv", derived_5),
    ("10clients/timing_run_summary.csv", timing_10),
    ("10clients/metadata_summary.csv", metadata_10),
    ("10clients/accuracy_by_round.csv", accuracy_10),
    ("10clients/derived_costs.csv", derived_10),
]:
    check_schemes(df, name)


# Structural checks on the final outputs used by the figures.
assert len(derived_5) == 80
assert len(derived_10) == 80
assert len(metadata_5) == 8
assert len(metadata_10) == 8
assert len(accuracy_5) == 248
assert len(accuracy_10) == 248

assert set(accuracy_5["round"]) == set(range(31))
assert set(accuracy_10["round"]) == set(range(31))


print("\nGenerating final thesis figures...\n")


# =====================================================================
# Figure 1
# Security-wrapper processing cost, five-client campaign
# =====================================================================

wrapper_5 = wrapper_summary(derived_5).sort_values("mean")

fig, ax = plt.subplots(figsize=(8.2, 4.8))

y = np.arange(len(wrapper_5))

ax.barh(
    y,
    wrapper_5["mean"],
    xerr=wrapper_5["std"],
    color=[COLORS[s] for s in wrapper_5.index],
    edgecolor="black",
    linewidth=0.5,
    capsize=3,
)

ax.set_yticks(y)
ax.set_yticklabels([DISPLAY_NAMES[s] for s in wrapper_5.index])
ax.set_xlabel("Security-wrapper cost per update (ms)")
ax.set_xscale("log")
ax.grid(axis="x", which="both", alpha=0.25)
ax.set_axisbelow(True)

for yi, (scheme, row) in enumerate(wrapper_5.iterrows()):
    ax.text(
        row["mean"] * 1.06,
        yi,
        f'{row["mean"]:.2f}',
        va="center",
        fontsize=8,
    )

ax.set_xlim(
    wrapper_5["mean"].min() / 1.8,
    wrapper_5["mean"].max() * 1.8,
)

save_figure(fig, "security_wrapper_cost_5clients")


# =====================================================================
# Figure 2
# Cryptographic metadata per signed update
# =====================================================================

metadata_plot = (
    metadata_5[metadata_5["scheme"].isin(SIGNED_SCHEMES)]
    .set_index("scheme")
    .loc[SIGNED_SCHEMES]
    .sort_values("metadata_mean_B")
)

fig, ax = plt.subplots(figsize=(8.2, 4.8))

y = np.arange(len(metadata_plot))

ax.barh(
    y,
    metadata_plot["metadata_mean_B"],
    color=[COLORS[s] for s in metadata_plot.index],
    edgecolor="black",
    linewidth=0.5,
)

ax.set_yticks(y)
ax.set_yticklabels([DISPLAY_NAMES[s] for s in metadata_plot.index])
ax.set_xlabel("Cryptographic metadata per update (bytes)")
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

max_metadata = metadata_plot["metadata_mean_B"].max()

for yi, (scheme, row) in enumerate(metadata_plot.iterrows()):
    value = row["metadata_mean_B"]

    if value < 1000:
        label = f"{value:.0f} B"
    else:
        label = f"{value:,.0f} B".replace(",", " ")

    ax.text(
        value + max_metadata * 0.015,
        yi,
        label,
        va="center",
        fontsize=8,
    )

ax.set_xlim(0, max_metadata * 1.18)

save_figure(fig, "cryptographic_metadata_per_update")


# =====================================================================
# Figure 3
# Accuracy by round, five- and ten-client campaigns
# =====================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.0, 4.8),
    sharex=True,
    sharey=True,
)

for ax, df, title in [
    (axes[0], accuracy_5, "Five clients"),
    (axes[1], accuracy_10, "Ten clients"),
]:
    for scheme in SCHEME_ORDER:
        subset = (
            df[df["scheme"] == scheme]
            .sort_values("round")
            .reset_index(drop=True)
        )

        rounds = subset["round"].to_numpy()
        mean = subset["mean_accuracy"].to_numpy()
        std = subset["std_accuracy"].to_numpy()

        ax.plot(
            rounds,
            mean,
            label=DISPLAY_NAMES[scheme],
            color=COLORS[scheme],
            linestyle=LINESTYLES[scheme],
            linewidth=1.7,
        )

        ax.fill_between(
            rounds,
            mean - std,
            mean + std,
            color=COLORS[scheme],
            alpha=0.035,
            linewidth=0,
        )

    ax.set_title(title)
    ax.set_xlabel("Federated round")
    ax.set_xlim(0, 30)
    ax.grid(alpha=0.20)
    ax.set_axisbelow(True)
    ax.set_ylim(0.08, 0.75),

axes[0].set_ylabel("Central test accuracy")

handles, labels = axes[1].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.09),
    ncol=4,
    frameon=False,
)

fig.subplots_adjust(bottom=0.24, wspace=0.08)

save_figure(fig, "accuracy_by_round")


# =====================================================================
# Figure 4
# Aggregate wrapper work per round, 5 vs 10 clients
# =====================================================================

aggregate_5 = derived_5[derived_5["scheme"].isin(SIGNED_SCHEMES)].copy()
aggregate_10 = derived_10[derived_10["scheme"].isin(SIGNED_SCHEMES)].copy()

aggregate_5["aggregate_round_ms"] = (
    aggregate_5["security_wrapper_cost_ms"] * 5
)

aggregate_10["aggregate_round_ms"] = (
    aggregate_10["security_wrapper_cost_ms"] * 10
)

summary_5 = (
    aggregate_5.groupby("scheme")["aggregate_round_ms"]
    .agg(["mean", "std"])
    .reindex(SIGNED_SCHEMES)
)

summary_10 = (
    aggregate_10.groupby("scheme")["aggregate_round_ms"]
    .agg(["mean", "std"])
    .reindex(SIGNED_SCHEMES)
)

# Sort schemes by the five-client aggregate value while preserving
# the same ordering for both campaigns.
scale_order = summary_5.sort_values("mean").index.tolist()

summary_5 = summary_5.loc[scale_order]
summary_10 = summary_10.loc[scale_order]

fig, ax = plt.subplots(figsize=(8.6, 5.0))

y = np.arange(len(scale_order))
height = 0.34

ax.barh(
    y - height / 2,
    summary_5["mean"],
    height=height,
    xerr=summary_5["std"],
    capsize=2.5,
    label="5 clients",
    alpha=0.72,
    edgecolor="black",
    linewidth=0.4,
)

ax.barh(
    y + height / 2,
    summary_10["mean"],
    height=height,
    xerr=summary_10["std"],
    capsize=2.5,
    label="10 clients",
    alpha=0.92,
    edgecolor="black",
    linewidth=0.4,
)

ax.set_yticks(y)
ax.set_yticklabels([DISPLAY_NAMES[s] for s in scale_order])
ax.set_xlabel("Aggregate security-wrapper work per round (ms)")
ax.set_xscale("log")
ax.grid(axis="x", which="both", alpha=0.25)
ax.set_axisbelow(True)
ax.legend(frameon=False)

minimum = min(summary_5["mean"].min(), summary_10["mean"].min())
maximum = max(summary_5["mean"].max(), summary_10["mean"].max())

ax.set_xlim(minimum / 1.8, maximum * 1.5)

save_figure(fig, "scaling_wrapper_cost")


# =====================================================================
# Figure 5
# Processing-cost / cryptographic-metadata trade-off
# =====================================================================

wrapper_mean_5 = (
    derived_5[derived_5["scheme"].isin(SIGNED_SCHEMES)]
    .groupby("scheme")["security_wrapper_cost_ms"]
    .mean()
)

metadata_tradeoff = (
    metadata_5[metadata_5["scheme"].isin(SIGNED_SCHEMES)]
    .set_index("scheme")["metadata_mean_B"]
)

tradeoff = pd.DataFrame(
    {
        "metadata_B": metadata_tradeoff,
        "wrapper_ms": wrapper_mean_5,
    }
).loc[SIGNED_SCHEMES]


fig, ax = plt.subplots(figsize=(8.2, 5.4))

for scheme, row in tradeoff.iterrows():
    ax.scatter(
        row["metadata_B"],
        row["wrapper_ms"],
        s=60,
        color=COLORS[scheme],
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )

# Offsets are intentionally scheme-specific to prevent labels from
# overlapping while keeping the data points unchanged.
ANNOTATION_SETTINGS = {
    "RSA-2048": {
        "offset": (10, 8),
        "ha": "left",
        "va": "bottom",
    },
    "ECDSA-256": {
        "offset": (10, 8),
        "ha": "left",
        "va": "bottom",
    },
    "ML-DSA-44": {
        "offset": (10, 8),
        "ha": "left",
        "va": "bottom",
    },
    "ML-DSA-65": {
        "offset": (10, -10),
        "ha": "left",
        "va": "top",
    },
    "ML-DSA-87": {
        "offset": (10, 8),
        "ha": "left",
        "va": "bottom",
    },
    "Falcon-padded-512": {
        "offset": (10, 8),
        "ha": "left",
        "va": "bottom",
    },
    "SLH-DSA-SHA2-128s": {
        "offset": (-12, -8),
        "ha": "right",
        "va": "top",
    },
}

for scheme, row in tradeoff.iterrows():
    settings = ANNOTATION_SETTINGS[scheme]

    ax.annotate(
        DISPLAY_NAMES[scheme],
        (row["metadata_B"], row["wrapper_ms"]),
        xytext=settings["offset"],
        textcoords="offset points",
        ha=settings["ha"],
        va=settings["va"],
        fontsize=8,
    )

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(120, 12000)
ax.set_ylim(11, 1200)

ax.set_xlabel("Cryptographic metadata per update (bytes)")
ax.set_ylabel("Security-wrapper cost per update (ms)")

ax.grid(which="both", alpha=0.22)
ax.set_axisbelow(True)

save_figure(fig, "processing_metadata_tradeoff")


# ---------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------

print("\nAll final thesis figures generated successfully.")
print(f"Output directory: {FIGURE_DIR}")

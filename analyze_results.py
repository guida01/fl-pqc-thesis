import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = "results"
SCHEMES = ["RSA-2048", "ECDSA-256", "ML-DSA-44", "ML-DSA-65", 
           "ML-DSA-87", "Falcon-512", "SPHINCS+-SHA2-128s-simple"]

# Ler todos os CSVs
all_data = []
for scheme in SCHEMES:
    path = os.path.join(RESULTS_DIR, f"{scheme.replace('/', '_')}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        all_data.append(df)

df = pd.concat(all_data, ignore_index=True)

# Estatísticas por scheme
stats = df.groupby("scheme").agg({
    "sign_time": ["mean", "std", "min", "max"],
    "verify_time": ["mean", "std", "min", "max"],
    "sig_size": ["mean", "std"],
    "pubkey_size": ["mean"],
}).round(4)

# Salvar estatísticas
stats.to_csv(os.path.join(RESULTS_DIR, "summary_statistics.csv"))
print(stats)

# Gráfico 1: Signing Time
fig, ax = plt.subplots(figsize=(10, 6))
summary = df.groupby("scheme")["sign_time"].mean() * 1000  # ms
summary.plot(kind="bar", ax=ax, color="steelblue")
ax.set_ylabel("Milliseconds")
ax.set_title("Mean Signing Time by Scheme")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_signing_time.png"), dpi=150)

# Gráfico 2: Signature Size
fig, ax = plt.subplots(figsize=(10, 6))
summary = df.groupby("scheme")["sig_size"].mean()
summary.plot(kind="bar", ax=ax, color="coral")
ax.set_ylabel("Bytes")
ax.set_title("Mean Signature Size by Scheme")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_signature_size.png"), dpi=150)

# Gráfico 3: Trade-off (Signing Time vs Size)
fig, ax = plt.subplots(figsize=(10, 8))
summary = df.groupby("scheme").agg({"sign_time": "mean", "sig_size": "mean"})
for scheme in summary.index:
    x = summary.loc[scheme, "sign_time"] * 1000
    y = summary.loc[scheme, "sig_size"]
    ax.scatter(x, y, s=100)
    ax.annotate(scheme, (x, y), fontsize=9, ha="left")
ax.set_xlabel("Signing Time (ms)")
ax.set_ylabel("Signature Size (bytes)")
ax.set_title("Trade-off: Signing Time vs Signature Size")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_tradeoff.png"), dpi=150)

# Gráfico 4: Communication Overhead (10 clients × 10 rounds)
fig, ax = plt.subplots(figsize=(10, 6))
summary = df.groupby("scheme")["sig_size"].mean() * 100  # 10 clients × 10 rounds
summary_mb = summary / (1024 * 1024)  # Convert to MB
summary_mb.plot(kind="bar", ax=ax, color="green")
ax.set_ylabel("MB")
ax.set_title("Total Communication Overhead (100 signatures)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_overhead.png"), dpi=150)

print(f"\n Análise completa! Ficheiros em {RESULTS_DIR}/")
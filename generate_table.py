import pandas as pd

df = pd.read_csv("results/summary_statistics.csv")

print("% Copiar para a tese:")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Performance Comparison of Digital Signature Schemes}")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("Scheme & Sign (ms) & Verify (ms) & Sig (bytes) & PK (bytes) \\\\")
print("\\midrule")

schemes = df.index
for scheme in schemes:
    sign = df.loc[scheme, ("sign_time", "mean")] * 1000
    verify = df.loc[scheme, ("verify_time", "mean")] * 1000
    sig_size = df.loc[scheme, ("sig_size", "mean")]
    pk_size = df.loc[scheme, ("pubkey_size", "mean")]
    print(f"{scheme} & {sign:.2f} & {verify:.2f} & {sig_size:.0f} & {pk_size:.0f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:performance}")
print("\\end{table}")
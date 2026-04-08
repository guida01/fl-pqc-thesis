import pandas as pd
import os

RESULTS_DIR = "results"
SCHEMES = ["RSA-2048", "ECDSA-256", "ML-DSA-44", "ML-DSA-65", 
           "ML-DSA-87", "Falcon-512", "SPHINCS+-SHA2-128s-simple"]

def generate_latex_table():
    print("A ler CSVs diretamente...\n")
    
    # Ler todos os CSVs
    all_data = []
    for scheme in SCHEMES:
        path = os.path.join(RESULTS_DIR, f"{scheme.replace('/', '_')}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_data.append(df)
            print(f"  ✓ {scheme}: {len(df)} rows")
        else:
            print(f"  ✗ {scheme}: NOT FOUND")
    
    if not all_data:
        print("\n❌ Nenhum CSV encontrado!")
        return
    
    # Concatenar tudo
    df = pd.concat(all_data, ignore_index=True)
    
    # Calcular estatísticas por scheme
    stats = df.groupby("scheme").agg({
        "sign_time": ["mean", "std", "min", "max"],
        "verify_time": ["mean", "std", "min", "max"],
        "sig_size": ["mean", "std"],
        "pubkey_size": ["mean"],
    }).round(6)
    
    print("\n" + "="*80)
    print("📋 TABELA LATEX - Copiar para a tese:")
    print("="*80 + "\n")
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Performance Comparison of Digital Signature Schemes in FL}")
    print("\\label{tab:performance}")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("Scheme & Sign (ms) & Verify (ms) & Sig (bytes) & PK (bytes) \\\\")
    print("\\midrule")
    
    for scheme in stats.index:
        sign = stats.loc[scheme, ("sign_time", "mean")] * 1000
        verify = stats.loc[scheme, ("verify_time", "mean")] * 1000
        sig_size = stats.loc[scheme, ("sig_size", "mean")]
        pk_size = stats.loc[scheme, ("pubkey_size", "mean")]
        
        # Clean scheme name for LaTeX
        scheme_clean = scheme.replace("_", "\\_").replace("+-", "+")
        
        print(f"{scheme_clean} & {sign:.2f} & {verify:.2f} & {sig_size:.0f} & {pk_size:.0f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    print("\n" + "="*80)
    print("\n📊 RESUMO DETALHADO:\n")
    print(f"{'Scheme':<35} {'Sign(ms)':<12} {'Verify(ms)':<12} {'Sig(B)':<10} {'PK(B)':<10}")
    print("-" * 85)
    
    for scheme in stats.index:
        sign_mean = stats.loc[scheme, ("sign_time", "mean")] * 1000
        sign_std = stats.loc[scheme, ("sign_time", "std")] * 1000
        verify_mean = stats.loc[scheme, ("verify_time", "mean")] * 1000
        verify_std = stats.loc[scheme, ("verify_time", "std")] * 1000
        sig_size = stats.loc[scheme, ("sig_size", "mean")]
        pk_size = stats.loc[scheme, ("pubkey_size", "mean")]
        
        print(f"{scheme:<35} {sign_mean:>6.2f}±{sign_std:<4.2f} {verify_mean:>6.2f}±{verify_std:<4.2f} {sig_size:<10.0f} {pk_size:<10.0f}")
    
    # Mostrar também alguns insights
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    fastest_sign = stats["sign_time"]["mean"].idxmin()
    slowest_sign = stats["sign_time"]["mean"].idxmax()
    smallest_sig = stats["sig_size"]["mean"].idxmin()
    largest_sig = stats["sig_size"]["mean"].idxmax()
    
    print(f"\nFastest signing:  {fastest_sign} ({stats.loc[fastest_sign, ('sign_time', 'mean')]*1000:.2f} ms)")
    print(f"Slowest signing:  {slowest_sign} ({stats.loc[slowest_sign, ('sign_time', 'mean')]*1000:.2f} ms)")
    print(f"Smallest signature: {smallest_sig} ({stats.loc[smallest_sig, ('sig_size', 'mean')]:.0f} bytes)")
    print(f"Largest signature:  {largest_sig} ({stats.loc[largest_sig, ('sig_size', 'mean')]:.0f} bytes)")
    
    # Salvar numa pasta tables
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    table_path = os.path.join(RESULTS_DIR, "tables", "latex_table.txt")
    
    with open(table_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Comparison of Digital Signature Schemes in FL}\n")
        f.write("\\label{tab:performance}\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Scheme & Sign (ms) & Verify (ms) & Sig (bytes) & PK (bytes) \\\\\n")
        f.write("\\midrule\n")
        
        for scheme in stats.index:
            sign = stats.loc[scheme, ("sign_time", "mean")] * 1000
            verify = stats.loc[scheme, ("verify_time", "mean")] * 1000
            sig_size = stats.loc[scheme, ("sig_size", "mean")]
            pk_size = stats.loc[scheme, ("pubkey_size", "mean")]
            scheme_clean = scheme.replace("_", "\\_").replace("+-", "+")
            f.write(f"{scheme_clean} & {sign:.2f} & {verify:.2f} & {sig_size:.0f} & {pk_size:.0f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\n✅ Tabela LaTeX salva em: {table_path}")
    
    # Salvar também as estatísticas completas
    stats_path = os.path.join(RESULTS_DIR, "tables", "summary_statistics.csv")
    stats.to_csv(stats_path)
    print(f"Estatísticas completas salvas em: {stats_path}")

if __name__ == "__main__":
    generate_latex_table()
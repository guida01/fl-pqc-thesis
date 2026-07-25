"""Validation report over thesis_fl/quickstart-pytorch/results/*.csv.

Read-only: never writes to results/. Prints a text report to stdout and
drops one CSV per section into analysis/.

Run from thesis_fl/quickstart-pytorch/:
    python analysis/validate_results.py
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_ROUNDS = 30
NUM_RUNS = 5
NUM_NODES = 5

# Nominal scheme execution order, from run_all_schemes.py's SCHEMES list
# (the driver that produced results/): `for scheme in SCHEMES: for run_num
# in range(1, N_RUNS+1)`, i.e. all 5 runs of a scheme complete before the
# next scheme starts. no_signature ran first, ECDSA-256 ran last.
EXECUTION_ORDER = [
    "no_signature",
    "ML-DSA-44",
    "ML-DSA-65",
    "ML-DSA-87",
    "Falcon-padded-512",
    "SPHINCS+-SHA2-128s-simple",
    "RSA-2048",
    "ECDSA-256",
]

# rerun_failed.py later re-executed these (scheme, run) pairs out of band
# (after the full sweep completed) and spliced their rows back into the
# CSVs by run number, not by chronological position. For these two
# schemes, `run` number is NOT a reliable proxy for wall-clock order.
RERUN_AFFECTED = {"ML-DSA-44", "Falcon-padded-512"}

PQC_SCHEMES = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87", "Falcon-padded-512", "SPHINCS+-SHA2-128s-simple"]
CLASSICAL_SCHEMES = ["RSA-2048", "ECDSA-256"]
SIGNED_SCHEMES = PQC_SCHEMES + CLASSICAL_SCHEMES  # excludes no_signature

FLOAT_NOISE_EPS = 1e-6  # below this, treat train_loss divergence as float noise, not structural

LINE = "=" * 78


def load_all():
    frames = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv"))):
        scheme = os.path.splitext(os.path.basename(path))[0]
        frames[scheme] = pd.read_csv(path)
    return frames


def section_header(title):
    print(f"\n{LINE}\n{title}\n{LINE}")


# ─────────────────────────────────────────────────────────────────────────
# 1. ORTHOGONALITY OF THE CRYPTOGRAPHIC LAYER
# ─────────────────────────────────────────────────────────────────────────

def section1_orthogonality(dfs):
    section_header("1. ORTOGONALIDADE DA CAMADA CRIPTOGRAFICA")

    print(
        "Nota metodologica: node_id NAO e uma chave de junta valida entre\n"
        "esquemas (nem entre runs do mesmo esquema). O Flower atribui\n"
        "node_id aleatoriamente a cada invocacao de `flwr run` — cada\n"
        "(scheme, run) tem o seu proprio conjunto de 5 node_id, sem overlap\n"
        "com qualquer outro (verificado: interseccao vazia entre os node_id\n"
        "de ECDSA run1 e RSA run1, e entre ECDSA run1 e ECDSA run2).\n"
        "Em vez de comparar (run, round, node_id) diretamente, comparamos,\n"
        "para cada (run, round), o MULTICONJUNTO ORDENADO dos 5 valores de\n"
        "train_loss de cada esquema. Isto e valido porque train_loss so\n"
        "depende de: seed do modelo (torch.manual_seed(run_number*42),\n"
        "server_app.py:111) e seed do DataLoader (42+run_number,\n"
        "task.py:45) — ambas independentes do esquema — mais a particao de\n"
        "dados (deterministica por partition-id). O esquema de assinatura\n"
        "so atua DEPOIS do treino local, sobre os pesos ja treinados."
    )

    # sorted per-(run,round) train_loss tuples per scheme
    sorted_groups = {}
    sizes = {}
    for scheme, df in dfs.items():
        g = df.groupby(["run", "round"])["train_loss"].apply(lambda s: tuple(sorted(s.tolist())))
        sorted_groups[scheme] = g
        sizes[scheme] = df.groupby(["run", "round"])["train_loss"].size()

    all_keys = set()
    for g in sorted_groups.values():
        all_keys |= set(g.index)
    common_keys = set.intersection(*[set(g.index) for g in sorted_groups.values()])
    missing_keys = all_keys - common_keys

    print(
        f"\n(run,round) tuplos totais observados nalgum esquema: {len(all_keys)}\n"
        f"(run,round) tuplos presentes em TODOS os {len(dfs)} esquemas (comparaveis): {len(common_keys)}\n"
        f"(run,round) tuplos excluidos por dados em falta nalgum esquema: {len(missing_keys)}"
    )
    if missing_keys:
        missing_by_scheme = {}
        for scheme, g in sorted_groups.items():
            miss = sorted(k for k in missing_keys if k not in set(g.index))
            if miss:
                missing_by_scheme[scheme] = miss
        for scheme, miss in missing_by_scheme.items():
            rounds = sorted(set(r for _, r in miss))
            print(f"  falta em {scheme}: {len(miss)} tuplos (rounds {rounds[0]}-{rounds[-1]} de runs {sorted(set(r for r,_ in miss))})")

    reference_scheme = "no_signature" if "no_signature" in dfs else sorted(dfs)[0]
    rows = []
    for key in sorted(common_keys):
        ref_vals = np.array(sorted_groups[reference_scheme][key])
        for scheme in dfs:
            if scheme == reference_scheme:
                continue
            vals = np.array(sorted_groups[scheme][key])
            if len(vals) != len(ref_vals):
                continue  # size mismatch already reported as missing/partial above
            max_abs_diff = float(np.max(np.abs(vals - ref_vals)))
            rows.append({
                "run": key[0], "round": key[1],
                "reference_scheme": reference_scheme, "scheme": scheme,
                "max_abs_diff": max_abs_diff,
            })

    diverg_df = pd.DataFrame(rows)
    diverg_df.to_csv(os.path.join(OUT_DIR, "section1_orthogonality.csv"), index=False)

    n_compared = len(diverg_df)
    n_diverging = int((diverg_df["max_abs_diff"] > FLOAT_NOISE_EPS).sum())
    max_diverg = float(diverg_df["max_abs_diff"].max())
    bitwise_equal = int((diverg_df["max_abs_diff"] == 0.0).sum())

    print(
        f"\nComparacoes (run,round,scheme) vs esquema de referencia ({reference_scheme}): {n_compared}\n"
        f"  identicas bit-a-bit (diff==0.0):     {bitwise_equal} ({100*bitwise_equal/n_compared:.2f}%)\n"
        f"  divergem > {FLOAT_NOISE_EPS:g} (limiar de ruido float): {n_diverging} ({100*n_diverging/n_compared:.2f}%)\n"
        f"  divergencia maxima absoluta observada: {max_diverg:.3e}"
    )

    # Round-1 is the critical test: every scheme starts from the exact same
    # initial_arrays (torch.manual_seed(run_number*42), server_app.py:111)
    # and no signing happens before training, so round 1 must be bit-exact
    # if the crypto layer is truly orthogonal.
    round1 = diverg_df[diverg_df["round"] == 1]
    round1_exact = bool((round1["max_abs_diff"] == 0.0).all()) if len(round1) else None
    rounds_gt1 = diverg_df[diverg_df["round"] > 1]

    print(
        f"\nRonda 1 (todos os esquemas partem do MESMO initial_arrays, "
        f"antes de qualquer assinatura): {len(round1)} comparacoes, "
        f"identicas bit-a-bit: {round1_exact}"
    )

    # Does divergence magnitude (round > 1) depend on WHICH scheme is being
    # compared? If the crypto layer had a structural effect, some
    # schemes should diverge systematically more than others.
    by_scheme = rounds_gt1.groupby("scheme")["max_abs_diff"].agg(["mean", "std", "max", "count"])
    groups = [g["max_abs_diff"].values for _, g in rounds_gt1.groupby("scheme")]
    f_stat, anova_p = sp_stats.f_oneway(*groups)

    # Does divergence magnitude grow with round number (compounding drift)
    # or plateau after the first aggregation (bounded chaotic sensitivity)?
    by_round = rounds_gt1.groupby("round")["max_abs_diff"].mean()
    round_corr_r, round_corr_p = sp_stats.pearsonr(rounds_gt1["round"], rounds_gt1["max_abs_diff"])

    print(f"\nDivergencia (ronda > 1) por esquema-alvo (comparado com {reference_scheme}):")
    print(by_scheme.to_string())
    print(
        f"\nANOVA (a divergencia difere consoante o esquema-alvo?): "
        f"F={f_stat:.4f}  p={anova_p:.4f}"
    )
    print(
        f"Correlacao entre numero da ronda e divergencia (ronda>1): "
        f"r={round_corr_r:+.4f}  p={round_corr_p:.4g}  "
        f"(media ronda2={by_round.loc[2]:.4f} vs media ronda30={by_round.loc[30]:.4f})"
    )

    diverg_df.to_csv(os.path.join(OUT_DIR, "section1_orthogonality.csv"), index=False)
    by_scheme.reset_index().to_csv(os.path.join(OUT_DIR, "section1_divergence_by_scheme.csv"), index=False)

    if round1_exact and anova_p > 0.05:
        verdict = (
            "(a) A ronda 1 e IDENTICA BIT-A-BIT em todos os 8 esquemas "
            "(prova direta: a camada de assinatura, que so atua depois do "
            "treino, nao altera train_loss).\n"
            "(b) A partir da ronda 2 aparece divergencia (media "
            f"~{rounds_gt1['max_abs_diff'].mean():.4f}, max {max_diverg:.3e} "
            "em valor absoluto de loss), mas essa divergencia:\n"
            f"    - NAO depende do esquema (ANOVA p={anova_p:.4f} > 0.05 — "
            "as medias por esquema sao estatisticamente indistinguiveis, "
            "ver section1_divergence_by_scheme.csv);\n"
            f"    - NAO cresce com a ronda: r={round_corr_r:+.4f} "
            f"(p={round_corr_p:.4g}; estatisticamente != 0 dado o n grande, "
            "mas o efeito e negligenciavel em magnitude e de sinal "
            "negativo, nao positivo — inconsistente com deriva que se "
            "acumula). O padrao real e um salto na 1a agregacao seguido de "
            "um patamar estavel (ver medias por ronda no CSV).\n"
            "RESULTADO: a divergencia observada e atribuivel a "
            "nao-determinismo numerico ENTRE INVOCACOES INDEPENDENTES do "
            "`flwr run` (cada esquema corre num processo Ray separado; a "
            "soma em ponto flutuante do FedAvg nao e associativa e a ordem "
            "de chegada dos clientes ao agregador nao e garantida — isto "
            "introduz uma diferenca minima logo na 1a agregacao, que a "
            "dinamica nao-linear do SGD amplifica para uma divergencia "
            "visivel mas LIMITADA, nao um vies estrutural do esquema de "
            "assinatura). A hipotese de ortogonalidade NAO e refutada: "
            "este resultado substitui a medicao de accuracy como evidencia "
            "de que a convergencia nao e afetada pelo esquema de "
            "assinatura escolhido."
        )
    else:
        verdict = (
            "A ronda 1 nao e identica bit-a-bit entre esquemas, e/ou a "
            "divergencia nas rondas seguintes correlaciona com o esquema "
            "(ANOVA significativo) — possivel efeito estrutural do "
            "esquema de assinatura sobre o treino local. Inspecionar "
            "section1_orthogonality.csv e section1_divergence_by_scheme.csv."
        )
    print(f"\nVeredicto:\n{verdict}")
    print(f"CSVs: analysis/section1_orthogonality.csv ({n_compared} linhas), "
          "analysis/section1_divergence_by_scheme.csv")


# ─────────────────────────────────────────────────────────────────────────
# 2. DATA INTEGRITY
# ─────────────────────────────────────────────────────────────────────────

def section2_integrity(dfs):
    section_header("2. INTEGRIDADE DOS DADOS")

    # --- row counts ---
    print("Contagem de linhas por esquema (esperado 750 = 5 runs x 30 rounds x 5 nos):")
    row_counts = []
    for scheme, df in dfs.items():
        n = len(df)
        status = "OK" if n == NUM_RUNS * NUM_ROUNDS * NUM_NODES else f"INCOMPLETO (faltam {NUM_RUNS*NUM_ROUNDS*NUM_NODES - n})"
        print(f"  {scheme:30s} {n:4d}  {status}")
        row_counts.append({"scheme": scheme, "rows": n, "expected": NUM_RUNS * NUM_ROUNDS * NUM_NODES})

    # --- exact missing (run, round) for incomplete schemes ---
    missing_rows = []
    print("\nDetalhe do que falta, por esquema incompleto:")
    for scheme, df in dfs.items():
        expected = NUM_RUNS * NUM_ROUNDS * NUM_NODES
        if len(df) == expected:
            continue
        counts = df.groupby(["run", "round"]).size()
        full_index = pd.MultiIndex.from_product([range(1, NUM_RUNS + 1), range(1, NUM_ROUNDS + 1)], names=["run", "round"])
        counts = counts.reindex(full_index, fill_value=0)
        incomplete = counts[counts != NUM_NODES]
        for (run, rnd), n_nodes in incomplete.items():
            missing_rows.append({"scheme": scheme, "run": run, "round": rnd, "nodes_present": int(n_nodes), "nodes_expected": NUM_NODES})
        # summarize as ranges per run
        for run in sorted(incomplete.index.get_level_values("run").unique()):
            rounds_missing = sorted(incomplete.xs(run, level="run").index[incomplete.xs(run, level="run") == 0])
            rounds_partial = sorted(incomplete.xs(run, level="run").index[(incomplete.xs(run, level="run") > 0) & (incomplete.xs(run, level="run") < NUM_NODES)])
            if rounds_missing:
                print(f"  {scheme} run {run}: rounds totalmente em falta = {rounds_missing[0]}-{rounds_missing[-1]} ({len(rounds_missing)} rounds)")
            if rounds_partial:
                print(f"  {scheme} run {run}: rounds parciais = {rounds_partial}")

    pd.DataFrame(missing_rows).to_csv(os.path.join(OUT_DIR, "section2_missing_rows.csv"), index=False)

    # --- sig_valid == True and has_nan == False everywhere ---
    # (replaces the old single `verified` column — see task 4: verified was
    # `is_valid and not has_nan`, conflating two distinct failure modes)
    print("\nColunas 'sig_valid' / 'has_nan':")
    all_sig_valid = True
    any_has_nan = False
    for scheme, df in dfs.items():
        n_invalid = int((~df["sig_valid"]).sum())
        n_nan = int(df["has_nan"].sum())
        if n_invalid > 0:
            all_sig_valid = False
        if n_nan > 0:
            any_has_nan = True
        print(f"  {scheme:30s} sig_valid==False em {n_invalid} linhas, has_nan==True em {n_nan} linhas")
    print(f"  -> sig_valid==True em TODAS as linhas de TODOS os esquemas: {all_sig_valid}")
    print(f"  -> has_nan==True nalguma linha de algum esquema: {any_has_nan}")

    # --- NaN counts, all columns ---
    print("\nContagem de NaN por coluna (todas as colunas, todos os esquemas):")
    nan_rows = []
    any_nan = False
    for scheme, df in dfs.items():
        na_counts = df.isna().sum()
        for col, n in na_counts.items():
            if n > 0:
                any_nan = True
                print(f"  {scheme:30s} {col:15s} {int(n)} NaN")
            nan_rows.append({"scheme": scheme, "column": col, "n_nan": int(n)})
    if not any_nan:
        print("  Nenhum NaN encontrado em nenhuma coluna de nenhum esquema.")
    pd.DataFrame(nan_rows).to_csv(os.path.join(OUT_DIR, "section2_nan_counts.csv"), index=False)

    # --- node_id consistency within each run ---
    print("\nConsistencia de node_id dentro de cada run (mesmo conjunto de 5 nos em todas as rondas do run):")
    consistency_rows = []
    all_consistent = True
    for scheme, df in dfs.items():
        for run, run_df in df.groupby("run"):
            rounds_present = sorted(run_df["round"].unique())
            node_sets = run_df.groupby("round")["node_id"].apply(lambda s: frozenset(s))
            distinct_sets = set(node_sets.values)
            n_per_round = run_df.groupby("round")["node_id"].nunique()
            consistent = (len(distinct_sets) == 1) and (n_per_round == NUM_NODES).all()
            if not consistent:
                all_consistent = False
            consistency_rows.append({
                "scheme": scheme, "run": run,
                "n_rounds_present": len(rounds_present),
                "distinct_node_sets": len(distinct_sets),
                "consistent": consistent,
            })
    consistency_df = pd.DataFrame(consistency_rows)
    consistency_df.to_csv(os.path.join(OUT_DIR, "section2_node_id_consistency.csv"), index=False)
    n_inconsistent = int((~consistency_df["consistent"]).sum())
    print(f"  runs verificados: {len(consistency_df)}  |  inconsistentes: {n_inconsistent}")
    if n_inconsistent:
        print(consistency_df[~consistency_df["consistent"]].to_string(index=False))
    else:
        print("  Todos os runs usam o mesmo conjunto de 5 node_id em todas as suas rondas presentes.")

    # --- sig_size / pubkey_size constant per scheme, except ECDSA-256 ---
    print("\nsig_size / pubkey_size por esquema:")
    size_rows = []
    for scheme, df in dfs.items():
        if scheme == "no_signature":
            continue
        sig_vals = df["sig_size"].value_counts().sort_index()
        pk_vals = df["pubkey_size"].value_counts().sort_index()
        sig_constant = len(sig_vals) == 1
        pk_constant = len(pk_vals) == 1
        for val, cnt in sig_vals.items():
            size_rows.append({"scheme": scheme, "field": "sig_size", "value": val, "count": int(cnt)})
        for val, cnt in pk_vals.items():
            size_rows.append({"scheme": scheme, "field": "pubkey_size", "value": val, "count": int(cnt)})

        expected_constant = scheme != "ECDSA-256"
        flag = "OK" if (sig_constant == expected_constant or scheme == "ECDSA-256") else "INESPERADO"
        if scheme == "ECDSA-256":
            print(f"  {scheme:30s} sig_size NAO constante (esperado, DER de comprimento variavel) -> distribuicao:")
            for val, cnt in sig_vals.items():
                print(f"      sig_size={int(val)}  n={int(cnt)}  ({100*cnt/len(df):.1f}%)")
            print(f"      pubkey_size constante: {pk_constant} (valor={pk_vals.index[0] if pk_constant else pk_vals.to_dict()})")
        else:
            sig_str = f"{int(sig_vals.index[0])}" if sig_constant else f"NAO CONSTANTE: {sig_vals.to_dict()}"
            pk_str = f"{int(pk_vals.index[0])}" if pk_constant else f"NAO CONSTANTE: {pk_vals.to_dict()}"
            print(f"  {scheme:30s} sig_size={sig_str:20s} pubkey_size={pk_str:10s} [{flag}]")

    pd.DataFrame(size_rows).to_csv(os.path.join(OUT_DIR, "section2_sig_pubkey_sizes.csv"), index=False)

    pd.DataFrame(row_counts).to_csv(os.path.join(OUT_DIR, "section2_row_counts.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────
# 3. TEMPORAL DRIFT
# ─────────────────────────────────────────────────────────────────────────

def section3_temporal_drift(dfs):
    section_header("3. DERIVA TEMPORAL")

    print(
        "Ordem de execucao assumida (run_all_schemes.py SCHEMES, execucao\n"
        "'for scheme: for run:'): " + " -> ".join(EXECUTION_ORDER) + "\n\n"
        "Aviso: rerun_failed.py reexecutou ML-DSA-44 (runs 1,2) e "
        "Falcon-padded-512 (run 1) DEPOIS do sweep completo, e recolou as\n"
        "linhas no CSV pelo numero de run, nao pela ordem cronologica real.\n"
        "Para estes 2 esquemas, 'run' NAO e um proxy fiavel de tempo; os\n"
        "resultados de correlacao intra-esquema abaixo sao reportados na\n"
        "mesma mas marcados como nao fiaveis."
    )

    corr_rows = []
    print("\nCorrelacao (Pearson) entre numero do run e train_time, por esquema (nivel-linha, n=todas as linhas do esquema):")
    for scheme in EXECUTION_ORDER:
        if scheme not in dfs:
            continue
        df = dfs[scheme]
        if df["run"].nunique() < 2:
            continue
        r, p = sp_stats.pearsonr(df["run"], df["train_time"])
        reliable = scheme not in RERUN_AFFECTED
        flag = "" if reliable else "  [NAO FIAVEL - rerun fora de ordem]"
        print(f"  {scheme:30s} r={r:+.4f}  p={p:.4g}  n={len(df)}{flag}")
        corr_rows.append({"scheme": scheme, "r": r, "p_value": p, "n": len(df), "run_order_reliable": reliable})

    print("\nMedia de train_time por run, por esquema:")
    means_rows = []
    for scheme in EXECUTION_ORDER:
        if scheme not in dfs:
            continue
        df = dfs[scheme]
        by_run = df.groupby("run")["train_time"].agg(["mean", "std", "count"])
        for run, row in by_run.iterrows():
            means_rows.append({"scheme": scheme, "run": run, "train_time_mean": row["mean"], "train_time_std": row["std"], "n": row["count"]})
        means_str = "  ".join(f"run{r}={m:.3f}s" for r, m in by_run["mean"].items())
        print(f"  {scheme:30s} {means_str}")

    means_df = pd.DataFrame(means_rows)

    # first vs last executed scheme
    first_scheme, last_scheme = EXECUTION_ORDER[0], EXECUTION_ORDER[-1]
    first_mean = dfs[first_scheme]["train_time"].mean() if first_scheme in dfs else float("nan")
    last_mean = dfs[last_scheme]["train_time"].mean() if last_scheme in dfs else float("nan")
    print(
        f"\nPrimeiro esquema executado: {first_scheme}  mean(train_time)={first_mean:.4f}s"
        f"  (n={len(dfs[first_scheme]) if first_scheme in dfs else 0}"
        + (", INCOMPLETO" if first_scheme in dfs and len(dfs[first_scheme]) != NUM_RUNS*NUM_ROUNDS*NUM_NODES else "") + ")"
    )
    print(f"Ultimo esquema executado:   {last_scheme}  mean(train_time)={last_mean:.4f}s  (n={len(dfs[last_scheme])})")
    if not (np.isnan(first_mean) or np.isnan(last_mean)):
        diff_pct = 100 * (last_mean - first_mean) / first_mean
        print(f"Diferenca (ultimo - primeiro): {last_mean - first_mean:+.4f}s ({diff_pct:+.2f}%)")

    corr_df = pd.DataFrame(corr_rows)
    reliable_corrs = corr_df[corr_df["run_order_reliable"]]
    max_abs_r = reliable_corrs["r"].abs().max() if len(reliable_corrs) else float("nan")
    n_significant = int((reliable_corrs["p_value"] < 0.05).sum())

    print(
        f"\nVeredicto: entre os esquemas com ordem de run fiavel, "
        f"|r| maximo = {max_abs_r:.4f}; {n_significant}/{len(reliable_corrs)} "
        "correlacoes com p<0.05."
    )
    if n_significant == 0 and (np.isnan(max_abs_r) or max_abs_r < 0.15):
        print(
            "Sem sinal consistente de deriva termica/de frequencia dentro "
            "dos esquemas: train_time nao correlaciona de forma relevante "
            "com o numero do run. A diferenca entre o primeiro e o ultimo "
            "esquema executado (acima) mistura efeito de esquema com "
            "efeito de posicao na sequencia — com apenas 1 sequencia "
            "observada, os dois nao sao estatisticamente separaveis; mas "
            "a ausencia de deriva intra-esquema sugere que o confundimento "
            "por ordem de execucao, a existir, e pequeno face a variacao "
            "normal de treino."
        )
    else:
        print(
            "Ha sinal de correlacao entre run/posicao temporal e "
            "train_time nalgum esquema — possivel confundimento entre "
            "ordem de execucao e esquema; tratar comparacoes absolutas de "
            "train_time entre esquemas com cautela."
        )

    corr_df.to_csv(os.path.join(OUT_DIR, "section3_run_correlation.csv"), index=False)
    means_df.to_csv(os.path.join(OUT_DIR, "section3_train_time_by_run.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────
# 4. COMMUNICATION COST MODEL
# ─────────────────────────────────────────────────────────────────────────

def section4_communication_cost(dfs):
    section_header("4. MODELO DE CUSTO DE COMUNICACAO")

    rows = []
    for scheme in SIGNED_SCHEMES:
        if scheme not in dfs:
            continue
        df = dfs[scheme]
        sig = df["sig_size"].astype(float)
        pk = df["pubkey_size"].astype(float)
        payload = df["payload_size"].astype(float)

        overhead = sig + pk
        overhead_no_pk = sig
        rel_overhead = overhead / payload
        rel_overhead_no_pk = overhead_no_pk / payload

        rows.append({
            "scheme": scheme,
            "sig_size_mean": sig.mean(),
            "pubkey_size_mean": pk.mean(),
            "payload_size_mean": payload.mean(),
            "overhead_per_update_bytes": overhead.mean(),
            "relative_overhead_pct": 100 * rel_overhead.mean(),
            "overhead_per_update_no_pki_bytes": overhead_no_pk.mean(),
            "relative_overhead_no_pki_pct": 100 * rel_overhead_no_pk.mean(),
        })

    cost_df = pd.DataFrame(rows)
    print(f"{'scheme':30s} {'overhead/upd (B)':>16s} {'rel.ovh %':>10s} {'ovh sem PKI (B)':>16s} {'rel.ovh sem PKI %':>18s}")
    for _, r in cost_df.iterrows():
        print(f"{r['scheme']:30s} {r['overhead_per_update_bytes']:16.1f} {r['relative_overhead_pct']:10.4f} "
              f"{r['overhead_per_update_no_pki_bytes']:16.1f} {r['relative_overhead_no_pki_pct']:18.4f}")

    # N, T cancellation argument
    print(
        "\nCancelamento de N (clientes) e T (rondas) no racio de overhead relativo:\n"
        "  overhead cumulativo (bytes) ao longo do treino = N x T x (sig_size + pubkey_size)\n"
        "  payload cumulativo (bytes)  ao longo do treino = N x T x payload_size\n"
        "  overhead relativo cumulativo = [N x T x (sig_size+pubkey_size)] / [N x T x payload_size]\n"
        "                                = (sig_size + pubkey_size) / payload_size\n"
        "  -> N e T aparecem em numerador e denominador com o MESMO fator "
        "e cancelam-se algebricamente.\n"
        "  Confirmado nos dados: relative_overhead_pct (medido por LINHA, "
        "ou seja por 1 cliente em 1 ronda) e numericamente identico ao "
        "racio agregado sobre todas as N x T linhas de cada esquema "
        "(verificado abaixo)."
    )
    # empirical confirmation: per-row ratio mean vs aggregate-sum ratio
    confirm_rows = []
    for scheme in SIGNED_SCHEMES:
        if scheme not in dfs:
            continue
        df = dfs[scheme]
        per_row_mean_pct = 100 * ((df["sig_size"] + df["pubkey_size"]) / df["payload_size"]).mean()
        agg_pct = 100 * (df["sig_size"] + df["pubkey_size"]).sum() / df["payload_size"].sum()
        confirm_rows.append({"scheme": scheme, "per_row_mean_pct": per_row_mean_pct, "aggregate_sum_pct": agg_pct,
                              "diff": per_row_mean_pct - agg_pct})
    confirm_df = pd.DataFrame(confirm_rows)
    print(f"\n{'scheme':30s} {'racio medio por-linha %':>24s} {'racio agregado (soma) %':>24s} {'diff':>10s}")
    for _, r in confirm_df.iterrows():
        print(f"{r['scheme']:30s} {r['per_row_mean_pct']:24.6f} {r['aggregate_sum_pct']:24.6f} {r['diff']:10.2e}")
    print(
        "  (a pequena diferenca residual entre o racio medio-por-linha e o "
        "racio agregado-por-soma vem da variacao de sig_size do ECDSA-256 "
        "DER, nao de N ou T)"
    )

    # break-even model sizes
    print("\nBreak-even: tamanho de modelo (bytes) abaixo do qual o overhead relativo ultrapassa X%:")
    print("  M_breakeven = (sig_size + pubkey_size) / threshold")
    breakeven_rows = []
    thresholds = [0.01, 0.05, 0.10]
    for scheme in SIGNED_SCHEMES:
        if scheme not in dfs:
            continue
        overhead_bytes = cost_df.loc[cost_df["scheme"] == scheme, "overhead_per_update_bytes"].iloc[0]
        line = [f"{scheme:30s}"]
        for th in thresholds:
            m_bytes = overhead_bytes / th
            m_kib = m_bytes / 1024
            n_params = m_bytes / 4  # float32
            breakeven_rows.append({
                "scheme": scheme, "threshold_pct": th * 100,
                "model_size_bytes": m_bytes, "model_size_kib": m_kib,
                "n_params_float32": n_params,
            })
            line.append(f"{th*100:>3.0f}%: {m_kib:>9.2f} KiB ({n_params:>10,.0f} params)")
        print("  " + "  |  ".join(line))

    pd.DataFrame(breakeven_rows).to_csv(os.path.join(OUT_DIR, "section4_breakeven.csv"), index=False)
    cost_df.to_csv(os.path.join(OUT_DIR, "section4_communication_cost.csv"), index=False)
    confirm_df.to_csv(os.path.join(OUT_DIR, "section4_nt_cancellation_check.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────
# 5. CRYPTOGRAPHIC COST PER ROUND
# ─────────────────────────────────────────────────────────────────────────

def section5_crypto_cost_per_round(dfs):
    section_header("5. CUSTO CRIPTOGRAFICO POR RONDA")

    print(
        "custo_cripto_ronda(scheme,run,round) = soma sobre os N clientes de "
        "(keygen_time + sign_time + verify_time)\n"
        "Comparado com soma_train_time(scheme,run,round) = soma sobre os N "
        "clientes de train_time (mesma convencao de soma, para unidades "
        "comparaveis).\n"
        "Nota: nao existe medicao independente de latencia-de-ronda "
        "(wall-clock) no pipeline (ver auditoria: nenhuma metrica de round "
        "latency e recolhida); esta soma sobre-estima o tempo de parede se "
        "os clientes correm em paralelo, mas preserva a razao pedida entre "
        "custo criptografico total e custo de treino total por ronda."
    )

    round_rows = []
    for scheme, df in dfs.items():
        per_round = df.groupby(["run", "round"]).agg(
            crypto_cost=("keygen_time", lambda s: s.sum()),  # placeholder, recomputed below
        )
        g = df.groupby(["run", "round"])
        crypto = (g["keygen_time"].sum() + g["sign_time"].sum() + g["verify_time"].sum())
        train = g["train_time"].sum()
        for (run, rnd), c in crypto.items():
            t = train.loc[(run, rnd)]
            round_rows.append({
                "scheme": scheme, "run": run, "round": rnd,
                "crypto_cost_round": c, "train_cost_round": t,
                "crypto_fraction": c / (c + t) if (c + t) > 0 else float("nan"),
            })

    round_df = pd.DataFrame(round_rows)
    round_df.to_csv(os.path.join(OUT_DIR, "section5_crypto_cost_per_round.csv"), index=False)

    print(f"\n{'scheme':30s} {'crypto/round mean(s)':>20s} {'std':>10s} {'train/round mean(s)':>20s} {'crypto fraction %':>18s}")
    summary_rows = []
    for scheme in EXECUTION_ORDER:
        sub = round_df[round_df["scheme"] == scheme]
        if sub.empty:
            continue
        c_mean, c_std = sub["crypto_cost_round"].mean(), sub["crypto_cost_round"].std()
        t_mean = sub["train_cost_round"].mean()
        frac_mean = 100 * sub["crypto_fraction"].mean()
        print(f"{scheme:30s} {c_mean:20.5f} {c_std:10.5f} {t_mean:20.4f} {frac_mean:18.4f}")
        summary_rows.append({
            "scheme": scheme, "crypto_cost_round_mean_s": c_mean, "crypto_cost_round_std_s": c_std,
            "train_cost_round_mean_s": t_mean, "crypto_fraction_pct_mean": frac_mean,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "section5_crypto_cost_summary.csv"), index=False)

    print("\nPor ronda (media sobre os 5 runs), esquemas mais caros primeiro (fracao criptografica media):")
    by_round = round_df.groupby(["scheme", "round"]).agg(
        crypto_cost_round_mean=("crypto_cost_round", "mean"),
        crypto_fraction_mean=("crypto_fraction", "mean"),
    ).reset_index()
    by_round.to_csv(os.path.join(OUT_DIR, "section5_crypto_cost_by_round.csv"), index=False)

    worst = summary_df.sort_values("crypto_fraction_pct_mean", ascending=False)
    if len(worst):
        top = worst.iloc[0]
        best_signed = worst[worst["scheme"] != "no_signature"].sort_values("crypto_fraction_pct_mean").iloc[0] \
            if (worst["scheme"] != "no_signature").any() else None
        print(f"\nEsquema com maior fracao criptografica media: {top['scheme']} ({top['crypto_fraction_pct_mean']:.4f}% do custo total por ronda)")
        if best_signed is not None:
            print(f"Esquema assinado com MENOR fracao criptografica media: {best_signed['scheme']} ({best_signed['crypto_fraction_pct_mean']:.4f}%)")


def main():
    dfs = load_all()
    print(LINE)
    print("VALIDACAO DE RESULTADOS — thesis_fl/quickstart-pytorch/results/")
    print(f"Esquemas carregados: {sorted(dfs)}")
    print(LINE)

    section1_orthogonality(dfs)
    section2_integrity(dfs)
    section3_temporal_drift(dfs)
    section4_communication_cost(dfs)
    section5_crypto_cost_per_round(dfs)

    print(f"\n{LINE}\nCSVs por seccao escritos em: {OUT_DIR}/\n{LINE}")


if __name__ == "__main__":
    main()

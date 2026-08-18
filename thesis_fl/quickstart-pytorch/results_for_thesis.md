# Results for Thesis — Verification Report

Generated read-only from the repository state on 2026-08-18. No experiment was
re-run and no code was modified to produce this report. Sources: committed
`environment.txt` / `requirements-frozen.txt`, `pyproject.toml`, the
`pytorchexample/*.py` source, `results/*.csv` (5-client campaign, main),
`results_10clients/*.csv` (10-client cross-check), `results/execution_order.log`,
`campaign_5clients_20260725_171224.log`, and the outputs of
`analysis/validate_results.py` run against both results directories (that script
is read-only itself: it only writes into `analysis/`, never into `results/`).

## Assumptions

Recorded because the task said not to stop for clarification. None of these are
load-bearing surprises — they're the natural reading in each case.

1. **B1 aggregation** — "mean/std/median per scheme" is computed by pooling all
   750 rows per scheme (5 runs × 30 rounds × 5 clients), not by averaging
   per-round or per-run means first.
2. **B4 units** — the thesis's hard-coded break-even values (e.g. Falcon-padded-512
   = 156/31/16 "KB") use **decimal kilobytes (1 KB = 1000 B)**. The repo's own
   `analysis/validate_results.py` computes the same quantity in **KiB (1024 B)**
   and labels the column `model_size_kib`. Both are reported below; see B4 for
   the reconciliation.
3. **B6 Kruskal–Wallis on train_loss** — "train_loss across schemes" is
   ambiguous between "final round only" and "all rounds pooled". Both are
   reported (see B6); the headline number matches the same convention as the
   final-accuracy test (final round, n=5 runs per scheme) for direct
   comparability.
4. **A9** — the repository's analysis code does not contain a Kruskal–Wallis
   test anywhere; it uses one-way ANOVA (`scipy.stats.f_oneway`) at α=0.05 for
   every cross-scheme comparison. This is reported as a **correction**, not a
   confirmation — see A9. The B6 Kruskal–Wallis numbers were computed
   separately for this report (standalone `scipy.stats.kruskal` calls over the
   already-collected CSV data), since the user explicitly asked for that test;
   they are not part of `validate_results.py`'s output.
5. **A1 hardware** — `environment.txt` (committed at the `v1-experimental-baseline`
   tag) and a fresh live `lscpu`/`uname -a`/`free -h` agree exactly, so the
   machine has not changed between the tag and today.

---

## A. SETUP

### A1. Hardware

| Item | Value |
|---|---|
| CPU model | Intel(R) Core(TM) i7-4770 CPU @ 3.40GHz |
| Physical cores | 4 (1 socket × 4 cores/socket) |
| Logical cores (threads) | 8 (2 threads/core, hyperthreading on) |
| Base/max clock | 3.40 GHz base; CPU max MHz 3900.0 (boost), CPU min MHz 800.0 |
| Total RAM | 15.6 GiB (16.71 GB decimal) — `free -b` total = 16,707,936,256 bytes |
| OS | Debian GNU/Linux 12 (bookworm) |
| Kernel | Linux fabiana 6.1.0-44-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.1.164-1 (2026-03-09) x86_64 |

Source: committed `environment.txt` (frozen at tag `v1-experimental-baseline`,
commit `46e4c22`), cross-checked against a live `lscpu` / `uname -a` / `free -h`
run today — identical CPU, kernel, and core counts, confirming the hardware is
unchanged since the tag.

### A2. Installed package versions

| Package | requirements-frozen.txt | pyproject.toml pin | `pip show` (live venv) | Match? |
|---|---|---|---|---|
| python | (not a pip package; environment.txt) 3.11.2 | — | 3.11.2 | ✅ |
| flwr | 1.29.0 | `>=1.29.0` | 1.29.0 | ✅ |
| torch | 2.8.0 | `==2.8.0` | 2.8.0 | ✅ |
| liboqs-python | 0.14.1 | `==0.14.1` | 0.14.1 | ✅ |
| cryptography | 46.0.7 | `==46.0.7` | 46.0.7 | ✅ |
| numpy | 2.4.4 | `==2.4.4` | 2.4.4 | ✅ |

No mismatch found. Note: `pyproject.toml`'s `flwr` dependency is `>=1.29.0`
(a range, not an exact pin) — commit `c98aeb9` deliberately relaxed it from an
exact pin because the exact pin broke `flwr run`. The exact installed version
(1.29.0) is still what `requirements-frozen.txt` records and what `pip show`
confirms, so there is no discrepancy in what was actually used.

### A3. Repository / tag state — ⚠️ MISMATCH, flagged

| Item | Value |
|---|---|
| Remote URL | `git@github.com:guida01/fl-pqc-thesis.git` |
| `git describe --tags` | `v1-experimental-baseline-21-gec7768a` |
| Current branch | `implementation/scratch`, 18 commits ahead of `origin/implementation/scratch` |
| HEAD commit | `ec7768a` (2026-07-27T21:49:21+01:00) |
| Tag commit | `29f0794` ("results: complete 8-scheme sweep (5 runs x 30 rounds x 5 nodes)") |

**The working tree does NOT match tag `v1-experimental-baseline`.** HEAD is 21
commits ahead of the tag, and the working tree additionally has uncommitted
local changes on top of HEAD:

- **Code changed since the tag** (`git diff --stat v1-experimental-baseline..HEAD`
  touches `pytorchexample/client_app.py`, `server_app.py`, `task.py`,
  `pyproject.toml`): centralized server-side evaluation was added
  (`523871b`), the `verified` column was split into `sig_valid`/`has_nan`
  (`3399932`), the CIFAR-10 partition seed/logic was fixed
  (`bfce4fa`), `node_id` was fixed to encode as 8 bytes in the signed payload
  (`d31e52e`), execution order was randomized and logged (`446cb4c`), and CPU
  governor is now checked/logged at campaign start (`eb4bcc7`).
- **Uncommitted changes right now** (`git status`): `results/*.csv` (all 8
  files) are modified relative to the last commit, and `pyproject.toml` is
  modified (`num-supernodes` 5→10, `results-dir` `results`→`results_10clients`
  — this is the leftover config state from launching the 10-client cross-check
  after the 5-client campaign finished). `results/execution_order.log` and all
  `*_eval.csv` files are untracked (never committed).

**Practical implication:** the `results/` data analyzed in this report (the
5-client campaign, dated 2026-07-26 by file mtime and by
`campaign_5clients_20260725_171224.log`) was produced by the **post-tag** code
(with centralized eval, sig_valid/has_nan, the fixed partition, and randomized
execution order) — features the tag's own snapshot (`results_v1_frozen/`,
committed at the tag) does not have. If the thesis cites
`v1-experimental-baseline` as the provenance of the `results/` numbers used in
this report, that citation is incorrect; the correct provenance is HEAD
(`ec7768a`) plus the uncommitted `results/` regeneration from 2026-07-26.
Recommend either re-tagging HEAD (after committing the current `results/`) or
citing the commit hash `ec7768a` directly instead of the tag.

### A4. Model size and signed payload

| Item | Value |
|---|---|
| Trainable parameters (`CIFAR10CNN`) | **620,362** (verified by summing `p.numel()` over `model.parameters()`) |
| Signed/transmitted payload size | **2,482,153 bytes**, constant across all schemes, all rounds, all runs, all clients (verified: `payload_size` column has exactly one unique value per scheme, and that value is identical across all 7 signed schemes) |

Breakdown of the 2,482,153 bytes (from `client_app.py` / `server_app.py`):
`payload = weights_to_bytes(state_dict) + round.to_bytes(4, "big") + node_id.to_bytes(8, "big")`
→ the `np.save` serialization of the model's state dict (as an object array of
per-tensor numpy arrays, `allow_pickle=True`) is **2,482,141 bytes**, plus 4
bytes for the round number and 8 bytes for the node id = 2,482,153 bytes. This
is the exact byte length passed to `SignatureManager.sign()`/`.verify()`, and
matches the `payload_size` field written to the CSVs on the client side.

Note: the model class in the code is named `CIFAR10CNN` (in
`pytorchexample/task.py`), not `SimpleCNN` — flagging in case the thesis
prose uses "SimpleCNN" as a name for this architecture; the numbers above are
for the actual model used in the experiment regardless of name.

### A5. CIFAR-10 partition — confirmed, with one refinement

| Claim | Verified |
|---|---|
| Seed 42 | ✅ `_PARTITION_SEED = 42` in `task.py`, used as `np.random.default_rng(42).permutation(...)` |
| 10,000 training images per client | ✅ exactly 10,000 per partition (50,000 CIFAR-10 training images ÷ 5 clients, integer split with no remainder) |

Refinement: the partitioning is a **fixed-seed random shuffle-then-split**, not
a stratified/class-balanced split. It is *approximately* IID (CIFAR-10's
training set is already class-balanced at 5,000 images/class, so a random
shuffle is close to uniform in expectation) but not exactly uniform per
partition. Recomputing the actual class counts per partition (same code path
as `report_class_distribution`) gives:

| Partition (client) | class0 | class1 | class2 | class3 | class4 | class5 | class6 | class7 | class8 | class9 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 977 | 1027 | 1027 | 950 | 1024 | 1023 | 987 | 1011 | 1027 | 947 | 10,000 |
| 1 | 1010 | 1038 | 938 | 1038 | 1013 | 1016 | 948 | 996 | 960 | 1043 | 10,000 |
| 2 | 1011 | 949 | 1078 | 991 | 995 | 908 | 1035 | 979 | 1044 | 1010 | 10,000 |
| 3 | 1017 | 992 | 966 | 997 | 997 | 1055 | 958 | 1027 | 1000 | 991 | 10,000 |
| 4 | 985 | 994 | 991 | 1024 | 971 | 998 | 1072 | 987 | 969 | 1009 | 10,000 |

Per-class counts range 908–1078 (nominal even split would be 1000/class) —
close to but not exactly balanced. If the thesis states the partition as
"IID", that is a reasonable characterization but should not be read as
"exactly class-stratified".

### A6. Training / run configuration (as actually used for the main 5-client campaign)

| Parameter | Value | Source |
|---|---|---|
| Local epochs | 1 | `pyproject.toml` `local-epochs = 1` |
| Batch size | 32 | `pyproject.toml` `batch-size = 32` |
| Optimiser | SGD | hardcoded, `task.py` `torch.optim.SGD(...)` |
| Learning rate | 0.1 | `pyproject.toml` `learning-rate = 0.1` |
| Momentum | 0.9 | hardcoded, `task.py`: `SGD(net.parameters(), lr=lr, momentum=0.9)` (not configurable via `pyproject.toml`) |
| Gradient clipping | max_norm = 1.0 | hardcoded, `task.py`: `clip_grad_norm_(..., max_norm=1.0)` |
| Rounds | 30 | `pyproject.toml` `num-server-rounds = 30` |
| Clients (main campaign) | 5 | `pyproject.toml` (committed/HEAD version) `num-supernodes = 5`; `run_all_schemes.py` `NUM_SUPERNODES = 5` |
| Participation | Full — `fraction_train = 1.0`, `min_train_nodes = min_available_nodes = num_supernodes` (`server_app.py`); no client subsampling |
| Federated (per-client) evaluation | Disabled — `fraction-evaluate = 0.0` in `pyproject.toml` |
| Centralized evaluation | Enabled — `eval-central = true`; server evaluates the full 10,000-image CIFAR-10 test set once per round (plus once before round 1, "round 0") |
| Tamper injection | Inert during the campaign — `tamper-node-index = -1` |

Note: `pyproject.toml` currently on disk (uncommitted, working tree) shows
`num-supernodes = 10` and `results-dir = .../results_10clients` — this is the
leftover state from the later 10-client cross-check run, **not** the config
used for the main campaign. The values above are the committed-at-HEAD values,
which match what `run_all_schemes.py` (`NUM_SUPERNODES = 5`,
`RESULTS_DIR = "results"`) actually rewrote into `pyproject.toml` before every
`flwr run` invocation during the 5-client campaign (verified against
`campaign_5clients_20260725_171224.log`).

### A7. Runs per configuration and execution order

- **5 runs** per (scheme) configuration — `N_RUNS = 5` in `run_all_schemes.py`.
  8 schemes × 5 runs = 40 `flwr run` invocations total, all present in
  `results/execution_order.log` (40 `[N/40] starting ... run ...` lines).
- **Execution order is randomized, confirmed**: each run's scheme order is an
  independent shuffle, `random.Random(1000 + run_num).shuffle(SCHEMES)`. Actual
  positions occupied by each scheme across the campaign (from
  `execution_order.log`), showing no fixed block structure:

| Scheme | Positions occupied (of 1–40) |
|---|---|
| no_signature | 8, 10, 17, 31, 33 |
| ML-DSA-44 | 6, 15, 21, 28, 34 |
| ML-DSA-65 | 1, 12, 22, 25, 35 |
| ML-DSA-87 | 4, 9, 20, 26, 38 |
| Falcon-padded-512 | 3, 11, 23, 29, 37 |
| SPHINCS+-SHA2-128s-simple | 2, 13, 18, 30, 36 |
| RSA-2048 | 7, 16, 19, 32, 39 |
| ECDSA-256 | 5, 14, 24, 27, 40 |

No correlation between execution position and `train_time` was found (Pearson
r = +0.0253, p = 0.877, n = 40; ANOVA scheme-vs-train_time F=0.751, p=0.6314) —
i.e., no detectable thermal/frequency drift over the campaign's duration.

### A8. CSV column headers

Main per-client CSV (`results/<scheme>.csv`), header:
`run,round,node_id,scheme,keygen_time,sign_time,verify_time,train_time,train_loss,payload_size,sig_size,pubkey_size,sig_valid,has_nan`

| Column | Meaning |
|---|---|
| `run` | Campaign run number, 1–5 (independent repetition of the full 30-round experiment) |
| `round` | Federated round number, 1–30 |
| `node_id` | Flower-assigned simulated client id for this (scheme, run) — random per `flwr run` invocation, not stable across schemes/runs |
| `scheme` | Signature scheme name for this row |
| `keygen_time` | Wall-clock seconds to generate a fresh keypair for this client/round (`SignatureManager._keygen`, `time.perf_counter()`) |
| `sign_time` | Wall-clock seconds to sign the payload (post-hash; hashing excluded from the timing window) |
| `verify_time` | Wall-clock seconds for the server to verify this update's signature |
| `train_time` | Wall-clock seconds for local training (1 epoch over this client's 10,000-image partition) |
| `train_loss` | Average training loss over the last local epoch's batches |
| `payload_size` | Byte length of the exact payload that was signed (model weights serialization + round + node_id) |
| `sig_size` | Signature length in bytes |
| `pubkey_size` | Public key length in bytes |
| `sig_valid` | Boolean — whether the server's signature verification succeeded for this update (always `True` for `no_signature`) |
| `has_nan` | Boolean — whether any weight tensor in this client's update contained NaN (independent poisoning/divergence check, unrelated to signature validity) |

Centralized-evaluation CSV (`results/<scheme>_eval.csv`), header:
`run,round,accuracy,loss`

| Column | Meaning |
|---|---|
| `run` | Campaign run number, 1–5 |
| `round` | Round number, **0–30** (round 0 = initial random model, evaluated once before any training) |
| `accuracy` | Global model's top-1 accuracy on the full 10,000-image CIFAR-10 test set |
| `loss` | Global model's cross-entropy loss on the same test set |

### A9. Statistical test for cross-scheme comparison — ⚠️ CORRECTION

**The stated "Kruskal–Wallis, α = 0.05" is not what the repository's analysis
code uses.** `analysis/validate_results.py` — the only cross-scheme
statistical-comparison code in the repo — uses **one-way ANOVA**
(`scipy.stats.f_oneway`) throughout, at an implicit **α = 0.05** (every
verdict branches on `p > 0.05`). This appears in:

- Section 1 (train_loss divergence vs. scheme): `F=0.9980, p=0.4252`
- Section 3 (train_time vs. scheme): `F=0.7510, p=0.6314`
- Section 6 (accuracy divergence vs. scheme): `F=0.3262, p=0.9234`

No `scipy.stats.kruskal` call, or any other non-parametric test, exists
anywhere in the repository. If the thesis text says "Kruskal–Wallis", that
should be corrected to **one-way ANOVA, α = 0.05** to match what was actually
run and is reproducible from `analysis/validate_results.py` — OR the thesis
should keep "Kruskal–Wallis" and note that these particular p-values (B6, and
the ANOVA numbers above) were computed separately for this report using
`scipy.stats.kruskal`, not by the shipped analysis script. The α = 0.05
threshold itself is correct either way.

---

## B. RESULTS

All tables aggregate the 5-client campaign (`results/`) across all runs,
rounds, and clients (750 rows/scheme for the 7 signed schemes + `no_signature`,
except where noted).

### B1. Timing (ms) — keygen, sign, verify

| Scheme | keygen mean | keygen std | keygen median | sign mean | sign std | sign median | verify mean | verify std | verify median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_signature | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ML-DSA-44 | 0.236 | 0.206 | 0.191 | 0.237 | 0.111 | 0.205 | 0.050 | 0.015 | 0.043 |
| ML-DSA-65 | 0.270 | 0.206 | 0.222 | 0.310 | 0.164 | 0.261 | 0.069 | 0.016 | 0.062 |
| ML-DSA-87 | 0.310 | 0.210 | 0.263 | 0.362 | 0.178 | 0.315 | 0.098 | 0.017 | 0.089 |
| Falcon-padded-512 | 9.070 | 2.871 | 8.523 | 0.446 | 0.123 | 0.398 | 0.065 | 0.014 | 0.058 |
| SPHINCS+-SHA2-128s-simple | 28.558 | 8.240 | 23.634 | 192.710 | 47.524 | 170.110 | 0.294 | 0.039 | 0.282 |
| RSA-2048 | 62.142 | 40.711 | 50.540 | 1.421 | 0.486 | 1.169 | 0.088 | 0.034 | 0.082 |
| ECDSA-256 | 0.282 | 0.721 | 0.160 | 0.230 | 0.140 | 0.184 | 0.151 | 0.038 | 0.139 |

(RSA-2048 keygen has high variance — expected, since RSA key generation
involves probabilistic prime search whose runtime is inherently variable.)

### B2. Sizes (bytes) — measured sig_size and pubkey_size

| Scheme | sig_size (measured) | pubkey_size (measured) | Matches thesis hard-coded value? |
|---|---:|---:|---|
| ML-DSA-44 | 2420 (constant) | 1312 (constant) | ✅ matches 2420/1312 |
| ML-DSA-65 | 3309 (constant) | 1952 (constant) | ✅ matches 3309/1952 |
| ML-DSA-87 | 4627 (constant) | 2592 (constant) | ✅ matches 4627/2592 |
| Falcon-padded-512 | 666 (constant) | 897 (constant) | ✅ matches 666/897 |
| SPHINCS+-SHA2-128s-simple | 7856 (constant) | 32 (constant) | ✅ matches 7856/32 |
| RSA-2048 | 256 (constant) | **294 (constant)** | ✅ sig matches hard-coded 256; **pubkey_size = 294 bytes** (DER `SubjectPublicKeyInfo` encoding of a 2048-bit RSA public key with e=65537) — this is the value to hard-code if not already |
| ECDSA-256 | **70.99 avg** (69–72, DER-variable) | **91 (constant)** | Not previously hard-coded — see distribution below |

ECDSA-256 signature size is **not constant** (expected: DER encoding of two
256-bit integers has variable length depending on whether the high bit of each
integer requires a leading zero byte). Distribution over all 750 rows:

| sig_size (bytes) | count | % |
|---:|---:|---:|
| 69 | 1 | 0.1% |
| 70 | 188 | 25.1% |
| 71 | 378 | 50.4% |
| 72 | 183 | 24.4% |

Mean ECDSA-256 sig_size = **70.99 bytes** (used for the "average" figures in
B3/B4 below). ECDSA-256 public key size = **91 bytes** (constant — DER
`SubjectPublicKeyInfo` encoding of a P-256 public key).

### B3. Communication overhead per scheme

Per-round bytes = 5 clients × (sig_size + pubkey_size); cumulative = per-round
× 30 rounds; relative % = (sig_size + pubkey_size) / 2,482,153 (the A4 model
payload). Note N (clients) and T (rounds) cancel algebraically in the relative
% (verified numerically in `analysis/results/section4_nt_cancellation_check.csv`:
per-row-mean % and aggregate-sum % agree to <1e-14), so the relative % column
is identical whether computed per-update or per-round or cumulative.

| Scheme | sig+pubkey (bytes) | Per-round (5 clients), bytes | Cumulative (30 rounds), bytes | Relative % of model payload |
|---|---:|---:|---:|---:|
| ML-DSA-44 | 3,732 | 18,660 | 559,800 | 0.1504% |
| ML-DSA-65 | 5,261 | 26,305 | 789,150 | 0.2120% |
| ML-DSA-87 | 7,219 | 36,095 | 1,082,850 | 0.2908% |
| Falcon-padded-512 | 1,563 | 7,815 | 234,450 | 0.0630% |
| SPHINCS+-SHA2-128s-simple | 7,888 | 39,440 | 1,183,200 | 0.3178% |
| RSA-2048 | 550 | 2,750 | 82,500 | 0.0222% |
| ECDSA-256 | 161.99 (avg) | 809.95 (avg) | 24,298.6 (avg) | 0.0065% |

PQC per-round/cumulative figures verified against `analysis/results/section4_communication_cost.csv` — all confirmed. RSA-2048 and ECDSA-256 rows added from the same source.

### B4. Break-even model size at 1% / 5% / 10% overhead

M_breakeven = (sig_size + pubkey_size) / threshold. Reported in **decimal KB
(1000 B)**, matching the convention implied by the thesis's hard-coded
Falcon-padded-512 value (156/31/16 KB); KiB (1024 B, as computed natively by
`analysis/validate_results.py`) given alongside for completeness.

| Scheme | 1% (KB) | 1% (KiB) | 5% (KB) | 5% (KiB) | 10% (KB) | 10% (KiB) |
|---|---:|---:|---:|---:|---:|---:|
| ML-DSA-44 | 373.2 | 364.45 | 74.6 | 72.89 | 37.3 | 36.45 |
| ML-DSA-65 | 526.1 | 513.77 | 105.2 | 102.75 | 52.6 | 51.38 |
| ML-DSA-87 | 721.9 | 704.98 | 144.4 | 141.00 | 72.2 | 70.50 |
| **Falcon-padded-512** | **156.3** | 152.64 | **31.3** | 30.53 | **15.6** | 15.26 |
| SPHINCS+-SHA2-128s-simple | 788.8 | 770.31 | 157.8 | 154.06 | 78.9 | 77.03 |
| RSA-2048 | 55.0 | 53.71 | 11.0 | 10.74 | 5.5 | 5.37 |
| ECDSA-256 | 16.2 | 15.82 | 3.2 | 3.16 | 1.6 | 1.58 |

**Verification: Falcon-padded-512 = 156.3 / 31.3 / 15.6 KB (decimal)** — matches
the thesis's hard-coded "156 / 31 / 16 KB" once rounded to the nearest integer.
⚠️ **Flag**: this match only holds under the decimal-KB (1000 B) convention.
Under KiB (1024 B) — which is what `model_size_kib` in
`analysis/results/section4_breakeven.csv` actually computes — the values are
152.64 / 30.53 / 15.26, which round to 153 / 31 / 15, **not** 156/31/16. Two of
the three digits still round-match by coincidence, but the thesis should state
explicitly which unit convention (KB vs KiB) it uses, since the script's own
CSV column is *named* `_kib` but the thesis figure only matches if read as
decimal KB.

### B5. Per-round cryptographic cost

crypto_cost_round(scheme, run, round) = Σ over 5 clients of
(keygen_time + sign_time + verify_time); train_cost_round = Σ over 5 clients of
train_time; % = crypto/(crypto+train). **Caveat** (from the codebase's own
audit, `validate_results.py`): no independent round-level wall-clock latency
is recorded anywhere in the pipeline — this sum overestimates true round
latency if clients run in parallel (they do, under Flower's simulation
engine), but it correctly preserves the ratio of total crypto cost to total
train cost per round, which is what's reported as "% of round time" below.

| Scheme | Crypto cost/round, mean (ms) | Crypto cost/round, std (ms) | Train cost/round, mean (ms) | Crypto as % of (crypto+train) |
|---|---:|---:|---:|---:|
| no_signature | 0.000 | 0.000 | 100,927.5 | 0.0000% |
| ML-DSA-44 | 2.612 | 0.960 | 100,338.6 | 0.0026% |
| ML-DSA-65 | 3.245 | 0.998 | 100,733.6 | 0.0032% |
| ML-DSA-87 | 3.848 | 0.968 | 100,731.8 | 0.0038% |
| Falcon-padded-512 | 47.907 | 5.884 | 100,216.2 | 0.0478% |
| SPHINCS+-SHA2-128s-simple | 1,107.807 | 101.910 | 100,483.5 | 1.0905% |
| RSA-2048 | 318.259 | 91.473 | 100,708.9 | 0.3151% |
| ECDSA-256 | 3.317 | 3.615 | 100,471.3 | 0.0033% |

SPHINCS+-SHA2-128s-simple has both the highest absolute per-round crypto cost
and the highest fraction of round cost (~1.09%); every scheme's crypto cost is
under 1.1% of the training cost — signing overhead is negligible relative to
local training time regardless of scheme.

### B6. Accuracy

**Final-round (round 30) central test accuracy, mean ± std across the 5 runs:**

| Scheme | Final accuracy (mean ± std) |
|---|---|
| no_signature | 0.7208 ± 0.0077 |
| ML-DSA-44 | 0.7210 ± 0.0064 |
| ML-DSA-65 | 0.7175 ± 0.0095 |
| ML-DSA-87 | 0.7226 ± 0.0065 |
| Falcon-padded-512 | 0.7197 ± 0.0049 |
| SPHINCS+-SHA2-128s-simple | 0.7228 ± 0.0045 |
| RSA-2048 | 0.7247 ± 0.0085 |
| ECDSA-256 | 0.7219 ± 0.0045 |

All 8 schemes land within [0.7175, 0.7247] — a spread of 0.0072 accuracy, well
inside the per-scheme run-to-run std (0.0045–0.0095). **Curves coincide**:
accuracy-by-round milestones (rounds 0, 1, 5, 10, 15, 20, 25, 30) are visually
indistinguishable across all 8 schemes (see
`analysis/results/section6_accuracy_by_round.csv` for the full per-round
series); e.g. at round 5 all schemes are within [0.6356, 0.6404], at round 30
within the range above.

**Round-0/round-1 bit-identical check**: the evaluation CSVs record round 0 =
the initial (pre-training, pre-signing) global model. Its accuracy is
**bit-identical across all 8 schemes** (max abs diff = 0.0, verified directly
by re-reading `*_eval.csv` round-0 rows). This is the cleanest possible
orthogonality check, since round 0 depends only on `torch.manual_seed(run_number*42)`
and precedes any training or signing. (`validate_results.py`'s own section 6
reports the same result under the label "Ronda 0".) There is no "round 1"
column distinct from round 0 in the eval CSVs — round numbering there is
0–30, with round 0 being the state before any training round runs; if "round
1" in the request meant the initial global model (as opposed to the
post-round-1 model), that is what round 0 captures and it is confirmed
bit-identical. The post-round-1 models (round 1 in the 0–30 numbering) are
**not** bit-identical across schemes (max abs diff up to 0.0324) — but this
divergence does not correlate with which scheme was used (ANOVA F=0.3262,
p=0.9234), and is attributable to FedAvg's non-associative floating-point
summation across independent `flwr run`/Ray subprocess invocations, not to the
signature layer.

**Kruskal–Wallis test (computed separately for this report; not part of the
shipped analysis code — see A9):**

| Test | H statistic | p-value | Groups / n |
|---|---:|---:|---|
| Final accuracy across 8 schemes | 2.7344 | **0.9084** | 8 groups, n=5 runs each |
| train_loss, final round only, across 8 schemes | 4.2658 | **0.7487** | 8 groups, n=25 (5 runs × 5 clients) each |
| train_loss, all rounds/runs/clients pooled, across 8 schemes | 0.2348 | **1.0000** | 8 groups, n=750 each |

All three are far from significant at α=0.05 — no evidence that the signature
scheme affects either accuracy or training loss. This is consistent with the
ANOVA-based orthogonality conclusions already in `validate_results.py`
(Section 1: F=0.9980, p=0.4252 on train_loss divergence; Section 6: F=0.3262,
p=0.9234 on accuracy divergence).

---

## C. DATA HYGIENE

### C1. sig_valid

**`sig_valid == True` for every recorded row, in every scheme, with zero
exceptions.** Verified across all 750×8 = 6,000 rows of the 5-client campaign
and all 1,500×8 = 12,000 rows of the 10-client cross-check. No client update
was ever rejected for signature failure in either campaign.

### C2. NaN / has_nan

**No `has_nan == True` rows and no NaN `train_loss` values anywhere.** Checked
across every column of every scheme's CSV, both campaigns (`validate_results.py`
section 2's full per-column NaN scan reports zero for every scheme × column
combination). No exclusion is needed and B1–B6 above use the full, unfiltered
dataset. No filter or re-run is required for data-hygiene reasons.

### C3. Completeness

**5-client campaign (`results/`) — complete.**

| Scheme | Main CSV rows | Expected (5×30×5) | Eval CSV rows | Expected (5×31) |
|---|---:|---:|---:|---:|
| no_signature | 750 | 750 ✅ | 155 | 155 ✅ |
| ML-DSA-44 | 750 | 750 ✅ | 155 | 155 ✅ |
| ML-DSA-65 | 750 | 750 ✅ | 155 | 155 ✅ |
| ML-DSA-87 | 750 | 750 ✅ | 155 | 155 ✅ |
| Falcon-padded-512 | 750 | 750 ✅ | 155 | 155 ✅ |
| SPHINCS+-SHA2-128s-simple | 750 | 750 ✅ | 155 | 155 ✅ |
| RSA-2048 | 750 | 750 ✅ | 155 | 155 ✅ |
| ECDSA-256 | 750 | 750 ✅ | 155 | 155 ✅ |

`execution_order.log` also confirms all 40/40 (8 schemes × 5 runs) launches
completed (`[N/40] starting ...` present for N=1..40, and each pairs with a
`-> completed` status line). `node_id` consistency check: all 40 (scheme, run)
pairs use the same set of 5 node ids across all their rounds — no gaps.

**10-client cross-check (`results_10clients/`) — also complete.**

| Scheme | Main CSV rows | Expected (5×30×10) | Eval CSV rows | Expected (5×31) |
|---|---:|---:|---:|---:|
| no_signature | 1500 | 1500 ✅ | 155 | 155 ✅ |
| ML-DSA-44 | 1500 | 1500 ✅ | 155 | 155 ✅ |
| ML-DSA-65 | 1500 | 1500 ✅ | 155 | 155 ✅ |
| ML-DSA-87 | 1500 | 1500 ✅ | 155 | 155 ✅ |
| Falcon-padded-512 | 1500 | 1500 ✅ | 155 | 155 ✅ |
| SPHINCS+-SHA2-128s-simple | 1500 | 1500 ✅ | 155 | 155 ✅ |
| RSA-2048 | 1500 | 1500 ✅ | 155 | 155 ✅ |
| ECDSA-256 | 1500 | 1500 ✅ | 155 | 155 ✅ |

No gaps in either campaign; no missing (run, round) tuples in either
`section2_missing_rows.csv` output (both empty).

For reference, final-round accuracy on the 10-client cross-check (not part of
the requested B tables, included here only to support C3's "no gaps" claim):
0.7285–0.7330 across all 8 schemes — same pattern of scheme-independence as
the 5-client campaign.

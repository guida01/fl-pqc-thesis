# Securing Federated Model Updates with Post-Quantum Digital Signatures

Experimental implementation for the thesis **Securing Federated Model Updates with Post-Quantum Digital Signatures**.

The project integrates digital signatures into a Flower/PyTorch federated learning workflow using CIFAR-10.

## Signature configurations

The final experiments evaluate:

- `no_signature`
- `ML-DSA-44`
- `ML-DSA-65`
- `ML-DSA-87`
- `Falcon-padded-512`
- `SLH_DSA_PURE_SHA2_128S`
- `RSA-2048`
- `ECDSA-256`

## Environment

The final experiments use:

- `liboqs 0.16.0`
- `liboqs-python 0.16.0`

From `thesis_fl/quickstart-pytorch`, activate the pinned environment with:

```bash
source scripts/activate_final_env.sh
```

A successful activation must report:

```text
liboqs:          0.16.0
liboqs-python:   0.16.0
OQS environment: OK
```

Do not rely on the system-wide liboqs installation.

The native liboqs installation used by the final experiments is expected at:

```text
$VIRTUAL_ENV/oqs-0.16.0
```

## Tests

Run the complete test suite with:

```bash
python -m pytest -q
```

Run the protocol smoke test with:

```bash
python smoke_test.py
```

The current test suite contains 64 tests.

The protocol smoke test checks:

- valid signature acceptance;
- modified model-update rejection;
- modified signature rejection;
- cross-round replay rejection;
- the expected experiment CSV schema.

## Signed update

For signed configurations, the signed message consists of:

```text
serialized model weights
+ server round
+ client node ID
```

The server reconstructs the same representation before signature verification.

Signatures and public keys are transported as binary values through Flower `ConfigRecord`.

The public key accompanies each signed update. The implementation therefore evaluates signed-update integrity under an assumed trusted/authenticated key-delivery context. It does not implement a full PKI or server-side identity-to-key binding.

A fresh signing keypair is generated for every signed client update. Key-generation results must therefore be interpreted under this experimental ephemeral-key-per-update policy.

## Recorded metrics

Each client-update CSV row records:

```text
run
round
node_id
scheme
keygen_time
client_serialize_time
sign_time
server_serialize_time
verify_time
server_verify_total_time
train_time
train_loss
signed_payload_size
sig_size
pubkey_size
num_examples
sig_valid
has_nan
```

Metric interpretation:

- `keygen_time`: client key-generation time under the ephemeral-key-per-update policy.
- `client_serialize_time`: time required by the client to construct the deterministic signed representation.
- `sign_time`: signing backend time over the complete signed representation.
- `server_serialize_time`: time required by the server to reconstruct that representation.
- `verify_time`: verification timing reported by `SignatureManager`.
- `server_verify_total_time`: total server-side verifier setup and verification processing time.
- `signed_payload_size`: size of the deterministic representation submitted to the signature algorithm.
- `sig_size`: raw signature size in bytes.
- `pubkey_size`: raw public-key size in bytes.
- `num_examples`: number of examples used by the client for the local update.
- `sig_valid`: whether the received signature passed verification.
- `has_nan`: whether the submitted model update contained NaN values.

`signed_payload_size` is not a measurement of Flower, gRPC, TLS, or network-wire bytes.

Raw cryptographic metadata size can be derived as:

```text
sig_size + pubkey_size
```

The cryptographic-operation cost used in analysis is:

```text
keygen_time + sign_time + verify_time
```

The broader security-wrapper processing cost is:

```text
client_serialize_time
+ keygen_time
+ sign_time
+ server_serialize_time
+ server_verify_total_time
```

These summed compute-time quantities are not federated-round wall-clock latency when clients execute concurrently.

## CPU governor

CPU frequency scaling can affect timing measurements.

Check all available CPU governors before collecting final measurements:

```bash
grep -H . /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor
```

The final campaign should be run with all available CPUs using the `performance` governor.

On systems where `performance` is available, one way to set it manually is:

```bash
for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
    echo performance | sudo tee "$f" >/dev/null
done
```

Verify afterwards:

```bash
grep -H . /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor
```

The campaign runners report the observed governor but do not modify it automatically.

## Primary campaign: 5 clients

Configuration:

```text
clients:        5
server rounds:  30
runs:           10
configurations: 8
```

Activate the environment:

```bash
source scripts/activate_final_env.sh
```

Run:

```bash
python run_all_schemes.py
```

Results are stored in:

```text
results_final_5clients/
```

Expected update rows per configuration:

```text
10 runs x 30 rounds x 5 clients = 1500 rows
```

Expected centralized-evaluation rows per configuration:

```text
10 runs x (30 rounds + round 0) = 310 rows
```

The available execution-order log is stored in:

```text
results_final_5clients/execution_order.log
```

The scheme order is deterministically randomized independently for each run to reduce systematic confounding with campaign execution position.

## Secondary campaign: 10 clients

Configuration:

```text
clients:        10
server rounds:  30
runs:           10
configurations: 8
```

Activate the environment:

```bash
source scripts/activate_final_env.sh
```

Run:

```bash
python run_10clients.py
```

Results are stored in:

```text
results_final_10clients/
```

Expected update rows per configuration:

```text
10 runs x 30 rounds x 10 clients = 3000 rows
```

Expected centralized-evaluation rows per configuration:

```text
10 runs x (30 rounds + round 0) = 310 rows
```

The available execution-order log is stored in:

```text
results_final_10clients/execution_order.log
```

## Campaign safeguards

The final campaign runners:

- verify `liboqs == 0.16.0`;
- verify `liboqs-python == 0.16.0`;
- verify that the required post-quantum signature mechanisms are available;
- randomize scheme execution order deterministically per run;
- refuse to mix a new campaign with an existing non-empty final-results directory;
- stop if a Flower execution fails;
- validate expected CSV row counts;
- restore `pyproject.toml` when execution terminates.

## Recovering an interrupted campaign

The normal campaign runners are intended for a fresh final-results directory and refuse to mix new measurements with existing campaign data.

If a final campaign is interrupted, use the dedicated safe resume tool instead of restarting the normal runner:

```bash
python resume_campaign.py --clients 5
```

For the secondary campaign:

```bash
python resume_campaign.py --clients 10
```

A completed scheme/run pair is preserved and skipped.

A partially written scheme/run pair is removed from both its update CSV and centralized-evaluation CSV and then rerun from the beginning.

The recovery tool refuses unusual or inconsistent states rather than deleting data silently.

## Result validation

The main validation script is:

```bash
python analysis/validate_results.py
```

For the 5-client final campaign:

```bash
python analysis/validate_results.py results_final_5clients
```

For the 10-client final campaign:

```bash
python analysis/validate_results.py results_final_10clients
```

The validation and analysis code distinguishes raw cryptographic metadata size from actual network traffic.

Final campaign results must remain separate from historical pre-rerun results.

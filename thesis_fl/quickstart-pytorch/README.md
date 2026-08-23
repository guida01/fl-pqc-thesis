# Securing Federated Learning with Post-Quantum Digital Signatures

Experimental implementation for the thesis **Securing Federated Learning with Post-Quantum Digital Signatures**.

The project integrates digital signatures into a Flower/PyTorch federated learning workflow using CIFAR-10 and evaluates cryptographic computation, serialization overhead, metadata size, and model accuracy.

## Signature configurations

The final experiments evaluate eight configurations:

- `no_signature`
- `ML-DSA-44`
- `ML-DSA-65`
- `ML-DSA-87`
- `Falcon-padded-512`
- `SLH_DSA_PURE_SHA2_128S`
- `RSA-2048`
- `ECDSA-256`

## Final OQS environment

The final experiments use:

- `liboqs 0.16.0`
- `liboqs-python 0.16.0`

From `thesis_fl/quickstart-pytorch`, activate the pinned environment with:

```bash
source scripts/activate_final_env.sh

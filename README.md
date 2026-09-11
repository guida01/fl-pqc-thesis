# Securing Federated Model Updates with Post-Quantum Digital Signatures

Reproducibility artefact for the MSc thesis **Securing Federated Model Updates with Post-Quantum Digital Signatures**.

This repository contains the implementation, experimental campaign scripts, analysis pipeline, derived results, and figures used to evaluate classical and post-quantum digital signatures in a Flower/PyTorch federated learning workflow.

## Experimental setup

- Framework: Flower with FedAvg
- Dataset: CIFAR-10
- Model: SimpleCNN
- Client campaigns: 5 and 10 clients
- Federated rounds: 30
- Matched runs: 10
- Participation: full participation

The evaluated configurations are:

- `no_signature`
- `RSA-2048`
- `ECDSA-256`
- `ML-DSA-44`
- `ML-DSA-65`
- `ML-DSA-87`
- `Falcon-padded-512`
- `SLH-DSA-SHA2-128s`

## Security scope

For signed configurations, the protected payload contains the serialized model weights, the server round identifier, and the client/node identifier.

The prototype evaluates payload integrity and cross-round replay resistance under an assumed trusted or authenticated external public-key delivery mechanism. It does not implement a PKI or persistent client-identity-to-key binding.

A fresh key pair is generated for every signed client update as an experimental design choice.

## Repository structure

The main implementation and detailed reproduction instructions are available under:

`thesis_fl/quickstart-pytorch/`

Derived analysis outputs are available under:

`thesis_fl/quickstart-pytorch/analysis/results/`

Thesis figures are available under:

`thesis_fl/quickstart-pytorch/analysis/figures/`

Raw experimental measurements are available under:

`thesis_fl/quickstart-pytorch/results_final_5clients/`

and

`thesis_fl/quickstart-pytorch/results_final_10clients/`

Derived analysis outputs are available under:

`thesis_fl/quickstart-pytorch/analysis/results/`

## Reproduction

See `thesis_fl/quickstart-pytorch/README.md` for environment setup, experimental execution, validation, and analysis instructions.

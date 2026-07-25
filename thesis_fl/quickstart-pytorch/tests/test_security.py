"""Security tests for pytorchexample/signature_manager.py.

Parametrized over all 7 signature schemes (5 PQC via liboqs + 2 classical via
`cryptography`), pulled directly from SignatureManager.PQC_SCHEMES /
CLASSICAL_SCHEMES so this file never drifts from the scheme list the rest of
the codebase uses.

Payloads are built with `make_payload()`, which replicates byte-for-byte the
construction in client_app.py:49-53 (identically reconstructed in
server_app.py:56-60):

    payload = (
        weights_to_bytes(state_dict)
        + server_round.to_bytes(4, "big")
        + node_id.to_bytes(8, "big")
    )

using real CIFAR10CNN() weights, not random bytes.
"""

import pytest
import torch

from pytorchexample.signature_manager import SignatureManager
from pytorchexample.task import CIFAR10CNN, weights_to_bytes

ALL_SCHEMES = SignatureManager.PQC_SCHEMES + SignatureManager.CLASSICAL_SCHEMES

ROUND_T = 1
NODE_A = 1001
NODE_B = 2002


def make_payload(state_dict, server_round: int, node_id: int) -> bytes:
    """Exact replica of client_app.py:49-53 / server_app.py:56-60."""
    return (
        weights_to_bytes(state_dict)
        + server_round.to_bytes(4, "big")
        + node_id.to_bytes(8, "big")
    )


def do_verify(scheme: str, payload: bytes, signature: bytes, public_key: bytes):
    """Exact replica of the verification call in server_app.py:61-63 —
    a fresh SignatureManager is constructed for verification, same as
    production (the server never reuses the client's signer object)."""
    return SignatureManager(scheme).verify(payload, signature, public_key)


def assert_never_valid(scheme, payload, signature, public_key):
    """For inputs a backend may legitimately reject by raising instead of
    returning False (malformed/truncated/empty signatures): accept either
    outcome, but NEVER accept is_valid is True."""
    try:
        is_valid, _ = do_verify(scheme, payload, signature, public_key)
    except Exception:
        return
    assert is_valid is False


@pytest.fixture(scope="module", params=ALL_SCHEMES, ids=ALL_SCHEMES)
def base_case(request):
    """One (scheme, signed payload) fixture, built once per scheme and
    reused across all 8 tests below to avoid redundant keygen/sign calls."""
    scheme = request.param
    torch.manual_seed(0)
    state_dict = CIFAR10CNN().state_dict()
    payload = make_payload(state_dict, ROUND_T, NODE_A)

    sig_mgr = SignatureManager(scheme)
    signature, _ = sig_mgr.sign(payload)
    public_key = sig_mgr.public_key_bytes

    return {
        "scheme": scheme,
        "state_dict": state_dict,
        "payload": payload,
        "signature": signature,
        "public_key": public_key,
    }


# T1 — valid signature over intact payload -> verify == True
def test_t1_valid_signature_verifies(base_case):
    is_valid, _ = do_verify(
        base_case["scheme"], base_case["payload"],
        base_case["signature"], base_case["public_key"],
    )
    assert is_valid is True


# T2 — one byte of the weights region altered after signing -> verify == False
def test_t2_tampered_weights_rejected(base_case):
    bad_payload = bytearray(base_case["payload"])
    bad_payload[100] ^= 0xFF  # offset 100 is deep inside weights_to_bytes() output
    is_valid, _ = do_verify(
        base_case["scheme"], bytes(bad_payload),
        base_case["signature"], base_case["public_key"],
    )
    assert is_valid is False


# T3 — one byte of the signature altered -> verify == False
def test_t3_tampered_signature_rejected(base_case):
    bad_sig = bytearray(base_case["signature"])
    bad_sig[0] ^= 0xFF
    is_valid, _ = do_verify(
        base_case["scheme"], base_case["payload"],
        bytes(bad_sig), base_case["public_key"],
    )
    assert is_valid is False


# T4 — truncated signature (sig[:-1]) -> verify == False, or raises (never True)
#
# Observed behavior per scheme, measured directly against this liboqs 0.14.1 /
# cryptography 46.0.7 install (all 7 return False cleanly; none raise):
#   RSA-2048                     -> returns False (caught internally, signature_manager.py:96-101)
#   ECDSA-256                    -> returns False (caught internally, signature_manager.py:103-108)
#   ML-DSA-44 / ML-DSA-65 / ML-DSA-87   -> returns False (oqs.Signature.verify rejects malformed length)
#   Falcon-padded-512             -> returns False (oqs.Signature.verify rejects malformed length)
#   SPHINCS+-SHA2-128s-simple     -> returns False (oqs.Signature.verify rejects malformed length)
def test_t4_truncated_signature_rejected(base_case):
    truncated = base_case["signature"][:-1]
    assert_never_valid(
        base_case["scheme"], base_case["payload"],
        truncated, base_case["public_key"],
    )


# T5 — valid signature verified with the public key of a different keypair -> verify == False
def test_t5_wrong_keypair_rejected(base_case):
    other_public_key = SignatureManager(base_case["scheme"]).public_key_bytes
    is_valid, _ = do_verify(
        base_case["scheme"], base_case["payload"],
        base_case["signature"], other_public_key,
    )
    assert is_valid is False


# T6 — round-t signature verified against a payload reconstructed for round t+1 -> verify == False
def test_t6_replay_next_round_rejected(base_case):
    next_round_payload = make_payload(base_case["state_dict"], ROUND_T + 1, NODE_A)
    is_valid, _ = do_verify(
        base_case["scheme"], next_round_payload,
        base_case["signature"], base_case["public_key"],
    )
    assert is_valid is False


# T7 — node-A signature verified against a payload reconstructed with node B's id, same round -> verify == False
def test_t7_wrong_node_rejected(base_case):
    other_node_payload = make_payload(base_case["state_dict"], ROUND_T, NODE_B)
    is_valid, _ = do_verify(
        base_case["scheme"], other_node_payload,
        base_case["signature"], base_case["public_key"],
    )
    assert is_valid is False


# T8 — empty signature (b"") -> verify == False, or raises (never True)
#
# Observed behavior per scheme, measured directly against this liboqs 0.14.1 /
# cryptography 46.0.7 install (all 7 return False cleanly; none raise):
#   RSA-2048                     -> returns False (caught internally, signature_manager.py:96-101)
#   ECDSA-256                    -> returns False (caught internally, signature_manager.py:103-108)
#   ML-DSA-44 / ML-DSA-65 / ML-DSA-87   -> returns False (oqs.Signature.verify rejects zero-length signature)
#   Falcon-padded-512             -> returns False (oqs.Signature.verify rejects zero-length signature)
#   SPHINCS+-SHA2-128s-simple     -> returns False (oqs.Signature.verify rejects zero-length signature)
def test_t8_empty_signature_rejected(base_case):
    assert_never_valid(
        base_case["scheme"], base_case["payload"],
        b"", base_case["public_key"],
    )

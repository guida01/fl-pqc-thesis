import oqs
import time
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend

class SignatureManager:

    # NIST-standardized PQC schemes - handled via liboqs
    PQC_SCHEMES = [
        "ML-DSA-44",
        "ML-DSA-65",
        "ML-DSA-87",
        "Falcon-padded-512",
        "SPHINCS+-SHA2-128s-simple"]

    # Classical schemes — baseline for comparison
    CLASSICAL_SCHEMES = ["RSA-2048", "ECDSA-256"]

    def __init__(self, scheme: str, generate_keypair: bool = True):
        self.scheme = scheme
        self.public_key_bytes = None
        self.keygen_time = 0.0

        # Signing requires a private/public key pair, but verification only
        # needs the public key supplied by the caller.
        if generate_keypair:
            self._keygen()

    def _keygen(self):
        start = time.perf_counter()

        if self.scheme in self.PQC_SCHEMES:
            # generate_keypair() returns the public key and stores the private key internally
            self._signer = oqs.Signature(self.scheme)
            self.public_key_bytes = self._signer.generate_keypair()

        elif self.scheme == "RSA-2048":
            # exponent 65537 is standard — Fermat prime, efficient and secure
            self._private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )
            self._public_key = self._private_key.public_key()
            self.public_key_bytes = self._public_key.public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo
            )

        elif self.scheme == "ECDSA-256":
            # SECP256R1 = P-256 curve, equivalent to 128 bits of classical security
            self._private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
            self._public_key = self._private_key.public_key()
            self.public_key_bytes = self._public_key.public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo
            )

        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

        self.keygen_time = time.perf_counter() - start

    def sign(self, data: bytes):
        if self.public_key_bytes is None:
            raise RuntimeError(
                "Cannot sign with a verifier-only SignatureManager; "
                "construct it with generate_keypair=True."
            )
        start = time.perf_counter()

        if self.scheme in self.PQC_SCHEMES:
            signature = self._signer.sign(data)

        elif self.scheme == "RSA-2048":
            signature = self._private_key.sign(
                data,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )

        elif self.scheme == "ECDSA-256":
            signature = self._private_key.sign(
                data,
                ec.ECDSA(hashes.SHA256()),
            )
        sign_time = time.perf_counter() - start
        return signature, sign_time

    def verify(self, data: bytes, signature: bytes, public_key_bytes: bytes):
        from cryptography.hazmat.primitives.serialization import load_der_public_key
        # Verifier/public-key setup is kept outside the timing window.
        # verify_time measures the backend verification call over the full payload,
        # including message processing performed by the signature implementation.
        if self.scheme in self.PQC_SCHEMES:
            verifier = oqs.Signature(self.scheme)
        elif self.scheme in ("RSA-2048", "ECDSA-256"):
            pub = load_der_public_key(public_key_bytes, backend=default_backend())

        start = time.perf_counter()

        if self.scheme in self.PQC_SCHEMES:
            is_valid = verifier.verify(data, signature, public_key_bytes)
        elif self.scheme == "RSA-2048":
            try:
                pub.verify(
                    signature,
                    data,
                    padding.PKCS1v15(),
                    hashes.SHA256(),
                )
                is_valid = True
            except Exception:
                is_valid = False

        elif self.scheme == "ECDSA-256":
            try:
                pub.verify(
                    signature,
                    data,
                    ec.ECDSA(hashes.SHA256()),
                )
                is_valid = True
            except Exception:
                is_valid = False

        verify_time = time.perf_counter() - start
        return is_valid, verify_time
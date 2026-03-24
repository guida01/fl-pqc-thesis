import oqs
import time
import hashlib
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend

class SignatureManager:

    PQC_SCHEMES = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87", "Falcon-512", "SPHINCS+-SHA2-128s-simple"]
    CLASSICAL_SCHEMES = ["RSA-2048", "ECDSA-256"]

    def __init__(self, scheme: str):
        self.scheme = scheme
        self.public_key_bytes = None
        self._keygen()

    def _keygen(self):
        start = time.perf_counter()

        if self.scheme in self.PQC_SCHEMES:
            self._signer = oqs.Signature(self.scheme)
            self.public_key_bytes = self._signer.generate_keypair()

        elif self.scheme == "RSA-2048":
            self._private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )
            self._public_key = self._private_key.public_key()
            self.public_key_bytes = self._public_key.public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo
            )

        elif self.scheme == "ECDSA-256":
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
        digest = hashlib.sha256(data).digest()
        start = time.perf_counter()

        if self.scheme in self.PQC_SCHEMES:
            signature = self._signer.sign(digest)

        elif self.scheme == "RSA-2048":
            signature = self._private_key.sign(digest, padding.PKCS1v15(), hashes.SHA256())

        elif self.scheme == "ECDSA-256":
            signature = self._private_key.sign(digest, ec.ECDSA(hashes.SHA256()))

        sign_time = time.perf_counter() - start
        return signature, sign_time

    def verify(self, data: bytes, signature: bytes, public_key_bytes: bytes):
        digest = hashlib.sha256(data).digest()
        start = time.perf_counter()

        if self.scheme in self.PQC_SCHEMES:
            verifier = oqs.Signature(self.scheme)
            is_valid = verifier.verify(digest, signature, public_key_bytes)

        elif self.scheme == "RSA-2048":
            from cryptography.hazmat.primitives.serialization import load_der_public_key
            pub = load_der_public_key(public_key_bytes, backend=default_backend())
            try:
                pub.verify(signature, digest, padding.PKCS1v15(), hashes.SHA256())
                is_valid = True
            except Exception:
                is_valid = False

        elif self.scheme == "ECDSA-256":
            from cryptography.hazmat.primitives.serialization import load_der_public_key
            pub = load_der_public_key(public_key_bytes, backend=default_backend())
            try:
                pub.verify(signature, digest, ec.ECDSA(hashes.SHA256()))
                is_valid = True
            except Exception:
                is_valid = False

        verify_time = time.perf_counter() - start
        return is_valid, verify_time

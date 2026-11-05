"""Post-quantum cryptography for the PBFT consensus of CoSched-GNN.

Real cryptography, not a model:
  * signatures: ML-DSA-44 / 65 / 87 (CRYSTALS-Dilithium, NIST FIPS 204), and ECDSA P-256 as
    the classical comparison;
  * key exchange: ML-KEM-768 (CRYSTALS-Kyber, NIST FIPS 203) for every vehicle-to-server
    session, followed by AES-256-GCM encryption of the uploaded task descriptors;
  * hashes: SHA3-256.

Backends: liboqs (Open Quantum Safe, `pip install liboqs-python`) is used when available,
otherwise the `pqcrypto` wheel (slower signing); ECDSA and AES-GCM come from `cryptography`.

The time each operation takes is measured on the running machine (`measure`) and fed into
the simulated consensus delay, so the delay uses measured, not assumed, costs.

Quantum adversary: a cryptographically relevant quantum computer runs Shor's algorithm and
recovers an ECDSA secret key from its public key. `Signer.recover_secret` models exactly that
capability (it returns the secret key of a classical scheme); no such attack is known for
ML-DSA, so it returns None there.
"""
import hashlib
import os
import time

SCHEMES = {                       # name used in the code -> (algorithm, pqcrypto module, level)
    "ecdsa": ("ECDSA-P256", None, "classical (broken by Shor's algorithm)"),
    "dilithium2": ("ML-DSA-44", "ml_dsa_44", "post-quantum, NIST level 2"),
    "dilithium3": ("ML-DSA-65", "ml_dsa_65", "post-quantum, NIST level 3"),
    "dilithium5": ("ML-DSA-87", "ml_dsa_87", "post-quantum, NIST level 5"),
}
KEM_ALG, KEM_MOD = "ML-KEM-768", "ml_kem_768"
INSTALL_HINT = ("Post-quantum library missing. On Kaggle run first:\n"
                "  !pip install -q liboqs-python pqcrypto\n"
                "(liboqs-python builds liboqs on its first import, needs Internet on), "
                "or run with --set crypto=model to skip the real cryptography.")


def _load_oqs():
    try:
        import oqs                                          # noqa: F401
        return oqs
    except (Exception, SystemExit):                         # liboqs missing or failed to build
        return None


def _load_pqcrypto():
    try:
        import pqcrypto.sign                                # noqa: F401
        import pqcrypto.kem                                 # noqa: F401
        return True
    except Exception:
        return False


def backend_name():
    if _load_oqs() is not None:
        return "liboqs"
    if _load_pqcrypto():
        return "pqcrypto"
    return None


def sha3(*parts):
    h = hashlib.sha3_256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"|")
    return h.digest()


class Signer:
    """keygen / sign / verify for one signature scheme."""

    def __init__(self, scheme):
        self.scheme = scheme
        self.alg, self.mod_name, self.level = SCHEMES[scheme]
        self.post_quantum = scheme != "ecdsa"
        self._secret_of = {}                    # pk -> secret handle (only used by recover_secret)
        if scheme == "ecdsa":
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.exceptions import InvalidSignature
            self._ec, self._hashes, self._ser, self._bad = ec, hashes, serialization, InvalidSignature
            self.backend = "cryptography"
            self._pub_cache = {}
            self.sig_bytes = 72                 # DER-encoded ECDSA P-256 (at most)
            return
        oqs = _load_oqs()
        if oqs is not None:
            self.backend, self._oqs = "liboqs", oqs
            self._verifier = oqs.Signature(self.alg)
            self.sig_bytes = self._verifier.details["length_signature"]
        elif _load_pqcrypto():
            import importlib
            self.backend = "pqcrypto"
            self._m = importlib.import_module(f"pqcrypto.sign.{self.mod_name}")
            self.sig_bytes = self._m.SIGNATURE_SIZE
        else:
            raise RuntimeError(INSTALL_HINT)

    def keygen(self):
        if self.scheme == "ecdsa":
            sk = self._ec.generate_private_key(self._ec.SECP256R1())
            pk = sk.public_key().public_bytes(self._ser.Encoding.X962,
                                              self._ser.PublicFormat.UncompressedPoint)
        elif self.backend == "liboqs":
            sk = self._oqs.Signature(self.alg)
            pk = sk.generate_keypair()
        else:
            pk, sk = self._m.keygen()
        self._secret_of[pk] = sk
        return pk, sk

    def sign(self, sk, msg):
        if self.scheme == "ecdsa":
            return sk.sign(msg, self._ec.ECDSA(self._hashes.SHA256()))
        if self.backend == "liboqs":
            return sk.sign(msg)
        return self._m.sign(sk, msg)

    def verify(self, pk, msg, sig):
        try:
            if self.scheme == "ecdsa":
                key = self._pub_cache.get(pk)
                if key is None:
                    key = self._pub_cache[pk] = self._ec.EllipticCurvePublicKey.from_encoded_point(
                        self._ec.SECP256R1(), pk)
                key.verify(sig, msg, self._ec.ECDSA(self._hashes.SHA256()))
                return True
            if self.backend == "liboqs":
                return bool(self._verifier.verify(msg, sig, pk))
            self._m.verify(pk, msg, sig)
            return True
        except Exception:
            return False

    def recover_secret(self, pk):
        """Quantum adversary (Shor): recovers classical secret keys, not ML-DSA ones."""
        return None if self.post_quantum else self._secret_of.get(pk)


class KEM:
    """ML-KEM-768 key encapsulation + AES-256-GCM for vehicle-to-server sessions."""

    def __init__(self):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        self._aead = AESGCM
        oqs = _load_oqs()
        if oqs is not None:
            self.backend, self._oqs = "liboqs", oqs
            self.ct_bytes = oqs.KeyEncapsulation(KEM_ALG).details["length_ciphertext"]
        elif _load_pqcrypto():
            import importlib
            self.backend = "pqcrypto"
            self._m = importlib.import_module(f"pqcrypto.kem.{KEM_MOD}")
            self.ct_bytes = self._m.CIPHERTEXT_SIZE
        else:
            raise RuntimeError(INSTALL_HINT)

    def keygen(self):
        if self.backend == "liboqs":
            k = self._oqs.KeyEncapsulation(KEM_ALG)
            return k.generate_keypair(), k
        return self._m.keygen()

    def encaps(self, pk):
        if self.backend == "liboqs":
            ct, ss = self._oqs.KeyEncapsulation(KEM_ALG).encap_secret(pk)
            return ct, ss
        return self._m.encaps(pk)

    def decaps(self, sk, ct):
        if self.backend == "liboqs":
            return sk.decap_secret(ct)
        return self._m.decaps(sk, ct)

    def seal(self, key, plaintext, aad=b""):
        nonce = os.urandom(12)
        return nonce + self._aead(key[:32]).encrypt(nonce, plaintext, aad)

    def open(self, key, blob, aad=b""):
        return self._aead(key[:32]).decrypt(blob[:12], blob[12:], aad)


_SIGNERS, _COSTS = {}, {}


def signer(scheme):
    if scheme not in _SIGNERS:
        _SIGNERS[scheme] = Signer(scheme)
    return _SIGNERS[scheme]


def measure(scheme, reps=300):
    """Mean wall-clock seconds of sign / verify on this machine, plus sizes (cached)."""
    if scheme in _COSTS:
        return _COSTS[scheme]
    s = signer(scheme)
    pk, sk = s.keygen()
    msg = os.urandom(96)
    for _ in range(20):                                     # warm-up
        sig = s.sign(sk, msg); s.verify(pk, msg, sig)
    t = time.perf_counter()
    for _ in range(reps):
        sig = s.sign(sk, msg)
    t_sign = (time.perf_counter() - t) / reps
    t = time.perf_counter()
    for _ in range(reps):
        s.verify(pk, msg, sig)
    t_verify = (time.perf_counter() - t) / reps
    r3 = lambda v: float(f"{v:.3g}")                        # 3 significant digits
    _COSTS[scheme] = dict(sign_s=r3(t_sign), verify_s=r3(t_verify), sig_bytes=s.sig_bytes,
                          backend=s.backend, alg=s.alg, level=s.level)
    return _COSTS[scheme]

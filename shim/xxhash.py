"""Pure-Python stand-in for the `xxhash` package (local Windows workaround).

Why this exists
---------------
Windows **Smart App Control** blocks the unsigned native ``_xxhash.pyd`` that
ships inside the ``xxhash`` wheel::

    ImportError: DLL load failed while importing _xxhash:
    An Application Control policy has blocked this file.

LangChain imports xxhash transitively::

    langchain_core.tracers.schemas -> langsmith -> langsmith._internal._uuid

and langsmith uses exactly one function, ``xxh3_128``, to derive deterministic
replica UUIDs for LangSmith tracing. This project does not use LangSmith
tracing, so a deterministic BLAKE2b digest with the same return shape is a safe
substitute.

Digest *sizes* match the real algorithms (4 / 8 / 16 bytes) so code that slices
or hex-encodes results behaves identically. Digest *values* are NOT real XXH
hashes and must not be compared against another xxhash implementation.

This file is only needed on machines where the native extension is blocked.
Install it with ``python scripts/install_xxhash_shim.py``; revert with
``pip install --force-reinstall xxhash``.
"""

from __future__ import annotations

import hashlib
from typing import Final

__version__ = "4.1.0+shim"

XXH_VERSION_MAJOR: Final = 0
XXH_VERSION_MINOR: Final = 8
XXH_VERSION_RELEASE: Final = 3
VERSION: Final = XXH_VERSION_MAJOR * 10000 + XXH_VERSION_MINOR * 100 + XXH_VERSION_RELEASE


class _Hasher:
    """Mimics the object returned by the native ``xxh*`` constructors."""

    def __init__(self, data: bytes = b"", *, digest_size: int, seed: int = 0) -> None:
        self.digest_size = digest_size
        self._seed = seed
        self._buf = bytearray(data)

    # Native objects support incremental updates and copying.
    def update(self, data: bytes) -> "_Hasher":
        self._buf += data
        return self

    def copy(self) -> "_Hasher":
        clone = _Hasher(b"", digest_size=self.digest_size, seed=self._seed)
        clone._buf = bytearray(self._buf)
        return clone

    def _raw(self) -> bytes:
        return hashlib.blake2b(
            bytes(self._buf),
            digest_size=self.digest_size,
            person=str(self._seed).encode()[:16] or b"\x00",
        ).digest()

    def digest(self) -> bytes:
        return self._raw()

    def hexdigest(self) -> str:
        return self._raw().hex()

    def intdigest(self) -> int:
        return int.from_bytes(self._raw(), "big")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<xxhash shim digest_size={self.digest_size}>"


def _make(data: bytes, *, digest_size: int, seed: int = 0) -> _Hasher:
    return _Hasher(data, digest_size=digest_size, seed=seed)


def xxh32(data: bytes = b"", seed: int = 0) -> _Hasher:
    return _make(data, digest_size=4, seed=seed)


def xxh64(data: bytes = b"", seed: int = 0) -> _Hasher:
    return _make(data, digest_size=8, seed=seed)


def xxh3_64(data: bytes = b"", seed: int = 0) -> _Hasher:
    return _make(data, digest_size=8, seed=seed)


def xxh3_128(data: bytes = b"", seed: int = 0) -> _Hasher:
    return _make(data, digest_size=16, seed=seed)


xxh128 = xxh3_128


def xxh3_64_digest(data: bytes = b"", seed: int = 0) -> bytes:
    return xxh3_64(data, seed).digest()


def xxh3_128_digest(data: bytes = b"", seed: int = 0) -> bytes:
    return xxh3_128(data, seed).digest()

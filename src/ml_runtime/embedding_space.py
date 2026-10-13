"""Versioned embedding types and the guarded cosine boundary.

Numeric ``src.ml_runtime.embeddings.cosine`` remains for hash-era internals
and existing tests that pass anonymous float sequences. Every similarity
boundary that can mix representations must call :func:`compare_embeddings`.

Zero-vector policy (explicit): cosine against a zero vector is ``0.0``.
That matches the historical hash embedder (empty text -> zero vector ->
no-signal, never a false-identical match). NaN/Inf values raise.
Vectors are never silently truncated or padded at compare time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

# Canonical hash representation. Covers the complete transformation in
# embeddings.embed_text: blake2b feature hashing, token/bigram votes,
# sublinear TF, optional fitted IDF, L2, 512-d. IDF is process state and
# is part of the historical contract (unfitted -> uniform weights).
HASH_EMBEDDING_VERSION = (
    "blake2b-512-v1:feature-hash:tok2.0-bi0.5:sublinear-tf:idf-optional:l2:dim512"
)

OUTPUT_DIMENSION = 512
MINILM_NATIVE_DIMENSION = 384
MINILM_PAD_WIDTH = OUTPUT_DIMENSION - MINILM_NATIVE_DIMENSION  # 128
L2_TOLERANCE = 1e-5


class EmbeddingVersionMismatch(ValueError):
    """Raised when cosine is requested across distinct embedding versions."""


class EmbeddingDimensionError(ValueError):
    """Raised when a vector length does not match its declared dimension."""


class EmbeddingInvalidValuesError(ValueError):
    """Raised when a vector contains NaN, Inf, or is empty when values are required."""


class EmbeddingUnavailableError(RuntimeError):
    """Raised when a semantic embedder cannot load or cannot embed a record."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when an ONNX artifact fails manifest/checksum/shape validation."""


@dataclass(frozen=True)
class EmbeddedVector:
    """A vector plus the complete transformation identity that produced it."""

    values: tuple[float, ...]
    embedding_version: str
    native_dimension: int
    output_dimension: int
    model_key: str = ""
    padding_strategy: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(float(x) for x in self.values))
        if not self.embedding_version:
            raise EmbeddingInvalidValuesError("embedding_version is required")
        if self.output_dimension <= 0 or self.native_dimension <= 0:
            raise EmbeddingDimensionError("dimensions must be positive")
        if len(self.values) != self.output_dimension:
            raise EmbeddingDimensionError(
                f"values length {len(self.values)} != output_dimension {self.output_dimension}"
            )

    @property
    def is_zero(self) -> bool:
        return not any(self.values)

    def as_list(self) -> list[float]:
        return list(self.values)


@dataclass(frozen=True)
class ArtifactMetadata:
    """Inspectable identity of a loaded local artifact (no filesystem secrets)."""

    model_family: str
    model_revision: str
    tokenizer_revision: str
    model_sha256: str
    tokenizer_sha256: str
    quantization: str
    pooling: str
    normalization: str
    padding: str
    native_dimension: int
    output_dimension: int
    max_seq_length: int
    files_present: tuple[str, ...]


class Embedder(Protocol):
    @property
    def version(self) -> str: ...

    @property
    def native_dimension(self) -> int: ...

    @property
    def output_dimension(self) -> int: ...

    def embed_many(self, texts: Sequence[str]) -> list[EmbeddedVector]: ...

    def embed(self, text: str) -> EmbeddedVector: ...

    def ready(self) -> bool: ...

    def artifact_metadata(self) -> ArtifactMetadata | None: ...


def as_embedded(
    values: Sequence[float],
    embedding_version: str,
    *,
    native_dimension: int | None = None,
    output_dimension: int | None = None,
    model_key: str = "",
    padding_strategy: str = "",
) -> EmbeddedVector:
    vals = tuple(float(x) for x in values)
    out_dim = output_dimension if output_dimension is not None else len(vals)
    nat_dim = native_dimension if native_dimension is not None else out_dim
    return EmbeddedVector(
        values=vals,
        embedding_version=embedding_version,
        native_dimension=nat_dim,
        output_dimension=out_dim,
        model_key=model_key or "",
        padding_strategy=padding_strategy or "",
    )


def _finite(values: Sequence[float]) -> None:
    if not values:
        raise EmbeddingInvalidValuesError("empty vector")
    for x in values:
        if not math.isfinite(x):
            raise EmbeddingInvalidValuesError("vector contains NaN or Inf")


def l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in values))


def pad_native_to_output(
    native: Sequence[float],
    *,
    native_dimension: int,
    output_dimension: int,
) -> list[float]:
    """Zero-pad a native vector to the storage dimension.

    Padding is NOT a learned projection. Cosine of padded pairs equals
    cosine of the native pairs (zeros add no energy).
    """
    if len(native) != native_dimension:
        raise EmbeddingDimensionError(
            f"native length {len(native)} != {native_dimension}"
        )
    if output_dimension < native_dimension:
        raise EmbeddingDimensionError("output_dimension must be >= native_dimension")
    _finite(native)
    pad = output_dimension - native_dimension
    return [float(x) for x in native] + [0.0] * pad


def native_prefix(values: Sequence[float], native_dimension: int) -> list[float]:
    if len(values) < native_dimension:
        raise EmbeddingDimensionError(
            f"vector length {len(values)} < native_dimension {native_dimension}"
        )
    return [float(x) for x in values[:native_dimension]]


def compare_embeddings(a: EmbeddedVector, b: EmbeddedVector) -> float:
    """Guarded cosine similarity. Versions must match exactly.

    Policy:
    - mismatch of ``embedding_version`` -> EmbeddingVersionMismatch
    - length / declared-dimension mismatch -> EmbeddingDimensionError
    - NaN/Inf -> EmbeddingInvalidValuesError
    - no silent pad/truncate
    - either vector all zeros -> 0.0
    - otherwise clamp to [-1, 1]
    """
    if a.embedding_version != b.embedding_version:
        raise EmbeddingVersionMismatch(
            f"embedding version mismatch: {a.embedding_version!r} != {b.embedding_version!r}"
        )
    if (a.model_key or "") != (b.model_key or ""):
        raise EmbeddingVersionMismatch(
            f"embedding model_key mismatch: {a.model_key!r} != {b.model_key!r}"
        )
    if (a.padding_strategy or "") != (b.padding_strategy or ""):
        raise EmbeddingVersionMismatch(
            f"padding_strategy mismatch: {a.padding_strategy!r} != {b.padding_strategy!r}"
        )
    if a.output_dimension != b.output_dimension:
        raise EmbeddingDimensionError(
            f"output_dimension mismatch: {a.output_dimension} != {b.output_dimension}"
        )
    if len(a.values) != len(b.values):
        raise EmbeddingDimensionError(
            f"embedding dim mismatch: {len(a.values)} != {len(b.values)}"
        )
    _finite(a.values)
    _finite(b.values)
    na = l2_norm(a.values)
    nb = l2_norm(b.values)
    if na == 0.0 or nb == 0.0:
        return 0.0
    if a.values == b.values:
        return 1.0
    dot = float(sum(x * y for x, y in zip(a.values, b.values)))
    return max(-1.0, min(1.0, dot / (na * nb)))


__all__ = [
    "HASH_EMBEDDING_VERSION",
    "OUTPUT_DIMENSION",
    "MINILM_NATIVE_DIMENSION",
    "MINILM_PAD_WIDTH",
    "L2_TOLERANCE",
    "EmbeddedVector",
    "ArtifactMetadata",
    "Embedder",
    "EmbeddingVersionMismatch",
    "EmbeddingDimensionError",
    "EmbeddingInvalidValuesError",
    "EmbeddingUnavailableError",
    "ArtifactIntegrityError",
    "as_embedded",
    "compare_embeddings",
    "pad_native_to_output",
    "native_prefix",
    "l2_norm",
]

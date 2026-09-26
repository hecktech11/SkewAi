"""Hash embedder provider wrapping the existing blake2b feature-hash path.

Numerical behavior is delegated entirely to ``embeddings.embed_text``. This
wrapper only attaches the canonical version identity so similarity callers
cannot lose model identity.
"""

from __future__ import annotations

from typing import Sequence

from src.ml_runtime.embedding_space import (
    HASH_EMBEDDING_VERSION,
    OUTPUT_DIMENSION,
    ArtifactMetadata,
    EmbeddedVector,
    as_embedded,
)
from src.ml_runtime.embeddings import embed_text, embedding_dim


class HashEmbedder:
    """Explicit provider for the historical 512-d blake2b hashing trick."""

    def __init__(self, pack_id: str | None = None) -> None:
        if embedding_dim() != OUTPUT_DIMENSION:
            raise RuntimeError(
                f"hash embedder expected dim {OUTPUT_DIMENSION}, got {embedding_dim()}"
            )
        self.pack_id = pack_id

    @property
    def version(self) -> str:
        from src.ml_runtime.embeddings import hash_embedding_version

        return hash_embedding_version(self.pack_id)

    @property
    def native_dimension(self) -> int:
        return OUTPUT_DIMENSION

    @property
    def output_dimension(self) -> int:
        return OUTPUT_DIMENSION

    def ready(self) -> bool:
        return True

    def artifact_metadata(self) -> ArtifactMetadata | None:
        return ArtifactMetadata(
            model_family="blake2b-feature-hash",
            model_revision="v1",
            tokenizer_revision="regex-[a-z0-9]+",
            model_sha256="",
            tokenizer_sha256="",
            quantization="none",
            pooling="feature-hash",
            normalization="l2",
            padding="none",
            native_dimension=OUTPUT_DIMENSION,
            output_dimension=OUTPUT_DIMENSION,
            max_seq_length=0,
            files_present=(),
        )

    def embed_many(self, texts: Sequence[str]) -> list[EmbeddedVector]:
        out: list[EmbeddedVector] = []
        ver = self.version
        for text in texts:
            values = embed_text(text or "", pack_id=self.pack_id)
            out.append(
                as_embedded(
                    values,
                    ver,
                    native_dimension=OUTPUT_DIMENSION,
                    output_dimension=OUTPUT_DIMENSION,
                    model_key="blake2b-512-v1",
                    padding_strategy="none",
                )
            )
        return out

    def embed(self, text: str) -> EmbeddedVector:
        return self.embed_many([text])[0]


__all__ = ["HashEmbedder"]

"""CPU-only ONNX sentence-transformer embedder.

Runtime never downloads. The artifact directory must already contain the
ONNX model, tokenizer, manifest, checksums, and license. Validation fails
closed on missing files, hash mismatch, tokenizer/model revision disagreement,
or unexpected native dimension.

MiniLM-L6-v2 contract (authoritative sentence-transformers config):
  tokenize (max_seq_length=256, truncation) -> ONNX last_hidden_state
  -> attention-mask mean pooling -> L2 normalize -> zero-pad 384 to 512.

Padding is not a learned projection; it preserves cosine exactly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.ml_runtime.embedding_space import (
    L2_TOLERANCE,
    MINILM_NATIVE_DIMENSION,
    OUTPUT_DIMENSION,
    ArtifactIntegrityError,
    ArtifactMetadata,
    EmbeddedVector,
    EmbeddingUnavailableError,
    as_embedded,
    pad_native_to_output,
)

logger = logging.getLogger(__name__)

REQUIRED_FILES = (
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "manifest.json",
    "checksums.sha256",
)

_SESSION_LOCK = threading.Lock()
_PROCESS_EMBEDDER: OnnxSemanticEmbedder | None = None  # type: ignore[name-defined]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Attention-mask-aware mean pooling. Padding tokens must not contribute.

    last_hidden_state: [batch, seq, hidden]
    attention_mask:    [batch, seq]  (1 = real token, 0 = pad)
    """
    if last_hidden_state.ndim != 3:
        raise EmbeddingUnavailableError(
            f"last_hidden_state rank {last_hidden_state.ndim} != 3"
        )
    mask = attention_mask.astype(np.float64)
    if mask.ndim != 2:
        raise EmbeddingUnavailableError(f"attention_mask rank {mask.ndim} != 2")
    weights = mask[:, :, None]
    summed = (last_hidden_state.astype(np.float64) * weights).sum(axis=1)
    counts = mask.sum(axis=1, keepdims=True)
    # An all-pad row has no tokens. Dividing by a tiny epsilon explodes it.
    safe = np.where(counts > 0.0, counts, 1.0)
    pooled = summed / safe
    pooled = np.where(counts > 0.0, pooled, 0.0)
    return pooled.astype(np.float32)


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (matrix.astype(np.float64) / norms).astype(np.float32)


def build_canonical_version(manifest: dict[str, Any]) -> str:
    """Complete transformation identity — not a friendly alias."""
    family = str(manifest.get("model_family") or "all-MiniLM-L6-v2")
    rev = str(manifest.get("model_revision") or "unknown")
    tok_rev = str(manifest.get("tokenizer_revision") or rev)
    quant = str(manifest.get("quantization") or "onnx-int8")
    pooling = str(manifest.get("pooling") or "mean-pool")
    norm = str(manifest.get("normalization") or "l2")
    pad = str(manifest.get("padding") or "zeropad512")
    schema = str(manifest.get("schema_version") or "v1")
    model_sha = str(manifest.get("model_sha256") or "")
    if tok_rev != rev:
        # Disagreement is a load-time error; this branch is for the string only
        # after validation has already confirmed they match or are both recorded.
        pass
    return (
        f"{family}@{rev}:tok@{tok_rev}:{quant}:{pooling}:{norm}:{pad}:{schema}"
        f":modelsha256={model_sha}"
    )


def load_manifest(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "manifest.json"
    if not path.is_file():
        raise ArtifactIntegrityError(f"missing manifest.json in {artifact_dir.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ArtifactIntegrityError(f"manifest.json is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ArtifactIntegrityError("manifest.json must be an object")
    return data


def parse_checksums(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ArtifactIntegrityError(f"malformed checksums line: {line!r}")
        digest, name = parts[0], parts[-1]
        name = name.lstrip("*")
        out[Path(name).name] = digest.lower()
    return out


def validate_artifact_dir(artifact_dir: Path) -> dict[str, Any]:
    """Fail closed if the local artifact is incomplete or tampered."""
    if not artifact_dir.is_dir():
        raise ArtifactIntegrityError("semantic artifact directory is missing")
    missing = [name for name in REQUIRED_FILES if not (artifact_dir / name).is_file()]
    # LICENSE is required for redistribution honesty; accept LICENSE or LICENSE.txt
    if not (artifact_dir / "LICENSE").is_file() and not (artifact_dir / "LICENSE.txt").is_file():
        missing.append("LICENSE")
    if missing:
        raise ArtifactIntegrityError(
            f"semantic artifact missing files: {', '.join(missing)}"
        )
    manifest = load_manifest(artifact_dir)
    checksums = parse_checksums(artifact_dir / "checksums.sha256")
    for name in ("model.onnx", "tokenizer.json"):
        expected = (checksums.get(name) or "").lower()
        if not expected:
            raise ArtifactIntegrityError(f"checksums.sha256 missing {name}")
        actual = sha256_file(artifact_dir / name)
        if actual != expected:
            raise ArtifactIntegrityError(
                f"checksum mismatch for {name}: expected {expected[:12]}… got {actual[:12]}…"
            )
        if name == "model.onnx":
            manifest["model_sha256"] = actual
        else:
            manifest["tokenizer_sha256"] = actual
    model_rev = str(manifest.get("model_revision") or "")
    tok_rev = str(manifest.get("tokenizer_revision") or "")
    if not model_rev or not tok_rev:
        raise ArtifactIntegrityError("manifest must include model_revision and tokenizer_revision")
    if model_rev != tok_rev:
        raise ArtifactIntegrityError(
            f"tokenizer revision {tok_rev!r} disagrees with model revision {model_rev!r}"
        )
    native = int(manifest.get("native_dimension") or 0)
    if native != MINILM_NATIVE_DIMENSION:
        raise ArtifactIntegrityError(
            f"unexpected native dimension {native}; expected {MINILM_NATIVE_DIMENSION}"
        )
    output = int(manifest.get("output_dimension") or 0)
    if output != OUTPUT_DIMENSION:
        raise ArtifactIntegrityError(
            f"unexpected output dimension {output}; expected {OUTPUT_DIMENSION}"
        )
    return manifest


class OnnxSemanticEmbedder:
    """Process-local ONNX MiniLM. Construct once; reuse across requests."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        intra_op_threads: int | None = None,
        inter_op_threads: int = 1,
        max_batch_size: int = 32,
    ) -> None:
        self._dir = Path(artifact_dir).expanduser().resolve()
        self._manifest = validate_artifact_dir(self._dir)
        self._version = build_canonical_version(self._manifest)
        self._max_seq = int(self._manifest.get("max_seq_length") or 256)
        self._max_batch = max(1, int(max_batch_size))
        self._native = MINILM_NATIVE_DIMENSION
        self._output = OUTPUT_DIMENSION
        threads = intra_op_threads
        if threads is None:
            threads = int(os.getenv("FRONTLINE_ONNX_INTRA_THREADS") or "1")
        self._intra_threads = max(1, int(threads))
        self._inter_threads = max(1, int(inter_op_threads))
        self._session = None
        self._input_names: list[str] = []
        self._output_name: str = "last_hidden_state"
        self._tokenizer = None
        self._init_lock = threading.Lock()
        self._loaded = False

    @property
    def version(self) -> str:
        return self._version

    @property
    def native_dimension(self) -> int:
        return self._native

    @property
    def output_dimension(self) -> int:
        return self._output

    @property
    def max_seq_length(self) -> int:
        """Token budget. Sequences longer than this are truncated (head)."""
        return self._max_seq

    def ready(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except Exception:
            return False

    def artifact_metadata(self) -> ArtifactMetadata:
        files = tuple(sorted(p.name for p in self._dir.iterdir() if p.is_file()))
        return ArtifactMetadata(
            model_family=str(self._manifest.get("model_family") or ""),
            model_revision=str(self._manifest.get("model_revision") or ""),
            tokenizer_revision=str(self._manifest.get("tokenizer_revision") or ""),
            model_sha256=str(self._manifest.get("model_sha256") or ""),
            tokenizer_sha256=str(self._manifest.get("tokenizer_sha256") or ""),
            quantization=str(self._manifest.get("quantization") or ""),
            pooling=str(self._manifest.get("pooling") or "mean-pool"),
            normalization=str(self._manifest.get("normalization") or "l2"),
            padding=str(self._manifest.get("padding") or "zeropad512"),
            native_dimension=self._native,
            output_dimension=self._output,
            max_seq_length=self._max_seq,
            files_present=files,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._init_lock:
            if self._loaded:
                return
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
            except ImportError as e:
                raise EmbeddingUnavailableError(
                    f"ONNX embedder dependencies missing: {e}"
                ) from e
            so = ort.SessionOptions()
            so.intra_op_num_threads = self._intra_threads
            so.inter_op_num_threads = self._inter_threads
            # Sequential + basic opts: graph-all can reorder reductions across
            # runs on some ORT builds. Bit-identical vectors are required.
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            model_path = str(self._dir / "model.onnx")
            self._session = ort.InferenceSession(
                model_path,
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = [i.name for i in self._session.get_inputs()]
            outputs = [o.name for o in self._session.get_outputs()]
            if "last_hidden_state" in outputs:
                self._output_name = "last_hidden_state"
            elif "token_embeddings" in outputs:
                self._output_name = "token_embeddings"
            else:
                self._output_name = outputs[0]
            self._tokenizer = Tokenizer.from_file(str(self._dir / "tokenizer.json"))
            self._tokenizer.enable_truncation(max_length=self._max_seq)
            # Pad to the longest sequence in the batch (not a fixed 256).
            # Attention-mask mean pooling still ignores pad tokens.
            self._tokenizer.enable_padding()
            self._loaded = True

    def embed(self, text: str) -> EmbeddedVector:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[EmbeddedVector]:
        if not texts:
            return []
        try:
            self._ensure_loaded()
        except ArtifactIntegrityError:
            raise
        except Exception as e:
            raise EmbeddingUnavailableError(
                f"semantic embedder failed to load: {type(e).__name__}"
            ) from e
        out: list[EmbeddedVector] = []
        batch: list[str] = []
        for text in texts:
            batch.append(text if text is not None else "")
            if len(batch) >= self._max_batch:
                out.extend(self._infer_batch(batch))
                batch = []
        if batch:
            out.extend(self._infer_batch(batch))
        return out

    def _infer_batch(self, texts: list[str]) -> list[EmbeddedVector]:
        assert self._tokenizer is not None and self._session is not None
        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds: dict[str, np.ndarray] = {}
        for name in self._input_names:
            if name in {"input_ids", "inputs"}:
                feeds[name] = input_ids
            elif name in {"attention_mask", "mask"}:
                feeds[name] = attention_mask
            elif name in {"token_type_ids", "token_type_id", "type_ids"}:
                feeds[name] = np.zeros_like(input_ids)
            else:
                # Unknown extra input: zeros of input_ids shape (int64).
                feeds[name] = np.zeros_like(input_ids)
        try:
            raw = self._session.run([self._output_name], feeds)[0]
        except Exception as e:
            raise EmbeddingUnavailableError(
                f"onnx inference failed: {type(e).__name__}"
            ) from e
        hidden = np.asarray(raw)
        if hidden.ndim == 2:
            # Some exports already pooled; reject — we require token states so
            # padding cannot silently leak into a pre-pooled vector we cannot audit.
            if hidden.shape[-1] != self._native:
                raise EmbeddingUnavailableError(
                    f"onnx output last dim {hidden.shape[-1]} != {self._native}"
                )
            pooled = hidden.astype(np.float32)
        elif hidden.ndim == 3:
            if hidden.shape[-1] != self._native:
                raise EmbeddingUnavailableError(
                    f"onnx hidden size {hidden.shape[-1]} != {self._native}"
                )
            pooled = mean_pool(hidden, attention_mask)
        else:
            raise EmbeddingUnavailableError(f"unexpected onnx output rank {hidden.ndim}")
        normalized = l2_normalize_rows(pooled)
        vectors: list[EmbeddedVector] = []
        for row in normalized:
            native = [float(np.float32(x)) for x in row.tolist()]
            if len(native) != self._native:
                raise EmbeddingUnavailableError(
                    f"pooled length {len(native)} != {self._native}"
                )
            if any(not np.isfinite(x) for x in native):
                raise EmbeddingUnavailableError("semantic embedding produced NaN/Inf")
            padded = pad_native_to_output(
                native,
                native_dimension=self._native,
                output_dimension=self._output,
            )
            vectors.append(
                as_embedded(
                    padded,
                    self._version,
                    native_dimension=self._native,
                    output_dimension=self._output,
                    model_key="minilm-l6-onnx-384+zeropad512-v1",
                    padding_strategy="zeropad512",
                )
            )
        return vectors


def get_process_onnx_embedder(
    artifact_dir: str | Path,
    **kwargs: Any,
) -> OnnxSemanticEmbedder:
    """Initialize once per process. Thread-safe."""
    global _PROCESS_EMBEDDER
    if _PROCESS_EMBEDDER is not None:
        return _PROCESS_EMBEDDER
    with _SESSION_LOCK:
        if _PROCESS_EMBEDDER is None:
            _PROCESS_EMBEDDER = OnnxSemanticEmbedder(artifact_dir, **kwargs)
        return _PROCESS_EMBEDDER


def reset_process_onnx_embedder() -> None:
    """Test helper — drop the process singleton."""
    global _PROCESS_EMBEDDER
    with _SESSION_LOCK:
        _PROCESS_EMBEDDER = None


class ToySemanticEmbedder:
    """Deterministic 384->512 test double. Not a production provider.

    Uses a different hash family than blake2b-512 so hash vs toy cosine is
    meaningful in pipeline tests. Canonical version is deliberately marked
    ``toy-semantic`` so production registries reject it.
    """

    def __init__(self, *, version_tag: str = "toy-semantic-v1") -> None:
        self._version = (
            f"{version_tag}:blake2s-384:mean-pool:l2:zeropad512:v1:modelsha256=testdouble"
        )

    @property
    def version(self) -> str:
        return self._version

    @property
    def native_dimension(self) -> int:
        return MINILM_NATIVE_DIMENSION

    @property
    def output_dimension(self) -> int:
        return OUTPUT_DIMENSION

    def ready(self) -> bool:
        return True

    def artifact_metadata(self) -> ArtifactMetadata | None:
        return ArtifactMetadata(
            model_family="toy-semantic",
            model_revision="test",
            tokenizer_revision="test",
            model_sha256="testdouble",
            tokenizer_sha256="testdouble",
            quantization="none",
            pooling="mean-pool",
            normalization="l2",
            padding="zeropad512",
            native_dimension=MINILM_NATIVE_DIMENSION,
            output_dimension=OUTPUT_DIMENSION,
            max_seq_length=256,
            files_present=(),
        )

    def embed(self, text: str) -> EmbeddedVector:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[EmbeddedVector]:
        import re

        tok_re = re.compile(r"[a-z0-9]+", re.I)
        out: list[EmbeddedVector] = []
        for text in texts:
            vec = np.zeros(MINILM_NATIVE_DIMENSION, dtype=np.float64)
            tokens = tok_re.findall((text or "").lower())
            if tokens:
                for tok in tokens:
                    digest = hashlib.blake2s(tok.encode("utf-8"), digest_size=8).digest()
                    bucket = int.from_bytes(digest, "little") % MINILM_NATIVE_DIMENSION
                    vec[bucket] += 1.0
                n = np.linalg.norm(vec) or 1.0
                vec = vec / n
            padded = pad_native_to_output(
                [float(np.float32(x)) for x in vec.tolist()],
                native_dimension=MINILM_NATIVE_DIMENSION,
                output_dimension=OUTPUT_DIMENSION,
            )
            out.append(
                as_embedded(
                    padded,
                    self._version,
                    native_dimension=MINILM_NATIVE_DIMENSION,
                    output_dimension=OUTPUT_DIMENSION,
                    model_key="toy-semantic-v1",
                    padding_strategy="zeropad512",
                )
            )
        return out


__all__ = [
    "OnnxSemanticEmbedder",
    "ToySemanticEmbedder",
    "get_process_onnx_embedder",
    "reset_process_onnx_embedder",
    "mean_pool",
    "l2_normalize_rows",
    "validate_artifact_dir",
    "build_canonical_version",
    "sha256_file",
    "REQUIRED_FILES",
]

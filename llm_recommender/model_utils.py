"""Utility helpers for the LLM driven recommender system.

This module keeps all heavyweight imports optional and provides
utility helpers to build and load the machine learning artefacts used
by the recommendation engine (ALS collaborative filtering model,
semantic item embeddings, FAISS vector index and metadata loaders).

The functions are written so that unit tests can exercise the logic
without requiring the large, optional dependencies to be installed.
Imports are resolved lazily and informative errors are raised when a
dependency is missing.  Each helper also validates the existence of the
expected input data before attempting to train or load a model.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# Project wide default folders
DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")


class DependencyError(RuntimeError):
    """Raised when an optional dependency is not installed."""


class MissingDataError(RuntimeError):
    """Raised when an expected data file cannot be located."""


class MissingArtifactError(RuntimeError):
    """Raised when a persisted model artefact is missing."""


def _resolve_path(path: Path | str) -> Path:
    """Return a normalised :class:`Path` instance."""

    return Path(path).expanduser().resolve()


def ensure_directory(path: Path | str) -> Path:
    """Create a directory (and parents) if it does not already exist."""

    directory = _resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _import_module(name: str, install_hint: Optional[str] = None) -> Any:
    """Import *name* lazily and raise a helpful :class:`DependencyError`.

    Parameters
    ----------
    name:
        Fully qualified module name to import.
    install_hint:
        Optional package name or command to display to the user if the
        import fails.
    """

    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - exercised indirectly
        hint = install_hint or name
        raise DependencyError(
            f"The optional dependency '{name}' is required. Install it via 'pip install {hint}'."
        ) from exc


def _import_attr(module_name: str, attribute: str, install_hint: Optional[str] = None) -> Any:
    """Import *attribute* from *module_name* lazily."""

    module = _import_module(module_name, install_hint=install_hint)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise DependencyError(
            f"Module '{module_name}' does not provide attribute '{attribute}'."
        ) from exc


def _check_exists(path: Path, error_cls: type[Exception]) -> Path:
    """Ensure that *path* exists and return it."""

    if not path.exists():
        raise error_cls(f"Expected file not found: {path}")
    return path


def load_metadata(items_path: Path | str = DATA_DIR / "items.parquet"):
    """Load item metadata as a pandas ``DataFrame`` indexed by ``track_id``."""

    pandas = _import_module("pandas", install_hint="pandas")
    path = _check_exists(_resolve_path(items_path), MissingDataError)
    df = pandas.read_parquet(path)
    if "track_id" not in df.columns:
        raise ValueError("Metadata must include a 'track_id' column.")
    return df.set_index("track_id")


def build_cf_model(
    interactions_path: Path | str = DATA_DIR / "interactions.parquet",
    artifact_path: Path | str = ARTIFACT_DIR / "als.joblib",
    *,
    factors: int = 64,
    regularization: float = 0.08,
    iterations: int = 20,
    random_state: Optional[int] = 42,
) -> Path:
    """Train an implicit ALS model and persist the resulting artefact."""

    pandas = _import_module("pandas", install_hint="pandas")
    scipy_sparse = _import_module("scipy.sparse", install_hint="scipy")
    AlternatingLeastSquares = _import_attr(
        "implicit.als", "AlternatingLeastSquares", install_hint="implicit"
    )
    joblib = _import_module("joblib", install_hint="joblib")

    path = _check_exists(_resolve_path(interactions_path), MissingDataError)
    df = pandas.read_parquet(path)
    if df.empty:
        raise ValueError("Interaction data is empty; cannot train ALS model.")

    for column in ("user_id", "track_id", "strength"):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' missing from interactions data.")

    # Create contiguous id mappings
    u_codes = {uid: idx for idx, uid in enumerate(df["user_id"].unique())}
    i_codes = {tid: idx for idx, tid in enumerate(df["track_id"].unique())}
    df = df.assign(u=df["user_id"].map(u_codes), i=df["track_id"].map(i_codes))

    matrix = scipy_sparse.coo_matrix(
        (df["strength"], (df["i"], df["u"])),
        shape=(len(i_codes), len(u_codes)),
        dtype="float32",
    ).tocsr()

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
    )
    model.fit(matrix)

    ensure_directory(artifact_path if isinstance(artifact_path, Path) else ARTIFACT_DIR)
    artifact = _resolve_path(artifact_path)
    joblib.dump(
        {
            "item_factors": model.item_factors,
            "user_factors": model.user_factors,
            "u_codes": u_codes,
            "i_codes": i_codes,
        },
        artifact,
    )
    LOGGER.info("ALS model trained and saved to %s", artifact)
    return artifact


def load_cf_artifact(artifact_path: Path | str = ARTIFACT_DIR / "als.joblib") -> Dict[str, Any]:
    """Load the persisted ALS factors."""

    joblib = _import_module("joblib", install_hint="joblib")
    path = _check_exists(_resolve_path(artifact_path), MissingArtifactError)
    return joblib.load(path)


def build_item_embeddings(
    items_path: Path | str = DATA_DIR / "items.parquet",
    text_path: Path | str = DATA_DIR / "item_text.parquet",
    artifact_path: Path | str = ARTIFACT_DIR / "item_embs.joblib",
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_chars: int = 1_000,
) -> Path:
    """Create text embeddings for items and persist them."""

    pandas = _import_module("pandas", install_hint="pandas")
    numpy = _import_module("numpy", install_hint="numpy")
    SentenceTransformer = _import_attr(
        "sentence_transformers", "SentenceTransformer", install_hint="sentence-transformers"
    )
    joblib = _import_module("joblib", install_hint="joblib")

    items_df = pandas.read_parquet(_check_exists(_resolve_path(items_path), MissingDataError))
    texts_df = pandas.read_parquet(_check_exists(_resolve_path(text_path), MissingDataError))

    merged = items_df.merge(texts_df, on="track_id", how="left")

    def _safe_col(frame, column: str) -> Any:
        return frame[column].fillna("") if column in frame.columns else ""

    merged["blob"] = (
        _safe_col(merged, "title").astype(str)
        + " — "
        + _safe_col(merged, "artist").astype(str)
        + " | "
        + _safe_col(merged, "genres").astype(str)
        + " | "
        + _safe_col(merged, "tags").astype(str)
        + " | "
        + _safe_col(merged, "description").astype(str)
    ).str.slice(0, max_chars)

    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(merged["blob"].tolist(), normalize_embeddings=True)
    embeddings = numpy.asarray(embeddings, dtype="float32")

    ensure_directory(artifact_path if isinstance(artifact_path, Path) else ARTIFACT_DIR)
    artifact = _resolve_path(artifact_path)
    joblib.dump({"track_ids": merged["track_id"].tolist(), "embs": embeddings}, artifact)
    LOGGER.info("Item embeddings saved to %s", artifact)
    return artifact


def load_item_embeddings(artifact_path: Path | str = ARTIFACT_DIR / "item_embs.joblib") -> Dict[str, Any]:
    """Load the saved item embeddings."""

    joblib = _import_module("joblib", install_hint="joblib")
    path = _check_exists(_resolve_path(artifact_path), MissingArtifactError)
    return joblib.load(path)


def load_sentence_transformer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Instantiate a sentence-transformer model lazily."""

    SentenceTransformer = _import_attr(
        "sentence_transformers", "SentenceTransformer", install_hint="sentence-transformers"
    )
    return SentenceTransformer(model_name)


def build_faiss_index(
    embeddings_artifact: Path | str = ARTIFACT_DIR / "item_embs.joblib",
    index_path: Path | str = ARTIFACT_DIR / "item_faiss.index",
) -> Path:
    """Create a FAISS index from pre-computed embeddings."""

    faiss = _import_module("faiss", install_hint="faiss-cpu")
    numpy = _import_module("numpy", install_hint="numpy")
    joblib_art = load_item_embeddings(embeddings_artifact)

    embeddings = numpy.asarray(joblib_art["embs"], dtype="float32")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    ensure_directory(index_path if isinstance(index_path, Path) else ARTIFACT_DIR)
    path = _resolve_path(index_path)
    faiss.write_index(index, path)
    LOGGER.info("FAISS index persisted to %s", path)
    return path


def load_faiss_index(index_path: Path | str = ARTIFACT_DIR / "item_faiss.index"):
    """Load the FAISS index for similarity search."""

    faiss = _import_module("faiss", install_hint="faiss-cpu")
    path = _check_exists(_resolve_path(index_path), MissingArtifactError)
    return faiss.read_index(str(path))


def describe_available_artifacts(artifact_dir: Path | str = ARTIFACT_DIR) -> Dict[str, bool]:
    """Return a quick overview of which artefacts are present on disk."""

    directory = _resolve_path(artifact_dir)
    return {
        "als": (directory / "als.joblib").exists(),
        "item_embeddings": (directory / "item_embs.joblib").exists(),
        "faiss_index": (directory / "item_faiss.index").exists(),
    }


__all__ = [
    "ARTIFACT_DIR",
    "DATA_DIR",
    "DependencyError",
    "MissingDataError",
    "MissingArtifactError",
    "ensure_directory",
    "load_metadata",
    "build_cf_model",
    "load_cf_artifact",
    "build_item_embeddings",
    "load_item_embeddings",
    "load_sentence_transformer",
    "build_faiss_index",
    "load_faiss_index",
    "describe_available_artifacts",
]


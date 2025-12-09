"""
model_utils.py
Utility helpers for the LLM-driven recommender system.

This module handles:
  • Loading metadata from items.parquet
  • Training a PURE-PYTHON ALS model (no implicit dependency)
  • Generating item embeddings using Azure OpenAI ONLY
  • Building / loading a FAISS similarity index
  • Lazy / safe dependency importing with clean errors

Everything is designed to be stable in constrained environments
(Windows, limited build tools, no GPU, etc.).
"""

from __future__ import annotations
from datetime import datetime
import importlib
import logging
from pathlib import Path
from typing import Optional

from openai import AzureOpenAI
from config.settings import settings
from .pure_als import PureALS as AlternatingLeastSquares

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

LOGGER = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────

class DependencyError(RuntimeError):
    """Raised when a required dependency or configuration is missing."""
    pass

class MissingDataError(RuntimeError):
    """Raised when items.parquet / item_text.parquet / interactions files are missing."""
    pass

class MissingArtifactError(RuntimeError):
    """Raised when ALS, FAISS, or embedding artifacts are missing."""
    pass


# ─────────────────────────────────────────────
# SPOTIFY CLIENT
# ─────────────────────────────────────────────

_spotify = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=settings.SPOTIFY_CLIENT_ID,
        client_secret=settings.SPOTIFY_CLIENT_SECRET,
    )
)

# ─────────────────────────────────────────────
# METADATA SCHEMA (NEW + REQUIRED)
# ─────────────────────────────────────────────

METADATA_COLUMNS = [
    "track_id",
    "title",
    "artist",
    "album",
    "release_date",
    "popularity",
    "image_url",
    "source_type",
    "release_year",
]

DEFAULT_DF = pd.DataFrame(columns=METADATA_COLUMNS)

def enforce_metadata_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures metadata:
        - has all expected columns
        - contains only expected columns
        - track_id is string
        - release_year is nullable Int64
    """

    df = df.copy()

    # add missing columns
    for col in METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # drop unexpected columns
    df = df[METADATA_COLUMNS]

    # fix dtypes permanently
    df["track_id"] = df["track_id"].astype(str)
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")

    return df

# ─────────────────────────────────────────────
# METADATA FETCH / APPEND / REBUILD
# ─────────────────────────────────────────────

def fetch_track_metadata(track_id: str) -> dict:
    """Fetch metadata for one track from Spotify."""
    try:
        t = _spotify.track(track_id)
        return {
            "track_id": track_id,
            "title": t["name"],
            "artist": ", ".join(a["name"] for a in t["artists"]),
            "album": t["album"]["name"],
            "release_date": t["album"]["release_date"],
            "popularity": t["popularity"],
            "image_url": t["album"]["images"][0]["url"] if t["album"]["images"] else None,
            "source_type": "spotify_api",
            "release_year": (
                int(t["album"]["release_date"][:4])
                if t["album"]["release_date"] else None
            ),
        }

    except Exception as exc:
        LOGGER.warning(f"[WARN] Failed to fetch metadata for {track_id}: {exc}")
        return None

def load_metadata(path: Path | str = DATA_DIR / "items.parquet"):
    path = Path(path)

    if not path.exists():
        LOGGER.warning(f"Metadata not found → returning empty table: {path}")
        return DEFAULT_DF.copy()

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(
            f" items.parquet is corrupted: {exc}\n"
            f"Fix with: rebuild_items_parquet()"
        )

    # Remove leftover index column
    if "index" in df.columns:
        df = df.drop(columns=["index"])

    df = enforce_metadata_schema(df)

    # 🔥 FIX: normalize NA → None so JSON can serialize
    df = df.astype(object).where(pd.notna(df), None)

    # Set track_id as index
    if "track_id" in df.columns:
        df = df.set_index("track_id")

    return df

def append_metadata_row(parquet_path: str | Path, row: dict):
    if row is None:
        return

    path = Path(parquet_path)
    df = load_metadata(path)

    # Avoid duplicates
    if (df["track_id"] == row["track_id"]).any():
        LOGGER.info(f"Skipping duplicate track metadata: {row['track_id']}")
        return

    new = pd.DataFrame([row])
    new = enforce_metadata_schema(new)

    out = pd.concat([df, new], ignore_index=True)

    # Write WITHOUT index → prevents INT64 errors
    out.to_parquet(path, index=False)

    LOGGER.info(f"Appended metadata for: {row['track_id']}")

def rebuild_items_parquet(raw_path: Path | str, out_path: Path | str):
    """
    Hard rebuild metadata from raw source.
    Removes corrupted index & re-applies schema.
    """

    raw_path = Path(raw_path)
    out_path = Path(out_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Cannot rebuild; raw data missing: {raw_path}")

    if raw_path.suffix == ".parquet":
        df = pd.read_parquet(raw_path)
    elif raw_path.suffix == ".jsonl":
        df = pd.read_json(raw_path, lines=True)
    else:
        raise ValueError("Raw metadata must be parquet or jsonl")

    df = enforce_metadata_schema(df)
    out_path.parent.mkdir(exist_ok=True)
    df.to_parquet(out_path, index=False)

    LOGGER.info(f"items.parquet rebuilt → {out_path}")

# ─────────────────────────────────────────────
# ALS + Embeddings + FAISS (UNCHANGED)
# ─────────────────────────────────────────────

def _import_module(name: str, install_hint: Optional[str] = None):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        msg = install_hint or name
        raise RuntimeError(f"Missing dependency: pip install {msg}") from exc


def _resolve_path(path):
    return Path(path).expanduser().resolve()


def ensure_directory(path):
    p = _resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_cf_model(
    interactions_path: Path | str = DATA_DIR / "interactions.parquet",
    artifact_path: Path | str = ARTIFACT_DIR / "als.joblib",
    *,
    factors=64,
    regularization=0.08,
    iterations=20,
    random_state=42,
):
    import numpy as np
    import scipy.sparse as sp
    import joblib

    df = pd.read_parquet(_resolve_path(interactions_path))

    if df.empty:
        raise RuntimeError("Interactions are empty.")

    # map ids
    u_codes = {u: i for i, u in enumerate(df["user_id"].unique())}
    i_codes = {t: i for i, t in enumerate(df["track_id"].unique())}

    df["u"] = df["user_id"].map(u_codes)
    df["i"] = df["track_id"].map(i_codes)

    mat = sp.coo_matrix(
        (df["strength"], (df["i"], df["u"])),
        shape=(len(i_codes), len(u_codes)),
    ).tocsr()

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
    )
    model.fit(mat)

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

    LOGGER.info("ALS model saved → %s", artifact)
    return artifact


def load_cf_artifact(path=ARTIFACT_DIR / "als.joblib"):
    import joblib
    return joblib.load(_resolve_path(path))


# ─────────────────────────────────────────────
# AZURE OPENAI EMBEDDING LOGIC (unchanged)
# ─────────────────────────────────────────────

def _get_azure_embedding_client():
    endpoint = settings.AZURE_OPENAI_ENDPOINT
    key = settings.AZURE_OPENAI_API_KEY
    api_version = settings.AZURE_OPENAI_API_VERSION

    # IMPORTANT — use your existing settings variable
    deployment = getattr(
        settings,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        None
    )

    if not (endpoint and key and deployment):
        raise DependencyError(
            "Azure embedding configuration missing. "
            "Expected AZURE_OPENAI_EMBEDDING_DEPLOYMENT in settings."
        )

    client = AzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
    return client, deployment


def _azure_embed_texts(texts):
    import numpy as np
    client, model = _get_azure_embedding_client()

    if not texts:
        return np.zeros((0, 0), dtype="float32")

    vectors = []
    batch = 256
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        resp = client.embeddings.create(model=model, input=chunk)
        vectors.extend([v.embedding for v in resp.data])

    return np.asarray(vectors, dtype="float32")


def build_item_embeddings(
    items_path=DATA_DIR / "items.parquet",
    text_path=DATA_DIR / "item_text.parquet",
    artifact_path=ARTIFACT_DIR / "item_embs.joblib",
):
    import joblib

    # FIX: ensure metadata has track_id as column
    items = load_metadata(items_path).reset_index()

    # Load text metadata
    text = pd.read_parquet(text_path)
    
    # Always keep title from items (not text)
    text = text.drop(columns=["title", "artist"], errors="ignore")

    # Merge
    merged = items.merge(text, on="track_id", how="left")

    # Build embedding blob
    merged["blob"] = (
        merged["title"].fillna("") + " — " +
        merged["artist"].fillna("") + " | " +
        merged.get("genres", "").astype(str) + " | " +
        merged.get("tags", "").astype(str) + " | " +
        merged.get("description", "").astype(str)
    )

    # Generate Azure embeddings
    vectors = _azure_embed_texts(merged["blob"].tolist())

    # Save artifact
    joblib.dump(
        {"track_ids": merged["track_id"].tolist(), "embs": vectors},
        artifact_path,
    )

    LOGGER.info("Item embeddings saved → %s", artifact_path)
    return artifact_path



def load_item_embeddings(path=ARTIFACT_DIR / "item_embs.joblib"):
    import joblib
    return joblib.load(path)


# ─────────────────────────────────────────────
# FAISS INDEX (unchanged)
# ─────────────────────────────────────────────

def build_faiss_index(
    emb_path=ARTIFACT_DIR / "item_embs.joblib",
    index_path=ARTIFACT_DIR / "item_faiss.index",
):
    import faiss
    import numpy as np
    import joblib

    data = joblib.load(emb_path)
    embs = np.asarray(data["embs"], dtype="float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, str(index_path))
    LOGGER.info("FAISS index saved → %s", index_path)
    return index_path


def load_faiss_index(path=ARTIFACT_DIR / "item_faiss.index"):
    import faiss
    return faiss.read_index(str(path))


__all__ = [
    "load_metadata",
    "append_metadata_row",
    "rebuild_items_parquet",
    "build_cf_model",
    "load_cf_artifact",
    "build_item_embeddings",
    "load_item_embeddings",
    "build_faiss_index",
    "load_faiss_index",
    "fetch_track_metadata",
    "DATA_DIR",
    "ARTIFACT_DIR",
]

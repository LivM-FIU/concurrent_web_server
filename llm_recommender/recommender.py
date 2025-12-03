"""
recommender.py
Recommendation engine that wires together the hybrid retrieval stack.

The implementation mirrors the design sketched in the project plan:

* Collaborative filtering via an implicit ALS model trained on historic
  interactions.
* Semantic retrieval via sentence-transformer embeddings and a FAISS
  index for approximate nearest neighbours.
* An LLM-powered intent parser (Azure/OpenAI) that produces structured
  signals from natural-language queries.
* A ranking step that merges collaborative and semantic candidates,
  applies metadata + intent heuristics, and produces the final playlist.

All heavy resources (artefacts and ML models) are loaded lazily the
first time the engine is used.  If any component is missing the engine
falls back to a deterministic rule-based response, ensuring the public
API keeps functioning even in constrained environments.
"""

from __future__ import annotations

import importlib
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime
import os

import numpy as np
from pydantic import BaseModel, Field

from . import model_utils

LOGGER = logging.getLogger(__name__)

# Optional LLM clients (Azure/OpenAI)
try:
    from openai import AzureOpenAI, OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    AzureOpenAI = None
    OpenAI = None

_INTENT_CLIENT = None
_INTENT_BACKEND: Optional[str] = None


def get_intent_client():
    """
    Lazy initialise an LLM client for intent parsing.

    Preference order:
      1. Azure OpenAI (classic) via AZURE_OPENAI_* env vars
      2. OpenAI via OPENAI_API_KEY
      3. None (falls back to heuristic parse_nl)
    """
    global _INTENT_CLIENT, _INTENT_BACKEND

    if _INTENT_CLIENT is not None:
        return _INTENT_CLIENT

    # Try Azure first
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    azure_intent_deployment = os.getenv("AZURE_OPENAI_INTENT_DEPLOYMENT")

    if AzureOpenAI and azure_endpoint and azure_key and azure_intent_deployment:
        try:
            client = AzureOpenAI(
                api_key=azure_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
            )
            _INTENT_CLIENT = client
            _INTENT_BACKEND = "azure"
            LOGGER.info(
                "LLM intent parser: using Azure OpenAI (deployment=%s)",
                azure_intent_deployment,
            )
            return _INTENT_CLIENT
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to initialise Azure OpenAI intent client: %s", exc)

    # Fallback: public OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if OpenAI and openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            _INTENT_CLIENT = client
            _INTENT_BACKEND = "openai"
            LOGGER.info("LLM intent parser: using OpenAI.")
            return _INTENT_CLIENT
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to initialise OpenAI intent client: %s", exc)

    LOGGER.warning("No LLM backend available for intent parsing; using heuristic parser.")
    _INTENT_CLIENT = None
    _INTENT_BACKEND = None
    return None


class Range(BaseModel):
    """Numeric range used for user intent preferences."""

    min: Optional[float] = Field(default=None)
    max: Optional[float] = Field(default=None)


class Era(BaseModel):
    """Time range helper for intent parsing."""

    from_year: Optional[int] = Field(default=None)
    to_year: Optional[int] = Field(default=None)


class Intent(BaseModel):
    """Structured representation of the LLM extracted query intent."""

    moods: List[str] = Field(default_factory=list)
    avoid_vocals: bool = False
    genres: List[str] = Field(default_factory=list)
    energy: Range = Field(default_factory=Range)
    tempo_bpm: Range = Field(default_factory=Range)
    era: Era = Field(default_factory=Era)
    include_artists: List[str] = Field(default_factory=list)
    exclude_artists: List[str] = Field(default_factory=list)


def parse_nl(query: str) -> Intent:
    """Extremely small heuristic parser used as a fallback."""

    q = (query or "").lower()
    mood = ""
    if any(token in q for token in ("chill", "study", "focus")):
        mood = "chill"
    elif "workout" in q or "run" in q or "gym" in q:
        mood = "energetic"
    avoid_vocals = ("no vocals" in q) or ("instrumental" in q)
    return Intent(moods=[mood] if mood else [], avoid_vocals=avoid_vocals)


def parse_nl_llm(query: str) -> Intent:
    """
    LLM-powered intent extraction.

    If Azure/OpenAI is unavailable or fails, falls back to the heuristic parser.
    """
    if not query:
        return Intent()

    client = get_intent_client()
    if not client:
        return parse_nl(query)

    # Decide model / deployment based on backend
    if _INTENT_BACKEND == "azure":
        model_name = os.getenv("AZURE_OPENAI_INTENT_DEPLOYMENT")
    elif _INTENT_BACKEND == "openai":
        model_name = os.getenv("OPENAI_INTENT_MODEL", "gpt-4o-mini")
    else:
        model_name = None

    if not model_name:
        return parse_nl(query)

    system_msg = (
        "You are a music intent parser. Extract structured metadata from a user music "
        "request. Respond ONLY with valid JSON object matching this schema:\n\n"
        "{\n"
        '  "moods": ["chill", "energetic"],\n'
        '  "genres": ["lofi", "jazz"],\n'
        '  "avoid_vocals": true,\n'
        '  "era": {"from_year": 2000, "to_year": 2020},\n'
        '  "include_artists": ["artist1"],\n'
        '  "exclude_artists": []\n'
        "}\n"
    )

    user_msg = f'User query: "{query}"'

    try:
        resp = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        # unified response API: first output, first content block
        text = resp.output[0].content[0].text
        data = json.loads(text)
        return Intent(**data)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("LLM intent parsing failed, falling back to heuristic: %s", exc)
        return parse_nl(query)


_NUMPY = None


def _get_numpy():
    global _NUMPY
    if _NUMPY is None:
        try:
            _NUMPY = importlib.import_module("numpy")
        except ImportError as exc:  # pragma: no cover - exercised in fallback path
            raise model_utils.DependencyError("numpy is required for recommendation features.") from exc
    return _NUMPY


@dataclass
class RetrievalResult:
    """Wrapper around candidate identifiers and their scores."""

    ids: Any
    scores: Any

    def as_lists(self) -> Tuple[List[str], List[float]]:
        np = _get_numpy()
        return np.asarray(self.ids).tolist(), np.asarray(self.scores, dtype=float).tolist()


class HybridRetriever:
    """Combine collaborative filtering and semantic similarity search."""

    def __init__(self, artifact_dir: Path | str = model_utils.ARTIFACT_DIR):
        artifact_dir = Path(artifact_dir)
        np = _get_numpy()

        # Load semantic artefacts (embeddings + FAISS index)
        item_art = model_utils.load_item_embeddings(artifact_dir / "item_embs.joblib")
        self.item_ids = np.asarray(item_art["track_ids"])
        self.index = model_utils.load_faiss_index(artifact_dir / "item_faiss.index")

        # Encoder used for query -> embedding
        self.encoder = model_utils.load_sentence_transformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load collaborative filtering factors (ALS)
        als = model_utils.load_cf_artifact(artifact_dir / "als.joblib")
        self.user_factors = als["user_factors"]
        self.item_factors = als["item_factors"]
        self.u_codes = als["u_codes"]
        self.i_codes = als["i_codes"]
        self._idx_to_track = {idx: tid for tid, idx in self.i_codes.items()}

    # Collaborative filtering -------------------------------------------------
    def by_cf(self, user_id: str, k: int = 200) -> RetrievalResult:
        np = _get_numpy()
        if user_id not in self.u_codes:
            return RetrievalResult(np.array([]), np.array([]))

        uidx = self.u_codes[user_id]
        scores = self.item_factors @ self.user_factors[uidx]
        if scores.size == 0:
            return RetrievalResult(np.array([]), np.array([]))

        k = min(max(k, 0), scores.size)
        if k <= 0:
            return RetrievalResult(np.array([]), np.array([]))
        top = np.argpartition(-scores, k - 1)[:k]
        order = top[np.argsort(-scores[top])]
        track_ids = np.array([self._idx_to_track[idx] for idx in order])
        return RetrievalResult(track_ids, scores[order])

    # Semantic retrieval -------------------------------------------------------
    def by_nlq(self, text: str, k: int = 200) -> RetrievalResult:
        np = _get_numpy()
        if not text:
            return RetrievalResult(np.array([]), np.array([]))

        k = max(k, 0)
        if k == 0:
            return RetrievalResult(np.array([]), np.array([]))

        query_vec = self.encoder.encode([text], normalize_embeddings=True).astype("float32")
        distances, indices = self.index.search(query_vec, k)
        return RetrievalResult(self.item_ids[indices[0]], distances[0])


def rank(
    cf_candidates: RetrievalResult,
    sem_candidates: RetrievalResult,
    meta_df,
    intent: Optional[Intent] = None,
    user_history: Optional[Dict[str, Iterable[str]]] = None,
    weights: Tuple[float, float, float, float] = (0.45, 0.35, 0.15, 0.05),
    limit: int = 100,
) -> List[Tuple[str, float]]:
    """Blend collaborative and semantic scores with metadata + intent.

    weights = (w_cf, w_sem, w_meta, w_intent)
    """

    np = _get_numpy()

    w_cf, w_sem, w_meta, w_intent = weights
    user_history = user_history or {"skipped_ids": []}
    skipped_ids = set(user_history.get("skipped_ids", []))

    cf_map = {cid: score for cid, score in zip(cf_candidates.ids, cf_candidates.scores)}
    sem_map = {cid: score for cid, score in zip(sem_candidates.ids, sem_candidates.scores)}

    # union of all candidate ids
    seen: Dict[str, None] = {}
    for cid in cf_candidates.ids:
        seen.setdefault(cid, None)
    for cid in sem_candidates.ids:
        seen.setdefault(cid, None)
    candidates = list(seen.keys())

    results: List[Tuple[str, float]] = []

    for cid in candidates:
        meta_bonus = 0.0
        intent_bonus = 0.0

        s_cf = float(cf_map.get(cid, 0.0))
        s_sem = float(sem_map.get(cid, 0.0))

        row = None
        if meta_df is not None:
            try:
                row = meta_df.loc[cid]
            except Exception:
                row = None

        popularity = 0.0
        release_year = None

        if row is not None and hasattr(row, "get"):
            # Popularity normalised [0,1]
            popularity = float(row.get("popularity", 0) or 0) / 100.0
            meta_bonus += 0.4 * popularity

            # Recency bonus based on release year (if present)
            release_year = row.get("release_year")
            if release_year:
                try:
                    release_year_int = int(release_year)
                    current_year = datetime.now().year
                    recency = max(0.0, 1.0 - (current_year - release_year_int) / 30.0)
                    meta_bonus += 0.3 * recency
                except Exception:
                    pass

            # Playlist-sourced tracks get a small bump
            if row.get("source_type") == "playlist":
                meta_bonus += 0.1

            # Repeatedly skipped items get a penalty
            if cid in skipped_ids:
                meta_bonus -= 0.2

        # Intent-driven scoring
        if intent and row is not None and hasattr(row, "get"):
            genres_str = ""
            genres_val = row.get("genres")
            if isinstance(genres_val, (list, tuple)):
                genres_str = " ".join(str(g).lower() for g in genres_val)
            elif isinstance(genres_val, str):
                genres_str = genres_val.lower()

            # Genre alignment
            if intent.genres:
                if any(g.lower() in genres_str for g in intent.genres):
                    intent_bonus += 0.4

            # Era alignment
            if intent.era and (intent.era.from_year or intent.era.to_year) and release_year:
                try:
                    y = int(release_year)
                    from_y = intent.era.from_year or 1900
                    to_y = intent.era.to_year or datetime.now().year
                    if from_y <= y <= to_y:
                        intent_bonus += 0.3
                except Exception:
                    pass

            # Vocal preference
            vocals_flag = row.get("vocals")
            if intent.avoid_vocals and vocals_flag:
                intent_bonus -= 0.5

        score = (
            w_cf * s_cf +
            w_sem * s_sem +
            w_meta * meta_bonus +
            w_intent * intent_bonus
        )

        results.append((cid, score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results[:limit]


class RecommendationEngine:
    """High level orchestrator for the recommendation workflow."""

    def __init__(
        self,
        data_dir: Path | str = model_utils.DATA_DIR,
        artifact_dir: Path | str = model_utils.ARTIFACT_DIR,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.artifact_dir = Path(artifact_dir)
        self._lock = threading.RLock()
        self._retriever: Optional[HybridRetriever] = None
        self._meta = None
        self._fallback_reason: Optional[str] = None

    # Lazy loaders ------------------------------------------------------------
    def _load_retriever(self) -> Optional[HybridRetriever]:
        with self._lock:
            if self._retriever is not None:
                return self._retriever

            try:
                self._retriever = HybridRetriever(self.artifact_dir)
                LOGGER.info("HybridRetriever initialised successfully.")
            except (FileNotFoundError, model_utils.MissingArtifactError, model_utils.DependencyError) as exc:
                LOGGER.warning("Falling back to rule-based recommendations: %s", exc)
                self._fallback_reason = str(exc)
                self._retriever = None
            return self._retriever

    def _load_meta(self):
        if self._meta is None:
            try:
                self._meta = model_utils.load_metadata(self.data_dir / "items.parquet")
            except (FileNotFoundError, model_utils.MissingDataError, model_utils.DependencyError) as exc:
                LOGGER.warning("Metadata unavailable: %s", exc)
                self._meta = None
        return self._meta

    # Public API --------------------------------------------------------------
    def recommend(self, prompt: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        prompt = prompt or ""
        retriever = self._load_retriever()
        meta = self._load_meta()

        if retriever is None:
            return self._fallback(prompt, user_id)

        cf_result = RetrievalResult(np.array([]), np.array([]))
        sem_result = RetrievalResult(np.array([]), np.array([]))

        intent: Optional[Intent] = None
        if prompt:
            intent = parse_nl_llm(prompt)

        if user_id:
            cf_result = retriever.by_cf(user_id)
        if prompt:
            sem_result = retriever.by_nlq(prompt)

        scored = rank(
            cf_result,
            sem_result,
            meta,
            intent=intent,
            user_history={"skipped_ids": []},
        )
        payload = [self._payload_item(cid, score, meta) for cid, score in scored]
        explanation = self._build_explanation(
            intent,
            bool(cf_result.ids.size),
            bool(sem_result.ids.size),
        )

        return {
            "prompt": prompt,
            "user_id": user_id,
            "intent": json.loads(intent.json()) if intent else None,
            "recommendations": payload,
            "count": len(payload),
            "explanation": explanation,
        }

    # Helpers -----------------------------------------------------------------
    def _fallback(self, prompt: str, user_id: Optional[str]) -> Dict[str, Any]:
        """Fallback deterministic response when artefacts are unavailable."""

        baseline = [
            {"track_id": "mix_daily", "title": "Daily Mix", "artist": "Various", "score": 0.1},
            {"track_id": "coffee_morning", "title": "Morning Coffee", "artist": "Lo-Fi", "score": 0.05},
            {"track_id": "acoustic_essentials", "title": "Acoustic Essentials", "artist": "Indie", "score": 0.02},
        ]
        reason = self._fallback_reason or "Required ML artefacts are not available."
        return {
            "prompt": prompt,
            "user_id": user_id,
            "intent": None,
            "recommendations": baseline,
            "count": len(baseline),
            "explanation": reason,
        }

    def _payload_item(self, track_id: str, score: float, meta_df) -> Dict[str, Any]:
        data: Dict[str, Any] = {"track_id": track_id, "score": float(score)}
        if meta_df is not None:
            try:
                row = meta_df.loc[track_id]
            except Exception:
                row = None
            if row is not None and hasattr(row, "get"):
                # Safely pull common metadata fields if present
                for field in [
                    "title",
                    "artist",
                    "album",
                    "image_url",
                    "genres",
                    "source_type",
                    "popularity",
                    "release_year",
                    "preview_url",
                ]:
                    value = row.get(field) if hasattr(row, "get") else None
                    if value is not None:
                        data[field] = value
        return data

    @staticmethod
    def _build_explanation(intent: Optional[Intent], has_cf: bool, has_sem: bool) -> str:
        parts: List[str] = []
        if intent:
            if intent.moods:
                parts.append(f"Mood detected: {', '.join(intent.moods)}.")
            if intent.genres:
                parts.append(f"Genre preference: {', '.join(intent.genres)}.")
            if intent.era and (intent.era.from_year or intent.era.to_year):
                start = intent.era.from_year or "any"
                end = intent.era.to_year or "now"
                parts.append(f"Era preference: {start}–{end}.")
            if intent.avoid_vocals:
                parts.append("Filtered toward instrumental / low-vocal tracks.")

        if has_cf:
            parts.append("Incorporated collaborative filtering from user history.")
        if has_sem:
            parts.append("Used semantic similarity over track embeddings.")
        if not parts:
            return "Default ranking applied."
        return " ".join(parts)


_ENGINE: Optional[RecommendationEngine] = None
_ENGINE_LOCK = threading.Lock()


def _get_engine() -> RecommendationEngine:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = RecommendationEngine()
    return _ENGINE


def llm_recommender(prompt: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Public entry point used by the threaded HTTP server."""

    engine = _get_engine()
    return engine.recommend(prompt=prompt, user_id=user_id)


__all__ = [
    "Range",
    "Era",
    "Intent",
    "parse_nl",
    "parse_nl_llm",
    "HybridRetriever",
    "rank",
    "RecommendationEngine",
    "llm_recommender",
]

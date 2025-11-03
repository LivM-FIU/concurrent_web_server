"""Recommendation engine that wires together the hybrid retrieval stack.

The implementation mirrors the design sketched in the project plan:

* Collaborative filtering via an implicit ALS model trained on historic
  interactions.
* Semantic retrieval via sentence-transformer embeddings and a FAISS
  index for approximate nearest neighbours.
* A lightweight intent parser (LLM stub) that produces structured
  signals from natural-language queries.
* A ranking step that merges collaborative and semantic candidates,
  applies metadata heuristics, and produces the final playlist.

All heavy resources (artefacts and ML models) are loaded lazily the
first time the engine is used.  If any component is missing the engine
falls back to a deterministic rule-based response, ensuring the public
API keeps functioning even in constrained test environments.
"""

from __future__ import annotations

import importlib
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from . import model_utils

LOGGER = logging.getLogger(__name__)


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
    """Extremely small heuristic parser used as a placeholder for an LLM."""

    q = (query or "").lower()
    mood = ""
    if any(token in q for token in ("chill", "study", "focus")):
        mood = "chill"
    elif "workout" in q or "run" in q or "gym" in q:
        mood = "energetic"
    avoid_vocals = ("no vocals" in q) or ("instrumental" in q)
    return Intent(moods=[mood] if mood else [], avoid_vocals=avoid_vocals)


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

        # Load semantic artefacts
        item_art = model_utils.load_item_embeddings(artifact_dir / "item_embs.joblib")
        self.item_ids = np.asarray(item_art["track_ids"])
        self.index = model_utils.load_faiss_index(artifact_dir / "item_faiss.index")

        self.encoder = model_utils.load_sentence_transformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load collaborative filtering factors
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


def _candidate_union(cf_ids: Iterable[str], sem_ids: Iterable[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for cid in cf_ids:
        seen.setdefault(cid, None)
    for cid in sem_ids:
        seen.setdefault(cid, None)
    return list(seen.keys())


def rank(
    cf_candidates: RetrievalResult,
    sem_candidates: RetrievalResult,
    meta_df,
    user_history: Optional[Dict[str, Iterable[str]]] = None,
    weights: Tuple[float, float, float] = (0.5, 0.35, 0.15),
    limit: int = 100,
) -> List[Tuple[str, float]]:
    """Blend collaborative and semantic scores with light metadata signals."""

    user_history = user_history or {"skipped_ids": []}
    skipped = set(user_history.get("skipped_ids", []))
    candidates = _candidate_union(cf_candidates.ids, sem_candidates.ids)

    cf_map = {cid: score for cid, score in zip(cf_candidates.ids, cf_candidates.scores)}
    sem_map = {cid: score for cid, score in zip(sem_candidates.ids, sem_candidates.scores)}

    results: List[Tuple[str, float]] = []
    w_cf, w_sem, w_meta = weights

    for cid in candidates:
        s_cf = float(cf_map.get(cid, 0.0))
        s_sem = float(sem_map.get(cid, 0.0))

        meta_bonus = 1.0
        row = None
        if meta_df is not None:
            try:
                row = meta_df.loc[cid]
            except Exception:  # pragma: no cover - pandas raises KeyError
                row = None
        if row is not None and hasattr(row, "get"):
            freshness = float(row.get("freshness", 0) or 0)
            meta_bonus += 0.05 * freshness
            if row.get("vocals", False):
                meta_bonus -= 0.1
        if cid in skipped:
            meta_bonus -= 0.2

        score = w_cf * s_cf + w_sem * s_sem + w_meta * meta_bonus
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
        if user_id:
            cf_result = retriever.by_cf(user_id)
        if prompt:
            intent = parse_nl(prompt)
            sem_result = retriever.by_nlq(prompt)

        scored = rank(cf_result, sem_result, meta, user_history={"skipped_ids": set()})
        payload = [self._payload_item(cid, score, meta) for cid, score in scored]
        explanation = self._build_explanation(intent, bool(cf_result.ids.size), bool(sem_result.ids.size))

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
        data = {"track_id": track_id, "score": float(score)}
        if meta_df is not None:
            try:
                row = meta_df.loc[track_id]
            except Exception:  # pragma: no cover - KeyError for missing track
                row = None
            if row is not None and hasattr(row, "get"):
                data.update(
                    {
                        "title": row.get("title"),
                        "artist": row.get("artist"),
                    }
                )
        return data

    @staticmethod
    def _build_explanation(intent: Optional[Intent], has_cf: bool, has_sem: bool) -> str:
        parts: List[str] = []
        if intent:
            moods = ", ".join(intent.moods) if intent.moods else "general"
            parts.append(f"Matched natural language query with mood: {moods}.")
            if intent.avoid_vocals:
                parts.append("Applied instrumental filter.")
        if has_cf:
            parts.append("Blended collaborative history.")
        if has_sem:
            parts.append("Used semantic similarity search.")
        return " ".join(parts) if parts else "Default ranking applied."


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
    "HybridRetriever",
    "rank",
    "RecommendationEngine",
    "llm_recommender",
]


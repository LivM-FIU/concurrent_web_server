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
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - import guarded for environments without pydantic
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - dependency optional for tests
    BaseModel = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]

from . import model_utils

LOGGER = logging.getLogger(__name__)


if BaseModel is not None:
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

else:

    @dataclass
    class Range:
        """Numeric range used for user intent preferences."""

        min: Optional[float] = None
        max: Optional[float] = None


    @dataclass
    class Era:
        """Time range helper for intent parsing."""

        from_year: Optional[int] = None
        to_year: Optional[int] = None


    @dataclass
    class Intent:
        """Structured representation of the LLM extracted query intent."""

        moods: List[str] = field(default_factory=list)
        avoid_vocals: bool = False
        genres: List[str] = field(default_factory=list)
        energy: Range = field(default_factory=Range)
        tempo_bpm: Range = field(default_factory=Range)
        era: Era = field(default_factory=Era)
        include_artists: List[str] = field(default_factory=list)
        exclude_artists: List[str] = field(default_factory=list)

        def json(self) -> str:
            return json.dumps(asdict(self))

        def dict(self) -> Dict[str, Any]:
            return asdict(self)


_MOOD_KEYWORDS = {
    "chill": {"chill", "study", "focus", "calm", "relax"},
    "energetic": {"energetic", "workout", "gym", "run", "hype"},
    "happy": {"happy", "uplifting", "sunny", "feel good"},
    "melancholy": {"sad", "moody", "melancholy", "blue"},
}

_GENRE_KEYWORDS = {
    "lofi": {"lofi", "lo-fi", "lo fi"},
    "piano": {"piano", "keys"},
    "ambient": {"ambient"},
    "jazz": {"jazz"},
    "rock": {"rock"},
    "synthwave": {"synthwave", "retro wave", "retrowave"},
    "house": {"house", "deep house", "progressive house"},
    "hip hop": {"hip hop", "hip-hop", "rap"},
    "pop": {"pop"},
    "classical": {"classical", "orchestral"},
}

_TEMPO_WORDS = {
    "slow": (0, 90),
    "mid": (90, 120),
    "medium": (90, 120),
    "moderate": (90, 120),
    "fast": (120, 999),
    "upbeat": (120, 999),
    "rapid": (130, 999),
}

_ENERGY_WORDS = {
    "low": (0.0, 0.4),
    "medium": (0.3, 0.7),
    "mid": (0.3, 0.7),
    "high": (0.6, 1.0),
    "intense": (0.6, 1.0),
}

_DECADE_PATTERN = re.compile(r"(?P<decade>(?:19|20)\d0)s")
_YEAR_RANGE_PATTERN = re.compile(r"(?P<start>19\d{2}|20\d{2})\s*[-to]{1,3}\s*(?P<end>19\d{2}|20\d{2})")
_BPM_PATTERN = re.compile(r"(?P<value>\d{2,3})\s*(?:bpm|beats?)")
_ARTIST_INCLUDE_PATTERN = re.compile(
    r"(?:like|include|featuring|featuring\s+artist)\s+([a-z0-9'&\s]+?)(?=(?:,|;|and|but|with|without|except|$))"
)
_ARTIST_EXCLUDE_PATTERN = re.compile(
    r"(?:no|exclude|avoid)\s+([a-z0-9'&\s]+?)(?=(?:,|;|and|but|with|without|except|$))"
)


def parse_nl(query: str) -> Intent:
    """Heuristic natural-language parser approximating the LLM output schema."""

    q_raw = query or ""
    q = q_raw.lower()

    moods: List[str] = []
    for label, keywords in _MOOD_KEYWORDS.items():
        if any(word in q for word in keywords):
            moods.append(label)

    genres: List[str] = []
    for genre, keywords in _GENRE_KEYWORDS.items():
        if any(word in q for word in keywords):
            genres.append(genre)

    tempo_range = Range()
    bpm_match = _BPM_PATTERN.search(q)
    if bpm_match:
        bpm_value = float(bpm_match.group("value"))
        tempo_range = Range(min=max(0.0, bpm_value - 10), max=bpm_value + 10)
    else:
        for keyword, (low, high) in _TEMPO_WORDS.items():
            if keyword in q:
                tempo_range = Range(min=float(low), max=float(high))
                break

    energy_range = Range()
    for keyword, (low, high) in _ENERGY_WORDS.items():
        if keyword in q:
            energy_range = Range(min=low, max=high)
            break

    era = Era()
    decade_match = _DECADE_PATTERN.search(q)
    if decade_match:
        decade = decade_match.group("decade")
        era = Era(from_year=int(decade), to_year=int(decade) + 9)
    else:
        year_range = _YEAR_RANGE_PATTERN.search(q)
        if year_range:
            era = Era(from_year=int(year_range.group("start")), to_year=int(year_range.group("end")))
        else:
            single_year = re.search(r"(19\d{2}|20\d{2})", q)
            if single_year:
                year = int(single_year.group(1))
                era = Era(from_year=year, to_year=year)

    include_artists = [match.strip() for match in _ARTIST_INCLUDE_PATTERN.findall(q)]
    exclude_artists = [match.strip() for match in _ARTIST_EXCLUDE_PATTERN.findall(q)]

    avoid_vocals = ("no vocals" in q) or ("instrumental" in q) or ("without vocals" in q)

    return Intent(
        moods=moods,
        avoid_vocals=avoid_vocals,
        genres=genres,
        energy=energy_range,
        tempo_bpm=tempo_range,
        era=era,
        include_artists=include_artists,
        exclude_artists=exclude_artists,
    )


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


@dataclass
class _CacheEntry:
    """In-memory cache entry for previously computed recommendations."""

    timestamp: float
    payload: Dict[str, Any]


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
        cache_ttl: float = 300.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.artifact_dir = Path(artifact_dir)
        self._lock = threading.RLock()
        self._retriever: Optional[HybridRetriever] = None
        self._meta = None
        self._fallback_reason: Optional[str] = None
        self._cache_ttl = float(cache_ttl)
        self._user_cache: Dict[str, _CacheEntry] = {}

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
    def recommend(
        self,
        prompt: str,
        user_id: Optional[str] = None,
        *,
        refresh_cache: bool = False,
    ) -> Dict[str, Any]:
        prompt = prompt or ""

        if user_id and not prompt and not refresh_cache:
            cached = self._get_cached(user_id)
            if cached is not None:
                return cached

        retriever = self._load_retriever()

        if retriever is None:
            return self._fallback(prompt, user_id)

        meta = self._load_meta()

        np = _get_numpy()
        cf_result = RetrievalResult(np.array([]), np.array([]))
        sem_result = RetrievalResult(np.array([]), np.array([]))

        intent: Optional[Intent] = None
        if user_id:
            cf_result = retriever.by_cf(user_id)
        if prompt:
            intent = parse_nl(prompt)
            sem_result = retriever.by_nlq(prompt)

        cf_result, sem_result = self._filter_candidates(intent, cf_result, sem_result, meta)

        scored = rank(cf_result, sem_result, meta, user_history={"skipped_ids": set()})
        payload = [self._payload_item(cid, score, meta) for cid, score in scored]
        explanation = self._build_explanation(intent, bool(cf_result.ids.size), bool(sem_result.ids.size))

        response = {
            "prompt": prompt,
            "user_id": user_id,
            "intent": json.loads(intent.json()) if intent else None,
            "recommendations": payload,
            "count": len(payload),
            "explanation": explanation,
        }

        if user_id and not prompt:
            self._store_cache(user_id, response)

        return response

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

    # Cache utilities ---------------------------------------------------------
    def _get_cached(self, user_id: str) -> Optional[Dict[str, Any]]:
        entry = self._user_cache.get(user_id)
        if entry is None:
            return None
        if time.time() - entry.timestamp > self._cache_ttl:
            self._user_cache.pop(user_id, None)
            return None
        return entry.payload

    def _store_cache(self, user_id: str, payload: Dict[str, Any]) -> None:
        self._user_cache[user_id] = _CacheEntry(timestamp=time.time(), payload=payload)

    # Filtering ---------------------------------------------------------------
    def _filter_candidates(
        self,
        intent: Optional[Intent],
        cf_result: RetrievalResult,
        sem_result: RetrievalResult,
        meta_df,
    ) -> Tuple[RetrievalResult, RetrievalResult]:
        if intent is None or meta_df is None:
            return cf_result, sem_result

        np = _get_numpy()

        def _as_array(values: Any) -> Any:
            return np.asarray(values)

        def _filter(result: RetrievalResult) -> RetrievalResult:
            ids = _as_array(result.ids)
            scores = _as_array(result.scores)
            if ids.size == 0:
                return result

            mask = []
            for cid in ids:
                try:
                    row = meta_df.loc[cid]
                except Exception:
                    row = None
                mask.append(self._row_matches_intent(row, intent))

            if not any(mask):
                return RetrievalResult(ids[:0], scores[:0])

            mask_arr = np.asarray(mask)
            return RetrievalResult(ids[mask_arr], scores[mask_arr])

        return _filter(cf_result), _filter(sem_result)

    @staticmethod
    def _row_matches_intent(row, intent: Intent) -> bool:
        if row is None:
            return True

        if hasattr(row, "to_dict"):
            data = row.to_dict()
        elif hasattr(row, "items"):
            data = dict(row.items())
        else:
            data = dict(row)

        def _get_field(*names: str) -> Any:
            for name in names:
                if name in data:
                    return data[name]
            return None

        def _normalise_tokens(value: Any) -> List[str]:
            if value is None:
                return []
            if isinstance(value, str):
                tokens = re.split(r"[\s,;\/|]+", value.lower())
                return [token for token in tokens if token]
            if isinstance(value, (list, tuple, set)):
                return [str(item).lower() for item in value]
            return []

        def _artist_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.lower()
            if isinstance(value, (list, tuple, set)):
                return " ".join(str(item) for item in value).lower()
            return str(value).lower()

        if intent.avoid_vocals:
            vocals = _get_field("vocals", "has_vocals", "is_vocal")
            if isinstance(vocals, bool) and vocals:
                return False
            instrumentalness = _get_field("instrumentalness")
            if instrumentalness is not None and float(instrumentalness) < 0.5:
                return False

        if intent.genres:
            genres = set(_normalise_tokens(_get_field("genres", "genre")))
            if genres and not genres.intersection({g.lower() for g in intent.genres}):
                return False

        if intent.moods:
            moods = set(_normalise_tokens(_get_field("moods", "mood")))
            if moods and not moods.intersection({m.lower() for m in intent.moods}):
                return False

        if intent.include_artists:
            artist = _artist_text(_get_field("artist", "artists"))
            if artist:
                if not any(name.lower() in artist for name in intent.include_artists):
                    return False

        if intent.exclude_artists:
            artist = _artist_text(_get_field("artist", "artists"))
            if artist and any(name.lower() in artist for name in intent.exclude_artists):
                return False

        tempo = _get_field("tempo_bpm", "tempo", "bpm")
        if tempo is not None and (intent.tempo_bpm.min is not None or intent.tempo_bpm.max is not None):
            try:
                tempo_value = float(tempo)
            except (TypeError, ValueError):
                tempo_value = None
            if tempo_value is not None:
                if intent.tempo_bpm.min is not None and tempo_value < intent.tempo_bpm.min:
                    return False
                if intent.tempo_bpm.max is not None and tempo_value > intent.tempo_bpm.max:
                    return False

        energy = _get_field("energy", "energy_level")
        if energy is not None and (intent.energy.min is not None or intent.energy.max is not None):
            try:
                energy_value = float(energy)
            except (TypeError, ValueError):
                energy_value = None
            if energy_value is not None:
                if intent.energy.min is not None and energy_value < intent.energy.min:
                    return False
                if intent.energy.max is not None and energy_value > intent.energy.max:
                    return False

        year_value = _get_field("release_year", "year")
        if year_value and (intent.era.from_year is not None or intent.era.to_year is not None):
            try:
                year_int = int(str(year_value)[:4])
            except (TypeError, ValueError):
                year_int = None
            if year_int is not None:
                if intent.era.from_year is not None and year_int < intent.era.from_year:
                    return False
                if intent.era.to_year is not None and year_int > intent.era.to_year:
                    return False

        return True


_ENGINE: Optional[RecommendationEngine] = None
_ENGINE_LOCK = threading.Lock()


def _get_engine() -> RecommendationEngine:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = RecommendationEngine()
    return _ENGINE


def llm_recommender(
    prompt: str,
    user_id: Optional[str] = None,
    *,
    refresh_cache: bool = False,
) -> Dict[str, Any]:
    """Public entry point used by the threaded HTTP server."""

    engine = _get_engine()
    return engine.recommend(prompt=prompt, user_id=user_id, refresh_cache=refresh_cache)


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


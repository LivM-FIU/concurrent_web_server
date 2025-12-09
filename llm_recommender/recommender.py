"""
recommender.py
Hybrid LLM + CF + Azure Embeddings Recommendation Engine
Author: Livan Miranda

This engine connects:
  * ALS collaborative filtering (Pure Python ALS in model_utils)
  * Semantic similarity via FAISS + Azure embeddings
  * LLM intent parsing (Azure: chat.completions)
  * Metadata + intent-based ranking

This version is aligned with config.settings and your Azure setup.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from openai import AzureOpenAI
from config.settings import settings
from . import model_utils

LOGGER = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Azure Clients
# ─────────────────────────────────────────────

AZURE_EMB_CLIENT = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

AZURE_EMBED_MODEL = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT

AZURE_CHAT_CLIENT = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

AZURE_CHAT_MODEL = settings.AZURE_OPENAI_CHAT_MODEL  # can be base or fine-tuned


# ─────────────────────────────────────────────
# Intent Structures
# ─────────────────────────────────────────────

class Range(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None


class Era(BaseModel):
    from_year: Optional[int] = None
    to_year: Optional[int] = None


class Intent(BaseModel):
    moods: List[str] = Field(default_factory=list)
    avoid_vocals: bool = False
    genres: List[str] = Field(default_factory=list)
    energy: Range = Field(default_factory=Range)
    tempo_bpm: Range = Field(default_factory=Range)
    era: Era = Field(default_factory=Era)
    include_artists: List[str] = Field(default_factory=list)
    exclude_artists: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Helpers: numpy → native Python
# ─────────────────────────────────────────────

def _to_python(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays into native Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_python(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(x) for x in obj]
    return obj


# ─────────────────────────────────────────────
# Heuristic fallback parser
# ─────────────────────────────────────────────

def parse_nl_heuristic(query: str) -> Intent:
    """Emergency fallback for when Azure LLM fails."""
    q = (query or "").lower()
    mood = ""

    if any(t in q for t in ["chill", "study", "focus"]):
        mood = "chill"
    elif any(t in q for t in ["workout", "gym", "run"]):
        mood = "energetic"
    elif "jazz" in q:
        mood = "relaxed"

    avoid_vocals = "instrumental" in q or "no vocals" in q

    era = Era()
    # Very rough heuristic for era if decade mentioned
    if "1960" in q or "60s" in q or "60's" in q:
        era = Era(from_year=1960, to_year=1969)
    if "2010" in q or "2010s" in q:
        era = Era(from_year=2010, to_year=2019)

    return Intent(
        moods=[mood] if mood else [],
        avoid_vocals=avoid_vocals,
        era=era,
    )


# ─────────────────────────────────────────────
# Intent parsing via Azure Chat Completions
# ─────────────────────────────────────────────

def _extract_json_from_text(text: str) -> dict:
    """
    Extracts JSON from model output.
    Handles:
      - code fences (```json ... ```)
      - extra explanations
      - leading/trailing whitespace
    """
    text = text.strip()

    # Strip code fences if present
    if text.startswith("```"):
        # remove ```json or ``` and ending ```
        text = text.strip("`")
        # crude but effective:
        if text.startswith("json"):
            text = text[4:].strip()

    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])

    raise ValueError(f"Malformed JSON from LLM: {text}")


def _sanitize_intent_json(data: dict) -> dict:
    """Cleans LLM JSON output: dedupe, trim, fix invalid ranges & era."""
    def clean_list(lst, limit=5):
        out: List[str] = []
        for x in lst or []:
            s = str(x).strip().lower()
            if s and s not in out:
                out.append(s)
            if len(out) >= limit:
                break
        return out

    # Lists
    for field in ["moods", "genres", "include_artists", "exclude_artists"]:
        data[field] = clean_list(data.get(field, []))

    # Ranges
    for field, default_hi in [("energy", 1.0), ("tempo_bpm", 200.0)]:
        rng = data.get(field) or {}
        try:
            lo = float(rng.get("min", 0))
            hi = float(rng.get("max", default_hi))
            if lo > hi:
                lo, hi = hi, lo
            data[field] = {"min": lo, "max": hi}
        except Exception:
            data[field] = {"min": 0.0, "max": default_hi}

    # Era
    era = data.get("era") or {}
    fy = era.get("from_year")
    ty = era.get("to_year")
    try:
        fy = int(fy) if fy is not None else None
        ty = int(ty) if ty is not None else None
    except Exception:
        fy, ty = None, None

    if fy is not None and ty is not None and fy > ty:
        fy, ty = ty, fy

    data["era"] = {"from_year": fy, "to_year": ty}

    return data


def parse_nl_llm(text: str) -> Intent:
    """
    Uses Azure ChatCompletion to extract structured intent.
    Prints RAW LLM response for debugging, then parses & normalizes JSON.
    Falls back to heuristic parser on any error.
    """
    if not text:
        return Intent()  # empty intent

    try:
        resp = AZURE_CHAT_CLIENT.chat.completions.create(
            model=AZURE_CHAT_MODEL,
            temperature=0,
            max_tokens=400,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a music intent parser. "
                        "Return ONLY JSON, no additional text. "
                        "Schema:\n"
                        "{\n"
                        '  "moods": [string],\n'
                        '  "avoid_vocals": boolean,\n'
                        '  "genres": [string],\n'
                        '  "energy": {"min": number, "max": number},\n'
                        '  "tempo_bpm": {"min": number, "max": number},\n'
                        '  "era": {"from_year": number|null, "to_year": number|null},\n'
                        '  "include_artists": [string],\n'
                        '  "exclude_artists": [string]\n'
                        "}\n"
                    ),
                },
                {"role": "user", "content": text},
            ],
        )

        raw = resp.choices[0].message.content

        # Debug output
        print("\n========== RAW LLM RESPONSE ==========\n")
        print(raw)
        print("\n======================================\n")

        data = _extract_json_from_text(raw)
        data = _sanitize_intent_json(data)

        # Build Intent from normalized dict
        return Intent.parse_obj(data)

    except Exception as e:
        print(f"LLM intent parsing failed → heuristic used. Error: {e}")
        return parse_nl_heuristic(text)


# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────

@dataclass
class RetrievalResult:
    ids: np.ndarray
    scores: np.ndarray


class HybridRetriever:
    """Semantic (Azure embeddings + FAISS) + CF retrieval."""

    def __init__(self, artifact_dir: Path | str = model_utils.ARTIFACT_DIR):
        artifact_dir = Path(artifact_dir)

        # Load semantic artifacts
        emb = model_utils.load_item_embeddings(artifact_dir / "item_embs.joblib")
        # Enforce pure string IDs
        self.item_ids = np.asarray([str(t) for t in emb["track_ids"]], dtype=object)
        self.index = model_utils.load_faiss_index(artifact_dir / "item_faiss.index")

        # Load CF artifacts
        als = model_utils.load_cf_artifact(artifact_dir / "als.joblib")
        self.user_factors = als["user_factors"]
        self.item_factors = als["item_factors"]

        # Normalize keys to str for safety
        self.u_codes = {str(k): v for k, v in als["u_codes"].items()}
        self.i_codes = {str(k): v for k, v in als["i_codes"].items()}

        # Reverse lookup: idx → track_id (string)
        self._idx_to_track = {idx: str(tid) for tid, idx in self.i_codes.items()}

    # CF retrieval
    def by_cf(self, user_id: str, k: int = 200) -> RetrievalResult:
        user_id = str(user_id)
        if user_id not in self.u_codes:
            return RetrievalResult(np.array([], dtype=object), np.array([], dtype="float32"))

        uidx = self.u_codes[user_id]
        # item_factors: (n_items, factors), user_factors: (n_users, factors)
        scores = self.item_factors @ self.user_factors[uidx]

        k = min(k, len(scores))
        if k <= 0:
            return RetrievalResult(np.array([], dtype=object), np.array([], dtype="float32"))

        top = np.argpartition(-scores, k - 1)[:k]
        order = top[np.argsort(-scores[top])]
        ids = np.array([self._idx_to_track[i] for i in order], dtype=object)

        return RetrievalResult(ids, scores[order])

    # Semantic retrieval via Azure embeddings + FAISS
    def by_nlq(self, text: str, k: int = 200) -> RetrievalResult:
        if not text:
            return RetrievalResult(np.array([], dtype=object), np.array([], dtype="float32"))

        try:
            r = AZURE_EMB_CLIENT.embeddings.create(
                model=AZURE_EMBED_MODEL,
                input=[text],
            )
            qvec = np.asarray([r.data[0].embedding], dtype="float32")
        except Exception as exc:
            LOGGER.warning(f"Azure embedding failed: {exc}")
            return RetrievalResult(np.array([], dtype=object), np.array([], dtype="float32"))

        distances, indices = self.index.search(qvec, k)
        ids = self.item_ids[indices[0]]
        scores = distances[0]

        return RetrievalResult(ids.astype(object), scores.astype("float32"))


# ─────────────────────────────────────────────
# Ranking Layer
# ─────────────────────────────────────────────

def rank(
    cf: RetrievalResult,
    sem: RetrievalResult,
    meta_df,
    intent: Optional[Intent] = None,
    weights: Tuple[float, float, float, float] = (0.45, 0.35, 0.15, 0.05),
    limit: int = 100,
) -> List[Tuple[str, float]]:
    """
    Merge CF + semantic candidates and re-score with metadata + intent.
    Returns a list of (track_id, score) sorted descending.
    """
    # Merge IDs while preserving relative order
    all_ids: List[str] = []
    seen = set()

    for arr in [cf.ids, sem.ids]:
        for tid in arr:
            tid = str(tid)
            if tid not in seen:
                seen.add(tid)
                all_ids.append(tid)

    results: List[Tuple[str, float]] = []

    cf_id_to_idx = {str(tid): i for i, tid in enumerate(cf.ids)}
    sem_id_to_idx = {str(tid): i for i, tid in enumerate(sem.ids)}

    for tid in all_ids:
        idx_cf = cf_id_to_idx.get(tid)
        idx_sem = sem_id_to_idx.get(tid)

        s_cf = float(cf.scores[idx_cf]) if idx_cf is not None else 0.0
        s_sem = float(sem.scores[idx_sem]) if idx_sem is not None else 0.0

        meta_bonus = 0.0
        intent_bonus = 0.0

        row = None
        if meta_df is not None:
            try:
                row = meta_df.loc[tid]
            except Exception:
                row = None

        # Metadata scoring
        if row is not None:
            # Popularity (0-100)
            popularity = row.get("popularity", 0)
            try:
                popularity = float(popularity or 0) / 100.0
            except Exception:
                popularity = 0.0
            meta_bonus += 0.4 * popularity

            # Recency
            ry = row.get("release_year")
            try:
                if ry is not None and not (isinstance(ry, float) and np.isnan(ry)):
                    ry_int = int(ry)
                    rec = max(0.0, 1.0 - (datetime.now().year - ry_int) / 30.0)
                    meta_bonus += 0.3 * rec
            except Exception:
                pass

            if row.get("source_type") == "playlist":
                meta_bonus += 0.1

        # Intent alignment
        if intent and row is not None:
            genres_str = str(row.get("genres") or "").lower()

            if intent.genres and any(g in genres_str for g in intent.genres):
                intent_bonus += 0.4

            # Era filter
            if intent.era and row.get("release_year") is not None:
                try:
                    y = int(row.get("release_year"))
                    fy = intent.era.from_year or 1800
                    ty = intent.era.to_year or datetime.now().year
                    if fy <= y <= ty:
                        intent_bonus += 0.3
                except Exception:
                    pass

            # Very rough vocal penalty
            if intent.avoid_vocals and str(row.get("title", "")).lower().count("feat") > 0:
                intent_bonus -= 0.3

        w_cf, w_sem, w_meta, w_int = weights
        score = w_cf * s_cf + w_sem * s_sem + w_meta * meta_bonus + w_int * intent_bonus

        results.append((tid, float(score)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class RecommendationEngine:
    def __init__(self, data_dir=model_utils.DATA_DIR, artifact_dir=model_utils.ARTIFACT_DIR):
        self.data_dir = Path(data_dir)
        self.artifact_dir = Path(artifact_dir)
        self._retriever: Optional[HybridRetriever] = None
        self._meta = None
        self._lock = threading.RLock()

    def _load_retriever(self) -> HybridRetriever:
        with self._lock:
            if self._retriever is None:
                self._retriever = HybridRetriever(self.artifact_dir)
            return self._retriever

    def _load_meta(self):
        if self._meta is None:
            self._meta = model_utils.load_metadata(self.data_dir / "items.parquet")
        return self._meta

    def _format(self, track_id, score, meta_df):
        """
        Enriched formatter:
        - If metadata exists → return enriched output
        - If missing → auto-fetch from Spotify → append to parquet → reload metadata
        """
        # Normalize possible numpy types
        if not isinstance(track_id, str):
            track_id = str(track_id)

        data = {"track_id": track_id, "score": float(score)}

        # 1 — If metadata exists normally
        if track_id in meta_df.index:
            row = meta_df.loc[track_id]
            for f in [
                "title", "artist", "album", "release_date",
                "popularity", "image_url", "source_type", "release_year"
            ]:
                val = row.get(f)
                if hasattr(val, "item"):  # handle numpy scalar
                    val = val.item()
                data[f] = val
            return data

        # 2 — AUTO-FETCH METADATA FROM SPOTIFY
        print(f"[INFO] Metadata missing → fetching from Spotify: {track_id}")
        meta = model_utils.fetch_track_metadata(track_id)

        if meta:
            # Append to parquet
            parquet_path = self.data_dir / "items.parquet"
            model_utils.append_metadata_row(parquet_path, meta)

            # Reload metadata in engine
            self._meta = model_utils.load_metadata(parquet_path)

            # Hydrate formatting
            data.update({
                "title": meta["title"],
                "artist": meta["artist"],
                "album": meta["album"],
                "release_date": meta["release_date"],
                "popularity": meta["popularity"],
                "image_url": meta["image_url"],
                "source_type": meta["source_type"],
                "release_year": meta["release_year"],
            })
            return data

        # 3 — If everything fails
        data["error"] = f"metadata_fetch_failed: {track_id}"
        return data

    def recommend(self, prompt: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        retriever = self._load_retriever()
        meta = self._load_meta()

        cf_res = (
            retriever.by_cf(user_id)
            if user_id is not None
            else RetrievalResult(np.array([], dtype=object), np.array([], dtype="float32"))
        )
        sem_res = (
            retriever.by_nlq(prompt)
            if prompt
            else RetrievalResult(np.array([], dtype=object), np.array([], dtype="float32"))
        )

        intent = parse_nl_llm(prompt) if prompt else None
        ranked = rank(cf_res, sem_res, meta, intent=intent)

        recs = [self._format(tid, score, meta) for tid, score in ranked]

        result: Dict[str, Any] = {
            "prompt": prompt,
            "user_id": user_id,
            "intent": _to_python(json.loads(intent.json())) if intent else None,
            "recommendations": recs,
            "count": len(recs),
        }
        return _to_python(result)

# ─────────────────────────────────────────────
# Singleton API + Debug Helpers
# ─────────────────────────────────────────────

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
    """
    Public entry point used in your REPL tests and (eventually) HTTP API.
    """
    return _get_engine().recommend(prompt, user_id)


def debug_intent_llm(query: str) -> str:
    """
    Sends the query directly to the LLM and prints the *raw* response text
    before any JSON parsing or repair. Returns raw text.
    """
    print("\n DEBUG INTENT LLM MODE — RAW RESPONSE BELOW\n")

    resp = AZURE_CHAT_CLIENT.chat.completions.create(
        model=AZURE_CHAT_MODEL,
        temperature=0,
        max_tokens=400,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return JSON only. No natural language. "
                    "Keys: moods, avoid_vocals, genres, "
                    "energy:{min,max}, tempo_bpm:{min,max}, "
                    "era:{from_year,to_year}, include_artists, exclude_artists."
                ),
            },
            {"role": "user", "content": query},
        ],
    )

    raw = resp.choices[0].message.content

    print("RAW MODEL OUTPUT:\n")
    print(raw)
    print("\n END RAW OUTPUT\n")

    return raw

__all__ = [
    "Intent",
    "Range",
    "Era",
    "HybridRetriever",
    "rank",
    "RecommendationEngine",
    "llm_recommender",
    "parse_nl_llm",
    "debug_intent_llm",
]

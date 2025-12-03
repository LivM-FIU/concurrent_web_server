"""
data_ingestion.py
Spotify Data Ingestion with PKCE + Embedding-based Query Expansion
Author: Livan Miranda

This script:

  * Authenticates to Spotify using OAuth2 + PKCE (NO client secret needed)
  * Opens the browser for login, runs a local callback server on localhost:8888/callback
  * Saves refresh token to token.json and refreshes silently on future runs
  * Expands a small set of seed search queries using embeddings:
        - tries Azure OpenAI embeddings first
        - falls back to OpenAI embeddings
        - falls back to local sentence-transformers (all-MiniLM-L6-v2)
        - if nothing is available, uses seed queries only
  * Fetches:
        - tracks directly from search
        - tracks from top playlists per query
  * Stores rich, FREE metadata only (no /audio-features calls) in:
        - songs.json
        - songs_backup_<timestamp>.json
  * Builds:
        - items.parquet / item_text.parquet
        - embeddings + FAISS index via llm_recommender.model_utils

This version includes:
  * A Spotify API wrapper with bounded rate-limit handling (no more "stuck forever")
  * Exponential backoff with max wait per attempt
  * Centralized retry logic for all Spotify calls (search, playlists, artist, album)
"""

import os
import sys
import json
import math
import requests
import hashlib
import base64
import random
import string
import logging
import webbrowser
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Dict, Any, Optional, Iterable, Tuple, Set

from dotenv import load_dotenv
import spotipy
from spotipy.exceptions import SpotifyException
import schedule
import time

import pandas as pd
from pathlib import Path

# Ensure project root is on sys.path for llm_recommender imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Optional / soft dependencies
try:
    from openai import AzureOpenAI, OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    AzureOpenAI = None
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None

load_dotenv()

# ─────────────────────────────────────────────
# BASIC CONFIGURATION
# ─────────────────────────────────────────────
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TOKEN_FILE = os.path.join(DATA_DIR, "token.json")
SONGS_FILE = os.path.join(DATA_DIR, "songs.json")
LOG_FILE = os.path.join(DATA_DIR, "ingestion.log")

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Seed queries you defined
BASE_SEARCH_QUERIES: List[str] = [
    "lofi chill",
    "study beats",
    "piano instrumental",
    "jazz coffee",
    "workout motivation",
    "ambient focus",
    "pop hits",
    "classic rock",
]

SCOPE = "user-read-private"  # enough; we only call free public endpoints

# Embedding expansion configuration
USE_EMBEDDING_EXPANSION = True
MAX_TOTAL_QUERIES = 24
EXPANSION_CACHE_FILE = os.path.join(DATA_DIR, "query_expansion_cache.json")

# Playlist fetch configuration
SEARCH_TRACK_LIMIT = 20
SEARCH_PLAYLIST_LIMIT = 5
PLAYLIST_TRACKS_LIMIT = 50

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─────────────────────────────────────────────
# PKCE UTILITIES
# ─────────────────────────────────────────────
def generate_code_verifier(length: int = 96) -> str:
    chars = string.ascii_letters + string.digits + "-._~"
    return "".join(random.choice(chars) for _ in range(length))


def generate_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


# ─────────────────────────────────────────────
# RATE-LIMIT AWARE SPOTIFY WRAPPER
# ─────────────────────────────────────────────
def spotify_call_with_retry(fn, *args, **kwargs):
    """
    Wrap a spotipy call with bounded rate-limit handling.

    - Catches 429 errors
    - Uses Retry-After when available, but clamps to max_wait
    - Exponential backoff on repeated 429s
    - Raises RuntimeError after max_retries instead of hanging forever
    """
    max_retries = 6
    backoff = 2        # base backoff (seconds)
    max_wait = 120     # never wait more than 2 minutes per attempt

    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)

        except SpotifyException as e:
            # 429 = rate limit exceeded
            if e.http_status == 429:
                retry_after = getattr(e, "retry_after", None)
                headers = getattr(e, "headers", {}) or {}

                if retry_after is None:
                    retry_after = headers.get("Retry-After")

                if retry_after is None:
                    retry_after = backoff

                try:
                    retry_after = int(retry_after)
                except Exception:
                    retry_after = backoff

                wait = max(3, min(retry_after, max_wait))

                logging.warning(
                    "Rate limit hit (attempt %s/%s). Waiting %s seconds before retry.",
                    attempt,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                backoff = min(backoff * 2, max_wait)
                continue

            # Other Spotify errors: let them bubble
            logging.exception("SpotifyException in %s: %s", getattr(fn, "__name__", fn), e)
            raise

        except Exception as e:
            # Non-Spotify exceptions bubble as usual
            logging.exception("Non-Spotify exception in %s: %s", getattr(fn, "__name__", fn), e)
            raise

    logging.error(
        "Spotify call %s exceeded max retries; giving up.",
        getattr(fn, "__name__", fn),
    )
    raise RuntimeError("Spotify API call exceeded max retries")


class SpotifyAPI:
    """
    Thin wrapper around spotipy.Spotify that routes all API calls through
    spotify_call_with_retry, so we own the rate-limit/backoff behavior.
    """

    def __init__(self, client: spotipy.Spotify):
        self._client = client

    # --- Search helpers ---

    def search_tracks(self, query: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        return spotify_call_with_retry(
            self._client.search,
            q=query,
            type="track",
            limit=limit,
            offset=offset,
        )

    def search_playlists(self, query: str, limit: int = 5, offset: int = 0) -> Dict[str, Any]:
        return spotify_call_with_retry(
            self._client.search,
            q=query,
            type="playlist",
            limit=limit,
            offset=offset,
        )

    # --- Playlist helpers ---

    def playlist_tracks(
        self,
        playlist_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return spotify_call_with_retry(
            self._client.playlist_tracks,
            playlist_id=playlist_id,
            limit=limit,
            offset=offset,
            additional_types=("track",),
        )

    # --- Metadata helpers ---

    def artist(self, artist_id: str) -> Dict[str, Any]:
        return spotify_call_with_retry(self._client.artist, artist_id)

    def album(self, album_id: str) -> Dict[str, Any]:
        return spotify_call_with_retry(self._client.album, album_id)


# ─────────────────────────────────────────────
# PARQUET + EMBEDDING PIPELINE
# ─────────────────────────────────────────────
def persist_items_parquet(tracks: List[Dict[str, Any]]):
    """Create items.parquet + item_text.parquet used by the recommender."""
    items = []
    texts = []

    for t in tracks:
        items.append(
            {
                "track_id": t["track_id"],
                "title": t["title"],
                "artist": t["artists"][0] if t["artists"] else None,
                "album": t["album"],
                "release_date": t["release_date"],
                "popularity": t["popularity"],
                "image_url": t["image_url"],
            }
        )

        texts.append(
            {
                "track_id": t["track_id"],
                "title": t["title"],
                "artist": t["artists"][0] if t["artists"] else None,
                "genres": ",".join(t.get("artist_meta", {}).get("genres", [])),
                "tags": "",
                "description": "",
            }
        )

    pd.DataFrame(items).to_parquet(
        os.path.join(DATA_DIR, "items.parquet"), index=False
    )
    pd.DataFrame(texts).to_parquet(
        os.path.join(DATA_DIR, "item_text.parquet"), index=False
    )

    print("Saved items.parquet + item_text.parquet")


def build_embeddings_and_faiss():
    """Creates embeddings and FAISS index using your model_utils pipeline."""
    try:
        from llm_recommender import model_utils

        # 1. Build embeddings
        emb_artifact = model_utils.build_item_embeddings(
            items_path=Path(DATA_DIR) / "items.parquet",
            text_path=Path(DATA_DIR) / "item_text.parquet",
            artifact_path=Path(ARTIFACT_DIR) / "item_embs.joblib",
        )

        # 2. Build FAISS index
        _ = model_utils.build_faiss_index(
            embeddings_artifact=emb_artifact,
            index_path=Path(ARTIFACT_DIR) / "item_faiss.index",
        )

        print("✓ Embeddings + FAISS index built successfully.")
        logging.info("Embeddings + FAISS index built.")

    except Exception as exc:
        logging.exception("Failed to build embeddings + FAISS: %s", exc)
        print(f"⚠️ Skipped embeddings/FAISS (error: {exc})")


# ─────────────────────────────────────────────
# LOCAL CALLBACK SERVER
# ─────────────────────────────────────────────
auth_code: Optional[str] = None


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # type: ignore
        global auth_code

        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            auth_code = params["code"][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b"<h1>Spotify login successful!</h1>You may close this window."
            )
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<h1>Error: No authorization code</h1>")


def wait_for_redirect():
    server = HTTPServer(("localhost", 8888), CallbackHandler)
    print("Waiting for Spotify redirect on http://localhost:8888/callback ...")
    server.handle_request()
    server.server_close()


# ─────────────────────────────────────────────
# TOKEN MANAGEMENT
# ─────────────────────────────────────────────
def exchange_code_for_token(code: str, verifier: str) -> Dict[str, Any]:
    url = "https://accounts.spotify.com/api/token"
    payload = {
        "client_id": CLIENT_ID,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
    }

    res = requests.post(url, data=payload)
    data = res.json()

    if "error" in data:
        raise Exception(f"Token exchange failed: {data}")

    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


def refresh_token(refresh_token_value: str) -> Dict[str, Any]:
    url = "https://accounts.spotify.com/api/token"
    payload = {
        "client_id": CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token_value,
    }

    res = requests.post(url, data=payload)
    data = res.json()

    if "error" in data:
        raise Exception(f"Refresh failed: {data}")

    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


# ─────────────────────────────────────────────
# FULL PKCE OAUTH FLOW
# ─────────────────────────────────────────────
def perform_pkce_authentication() -> Dict[str, Any]:
    global auth_code

    verifier = generate_code_verifier()
    challenge = generate_code_challenge(verifier)

    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "scope": SCOPE,
    }

    auth_url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params)

    print("Opening browser for login...")
    webbrowser.open(auth_url)

    wait_for_redirect()

    if not auth_code:
        raise Exception("No authorization code received!")

    print("Authorization code received. Exchanging for tokens...")
    return exchange_code_for_token(auth_code, verifier)


def get_valid_access_token() -> str:
    if not os.path.exists(TOKEN_FILE):
        print("No token found — starting PKCE flow...")
        token_data = perform_pkce_authentication()
        return token_data["access_token"]

    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        token_data = json.load(f)

    refresh_value = token_data.get("refresh_token")
    if not refresh_value:
        print("Token missing refresh token — re-authenticating...")
        token_data = perform_pkce_authentication()
        return token_data["access_token"]

    print("Refreshing token silently...")
    token_data = refresh_token(refresh_value)
    return token_data["access_token"]


def get_spotify_client() -> SpotifyAPI:
    """
    Build a spotipy client with internal retries disabled and wrap it
    in our SpotifyAPI so all rate-limit logic is under our control.
    """
    access_token = get_valid_access_token()
    raw_client = spotipy.Spotify(
        auth=access_token,
        retries=0,          # disable spotipy's long internal retries
        status_retries=0,
        backoff_factor=0,
        requests_timeout=10,
    )
    return SpotifyAPI(raw_client)


# ─────────────────────────────────────────────
# EMBEDDINGS CLIENT (HYBRID: AZURE → OPENAI → LOCAL)
# ─────────────────────────────────────────────
class EmbeddingClient:
    def __init__(self):
        self.mode: Optional[str] = None
        self.azure_client = None
        self.azure_deployment: Optional[str] = None
        self.openai_client = None
        self.openai_model: Optional[str] = None
        self.local_model = None

        # Try Azure first
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        if AzureOpenAI and azure_endpoint and azure_key and azure_deployment:
            try:
                self.azure_client = AzureOpenAI(
                    api_key=azure_key,
                    api_version=azure_api_version,
                    azure_endpoint=azure_endpoint,
                )
                self.azure_deployment = azure_deployment
                self.mode = "azure"
                logging.info("EmbeddingClient: using Azure OpenAI embeddings.")
                return
            except Exception as exc:  # pragma: no cover
                logging.warning("Failed to init Azure OpenAI embeddings: %s", exc)

        # Then try OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        if OpenAI and openai_key:
            try:
                self.openai_client = OpenAI(api_key=openai_key)
                self.openai_model = openai_model
                self.mode = "openai"
                logging.info("EmbeddingClient: using OpenAI embeddings.")
                return
            except Exception as exc:  # pragma: no cover
                logging.warning("Failed to init OpenAI embeddings: %s", exc)

        # Finally try local sentence-transformers
        if SentenceTransformer:
            try:
                local_model_name = os.getenv(
                    "LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
                )
                self.local_model = SentenceTransformer(local_model_name)
                self.mode = "local"
                logging.info(
                    "EmbeddingClient: using local SentenceTransformer (%s).",
                    local_model_name,
                )
                return
            except Exception as exc:  # pragma: no cover
                logging.warning("Failed to init local embeddings: %s", exc)

        logging.warning("EmbeddingClient: no embedding backend available.")
        self.mode = None

    @property
    def available(self) -> bool:
        return self.mode is not None

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        texts_list = list(texts)
        if not texts_list:
            return []

        if self.mode == "azure" and self.azure_client and self.azure_deployment:
            resp = self.azure_client.embeddings.create(
                input=texts_list,
                model=self.azure_deployment,
            )
            return [d.embedding for d in resp.data]

        if self.mode == "openai" and self.openai_client and self.openai_model:
            resp = self.openai_client.embeddings.create(
                input=texts_list,
                model=self.openai_model,
            )
            return [d.embedding for d in resp.data]

        if self.mode == "local" and self.local_model:
            vectors = self.local_model.encode(texts_list, convert_to_numpy=False)
            # ensure list[list[float]]
            return [list(map(float, v)) for v in vectors]

        raise RuntimeError("EmbeddingClient.embed called but no backend is active.")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_candidate_strings(base_queries: List[str]) -> List[str]:
    """Heuristic expansion candidates around your base seeds."""
    candidates: Set[str] = set()

    generic_suffixes = [
        "music",
        "playlist",
        "mix",
        "songs",
        "hits",
        "vibes",
        "radio",
        "2025",
        "instrumental",
    ]

    for q in base_queries:
        q_clean = q.strip()
        if not q_clean:
            continue

        tokens = q_clean.split()
        # Use last token as "tag" (e.g. 'chill', 'beats', 'piano', 'jazz', 'rock')
        tag = tokens[-1].lower()

        # Basic variations
        candidates.add(tag)
        candidates.add(f"{tag} music")
        candidates.add(f"{tag} playlist")
        candidates.add(f"{tag} mix")
        candidates.add(f"{tag} songs")
        candidates.add(f"{tag} vibes")

        # Attach generic suffixes to the full phrase
        for suf in generic_suffixes:
            candidates.add(f"{q_clean} {suf}")

    # Add some global genre-ish candidates for more variety
    global_candidates = [
        "focus music",
        "deep focus",
        "coding lofi",
        "chillhop beats",
        "jazz lounge",
        "coffee shop jazz",
        "soft piano",
        "instrumental study",
        "sleep lofi",
        "chill edm",
    ]
    candidates.update(global_candidates)

    # Remove the original base queries from the pool
    lowered_base = {q.lower().strip() for q in base_queries}
    filtered = [
        c for c in candidates if c.lower().strip() not in lowered_base and c.strip()
    ]
    return filtered


def expand_search_queries_with_embeddings(
    base_queries: List[str],
    max_total: int,
    cache_file: str,
) -> List[str]:
    """Returns a list of queries: base + embedding-selected candidates."""
    # If cached, just use that
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list) and cached:
                logging.info(
                    "Using cached expanded search queries from %s", cache_file
                )
                return cached
        except Exception as exc:  # pragma: no cover
            logging.warning("Failed to read expansion cache: %s", exc)

    base = []
    seen_lower = set()
    for q in base_queries:
        q_clean = q.strip()
        if q_clean and q_clean.lower() not in seen_lower:
            base.append(q_clean)
            seen_lower.add(q_clean.lower())

    if not USE_EMBEDDING_EXPANSION or len(base) >= max_total:
        return base[:max_total]

    client = EmbeddingClient()
    if not client.available:
        logging.info("No embedding backend available; using base queries only.")
        return base[:max_total]

    candidates = build_candidate_strings(base)
    if not candidates:
        return base[:max_total]

    try:
        base_vecs = client.embed(base)
        cand_vecs = client.embed(candidates)
    except Exception as exc:  # pragma: no cover
        logging.warning("Embedding expansion failed: %s", exc)
        return base[:max_total]

    if not base_vecs or not cand_vecs:
        return base[:max_total]

    scored: List[Tuple[float, str]] = []
    for idx, c in enumerate(candidates):
        vec = cand_vecs[idx]
        sim = max(cosine_similarity(vec, bvec) for bvec in base_vecs)
        scored.append((sim, c))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[0], reverse=True)

    expanded = list(base)
    for sim, cand in scored:
        if len(expanded) >= max_total:
            break
        if cand.lower() in seen_lower:
            continue
        # Require at least some semantic similarity
        if sim < 0.65:
            continue
        expanded.append(cand)
        seen_lower.add(cand.lower())

    # Persist to cache for next runs
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(expanded, f, indent=2, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover
        logging.warning("Failed to write expansion cache: %s", exc)

    return expanded[:max_total]


def resolve_search_queries() -> List[str]:
    if not USE_EMBEDDING_EXPANSION:
        logging.info("Embedding expansion disabled; using base queries only.")
        return BASE_SEARCH_QUERIES
    return expand_search_queries_with_embeddings(
        BASE_SEARCH_QUERIES, MAX_TOTAL_QUERIES, EXPANSION_CACHE_FILE
    )


# ─────────────────────────────────────────────
# TRACK FLATTENING HELPERS (FREE-ONLY METADATA)
# ─────────────────────────────────────────────
def get_or_fetch_artist_meta(
    api: SpotifyAPI, artist_id: str, cache: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if artist_id in cache:
        return cache[artist_id]
    try:
        data = api.artist(artist_id)
        meta = {
            "artist_name": data.get("name"),
            "followers": data.get("followers", {}).get("total"),
            "popularity": data.get("popularity"),
            "genres": data.get("genres", []),
        }
        cache[artist_id] = meta
        return meta
    except Exception as exc:
        logging.warning("Error fetching artist %s: %s", artist_id, exc)
        return None


def get_or_fetch_album_meta(
    api: SpotifyAPI, album_id: str, cache: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if album_id in cache:
        return cache[album_id]
    try:
        data = api.album(album_id)
        meta = {
            "album_name": data.get("name"),
            "label": data.get("label"),
            "total_tracks": data.get("total_tracks"),
            "release_date": data.get("release_date"),
            "album_popularity": data.get("popularity"),
        }
        cache[album_id] = meta
        return meta
    except Exception as exc:
        logging.warning("Error fetching album %s: %s", album_id, exc)
        return None


def build_track_record(
    api: SpotifyAPI,
    track: Dict[str, Any],
    source_query: str,
    source_type: str,
    artist_cache: Dict[str, Dict[str, Any]],
    album_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    track_id = track.get("id")
    if not track_id:
        raise ValueError("Track has no ID")

    album = track.get("album", {}) or {}
    artists = track.get("artists", []) or []
    main_artist = artists[0] if artists else {}
    artist_id = main_artist.get("id")

    artist_meta = (
        get_or_fetch_artist_meta(api, artist_id, artist_cache) if artist_id else None
    )

    album_id = album.get("id")
    album_meta = (
        get_or_fetch_album_meta(api, album_id, album_cache) if album_id else None
    )

    images = album.get("images") or []
    image_url = images[0]["url"] if images else None

    return {
        "track_id": track_id,
        "title": track.get("name"),
        "artists": [a.get("name") for a in artists if a.get("name")],
        "artist_id": artist_id,
        "artist_meta": artist_meta,
        "album": album.get("name"),
        "album_id": album_id,
        "album_meta": album_meta,
        "release_date": album.get("release_date"),
        "duration_ms": track.get("duration_ms"),
        "popularity": track.get("popularity"),
        "preview_url": track.get("preview_url"),
        "explicit": track.get("explicit"),
        "image_url": image_url,
        "track_number": track.get("track_number"),
        "disc_number": track.get("disc_number"),
        "available_markets": track.get("available_markets"),
        "source_query": source_query,
        "source_type": source_type,  # "search" or "playlist"
    }


# ─────────────────────────────────────────────
# FETCHING TRACKS (SEARCH + PLAYLISTS)
# ─────────────────────────────────────────────
def fetch_tracks_from_search(
    api: SpotifyAPI,
    query: str,
    seen_ids: Set[str],
    artist_cache: Dict[str, Dict[str, Any]],
    album_cache: Dict[str, Dict[str, Any]],
    limit: int = SEARCH_TRACK_LIMIT,
) -> List[Dict[str, Any]]:
    logging.info("Searching tracks for query: %s", query)
    try:
        results = api.search_tracks(query=query, limit=limit)
    except Exception as exc:
        logging.warning("Search failed for '%s': %s", query, exc)
        return []

    tracks = results.get("tracks", {}).get("items", []) or []
    records: List[Dict[str, Any]] = []

    for t in tracks:
        tid = t.get("id")
        if not tid or tid in seen_ids:
            continue
        try:
            record = build_track_record(
                api,
                t,
                source_query=query,
                source_type="search",
                artist_cache=artist_cache,
                album_cache=album_cache,
            )
            records.append(record)
            seen_ids.add(tid)
        except Exception as exc:
            logging.warning(
                "Error building search track record (%s) for query '%s': %s",
                tid,
                query,
                exc,
            )

    return records


def fetch_tracks_from_playlists(
    api: SpotifyAPI,
    query: str,
    seen_ids: Set[str],
    artist_cache: Dict[str, Dict[str, Any]],
    album_cache: Dict[str, Dict[str, Any]],
    playlist_limit: int = SEARCH_PLAYLIST_LIMIT,
    tracks_per_playlist: int = PLAYLIST_TRACKS_LIMIT,
) -> List[Dict[str, Any]]:
    logging.info("Searching playlists for query: %s", query)
    try:
        results = api.search_playlists(query=query, limit=playlist_limit)
    except Exception as exc:
        logging.warning("Playlist search failed for '%s': %s", query, exc)
        return []

    playlists = results.get("playlists", {}).get("items", []) or []
    all_records: List[Dict[str, Any]] = []

    for pl in playlists:
        if not pl or not isinstance(pl, dict):
            logging.warning("Encountered invalid playlist item: %s", pl)
            continue

        pid = pl.get("id")
        pname = pl.get("name")

        if not pid:
            logging.warning("Playlist missing ID: %s", pl)
            continue

        logging.info("Fetching tracks from playlist '%s' (%s)", pname, pid)
        try:
            tracks_resp = api.playlist_tracks(
                playlist_id=pid,
                limit=tracks_per_playlist,
            )
        except Exception as exc:
            logging.warning("Failed to fetch playlist %s (%s): %s", pname, pid, exc)
            continue

        items = tracks_resp.get("items", []) or []
        for item in items:
            track = item.get("track") or {}
            tid = track.get("id")
            if not tid or tid in seen_ids:
                continue

            try:
                record = build_track_record(
                    api,
                    track,
                    source_query=query,
                    source_type="playlist",
                    artist_cache=artist_cache,
                    album_cache=album_cache,
                )
                all_records.append(record)
                seen_ids.add(tid)
            except Exception as exc:
                logging.warning(
                    "Error building playlist track record (%s) in playlist '%s': %s",
                    tid,
                    pname,
                    exc,
                )

    return all_records


# ─────────────────────────────────────────────
# MAIN INGESTION JOB
# ─────────────────────────────────────────────
def run_ingestion_job() -> int:
    logging.info("Starting Spotify ingestion job...")
    print("Authenticating with Spotify...")

    api = get_spotify_client()

    # Get base + embedding-expanded queries
    effective_queries = resolve_search_queries()
    print("Using search queries:")
    for q in effective_queries:
        print(f"  • {q}")
    logging.info("Effective search queries: %s", effective_queries)

    all_tracks: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    artist_cache: Dict[str, Dict[str, Any]] = {}
    album_cache: Dict[str, Dict[str, Any]] = {}

    for query in effective_queries:
        # 1) Tracks directly from search
        search_records = fetch_tracks_from_search(
            api,
            query=query,
            seen_ids=seen_ids,
            artist_cache=artist_cache,
            album_cache=album_cache,
        )
        all_tracks.extend(search_records)

        # 2) Tracks from playlists matching this query
        playlist_records = fetch_tracks_from_playlists(
            api,
            query=query,
            seen_ids=seen_ids,
            artist_cache=artist_cache,
            album_cache=album_cache,
        )
        all_tracks.extend(playlist_records)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file = os.path.join(DATA_DIR, f"songs_backup_{timestamp}.json")

    with open(SONGS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_tracks, f, indent=2, ensure_ascii=False)

    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump(all_tracks, f, indent=2, ensure_ascii=False)

    print(f"Ingestion complete: {len(all_tracks)} tracks saved.")
    logging.info("Ingestion complete: %d tracks saved.", len(all_tracks))

    # ─────────────────────────────────────────────
    #  Build parquet files for recommender
    # ─────────────────────────────────────────────
    try:
        persist_items_parquet(all_tracks)
    except Exception as exc:
        logging.exception("Failed to save parquet files: %s", exc)
        print("Could not save parquet files.")

    # ─────────────────────────────────────────────
    #  Build embeddings + FAISS index
    # ─────────────────────────────────────────────
    try:
        build_embeddings_and_faiss()
    except Exception as exc:
        logging.exception("Embedding/FAISS build failed: %s", exc)

    return len(all_tracks)


# ─────────────────────────────────────────────
# OPTIONAL DAILY SCHEDULER
# ─────────────────────────────────────────────
def schedule_daily(hour: str = "03:00"):
    schedule.every().day.at(hour).do(run_ingestion_job)
    print(f"Scheduled daily ingestion at {hour}")

    while True:
        schedule.run_pending()
        time.sleep(60)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_ingestion_job()
    # To enable daily scheduling instead:
    # schedule_daily("03:00")

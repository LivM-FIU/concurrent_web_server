"""
data_ingestion.py  
Spotify Ingestion + Azure Embedding Pipeline  
Author: Livan Miranda  

This script:

  • Authenticates to Spotify using OAuth2 + PKCE  
  • Expands search queries  
  • Fetches tracks via search + playlists  
  • Stores free metadata only (no audio features)  
  • Builds items.parquet, item_text.parquet  
  • Builds Azure OpenAI embeddings  
  • Builds FAISS index  
"""

import os
import sys
import json
import math
import time
import random
import base64
import hashlib
import logging
import webbrowser
import urllib.parse
import requests
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Any, Optional, Iterable, Tuple, Set
from pathlib import Path

import spotipy
from spotipy.exceptions import SpotifyException
import pandas as pd

# project root for model_utils import
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from config.settings import settings
from llm_recommender import model_utils

# ─────────────────────────────────────────────
# DIRECTORIES
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

TOKEN_FILE = DATA_DIR / "token.json"
SONGS_FILE = DATA_DIR / "songs.json"
LOG_FILE = DATA_DIR / "ingestion.log"
EXPANSION_CACHE_FILE = DATA_DIR / "query_expansion_cache.json"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLIENT_ID = settings.SPOTIFY_CLIENT_ID
REDIRECT_URI = settings.SPOTIFY_REDIRECT_URI

BASE_SEARCH_QUERIES = [
    "lofi chill",
    "study beats",
    "piano instrumental",
    "jazz coffee",
    "workout motivation",
    "ambient focus",
    "pop hits",
    "classic rock",
    "hip hop",
    "r&b vibes",
    "house edm",
    "deep house",
    "latin",
    "reggaeton",
    "salsa",
    "brazilian",
    "indie",
    "alt rock",
    "metal",
    "orchestral",
    "trap",
    "afrobeats",
    "meditation",
    "sleep",
    "gaming music"
]

USE_EMBEDDING_EXPANSION = True
MAX_TOTAL_QUERIES = 24

SEARCH_TRACK_LIMIT = 30
SEARCH_PLAYLIST_LIMIT = 20
PLAYLIST_TRACKS_LIMIT = 100

# Azure embedding configuration
AZURE_ENDPOINT = settings.AZURE_OPENAI_ENDPOINT
AZURE_KEY = settings.AZURE_OPENAI_API_KEY
AZURE_EMB_DEPLOY = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
AZURE_API_VERSION = settings.AZURE_OPENAI_API_VERSION

from openai import AzureOpenAI
azure_client = AzureOpenAI(
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION
)

# ─────────────────────────────────────────────
# PKCE UTILITIES (UNCHANGED)
# ─────────────────────────────────────────────
def generate_code_verifier(n=96):
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~"
    return "".join(random.choice(chars) for _ in range(n))

def generate_code_challenge(verifier: str):
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")

auth_code: Optional[str] = None

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if "code" in params:
            auth_code = params["code"][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Spotify login successful. You may close this window.")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing auth code")

def wait_for_redirect():
    server = HTTPServer(("localhost", 8888), CallbackHandler)
    print("Waiting for Spotify redirect on http://localhost:8888/callback ...")
    server.handle_request()
    server.server_close()

def exchange_code_for_token(code: str, verifier: str):
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

    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data

def refresh_token(refresh_token: str):
    url = "https://accounts.spotify.com/api/token"
    payload = {
        "client_id": CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    res = requests.post(url, data=payload)
    data = res.json()

    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data

def get_valid_access_token():
    if not TOKEN_FILE.exists():
        verifier = generate_code_verifier()
        challenge = generate_code_challenge(verifier)

        params = {
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
            "scope": "user-read-private",
        }

        url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params)
        webbrowser.open(url)
        wait_for_redirect()

        return exchange_code_for_token(auth_code, verifier)["access_token"]

    data = json.loads(TOKEN_FILE.read_text())
    if "refresh_token" not in data:
        return get_valid_access_token()

    refreshed = refresh_token(data["refresh_token"])
    return refreshed["access_token"]

# ─────────────────────────────────────────────
# SPOTIFY API (UNCHANGED)
# ─────────────────────────────────────────────
def spotify_call(fn, *args, **kwargs):
    max_retries = 6
    backoff = 2

    for _ in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except SpotifyException as e:
            if e.http_status == 429:
                wait = max(3, int(e.headers.get("Retry-After", backoff)))
                time.sleep(wait)
                backoff = min(backoff * 2, 120)
                continue
            raise
    raise RuntimeError("Spotify API failed too many times")

class SpotifyAPI:
    def __init__(self):
        token = get_valid_access_token()
        self.client = spotipy.Spotify(auth=token, retries=0, backoff_factor=0)

    def search_tracks(self, q, limit=20):
        return spotify_call(self.client.search, q=q, type="track", limit=limit)

    def search_playlists(self, q, limit=5):
        return spotify_call(self.client.search, q=q, type="playlist", limit=limit)

    def playlist_tracks(self, pid, limit=100):
        return spotify_call(self.client.playlist_tracks, playlist_id=pid, limit=limit)

    def artist(self, artist_id):
        return spotify_call(self.client.artist, artist_id)

    def album(self, album_id):
        return spotify_call(self.client.album, album_id)

# ─────────────────────────────────────────────
# QUERY EXPANSION (UNCHANGED)
# ─────────────────────────────────────────────
def azure_embed_texts(texts: List[str]) -> List[List[float]]:
    resp = azure_client.embeddings.create(
        model=AZURE_EMB_DEPLOY,
        input=texts
    )
    return [d.embedding for d in resp.data]

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def expand_queries(base: List[str]) -> List[str]:
    if not USE_EMBEDDING_EXPANSION:
        return base[:MAX_TOTAL_QUERIES]

    if EXPANSION_CACHE_FILE.exists():
        return json.loads(EXPANSION_CACHE_FILE.read_text())

    candidates = []
    for q in base:
        tag = q.split()[-1]
        candidates.extend([
            f"{tag} music", f"{tag} playlist", f"{tag} vibes",
            f"{q} instrumental", f"{q} mix"
        ])

    base_vecs = azure_embed_texts(base)
    cand_vecs = azure_embed_texts(candidates)

    scored = []
    for txt, vec in zip(candidates, cand_vecs):
        sim = max(cosine_similarity(vec, bv) for bv in base_vecs)
        if sim >= 0.65:
            scored.append((sim, txt))

    scored.sort(reverse=True)
    expanded = base + [t for _, t in scored][:MAX_TOTAL_QUERIES - len(base)]

    EXPANSION_CACHE_FILE.write_text(json.dumps(expanded, indent=2))
    return expanded

# ─────────────────────────────────────────────
# FIXED — TRACK RECORD CREATION
# ─────────────────────────────────────────────
def build_track_record(api, track, query, source_type):
    if not track:
        return None

    album = track.get("album", {}) or {}
    artists = track.get("artists", []) or []
    image_url = None

    if isinstance(album.get("images"), list) and album["images"]:
        image_url = album["images"][0].get("url")

    main_artist = artists[0] if artists else {}

    return {
        "track_id": track.get("id"),
        "title": track.get("name") or "",     # <── ALWAYS ENSURE TITLE EXISTS
        "artists": [a.get("name") for a in artists if a],
        "artist_id": main_artist.get("id"),
        "album": album.get("name"),
        "album_id": album.get("id"),
        "release_date": album.get("release_date"),
        "popularity": track.get("popularity"),
        "image_url": image_url,
        "preview_url": track.get("preview_url"),
        "source_query": query,
        "source_type": source_type,
    }

# ─────────────────────────────────────────────
# FETCH TRACKS (UNCHANGED)
# ─────────────────────────────────────────────
def fetch_from_search(api, query, seen):
    results = api.search_tracks(query, SEARCH_TRACK_LIMIT) or {}
    tracks_block = results.get("tracks") or {}
    items = tracks_block.get("items") or []

    out = []
    for t in items:
        if not t:
            continue
        tid = t.get("id")
        if tid and tid not in seen:
            seen.add(tid)
            rec = build_track_record(api, t, query, "search")
            if rec:
                out.append(rec)
    return out

def fetch_from_playlists(api, query, seen):
    results = api.search_playlists(query, SEARCH_PLAYLIST_LIMIT) or {}
    playlists_block = results.get("playlists") or {}
    playlists = playlists_block.get("items") or []

    out = []
    for pl in playlists:
        if not pl:
            continue

        pid = pl.get("id")
        if not pid:
            continue

        playlist_items = api.playlist_tracks(pid, PLAYLIST_TRACKS_LIMIT)
        if not playlist_items:
            continue

        items = playlist_items.get("items") or []

        for item in items:
            if not item:
                continue

            track = item.get("track") or {}
            if not track:
                continue

            tid = track.get("id")
            if tid and tid not in seen:
                seen.add(tid)
                rec = build_track_record(api, track, query, "playlist")
                if rec:
                    out.append(rec)

    return out

# ─────────────────────────────────────────────
# FIXED — PARQUET SAVING
# ─────────────────────────────────────────────
def save_parquet(tracks):
    items = []
    texts = []

    for t in tracks:
        title = t.get("title") or ""
        artists = t.get("artists") or []
        artist = artists[0] if artists else ""

        items.append({
            "track_id": t.get("track_id"),
            "title": title,
            "artist": artist,
            "album": t.get("album") or "",
            "release_date": t.get("release_date") or "",
            "popularity": t.get("popularity") or 0,
            "image_url": t.get("image_url"),
            "source_type": t.get("source_type"),
        })

        texts.append({
            "track_id": t.get("track_id"),
            "title": title,
            "artist": artist,
            "genres": "",
            "tags": "",
            "description": "",
        })

    pd.DataFrame(items).to_parquet(DATA_DIR / "items.parquet", index=False)
    pd.DataFrame(texts).to_parquet(DATA_DIR / "item_text.parquet", index=False)


# ─────────────────────────────────────────────
# MAIN INGESTION (UNCHANGED)
# ─────────────────────────────────────────────
def run_ingestion_job():
    api = SpotifyAPI()

    queries = expand_queries(BASE_SEARCH_QUERIES)

    all_tracks = []
    seen = set()

    for q in queries:
        all_tracks.extend(fetch_from_search(api, q, seen))
        all_tracks.extend(fetch_from_playlists(api, q, seen))

    SONGS_FILE.write_text(json.dumps(all_tracks, indent=2))
    save_parquet(all_tracks)

    emb_art = model_utils.build_item_embeddings()
    model_utils.build_faiss_index(emb_art)

    print(f"Ingestion complete: {len(all_tracks)} tracks")

if __name__ == "__main__":
    run_ingestion_job()

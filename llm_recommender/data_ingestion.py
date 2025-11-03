"""
Spotify Data Ingestion Job
Author: Livan Miranda

Description:
  Pulls track metadata and audio features from Spotify’s free API,
  builds a local songs.json dataset for the recommender,
  and optionally schedules itself to run daily.

Requirements:
  pip install spotipy schedule

Usage:
  python llm_recommender/data_ingestion.py
"""

import os
import json
from datetime import datetime
import time
import logging

import schedule
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
# Spotify credentials (from https://developer.spotify.com/dashboard)
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "your_spotify_client_id")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "your_spotify_client_secret")

# Queries to pull (genres/moods you want in your catalog)
SEARCH_QUERIES = [
    "lofi chill",
    "study beats",
    "piano instrumental",
    "jazz coffee",
    "workout motivation",
    "ambient focus",
    "pop hits",
    "classic rock",
]

# Storage paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
SONGS_FILE = os.path.join(DATA_DIR, "songs.json")

# Logging
LOG_FILE = os.path.join(DATA_DIR, "ingestion.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ─────────────────────────────────────────────
#  INGESTION LOGIC
# ─────────────────────────────────────────────
def fetch_tracks(sp, query, limit=20):
    """Fetch metadata + audio features for a given query."""
    results = sp.search(q=query, limit=limit, type="track")
    tracks = []
    for item in results["tracks"]["items"]:
        try:
            features = sp.audio_features(item["id"])[0]
            tracks.append(
                {
                    "id": item["id"],
                    "title": item["name"],
                    "artist": item["artists"][0]["name"],
                    "album": item["album"]["name"],
                    "genre_query": query,
                    "release_year": item["album"]["release_date"][:4],
                    "popularity": item["popularity"],
                    "energy": features["energy"],
                    "tempo": features["tempo"],
                    "valence": features["valence"],
                    "instrumentalness": features["instrumentalness"],
                    "danceability": features["danceability"],
                    "duration_ms": features["duration_ms"],
                }
            )
        except Exception as exc:  # pragma: no cover - Spotify API errors are external
            logging.warning("Error fetching features for %s: %s", item["name"], exc)
    return tracks


def run_ingestion_job():
    """Main job: authenticate, fetch tracks, save JSON and backup."""
    logging.info("Starting Spotify ingestion job...")
    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
    )

    all_tracks = []
    for query in SEARCH_QUERIES:
        logging.info("Fetching query: %s", query)
        all_tracks.extend(fetch_tracks(sp, query, limit=20))

    # Save main file + timestamped backup
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file = os.path.join(DATA_DIR, f"songs_backup_{timestamp}.json")

    with open(SONGS_FILE, "w", encoding="utf-8") as file:
        json.dump(all_tracks, file, indent=2, ensure_ascii=False)
    with open(backup_file, "w", encoding="utf-8") as file:
        json.dump(all_tracks, file, indent=2, ensure_ascii=False)

    logging.info("Saved %d tracks to %s", len(all_tracks), SONGS_FILE)
    print(f"✅ Ingestion complete: {len(all_tracks)} tracks saved.")
    return len(all_tracks)


# ─────────────────────────────────────────────
#  OPTIONAL: DAILY SCHEDULER
# ─────────────────────────────────────────────
def schedule_daily(hour="03:00"):
    """Run the ingestion job every day at the given hour (24h format)."""
    logging.info("Scheduling daily ingestion at %s", hour)
    schedule.every().day.at(hour).do(run_ingestion_job)
    print(f"🕒 Scheduled daily ingestion at {hour}")
    while True:
        schedule.run_pending()
        time.sleep(60)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Option 1: Run once
    run_ingestion_job()

    # Option 2: Enable daily scheduler
    # schedule_daily("03:00")

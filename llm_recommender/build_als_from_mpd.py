"""
Build ALS joblib model from Spotify MPD (Million Playlist Dataset)
Author: Livan Miranda

Steps:
1. Read MPD slices
2. Convert playlist → user interactions
3. Save interactions.parquet
4. Train PURE ALS → produces als.joblib
"""

import json, os, sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from llm_recommender.model_utils import (
    build_cf_model,     # ← FIXED
    ensure_directory,
    DATA_DIR,
)

# CHANGE THIS — your MPD slices directory
MPD_DIR = Path(r"C:\Users\L\Desktop\FIU classes\spotify\data")

OUTPUT_DIR = DATA_DIR
ensure_directory(OUTPUT_DIR)


def load_slice(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["playlists"]


def build_interactions(num_slices=50):
    """
    Convert MPD playlists → user interactions.
    """
    print(f"Building interactions from {num_slices} MPD slices...")

    records = []

    for i in range(num_slices):
        slice_path = MPD_DIR / f"mpd.slice.{i*1000}-{i*1000+999}.json"
        if not slice_path.exists():
            print("Missing:", slice_path)
            continue

        playlists = load_slice(slice_path)

        for pl in playlists:
            user_id = f"mpd_user_{pl['pid']}"

            for track in pl["tracks"]:
                track_id = track["track_uri"].replace("spotify:track:", "")
                records.append({
                    "user_id": user_id,
                    "track_id": track_id,
                    "strength": 1.0,
                })

    df = pd.DataFrame(records)
    out_path = OUTPUT_DIR / "interactions.parquet"
    df.to_parquet(out_path, index=False)

    print("✔ interactions.parquet saved:", out_path)
    print("Rows:", len(df))
    return out_path


def main():
    # Start small; increase after testing
    interactions_path = build_interactions(num_slices=150)

    print("\nTraining PURE ALS...")
    als_path = build_cf_model(
        interactions_path,
        DATA_DIR.parent / "artifacts" / "als.joblib",
        factors=64,
        iterations=8,
    )

    print("ALS model saved:", als_path)


if __name__ == "__main__":
    main()

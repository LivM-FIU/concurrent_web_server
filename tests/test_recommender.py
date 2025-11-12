import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_recommender import recommender


def test_parse_nl_extracts_structured_ranges_and_artists():
    intent = recommender.parse_nl(
        "Upbeat 120 bpm synthwave instrumentals from the 80s like Daft Punk"
    )
    assert "synthwave" in intent.genres
    assert intent.avoid_vocals is True
    assert intent.tempo_bpm.min == 110.0
    assert intent.tempo_bpm.max == 130.0
    assert intent.era.from_year == 1980 and intent.era.to_year == 1989
    assert any("daft punk" in artist for artist in intent.include_artists)


def test_recommendation_engine_falls_back_without_retriever(monkeypatch):
    engine = recommender.RecommendationEngine()

    monkeypatch.setattr(engine, "_load_retriever", lambda: None)

    meta_called = False

    def fake_meta_loader():
        nonlocal meta_called
        meta_called = True
        return None

    monkeypatch.setattr(engine, "_load_meta", fake_meta_loader)

    result = engine.recommend(prompt="morning focus", user_id="user-123")

    assert result["count"] == len(result["recommendations"]) == 3
    assert any(item["track_id"] == "mix_daily" for item in result["recommendations"])
    assert meta_called is False
    assert json.loads(json.dumps(result["recommendations"]))


def test_filter_candidates_respects_intent_constraints(monkeypatch):
    engine = recommender.RecommendationEngine()
    meta = pd.DataFrame(
        [
            {
                "track_id": "t1",
                "genres": "lofi;jazz",
                "vocals": False,
                "tempo": 82,
                "energy": 0.25,
                "artist": "Artist One",
                "release_year": 2021,
            },
            {
                "track_id": "t2",
                "genres": "rock",
                "vocals": True,
                "tempo": 140,
                "energy": 0.8,
                "artist": "Artist Two",
                "release_year": 2018,
            },
        ]
    ).set_index("track_id")

    intent = recommender.Intent(
        genres=["lofi"],
        avoid_vocals=True,
        tempo_bpm=recommender.Range(min=60, max=100),
        energy=recommender.Range(max=0.4),
    )

    cf = recommender.RetrievalResult(np.array(["t1", "t2"]), np.array([0.9, 0.8]))
    sem = recommender.RetrievalResult(np.array(["t2", "t3"]), np.array([0.7, 0.6]))

    filtered_cf, filtered_sem = engine._filter_candidates(intent, cf, sem, meta)

    assert list(filtered_cf.ids) == ["t1"]
    # t2 should be filtered from semantic results, t3 has no metadata so it remains
    assert list(filtered_sem.ids) == ["t3"]


def test_recommendation_engine_caches_cf_only_results(monkeypatch):
    engine = recommender.RecommendationEngine()

    class DummyRetriever:
        def __init__(self):
            self.cf_calls = 0
            self.sem_calls = 0

        def by_cf(self, user_id, k=200):
            self.cf_calls += 1
            return recommender.RetrievalResult(np.array(["track-x"]), np.array([0.95]))

        def by_nlq(self, text, k=200):
            self.sem_calls += 1
            return recommender.RetrievalResult(np.array(["track-y"]), np.array([0.85]))

    dummy = DummyRetriever()
    meta = pd.DataFrame(
        [
            {
                "track_id": "track-x",
                "title": "Song X",
                "artist": "Artist A",
                "freshness": 1.0,
            },
            {
                "track_id": "track-y",
                "title": "Song Y",
                "artist": "Artist B",
                "freshness": 1.0,
            },
        ]
    ).set_index("track_id")

    monkeypatch.setattr(engine, "_load_retriever", lambda: dummy)
    monkeypatch.setattr(engine, "_load_meta", lambda: meta)

    result_first = engine.recommend(prompt="", user_id="user-1")
    assert result_first["count"] >= 1
    assert dummy.cf_calls == 1

    result_cached = engine.recommend(prompt="", user_id="user-1")
    assert result_cached == result_first
    assert dummy.cf_calls == 1  # cache hit

    result_refresh = engine.recommend(prompt="", user_id="user-1", refresh_cache=True)
    assert result_refresh["count"] >= 1
    assert dummy.cf_calls == 2

    engine.recommend(prompt="need energetic tracks", user_id="user-1")
    assert dummy.sem_calls == 1

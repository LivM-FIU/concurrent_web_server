import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_recommender import recommender


def test_parse_nl_detects_mood_and_vocals():
    intent = recommender.parse_nl("Chill piano study mix, no vocals please")
    assert "chill" in intent.moods
    assert intent.avoid_vocals is True


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

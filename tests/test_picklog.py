"""The pick log: append, deduplicate, update."""

import pytest

from data import picklog


@pytest.fixture(autouse=True)
def temp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(picklog, "PICKS_DIR", tmp_path)
    monkeypatch.setattr(picklog, "PICKS_FILE", tmp_path / "picks.jsonl")


def _pick(outcome="home"):
    return {
        "match": "Arsenal v Chelsea",
        "kickoff": "2030-01-01T15:00:00Z",
        "selection": "Arsenal win",
        "outcome": outcome,
        "model_prob": 0.62,
        "fair_prob": 0.55,
        "edge": 0.07,
        "price": 1.75,
        "price_book": "Betway",
        "best_price": 1.80,
    }


def test_record_and_load():
    assert picklog.record("PL", [_pick()])["added"] == 1
    records = picklog.load_all()
    assert len(records) == 1
    assert records[0]["status"] == "open"
    assert records[0]["price_taken"] == 1.75


def test_same_pick_is_not_logged_twice():
    picklog.record("PL", [_pick()])
    second = picklog.record("PL", [_pick()])
    assert second["added"] == 0
    assert second["skipped"] == 1
    assert len(picklog.load_all()) == 1


def test_different_selections_on_one_fixture_both_log():
    picklog.record("PL", [_pick("home"), _pick("draw")])
    assert len(picklog.load_all()) == 2


def test_update_settles_a_pick():
    picklog.record("PL", [_pick()])
    pick_id = picklog.load_all()[0]["id"]
    picklog.update(pick_id, status="settled", profit_units=0.75)
    settled = picklog.load_all()[0]
    assert settled["status"] == "settled"
    assert settled["profit_units"] == 0.75


def test_missing_log_reads_as_empty():
    assert picklog.load_all() == []

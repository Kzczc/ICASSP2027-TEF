import pytest

from tef.data import append_jsonl
from tef.pipeline import cache_path, dimension_table, load_records, score_records


def _write_cache(tmp_path):
    path = cache_path(tmp_path, "demo", "original", "en")
    append_jsonl(path, [
        {"post_id": "p1", "lang": "en", "dimension": "Culture", "side": "change", "direct": 0.4, "sentences": [0.9, 0.35, 0.35, 0.35, 0.35]},
        {"post_id": "p2", "lang": "en", "dimension": "Culture", "side": "stability", "direct": 0.3, "sentences": [0.2, 0.6]},
        {"post_id": "p3", "lang": "en", "dimension": "Economy", "side": "change", "direct": 0.7, "sentences": [0.7]},
    ])
    return path


def test_scores_from_cache(tmp_path):
    records = load_records(_write_cache(tmp_path))
    assert len(records) == 3
    assert score_records(records, "tef").accuracy == pytest.approx(100.0)
    assert score_records(records, "majority_vote").accuracy == pytest.approx(200 / 3)
    assert score_records(records, "direct").accuracy == pytest.approx(200 / 3)


def test_dimension_table_order_and_average(tmp_path):
    records = load_records(_write_cache(tmp_path))
    table = dimension_table(records, ["tef", "direct"])
    assert list(table["tef"]) == ["Culture", "Economy", "Average"]
    direct = table["direct"]
    assert direct["Average"].accuracy == pytest.approx((direct["Culture"].accuracy + direct["Economy"].accuracy) / 2)


def test_missing_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_records(tmp_path / "missing.jsonl")

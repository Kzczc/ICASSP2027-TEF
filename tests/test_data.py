import json
from pathlib import Path

import pytest

from tef.data import DATA_FILES, data_dir, evaluation_items, load_posts
from tef.schema import get_dimension

RELEASED_POSTS = {"zh": 5474, "en": 2884}


def _label(dimension, subcategory, orientation, side):
    return {"dimension": dimension, "subcategory": subcategory, "orientation": orientation, "side": side}


def _write(directory, lang, rows):
    (directory / DATA_FILES[lang]).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_evaluation_items_skip_dimensions_with_both_sides(tmp_path):
    _write(tmp_path, "en", [
        {"id": "a", "lang": "en", "event": "E", "text": "text a", "labels": [
            _label("Economy", "Economic Models", "Free-Market", "change"),
            _label("Economy", "Distribution Justice", "Equality of Opportunity", "change"),
        ]},
        {"id": "b", "lang": "en", "event": "E", "text": "text b", "labels": [
            _label("Economy", "Economic Models", "Free-Market", "change"),
            _label("Economy", "Distribution Justice", "Equality of Outcome", "stability"),
            _label("Culture", "Value Orientations", "Collectivism", "stability"),
        ]},
    ])
    items = evaluation_items(load_posts("en", tmp_path))
    assert [(i.post_id, i.dimension, i.side) for i in items] == [("a", "Economy", "change"), ("b", "Culture", "stability")]


def test_invalid_label_is_rejected(tmp_path):
    _write(tmp_path, "en", [{"id": "a", "lang": "en", "event": "E", "text": "t", "labels": [
        _label("Economy", "Economic Models", "Free-Market", "left"),
    ]}])
    with pytest.raises(ValueError):
        load_posts("en", tmp_path)


def test_data_dir_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("MIND_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
    assert data_dir("elsewhere") == Path("elsewhere")
    with pytest.raises(FileNotFoundError):
        load_posts("zh")


@pytest.mark.parametrize("lang", sorted(RELEASED_POSTS))
def test_released_data_matches_schema(lang, monkeypatch):
    monkeypatch.delenv("MIND_DATA_DIR", raising=False)
    if not (data_dir() / DATA_FILES[lang]).exists():
        pytest.skip("released data not found")
    posts = load_posts(lang)
    assert len(posts) == RELEASED_POSTS[lang]
    assert len({p.id for p in posts}) == len(posts)
    for post in posts:
        assert post.lang == lang and post.text.strip() and post.labels
        for label in post.labels:
            subcategories = {s.name: s for s in get_dimension(label.dimension).subcategories}
            assert subcategories[label.subcategory].position(label.side).name == label.orientation

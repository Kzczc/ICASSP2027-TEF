"""Loading MIND and building the evaluation items."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .schema import DIMENSION_BY_NAME, SIDES

LANGUAGES: tuple[str, ...] = ("zh", "en")
DATA_FILES: dict[str, str] = {"zh": "mind_zh.jsonl", "en": "mind_en.jsonl"}
REPO_ROOT = Path(__file__).resolve().parent.parent


def data_dir(path: str | os.PathLike | None = None) -> Path:
    """Resolve the data directory: explicit argument, then ``$MIND_DATA_DIR``, then ``<repo>/data``."""
    return Path(path or os.environ.get("MIND_DATA_DIR") or REPO_ROOT / "data")


@dataclass(frozen=True)
class Label:
    dimension: str
    subcategory: str
    orientation: str
    side: str


@dataclass(frozen=True)
class Post:
    id: str
    lang: str
    event: str
    text: str
    labels: tuple[Label, ...]


@dataclass(frozen=True)
class Item:
    """One evaluation unit: a post and a dimension on which all of its labels agree."""

    post_id: str
    lang: str
    dimension: str
    side: str
    text: str


def load_posts(lang: str, directory: str | os.PathLike | None = None) -> list[Post]:
    if lang not in DATA_FILES:
        raise ValueError(f"unsupported language {lang!r}; expected one of {LANGUAGES}")
    path = data_dir(directory) / DATA_FILES[lang]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; set --data-dir or $MIND_DATA_DIR")
    posts = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            labels = tuple(Label(**label) for label in record["labels"])
            for label in labels:
                if label.dimension not in DIMENSION_BY_NAME or label.side not in SIDES:
                    raise ValueError(f"{path}:{line_no}: invalid label {label}")
            posts.append(Post(record["id"], record["lang"], record.get("event", ""), record["text"], labels))
    return posts


def evaluation_items(posts: Iterable[Post]) -> list[Item]:
    """Create one item per (post, dimension) whose labels share a single side.

    A post can carry labels from several dimensions. Pairs whose two subcategory labels point to
    different sides have no dimension-level target and are skipped.
    """
    items = []
    for post in posts:
        sides: dict[str, set[str]] = defaultdict(set)
        for label in post.labels:
            sides[label.dimension].add(label.side)
        for dimension, found in sides.items():
            if len(found) == 1:
                items.append(Item(post.id, post.lang, dimension, next(iter(found)), post.text))
    return items


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: str | os.PathLike, records: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

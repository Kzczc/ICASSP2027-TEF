# MIND: Multi-event Insight Network Dimensions

MIND is a cross-lingual benchmark for measuring public value orientations in long social media posts.
It contains **8,358 posts** (5,474 Chinese and 2,884 English) collected from January 2020 to January 2025
and annotated over **six value dimensions**.

| File | Content |
|---|---|
| `mind_zh.jsonl` | 5,474 Chinese posts, one JSON object per line |
| `mind_en.jsonl` | 2,884 English posts, one JSON object per line |
| `statistics.json` | Post, label, and evaluation-pair counts per language |

## Label schema

Each dimension has two subcategories, and each subcategory contrasts a **change-oriented (left)** position
with a **stability-oriented (right)** position.

| Dimension | Subcategory | Change-oriented (left) | Stability-oriented (right) |
|---|---|---|---|
| Politics (政治) | Policy Orientations (政策导向) | Revolutionism (革命主义) | Reformism (改良主义) |
| | Diplomatic Strategies (外交战略) | Globalism (全球主义) | Isolationism (孤立主义) |
| Economy (经济) | Economic Models (经济模式) | Free-Market (市场自由) | Interventionist (国家干预) |
| | Distribution Justice (分配正义) | Equality of Opportunity (机会平等) | Equality of Outcome (结果平等) |
| Culture (文化) | Ethical Orientations (道德取向) | Ethical Liberalism (伦理自由) | Ethical Conservatism (伦理保守) |
| | Value Orientations (社会价值) | Individualism (个人主义) | Collectivism (集体主义) |
| Society (社会) | Fairness Principles (公正理念) | Substantive Justice (结果公正) | Procedural Justice (程序正义) |
| | Judicial Philosophies (法律取向) | Progressive Jurisprudence (进步法理) | Conservative Jurisprudence (保守法理) |
| Environment (生态文明) | Sustainability Strategies (发展取向) | Conservation Priority (保护优先) | Development Priority (发展优先) |
| | Climate Stances (气候立场) | Climate Actionism (气候行动) | Climate Skepticism (气候怀疑) |
| Technology (科技) | Innovation Models (创新模式) | Inclusive Sharing (普惠共享) | IP Protection (知识产权保护) |
| | Governance Models (风险治理) | Technological Libertarianism (技术自由) | Technological Regulation (技术监管) |

The same schema is defined in code in [`tef/schema.py`](../tef/schema.py) and shown in [`assets/dimension.png`](../assets/dimension.png).
English names are used in both files; the Chinese names are the label strings of the original annotation.

## Record format

```json
{
  "id": "en-single-9",
  "lang": "en",
  "event": "COL",
  "text": "...",
  "labels": [
    {"dimension": "Economy", "subcategory": "Economic Models", "orientation": "Interventionist", "side": "stability"}
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | `<lang>-<single\|multi>-<source id>`; `single` and `multi` identify the annotation file (single-dimension or multi-dimension task) |
| `lang` | string | `zh` or `en` |
| `event` | string | Code of the public event around which the post was collected |
| `text` | string | Anonymized post text |
| `labels` | list | One entry per annotated subcategory |
| `labels[].dimension` | string | One of the six dimensions |
| `labels[].subcategory` | string | One of the two subcategories of that dimension |
| `labels[].orientation` | string | The annotated position |
| `labels[].side` | string | `change` or `stability`, derived from the schema above |

## Evaluation unit

A post can carry labels in several dimensions, and in one dimension it can carry labels for both subcategories.
The evaluation unit is a **(post, dimension) pair**: the target is the side shared by all labels of the post in that
dimension. Pairs whose two labels fall on different sides have no dimension-level target and are excluded.
This rule is implemented in `tef.data.evaluation_items`.

| | Chinese | English |
|---|---:|---:|
| Posts | 5,474 | 2,884 |
| Subcategory labels | 11,762 | 4,919 |
| Evaluation pairs | 10,330 | 4,621 |
| Excluded pairs (both sides in one dimension) | 208 | 40 |
| Mean post length | 138.2 characters | 86.8 words |
| Mean number of sentences per post* | 4.35 | 5.84 |

\* Sentences are counted with the segmentation rule of `tef.segment.split_sentences`.

Evaluation pairs per dimension (change / stability):

| Dimension | Chinese | English |
|---|---:|---:|
| Culture | 1,999 (669 / 1,330) | 1,049 (434 / 615) |
| Economy | 1,387 (672 / 715) | 1,102 (563 / 539) |
| Society | 2,176 (1,323 / 853) | 1,112 (775 / 337) |
| Politics | 2,514 (1,806 / 708) | 987 (659 / 328) |
| Technology | 1,456 (822 / 634) | 234 (107 / 127) |
| Environment | 798 (531 / 267) | 137 (106 / 31) |

Number of dimensions per post:

| Dimensions | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Chinese | 2,055 | 1,848 | 1,508 | 53 | 9 | 1 |
| English | 1,419 | 1,182 | 256 | 26 | 0 | 1 |

Labels per position (Chinese / English):

| Dimension | Subcategory | Change-oriented | Stability-oriented |
|---|---|---|---|
| Politics | Policy Orientations | Revolutionism 621 / 341 | Reformism 374 / 97 |
| | Diplomatic Strategies | Globalism 1,531 / 390 | Isolationism 477 / 272 |
| Economy | Economic Models | Free-Market 452 / 304 | Interventionist 602 / 352 |
| | Distribution Justice | Equality of Opportunity 224 / 278 | Equality of Outcome 206 / 263 |
| Culture | Ethical Orientations | Ethical Liberalism 316 / 198 | Ethical Conservatism 549 / 202 |
| | Value Orientations | Individualism 385 / 245 | Collectivism 832 / 426 |
| Society | Fairness Principles | Substantive Justice 1,152 / 693 | Procedural Justice 694 / 263 |
| | Judicial Philosophies | Progressive Jurisprudence 193 / 101 | Conservative Jurisprudence 192 / 80 |
| Environment | Sustainability Strategies | Conservation Priority 201 / 67 | Development Priority 163 / 15 |
| | Climate Stances | Climate Actionism 427 / 49 | Climate Skepticism 121 / 16 |
| Technology | Innovation Models | Inclusive Sharing 621 / 42 | IP Protection 196 / 16 |
| | Governance Models | Technological Libertarianism 631 / 88 | Technological Regulation 602 / 121 |

Posts per event code:

| Language | Event codes (posts) |
|---|---|
| Chinese | RU 1,782 · GPT 1,046 · OL 775 · SY 579 · COVID 561 · COP 541 · FED 190 |
| English | FED 677 · COP 557 · RU 426 · COVID 411 · OL 340 · GPT 279 · COL 194 |

## Construction

MIND is built in three stages (Section 4 of the paper and [`assets/construction.png`](../assets/construction.png)).

1. **Data collection.** Candidate posts are collected from X, Weibo, Reddit, Zhihu, discussion forums, and news feeds.
   Keyword search for each value dimension is anchored to high-search-volume public events, such as the
   Russia–Ukraine war and the rise of generative AI. Posts are filtered for relevance and opinion content,
   anonymized, de-duplicated, and screened by an automatic quality score.
2. **Data annotation.** Three annotation rounds label the value dimension, the subcategory, and the orientation in
   sequence. Each round uses three annotators and accepts a label when at least two agree. Disputed cases enter
   arbitration, samples without a stable majority are discarded, and accepted samples pass distribution and quality checks.
3. **Agent-driven data evolution.** A balancing agent generates synthetic samples for underrepresented categories.
   For low-confidence or ambiguous samples, a critic agent checks the alignment between text and label, and a
   semantic-edit agent revises the text so that the labeled orientation is explicit. Only verified samples are retained.

The release does not mark which posts were generated or revised in Stage III.

## Intended use and license

MIND is released for non-commercial research on value measurement, stance detection, and the aggregation of
LLM judgments. Posts are anonymized; do not attempt to re-identify authors. The labels describe the orientation
expressed in a text and must not be used to profile individuals.

The data are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The code of this
repository is licensed under the MIT License.

import pytest

from tef.schema import CHANGE, DIMENSION_BY_NAME, DIMENSIONS, STABILITY, TABLE_ORDER, get_dimension

# Dimension schema of MIND (assets/dimension.png): subcategory, change-oriented (left) position,
# stability-oriented (right) position.
EXPECTED = {
    "Politics": [
        ("Policy Orientations", "Revolutionism", "Reformism"),
        ("Diplomatic Strategies", "Globalism", "Isolationism"),
    ],
    "Economy": [
        ("Economic Models", "Free-Market", "Interventionist"),
        ("Distribution Justice", "Equality of Opportunity", "Equality of Outcome"),
    ],
    "Culture": [
        ("Ethical Orientations", "Ethical Liberalism", "Ethical Conservatism"),
        ("Value Orientations", "Individualism", "Collectivism"),
    ],
    "Society": [
        ("Fairness Principles", "Substantive Justice", "Procedural Justice"),
        ("Judicial Philosophies", "Progressive Jurisprudence", "Conservative Jurisprudence"),
    ],
    "Environment": [
        ("Sustainability Strategies", "Conservation Priority", "Development Priority"),
        ("Climate Stances", "Climate Actionism", "Climate Skepticism"),
    ],
    "Technology": [
        ("Innovation Models", "Inclusive Sharing", "IP Protection"),
        ("Governance Models", "Technological Libertarianism", "Technological Regulation"),
    ],
}

# Label strings of the original Chinese annotation.
EXPECTED_ZH = {
    "政治": {"政策导向": {"革命主义", "改良主义"}, "外交战略": {"全球主义", "孤立主义"}},
    "经济": {"经济模式": {"市场自由", "国家干预"}, "分配正义": {"机会平等", "结果平等"}},
    "文化": {"道德取向": {"伦理自由", "伦理保守"}, "社会价值": {"个人主义", "集体主义"}},
    "社会": {"公正理念": {"结果公正", "程序正义"}, "法律取向": {"进步法理", "保守法理"}},
    "生态文明": {"发展取向": {"保护优先", "发展优先"}, "气候立场": {"气候行动", "气候怀疑"}},
    "科技": {"创新模式": {"普惠共享", "知识产权保护"}, "风险治理": {"技术自由", "技术监管"}},
}


def test_english_names_and_sides():
    assert [d.name for d in DIMENSIONS] == list(EXPECTED)
    for dimension in DIMENSIONS:
        assert [(s.name, s.change.name, s.stability.name) for s in dimension.subcategories] == EXPECTED[dimension.name]
        for sub in dimension.subcategories:
            assert sub.change.side == CHANGE and sub.stability.side == STABILITY
            assert sub.position(CHANGE) is sub.change and sub.position(STABILITY) is sub.stability


def test_chinese_names():
    found = {d.name_zh: {s.name_zh: {s.change.name_zh, s.stability.name_zh} for s in d.subcategories} for d in DIMENSIONS}
    assert found == EXPECTED_ZH


def test_translation_pairs():
    pairs = {
        ("Economy", "经济"), ("Environment", "生态文明"), ("Free-Market", "市场自由"), ("Interventionist", "国家干预"),
        ("Individualism", "个人主义"), ("Collectivism", "集体主义"), ("Value Orientations", "社会价值"),
        ("Sustainability Strategies", "发展取向"), ("Governance Models", "风险治理"),
        ("Technological Libertarianism", "技术自由"), ("Technological Regulation", "技术监管"),
    }
    found = set()
    for d in DIMENSIONS:
        found.add((d.name, d.name_zh))
        for s in d.subcategories:
            found |= {(s.name, s.name_zh), (s.change.name, s.change.name_zh), (s.stability.name, s.stability.name_zh)}
    assert pairs <= found


def test_positions_are_unique():
    positions = [p for d in DIMENSIONS for s in d.subcategories for p in (s.change, s.stability)]
    assert len(positions) == 24
    assert len({p.name for p in positions}) == 24
    assert len({p.name_zh for p in positions}) == 24


def test_table_order_and_abbreviations():
    assert sorted(TABLE_ORDER) == sorted(EXPECTED)
    assert [DIMENSION_BY_NAME[d].abbreviation for d in TABLE_ORDER] == ["Cul.", "Eco.", "Soc.", "Pol.", "Tech.", "Env."]


def test_unknown_dimension():
    with pytest.raises(KeyError):
        get_dimension("Religion")
    with pytest.raises(ValueError):
        get_dimension("Politics").subcategories[0].position("left")

"""Label schema of MIND.

English names, their order, and the change-oriented (left) / stability-oriented (right) assignment
follow the dimension schema of MIND (assets/dimension.png). Chinese names are the label strings of the
original annotation.
Each of the six value dimensions has two subcategories, and every subcategory contrasts a
change-oriented position with a stability-oriented position.
"""

from __future__ import annotations

from dataclasses import dataclass

CHANGE = "change"
STABILITY = "stability"
SIDES: tuple[str, str] = (CHANGE, STABILITY)


@dataclass(frozen=True)
class Position:
    name: str
    name_zh: str
    side: str


@dataclass(frozen=True)
class Subcategory:
    name: str
    name_zh: str
    change: Position
    stability: Position

    def position(self, side: str) -> Position:
        if side not in SIDES:
            raise ValueError(f"unknown side: {side!r}")
        return self.change if side == CHANGE else self.stability


@dataclass(frozen=True)
class Dimension:
    name: str
    name_zh: str
    abbreviation: str
    subcategories: tuple[Subcategory, Subcategory]

    def positions(self, side: str) -> tuple[Position, Position]:
        return tuple(sub.position(side) for sub in self.subcategories)  # type: ignore[return-value]


def _subcategory(name: str, name_zh: str, change: tuple[str, str], stability: tuple[str, str]) -> Subcategory:
    return Subcategory(
        name=name,
        name_zh=name_zh,
        change=Position(change[0], change[1], CHANGE),
        stability=Position(stability[0], stability[1], STABILITY),
    )


# Order of the dimension schema: Politics, Economy, Culture, Society, Environment, Technology.
DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("Politics", "政治", "Pol.", (
        _subcategory("Policy Orientations", "政策导向", ("Revolutionism", "革命主义"), ("Reformism", "改良主义")),
        _subcategory("Diplomatic Strategies", "外交战略", ("Globalism", "全球主义"), ("Isolationism", "孤立主义")),
    )),
    Dimension("Economy", "经济", "Eco.", (
        _subcategory("Economic Models", "经济模式", ("Free-Market", "市场自由"), ("Interventionist", "国家干预")),
        _subcategory("Distribution Justice", "分配正义", ("Equality of Opportunity", "机会平等"), ("Equality of Outcome", "结果平等")),
    )),
    Dimension("Culture", "文化", "Cul.", (
        _subcategory("Ethical Orientations", "道德取向", ("Ethical Liberalism", "伦理自由"), ("Ethical Conservatism", "伦理保守")),
        _subcategory("Value Orientations", "社会价值", ("Individualism", "个人主义"), ("Collectivism", "集体主义")),
    )),
    Dimension("Society", "社会", "Soc.", (
        _subcategory("Fairness Principles", "公正理念", ("Substantive Justice", "结果公正"), ("Procedural Justice", "程序正义")),
        _subcategory("Judicial Philosophies", "法律取向", ("Progressive Jurisprudence", "进步法理"), ("Conservative Jurisprudence", "保守法理")),
    )),
    Dimension("Environment", "生态文明", "Env.", (
        _subcategory("Sustainability Strategies", "发展取向", ("Conservation Priority", "保护优先"), ("Development Priority", "发展优先")),
        _subcategory("Climate Stances", "气候立场", ("Climate Actionism", "气候行动"), ("Climate Skepticism", "气候怀疑")),
    )),
    Dimension("Technology", "科技", "Tech.", (
        _subcategory("Innovation Models", "创新模式", ("Inclusive Sharing", "普惠共享"), ("IP Protection", "知识产权保护")),
        _subcategory("Governance Models", "风险治理", ("Technological Libertarianism", "技术自由"), ("Technological Regulation", "技术监管")),
    )),
)

DIMENSION_BY_NAME: dict[str, Dimension] = {d.name: d for d in DIMENSIONS}

# Column order of the main results and ablation tables in the paper.
TABLE_ORDER: tuple[str, ...] = ("Culture", "Economy", "Society", "Politics", "Technology", "Environment")


def get_dimension(name: str) -> Dimension:
    try:
        return DIMENSION_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown dimension {name!r}; expected one of {sorted(DIMENSION_BY_NAME)}") from exc

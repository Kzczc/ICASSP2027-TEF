"""Prompts for sentence-level and document-level value judgments.

As described in the paper, each prompt names the value dimension and its two positions and
requests a single answer token. ``original`` is used for all main experiments; ``verbose``
adds role background and annotation guidelines; ``minimal`` keeps only the classification
directive. The answer letters are C (change-oriented) and S (stability-oriented).
"""

from __future__ import annotations

from .schema import CHANGE, STABILITY, Dimension

ANSWER_TOKENS: dict[str, str] = {CHANGE: "C", STABILITY: "S"}
PROMPT_VARIANTS: tuple[str, ...] = ("original", "verbose", "minimal")

_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "original": (
            "You are a careful annotator who measures public value orientations in social media text.\n\n"
            "Decide which position the text supports on the given value dimension.\n\n"
            "Dimension: {dimension}\n"
            "C (change-oriented): {change}\n"
            "S (stability-oriented): {stability}\n\n"
            "Text: {text}\n\n"
            "Answer with a single letter, C or S.\n"
            "Answer:"
        ),
        "verbose": (
            "You are a political scientist and a trained annotator in computational social science. "
            "You measure the value orientation an author expresses in a social media post. "
            "Each value dimension contains two subcategories, and each subcategory contrasts a "
            "change-oriented position with a stability-oriented position.\n\n"
            "Annotation guidelines:\n"
            "1. Judge the position the author supports, not positions the author quotes, reports, or rejects.\n"
            "2. Rely on explicit arguments first, then on tone, framing, and conclusions.\n"
            "3. If both positions appear, choose the one the author endorses more strongly.\n"
            "4. Use the position definitions below rather than general political labels.\n\n"
            "Dimension: {dimension}\n"
            "C (change-oriented): {change}\n"
            "S (stability-oriented): {stability}\n\n"
            "Text: {text}\n\n"
            "Answer with a single letter, C or S, and nothing else.\n"
            "Answer:"
        ),
        "minimal": (
            "Dimension: {dimension}. C: {change}. S: {stability}.\n"
            "Text: {text}\n"
            "Answer C or S:"
        ),
    },
    "zh": {
        "original": (
            "你是一名严谨的标注员，负责测量社交媒体文本中的公共价值取向。\n\n"
            "请判断文本在给定价值维度上支持哪一种立场。\n\n"
            "维度：{dimension}\n"
            "C（变革取向）：{change}\n"
            "S（稳定取向）：{stability}\n\n"
            "文本：{text}\n\n"
            "只回答一个字母：C 或 S。\n"
            "答案："
        ),
        "verbose": (
            "你是一名政治学研究者，也是计算社会科学领域受过训练的标注员，负责测量社交媒体帖子中作者表达的价值取向。"
            "每个价值维度包含两个子类，每个子类都在变革取向立场与稳定取向立场之间作出区分。\n\n"
            "标注准则：\n"
            "1. 判断作者本人支持的立场，而不是作者引用、转述或反对的立场。\n"
            "2. 优先依据明确的论点，其次依据语气、框架和结论。\n"
            "3. 如果两种立场同时出现，选择作者更明确支持的一方。\n"
            "4. 以下方给出的立场定义为准，不要套用笼统的政治标签。\n\n"
            "维度：{dimension}\n"
            "C（变革取向）：{change}\n"
            "S（稳定取向）：{stability}\n\n"
            "文本：{text}\n\n"
            "只回答一个字母：C 或 S，不要输出其他内容。\n"
            "答案："
        ),
        "minimal": (
            "维度：{dimension}。C：{change}。S：{stability}。\n"
            "文本：{text}\n"
            "回答 C 或 S："
        ),
    },
}


def describe_side(dimension: Dimension, side: str, lang: str) -> str:
    """Describe one side of a dimension by its two subcategory positions."""
    if lang == "zh":
        return "；".join(f"{sub.position(side).name_zh}（{sub.name_zh}）" for sub in dimension.subcategories)
    return "; ".join(f"{sub.position(side).name} ({sub.name})" for sub in dimension.subcategories)


def build_prompt(text: str, dimension: Dimension, lang: str, variant: str = "original") -> str:
    """Build the prompt for one text (a sentence or a whole post) on one dimension."""
    if lang not in _TEMPLATES:
        raise ValueError(f"unsupported language {lang!r}")
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; expected one of {PROMPT_VARIANTS}")
    return _TEMPLATES[lang][variant].format(
        dimension=dimension.name_zh if lang == "zh" else dimension.name,
        change=describe_side(dimension, CHANGE, lang),
        stability=describe_side(dimension, STABILITY, lang),
        text=text.strip(),
    )

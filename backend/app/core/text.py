"""文本工具：句子切分 + 中英文混合分词（BM25 与 mock 模型共用）。

中文检索的坑：BM25 依赖分词质量。这里 jieba 可选，
没有 jieba 时退化为「汉字二元组 + 单字 + 英文词」，
对中文关键词召回依然有效（只是索引会大一些）。
"""
from __future__ import annotations

import re
import unicodedata

_CJK_RANGE = r"一-鿿"
_CJK = re.compile(f"[{_CJK_RANGE}]")
_CJK_RUN = re.compile(f"[{_CJK_RANGE}]+")
_LATIN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_PUNCT = re.compile(r"[，。；：、！？,.;:!?（）()\[\]【】「」《》\"'“”‘’—\-·/\\|]")

# 停用词只保留高频虚词，避免把「不」「未」这类否定词删掉影响语义
STOPWORDS = {
    "的", "了", "和", "是", "在", "我", "你", "他", "她", "它", "们", "这", "那",
    "有", "就", "也", "都", "而", "及", "与", "或", "被", "把", "对", "从", "到",
    "一", "个", "上", "下", "中", "吗", "呢", "啊", "吧", "着", "过", "地", "得",
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "for", "in", "on",
    "what", "how", "please",
}

try:  # jieba 是可选依赖
    import jieba

    jieba.setLogLevel(60)

    _HAS_JIEBA = True
except Exception:  # pragma: no cover - 环境相关
    _HAS_JIEBA = False


def has_jieba() -> bool:
    return _HAS_JIEBA


def normalize_text(text: str) -> str:
    """统一全半角、去零宽字符、压缩空白，但不破坏段落结构。"""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("​", "").replace("﻿", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|(?<=\n)")


def split_sentences(text: str, min_len: int = 6) -> list[str]:
    """中文友好分句：按句末标点/换行切，再把过短的碎片并回前一句。"""
    raw = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    merged: list[str] = []
    for part in raw:
        if merged and len(part) < min_len:
            merged[-1] = merged[-1] + part
        else:
            merged.append(part)
    return merged


def tokenize(text: str) -> list[str]:
    """返回用于 BM25 / 词面匹配的 token 列表。"""
    text = text.lower()
    tokens: list[str] = []

    if _HAS_JIEBA:
        for token in jieba.lcut(text):
            token = token.strip()
            if not token or _PUNCT.fullmatch(token):
                continue
            if not re.search(f"[0-9a-z{_CJK_RANGE}]", token):
                continue
            tokens.append(token)
    else:
        for word in _LATIN.findall(text):
            tokens.append(word)
        for run in _CJK_RUN.findall(text):
            # 单字 + 二元组：解决无分词器时的中文召回
            tokens.extend(run)
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    return [t for t in tokens if t not in STOPWORDS and len(t) <= 24]


def lexical_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """查询词在文档中的覆盖率（去重后），用于 mock rerank 与兜底判断。"""
    if not query_tokens or not doc_tokens:
        return 0.0
    q = set(query_tokens)
    d = set(doc_tokens)
    return len(q & d) / len(q)


def char_tokens(text: str) -> list[str]:
    """字符级 token（中文单字 + 英文词）。

    中文里跨词二元组（如「票的」「的抬」）几乎不会出现在文档里，
    会把查询命中的分母撑大；字符级 + IDF 是更稳的兜底方案。
    """
    text = text.lower()
    tokens = _CJK.findall(text) + _LATIN.findall(text)
    return [t for t in tokens if t not in STOPWORDS]


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + suffix

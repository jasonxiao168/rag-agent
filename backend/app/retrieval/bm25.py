"""BM25 关键词检索（自实现，无三方依赖）。

为什么不用向量检索一条路走到黑？
- 向量检索对「ORD-2026-1001」「¥600」「0.5%」这类精确 token 不敏感；
- 用户问「住宿标准 600」时，BM25 能精准命中，而 embedding 容易漂移；
- 混合检索 = 语义兜底 + 关键词精确，两者互补。

实现为经典 Okapi BM25，索引常驻内存并在启动时从 children 重建（毫秒级）。
"""
from __future__ import annotations

import math
from collections import Counter

from app.core.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.doc_ids: list[str | None] = []
        self._term_freqs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._doc_freq: Counter = Counter()
        self._avgdl: float = 0.0

    def build(
        self, ids: list[str], texts: list[str], doc_ids: list[str | None] | None = None
    ) -> None:
        self.ids = list(ids)
        self.doc_ids = list(doc_ids) if doc_ids is not None else [None] * len(ids)
        self._term_freqs = []
        self._doc_lens = []
        self._doc_freq = Counter()

        for text in texts:
            tokens = tokenize(text)
            tf = Counter(tokens)
            self._term_freqs.append(tf)
            self._doc_lens.append(len(tokens))
            self._doc_freq.update(tf.keys())

        total = sum(self._doc_lens)
        self._avgdl = total / len(self._doc_lens) if self._doc_lens else 0.0

    def __len__(self) -> int:
        return len(self.ids)

    def _idf(self, term: str) -> float:
        n = len(self.ids)
        df = self._doc_freq.get(term, 0)
        # 加 1 的平滑形式，避免高频词出现负分
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(
        self, query: str, top_k: int = 10, allowed_docs: set[str] | None = None
    ) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: list[tuple[float, int]] = []
        for index, tf in enumerate(self._term_freqs):
            # 权限过滤放在打分开头：无权文档不参与计算，也不会出现在 trace 里
            if allowed_docs is not None and self.doc_ids[index] not in allowed_docs:
                continue
            doc_len = self._doc_lens[index] or 1
            score = 0.0
            for term in query_terms:
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self._avgdl or 1))
                score += self._idf(term) * freq * (self.k1 + 1) / denom
            if score > 0:
                scores.append((score, index))

        scores.sort(reverse=True)
        return [(self.ids[i], round(score, 4)) for score, i in scores[:top_k]]

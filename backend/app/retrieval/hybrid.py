"""混合检索融合：RRF（Reciprocal Rank Fusion）。

为什么用 RRF 而不是把分数加权相加？
- 向量相似度（0~1 余弦）与 BM25 分数（无上界）量纲不同，直接加权需要调参、易过拟合；
- RRF 只看名次：score = Σ 1/(k + rank)，无需归一化，工程上更稳；
- k=60 是论文默认值，能压低「某一路召回特别强势」带来的偏差。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusedHit:
    child_id: str
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None

    def as_dict(self) -> dict:
        return {
            "child_id": self.child_id,
            "rrf_score": round(self.score, 6),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "dense_score": None if self.dense_score is None else round(self.dense_score, 4),
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 4),
        }


def reciprocal_rank_fusion(
    dense: list[tuple[str, float]],
    bm25: list[tuple[str, float]],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[FusedHit]:
    """输入两路已按相关性降序排好的 (child_id, score)，输出融合后的有序列表。"""
    fused: dict[str, FusedHit] = {}

    for rank, (child_id, score) in enumerate(dense, start=1):
        hit = fused.setdefault(child_id, FusedHit(child_id=child_id, score=0.0))
        hit.dense_rank = rank
        hit.dense_score = score
        hit.score += dense_weight / (k + rank)

    for rank, (child_id, score) in enumerate(bm25, start=1):
        hit = fused.setdefault(child_id, FusedHit(child_id=child_id, score=0.0))
        hit.bm25_rank = rank
        hit.bm25_score = score
        hit.score += bm25_weight / (k + rank)

    return sorted(fused.values(), key=lambda item: item.score, reverse=True)

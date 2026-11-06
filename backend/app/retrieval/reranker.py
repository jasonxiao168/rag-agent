"""Rerank 精排：粗排 Top-K → Cross-Encoder 精排 Top-N。

- 粗排（向量 + BM25）是「双塔/倒排」，速度快但 query 与 doc 不交互，容易主题相关而细节无关；
- 精排把 (query, doc) 拼起来送进 Cross-Encoder，逐条计算相关性，精度高但慢，
  所以只对粗排的 10 条做精排，成本可控。

默认用 DashScope 的 gte-rerank-v2；mock 模式用词面覆盖率 + 数字命中模拟。
想换成 BGE-Reranker / Cohere Rerank，只需实现同样的 `rerank()` 接口。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.tokens import Usage
from app.llm.client import LLMClient, get_llm


@dataclass
class RerankHit:
    child_id: str
    score: float | None
    original_rank: int


class Reranker:
    def __init__(self, llm: LLMClient | None = None, enabled: bool = True) -> None:
        self.llm = llm or get_llm()
        self.enabled = enabled

    def rerank(
        self, query: str, candidates: list[dict], *, top_n: int
    ) -> tuple[list[RerankHit], Usage | None]:
        if not candidates:
            return [], None

        if not self.enabled:
            hits = [
                RerankHit(child_id=c["child_id"], score=None, original_rank=i)
                for i, c in enumerate(candidates[:top_n])
            ]
            return hits, None

        documents = [c["text"] for c in candidates]
        ranked, usage = self.llm.rerank(query, documents, top_n=min(top_n, len(documents)))

        hits: list[RerankHit] = []
        seen: set[str] = set()
        for index, score in ranked:
            if index < 0 or index >= len(candidates):
                continue
            child = candidates[index]
            if child["child_id"] in seen:
                continue
            seen.add(child["child_id"])
            hits.append(RerankHit(child_id=child["child_id"], score=score, original_rank=index))

        # Rerank 服务偶尔返回数量不足，用原顺序补位，保证上下文不空
        if len(hits) < top_n:
            for i, candidate in enumerate(candidates[:top_n]):
                if candidate["child_id"] in seen:
                    continue
                hits.append(RerankHit(child_id=candidate["child_id"], score=None, original_rank=i))
                if len(hits) >= top_n:
                    break

        return hits[:top_n], usage

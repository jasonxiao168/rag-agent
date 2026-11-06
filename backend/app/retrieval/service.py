"""知识库检索服务：父子分块 + 混合检索 + Rerank + 文档级权限 + 增量索引。

一次 search() 的完整链路：
    query → 向量检索（按 ACL 过滤）┐
                                  ├→ RRF 融合 → Top10 → Rerank → Top3 父块
    query → BM25（按 ACL 过滤）   ┘

ACL 在检索阶段生效：无权文档不参与打分、不进入候选、不会出现在返回的 trace 里。
`departments=None` 表示「脚本内部调用，不做过滤」，HTTP 入口永远会传真实部门。

build() 支持增量：按 content_hash 对比，只对新增/变更文档重新切分与向量化，
删除的部分从向量库按 child_id 清除。
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.core.logging import get_logger
from app.core.masking import mask_text
from app.core.tokens import Usage
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_documents
from app.llm.client import LLMClient, get_llm
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import FusedHit, reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore, build_vector_store

logger = get_logger("retrieval.service")


class IndexNotReady(RuntimeError):
    """索引不存在，提示用户先跑 ingest。"""


class IndexDimensionMismatch(RuntimeError):
    """索引向量维度与当前 embedding 模型不一致（通常是换了模型没重建索引）。"""


@dataclass
class RetrievalConfig:
    top_k: int = field(default_factory=lambda: settings.retrieve_top_k)
    rerank_top_n: int = field(default_factory=lambda: settings.rerank_top_n)
    use_hybrid: bool = field(default_factory=lambda: settings.use_hybrid)
    use_rerank: bool = field(default_factory=lambda: settings.use_rerank)
    context_max_chars: int = field(default_factory=lambda: settings.context_max_chars)
    rrf_k: int = field(default_factory=lambda: settings.rrf_k)
    min_score: float = field(default_factory=lambda: settings.min_rerank_score)
    pool_multiplier: int = 2


@dataclass
class RetrievedContext:
    index: int
    parent_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float | None
    child_ids: list[str] = field(default_factory=list)
    acl: str = "all"

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "text": self.text,
            "score": None if self.score is None else round(self.score, 4),
            "child_ids": self.child_ids,
            "acl": self.acl,
        }


@dataclass
class RetrievalResult:
    query: str
    contexts: list[RetrievedContext]
    candidates: list[dict]
    timings: dict
    usage: list[Usage]
    best_score: float | None
    low_confidence: bool
    acl: dict = field(default_factory=dict)

    @property
    def citations(self) -> list[dict]:
        return [
            {
                "index": ctx.index,
                "title": ctx.title,
                "section": ctx.section,
                "score": None if ctx.score is None else round(ctx.score, 4),
                # 引用带上全文：前端只展示标题，评测脚本用全文判断「证据是否进入上下文」
                "preview": ctx.text,
                "acl": ctx.acl,
            }
            for ctx in self.contexts
        ]

    def trace(self) -> dict:
        return {
            "query": self.query,
            "hit_count": len(self.candidates),
            "selected_count": len(self.contexts),
            "best_score": None if self.best_score is None else round(self.best_score, 4),
            "low_confidence": self.low_confidence,
            "timings": self.timings,
            "acl": self.acl,
        }


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("索引文件损坏，按空处理：%s", path)
        return default


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


class KnowledgeBase:
    def __init__(
        self,
        index_dir: Path | None = None,
        config: RetrievalConfig | None = None,
        llm: LLMClient | None = None,
        vector_backend: str | None = None,
    ) -> None:
        self.index_dir = Path(index_dir or settings.index_dir)
        self.config = config or RetrievalConfig()
        self.llm = llm or get_llm()
        self.vector_backend = vector_backend or os.environ.get("VECTOR_BACKEND", "auto")
        self.children: dict[str, dict] = {}
        self.parents: dict[str, dict] = {}
        self.docs: list[dict] = []
        self.manifest: dict = {}
        self._bm25 = BM25Index()
        self._store: VectorStore | None = None
        self._loaded = False
        self._index_dim: int | None = None
        self._query_cache: dict[str, list[float]] = {}
        self.reranker = Reranker(self.llm, enabled=self.config.use_rerank)

    # ------------------------------------------------------------------ #
    # 索引读写
    # ------------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        return self._loaded and bool(self.children)

    def load(self) -> "KnowledgeBase":
        children_file = self.index_dir / "children.jsonl"
        if not children_file.exists():
            raise IndexNotReady(
                f"索引不存在：{children_file}，请先执行 `python -m scripts.ingest`"
            )

        self.children = {item["child_id"]: item for item in _read_jsonl(children_file)}
        self.parents = _read_json(self.index_dir / "parents.json", {})
        self.docs = _read_json(self.index_dir / "docs.json", [])
        self.manifest = _read_json(self.index_dir / "manifest.json", {})
        self._index_dim = self.manifest.get("dim") or None

        # 嵌入式 Qdrant 会持有目录锁，重载前必须先释放旧客户端
        if self._store is not None:
            close = getattr(self._store, "close", None)
            if callable(close):
                close()

        self._store = build_vector_store(self.index_dir, self.vector_backend)
        if self._store.count() == 0:
            logger.warning("向量库为空，本次仅能使用 BM25 检索：%s", self.index_dir)

        # BM25 常驻内存，启动时从子块文本重建（毫秒级，避免额外持久化带来的不一致）
        self._bm25.build(
            list(self.children.keys()),
            [item["embed_text"] for item in self.children.values()],
            [item.get("doc_id") for item in self.children.values()],
        )

        provider = self.manifest.get("provider")
        if provider and provider != self.llm.name:
            logger.warning(
                "索引由 %s 构建，当前模型提供方为 %s，向量空间不一致，建议重新 ingest",
                provider,
                self.llm.name,
            )

        self._loaded = True
        logger.info(
            "索引加载完成：docs=%d parents=%d children=%d backend=%s",
            len(self.docs),
            len(self.parents),
            len(self.children),
            getattr(self._store, "name", "unknown"),
        )
        return self

    def allowed_doc_ids(self, departments: list[str] | None) -> set[str] | None:
        """返回该身份可见的 doc_id 集合；None 表示不做过滤（脚本内部）。"""
        if departments is None:
            return None
        allowed = set()
        for doc in self.docs:
            acl = doc.get("acl_list") or ["all"]
            if "all" in acl or set(departments) & set(acl):
                allowed.add(doc["doc_id"])
        return allowed

    def build(self, docs: list | None = None, incremental: bool = True) -> dict:
        docs = docs or load_documents(settings.raw_dir)
        if not docs:
            raise ValueError(f"没有可索引的文档，请把 PDF/Markdown 放到 {settings.raw_dir}")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        previous_manifest = (
            _read_json(self.index_dir / "manifest.json", {}) if incremental else {}
        )

        # 嵌入式 Qdrant 同一目录只允许一个客户端：构建前先释放当前实例
        if self._store is not None:
            close = getattr(self._store, "close", None)
            if callable(close):
                close()
            self._store = None

        # 1) 读取旧索引（仅增量需要）
        if incremental:
            old_children = {item["child_id"]: item for item in _read_jsonl(self.index_dir / "children.jsonl")}
            old_parents = _read_json(self.index_dir / "parents.json", {})
            old_docs = {item["doc_id"]: item for item in _read_json(self.index_dir / "docs.json", [])}
        else:
            old_children, old_parents, old_docs = {}, {}, {}

        incoming = {doc.doc_id: doc for doc in docs}
        added = [doc for doc_id, doc in incoming.items() if doc_id not in old_docs]
        changed = [
            doc
            for doc_id, doc in incoming.items()
            if doc_id in old_docs and old_docs[doc_id].get("content_hash") != doc.content_hash
        ]
        unchanged = [
            doc
            for doc_id, doc in incoming.items()
            if doc_id in old_docs and old_docs[doc_id].get("content_hash") == doc.content_hash
        ]
        removed = [doc_id for doc_id in old_docs if doc_id not in incoming]

        # 2) 先切分 + 向量化。模型下载/调用失败时，现有索引完全不受影响。
        targets = added + changed
        if targets:
            parents, children = chunk_documents(
                targets,
                parent_size=settings.parent_size,
                child_size=settings.child_size,
                parent_overlap=settings.parent_overlap,
                child_overlap=settings.child_overlap,
            )
        else:
            parents, children = [], []

        embed_ms = 0.0
        embed_usage = Usage(model=self.llm.name, stage="index")
        vectors: list[list[float]] = []
        if children:
            started = time.perf_counter()
            vectors, embed_usage = self.llm.embed(
                [child.embed_text for child in children], stage="index"
            )
            embed_ms = _ms(started)

        # 3) 向量就绪后再动存储：全量重建此时才清空旧集合并 reset
        if incremental:
            store = build_vector_store(self.index_dir, self.vector_backend)
        else:
            self._clear_index()
            store = build_vector_store(self.index_dir, self.vector_backend, load_existing=False)
            reset = getattr(store, "reset", None)
            if callable(reset):
                reset()

        # 4) 清理变更/删除文档的旧块与旧向量
        stale_doc_ids = {doc.doc_id for doc in changed} | set(removed)
        stale_children = [
            child_id for child_id, meta in old_children.items() if meta.get("doc_id") in stale_doc_ids
        ]
        deleted_vectors = store.delete(stale_children) if stale_children else 0
        for child_id in stale_children:
            old_children.pop(child_id, None)
        for parent_id in [pid for pid, meta in old_parents.items() if meta.get("doc_id") in stale_doc_ids]:
            old_parents.pop(parent_id, None)
        for doc_id in stale_doc_ids:
            old_docs.pop(doc_id, None)

        if children:
            store.add(
                [child.child_id for child in children],
                vectors,
                [child.metadata() for child in children],
            )
            for child in children:
                old_children[child.child_id] = {
                    "child_id": child.child_id,
                    "parent_id": child.parent_id,
                    "doc_id": child.doc_id,
                    "title": child.title,
                    "section": child.section,
                    "text": child.text,
                    "embed_text": child.embed_text,
                    "acl": child.acl,
                }
            for parent in parents:
                old_parents[parent.parent_id] = parent.as_dict()

        now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        for doc in targets:
            old_docs[doc.doc_id] = {**doc.as_dict(), "indexed_at": now}

        # 3) 补全每篇文档的块数（便于排查“为什么这篇没被召回”）
        for doc_id, meta in old_docs.items():
            meta["children"] = sum(1 for child in old_children.values() if child.get("doc_id") == doc_id)
            meta["parents"] = sum(1 for parent in old_parents.values() if parent.get("doc_id") == doc_id)

        store.persist()

        # 4) 落盘并重载（BM25 从全部子块重建）
        with (self.index_dir / "children.jsonl").open("w", encoding="utf-8") as fh:
            for child_id in sorted(old_children):
                fh.write(json.dumps(old_children[child_id], ensure_ascii=False) + "\n")
        (self.index_dir / "parents.json").write_text(
            json.dumps(old_parents, ensure_ascii=False), encoding="utf-8"
        )
        (self.index_dir / "docs.json").write_text(
            json.dumps(sorted(old_docs.values(), key=lambda item: item["doc_id"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        build_stats = {
            "mode": "incremental" if incremental else "full",
            "added": len(added),
            "updated": len(changed),
            "unchanged": len(unchanged),
            "removed": len(removed),
            "children_embedded": len(children),
            "children_deleted": deleted_vectors,
        }

        manifest = {
            "built_at": now,
            "provider": self.llm.name,
            "embedding_model": (
                getattr(self.llm, "describe", lambda: {})().get("embedding")
                or (settings.model_embedding if self.llm.name != "mock" else "mock-hash")
            ),
            "dim": len(vectors[0]) if children else previous_manifest.get("dim", settings.embed_dim),
            "docs": len(old_docs),
            "parents": len(old_parents),
            "children": len(old_children),
            "vector_backend": getattr(store, "name", "unknown"),
            "chunking": {
                "parent_size": settings.parent_size,
                "child_size": settings.child_size,
                "parent_overlap": settings.parent_overlap,
                "child_overlap": settings.child_overlap,
            },
            "last_build": build_stats,
            "embed_ms": embed_ms,
            "embed_tokens": embed_usage.prompt_tokens,
        }
        (self.index_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info(
            "索引构建完成（%s）：新增 %d，更新 %d，未变 %d，删除 %d，重算向量 %d 条",
            build_stats["mode"],
            build_stats["added"],
            build_stats["updated"],
            build_stats["unchanged"],
            build_stats["removed"],
            build_stats["children_embedded"],
        )

        # 释放构建期的客户端（嵌入式 Qdrant 的目录锁），再重载
        close = getattr(store, "close", None)
        if callable(close):
            close()

        self.load()
        return manifest

    def _clear_index(self) -> None:
        # 清理历史遗留的 Chroma 目录（已切换到 Qdrant）
        legacy = self.index_dir / "chroma"
        if legacy.exists():
            shutil.rmtree(legacy, ignore_errors=True)

        for name in (
            "children.jsonl",
            "parents.json",
            "docs.json",
            "manifest.json",
            "vectors.npy",
            "vectors.json",
            "vectors_meta.jsonl",
        ):
            path = self.index_dir / name
            if path.exists():
                path.unlink()

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def _embed_query(self, query: str) -> tuple[list[float], Usage | None]:
        if query in self._query_cache:
            return self._query_cache[query], None
        vectors, usage = self.llm.embed([query], stage="query_embed")
        if len(self._query_cache) > 256:
            self._query_cache.clear()
        self._query_cache[query] = vectors[0]
        return vectors[0], usage

    def search(
        self,
        query: str,
        config: RetrievalConfig | None = None,
        departments: list[str] | None = None,
        permissions: list[str] | None = None,
    ) -> RetrievalResult:
        if not self.ready:
            raise IndexNotReady("知识库尚未加载，请先调用 load() 或执行 ingest 脚本")

        cfg = config or self.config
        started = time.perf_counter()
        timings: dict[str, float] = {}
        usages: list[Usage] = []

        allowed = self.allowed_doc_ids(departments)
        acl_summary = {
            "departments": departments,
            "documents_visible": len(allowed) if allowed is not None else len(self.docs),
            "documents_total": len(self.docs),
            "restricted": allowed is not None and len(allowed) < len(self.docs),
            "pii_allowed": bool(permissions and "pii" in permissions),
            "pii_masked": 0,
        }
        if allowed is not None and not allowed:
            timings["total_ms"] = _ms(started)
            return RetrievalResult(
                query=query,
                contexts=[],
                candidates=[],
                timings=timings,
                usage=[],
                best_score=None,
                low_confidence=True,
                acl=acl_summary,
            )

        # 1) 查询向量化（带缓存，重复问题不再付费）
        step = time.perf_counter()
        query_vec, embed_usage = self._embed_query(query)
        if self._index_dim and len(query_vec) != self._index_dim:
            embedding_name = (
                getattr(self.llm, "describe", lambda: {})().get("embedding") or self.llm.name
            )
            raise IndexDimensionMismatch(
                f"索引向量维度 {self._index_dim}（{self.manifest.get('embedding_model', '未知')}）"
                f"与当前 embedding 维度 {len(query_vec)}（{embedding_name}）不一致。"
                "通常是换了模型或开关了 MOCK 模式但没有重建索引，"
                "请执行：python -m scripts.ingest --full"
            )
        if embed_usage:
            usages.append(embed_usage)
        timings["embed_ms"] = _ms(step)

        # 2) 双路召回（都在检索阶段做 ACL 过滤）
        pool = max(cfg.top_k * cfg.pool_multiplier, cfg.top_k)
        step = time.perf_counter()
        dense_hits = (
            self._store.search(query_vec, pool, allowed_docs=allowed) if self._store else []
        )
        timings["dense_ms"] = _ms(step)

        step = time.perf_counter()
        bm25_hits = (
            self._bm25.search(query, pool, allowed_docs=allowed) if cfg.use_hybrid else []
        )
        timings["bm25_ms"] = _ms(step)

        # 3) RRF 融合
        step = time.perf_counter()
        dense_pairs = [(hit.child_id, hit.score) for hit in dense_hits]
        if cfg.use_hybrid:
            fused: list[FusedHit] = reciprocal_rank_fusion(dense_pairs, bm25_hits, k=cfg.rrf_k)
        else:
            fused = [
                FusedHit(child_id=child_id, score=score, dense_rank=i + 1, dense_score=score)
                for i, (child_id, score) in enumerate(dense_pairs)
            ]
        candidates = fused[: cfg.top_k]
        timings["fuse_ms"] = _ms(step)

        # 4) Rerank 精排（对子块文本，带上标题/章节，避免指代丢失）
        #    精排池比最终上下文数大一倍：多个子块常属于同一父块，
        #    去重后需要从更多候选里补足不同的父块。
        step = time.perf_counter()
        rerank_input = [
            {
                "child_id": hit.child_id,
                "text": self.children.get(hit.child_id, {}).get("embed_text", ""),
            }
            for hit in candidates
            if hit.child_id in self.children
        ]
        rerank_pool = min(len(rerank_input), max(cfg.rerank_top_n * 2, cfg.rerank_top_n))
        rerank_hits, rerank_usage = self.reranker.rerank(query, rerank_input, top_n=rerank_pool)
        if rerank_usage:
            usages.append(rerank_usage)
        timings["rerank_ms"] = _ms(step)

        if not rerank_hits:
            rerank_hits = Reranker(self.llm, enabled=False).rerank(
                query, rerank_input, top_n=rerank_pool
            )[0]

        rerank_scores = {hit.child_id: hit.score for hit in rerank_hits}

        # 5) 子块 → 父块，按预算拼上下文（父块去重，最多 rerank_top_n 个）
        selected: list[RetrievedContext] = []
        seen_parents: set[str] = set()
        used_chars = 0
        for hit in rerank_hits:
            if len(selected) >= cfg.rerank_top_n:
                break
            child = self.children.get(hit.child_id)
            if not child:
                continue
            parent = self.parents.get(child["parent_id"])
            if not parent or parent["parent_id"] in seen_parents:
                continue
            # 双保险：即便向量库过滤失效，也不允许无权文档进入上下文
            if allowed is not None and parent.get("doc_id") not in allowed:
                continue
            text = parent["text"]
            if selected and used_chars + len(text) > cfg.context_max_chars:
                continue
            seen_parents.add(parent["parent_id"])
            used_chars += len(text)
            selected.append(
                RetrievedContext(
                    index=len(selected) + 1,
                    parent_id=parent["parent_id"],
                    doc_id=child["doc_id"],
                    title=parent["title"],
                    section=parent["section"],
                    text=text,
                    score=rerank_scores.get(hit.child_id),
                    child_ids=[hit.child_id],
                    acl=child.get("acl", "all"),
                )
            )

        # 同一个父块被多个子块命中时，补全 child_ids（便于解释「为什么召回它」）
        for hit in rerank_hits:
            child = self.children.get(hit.child_id)
            if not child:
                continue
            for ctx in selected:
                if ctx.parent_id == child["parent_id"] and hit.child_id not in ctx.child_ids:
                    ctx.child_ids.append(hit.child_id)

        # 6) 可观测性：给每条候选补齐来源与各路分数
        candidate_trace = []
        for rank, hit in enumerate(candidates, start=1):
            child = self.children.get(hit.child_id, {})
            candidate_trace.append(
                {
                    **hit.as_dict(),
                    "rank": rank,
                    "title": child.get("title", ""),
                    "section": child.get("section", ""),
                    "parent_id": child.get("parent_id", ""),
                    "preview": child.get("text", "")[:90],
                    "rerank_score": (
                        None
                        if rerank_scores.get(hit.child_id) is None
                        else round(rerank_scores[hit.child_id], 4)
                    ),
                    "selected": any(hit.child_id in ctx.child_ids for ctx in selected),
                }
            )

        scores = [hit.score for hit in rerank_hits if hit.score is not None]
        best_score = max(scores) if scores else (candidates[0].score if candidates else None)
        timings["total_ms"] = _ms(started)

        # 7) 字段级脱敏：在进入 Prompt / trace / 引用之前就替换掉明文
        if settings.mask_pii and not acl_summary["pii_allowed"]:
            masked = 0
            for ctx in selected:
                ctx.text, count = mask_text(ctx.text)
                masked += count
            for candidate in candidate_trace:
                candidate["preview"], count = mask_text(candidate["preview"])
                masked += count
            acl_summary["pii_masked"] = masked

        return RetrievalResult(
            query=query,
            contexts=selected,
            candidates=candidate_trace,
            timings=timings,
            usage=usages,
            best_score=best_score,
            low_confidence=best_score is None or best_score < cfg.min_score,
            acl=acl_summary,
        )

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        return {
            "ready": self.ready,
            "docs": len(self.docs),
            "parents": len(self.parents),
            "children": len(self.children),
            "vector_backend": getattr(self._store, "name", None),
            "vector_location": getattr(self._store, "location", None),
            "manifest": self.manifest,
        }

    def close(self) -> None:
        """释放向量库客户端（嵌入式 Qdrant 会持有目录锁）。"""
        if self._store is not None:
            close = getattr(self._store, "close", None)
            if callable(close):
                close()
            self._store = None
        self._loaded = False

    def document_index(self) -> list[dict]:
        """给前端/运维看的文档清单（含 ACL 与块数）。"""
        return [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "acl": doc.get("acl", "all"),
                "department": doc.get("department", ""),
                "children": doc.get("children", 0),
                "updated_at": doc.get("indexed_at"),
            }
            for doc in self.docs
        ]


_KB: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    """进程级单例；索引缺失时不抛异常，让 /api/health 能报告 not_ready。"""
    global _KB
    if _KB is None:
        kb = KnowledgeBase()
        _KB = kb  # 先挂载再加载，避免加载失败时单例为 None
        try:
            kb.load()
        except IndexNotReady as exc:
            logger.warning("%s", exc)
    return _KB


def reindex(incremental: bool = True) -> dict:
    """重建索引并热替换单例（管理接口 / 脚本复用）。

    先释放旧客户端再重建：Qdrant 嵌入式模式会持有目录锁。
    构建失败时旧索引仍可用，finally 会把它重新加载回来。
    """
    global _KB
    if _KB is not None:
        _KB.close()

    try:
        manifest = rebuild_atomic(settings.index_dir, incremental=incremental)
    finally:
        kb = KnowledgeBase()
        try:
            kb.load()
        except IndexNotReady:
            kb = None
        _KB = kb

    return manifest


def _effective_backend() -> str:
    """判断当前会落到哪个向量后端（不真正建客户端）。"""
    backend = os.environ.get("VECTOR_BACKEND", "auto").lower()
    if backend in {"local", "numpy"}:
        return "local"
    if backend == "qdrant":
        return "qdrant"
    if os.environ.get("QDRANT_URL", "").strip():
        return "qdrant"
    try:
        import qdrant_client  # noqa: F401

        return "qdrant"
    except Exception:  # noqa: BLE001
        return "local"


def rebuild_atomic(
    target_dir: Path, docs: list | None = None, incremental: bool = True
) -> dict:
    """在 staging 目录构建，完成后原子替换 target_dir。

    只有本地 numpy 后端启用原子替换。Qdrant（服务端或嵌入式）的存储在
    自己的集合/目录里，复制目录并不能复制向量，因此退化为原地重建。
    生产若需要原子切换，应使用版本化 collection + alias。
    """
    target = Path(target_dir)

    if _effective_backend() == "qdrant":
        logger.warning(
            "检测到 Qdrant 后端：向量存储不在索引目录内，跳过目录级原子替换，改为原地重建。"
            "生产环境可用版本化 collection + alias 做原子切换。"
        )
        kb = KnowledgeBase(index_dir=target)
        try:
            manifest = kb.build(docs=docs, incremental=incremental)
        finally:
            kb.close()
        manifest["swap"] = {"atomic": False, "reason": "qdrant"}
        return manifest

    staging = target.with_name(f"{target.name}.staging")
    backup = target.with_name(f"{target.name}.old")

    if staging.exists():
        shutil.rmtree(staging)
    if incremental and target.exists():
        shutil.copytree(target, staging)
    else:
        staging.mkdir(parents=True, exist_ok=True)

    try:
        kb = KnowledgeBase(index_dir=staging)
        manifest = kb.build(docs=docs, incremental=incremental)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)  # 目录 rename，同文件系统内是原子的
    os.replace(staging, target)
    shutil.rmtree(backup, ignore_errors=True)

    manifest["swap"] = {"atomic": True, "staging": staging.name}
    return manifest


def reset_kb() -> None:
    """测试用：清空单例。"""
    global _KB
    _KB = None

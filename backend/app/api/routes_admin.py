"""运维接口：健康检查、索引统计、重建索引、运行指标。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.agent.graph import LANGGRAPH_AVAILABLE
from app.config import settings
from app.core.metrics import metrics
from app.llm.client import get_llm
from app.retrieval.service import get_kb, reindex
from app.services.chat_service import get_chat_service

router = APIRouter(tags=["admin"])


@router.get("/health")
def health():
    kb = get_kb()
    llm = get_llm()
    backends = getattr(llm, "describe", lambda: {})()
    return {
        "status": "ok" if kb.ready else "degraded",
        "app": settings.app_name,
        "provider": get_llm().name,
        "mock_mode": settings.mock_mode,
        "langgraph": LANGGRAPH_AVAILABLE,
        "models": {
            "generation": settings.generation_model,
            "fast": settings.fast_model,
            "embedding": backends.get("embedding") or settings.model_embedding,
            "rerank": backends.get("rerank") or settings.model_rerank,
        },
        "backends": backends,
        "index": kb.stats(),
    }


@router.get("/metrics")
def get_metrics():
    return metrics.snapshot()


@router.get("/documents")
def documents():
    """文档清单与 ACL，方便排查「为什么这篇没被召回」。"""
    kb = get_kb()
    return {"ready": kb.ready, "items": kb.document_index()}


@router.post("/admin/reindex")
def rebuild_index(incremental: bool = True):
    try:
        manifest = reindex(incremental=incremental)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"重建索引失败：{exc}") from exc

    service = get_chat_service()
    service.kb = None
    service.runtime.kb = None
    service.runtime.tools._kb = None  # noqa: SLF001 - 演示项目，直接重置工具持有的引用
    return {"ok": True, "manifest": manifest}

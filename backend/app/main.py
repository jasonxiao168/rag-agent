"""FastAPI 入口。

启动：
    uvicorn app.main:app --reload --port 8000
接口文档：
    http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_admin, routes_chat
from app.config import REPO_DIR, settings
from app.core.logging import get_logger, setup_logging
from app.retrieval.service import IndexNotReady, get_kb

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings.ensure_dirs()
    if settings.mock_mode:
        logger.warning("当前为 MOCK 模式：未检测到 DASHSCOPE_API_KEY，回答由规则模拟")
    try:
        kb = get_kb()
        logger.info("知识库就绪：%s", kb.stats())
    except IndexNotReady as exc:
        logger.warning("知识库未就绪：%s", exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="RAG + LangGraph Agent + Function Calling 的企业知识库助手",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(routes_chat.router, prefix="/api")
    app.include_router(routes_admin.router, prefix="/api")

    # 构建过前端（npm run build）就直接托管，一条命令即可演示完整全栈
    dist_dir = REPO_DIR / "frontend" / "dist"
    if dist_dir.exists():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="ui")
    else:
        @app.get("/")
        def root() -> JSONResponse:
            return JSONResponse(
                {
                    "app": settings.app_name,
                    "docs": "/docs",
                    "api": "/api/chat",
                    "hint": "前端未构建：cd frontend && npm install && npm run dev",
                }
            )

    return app


app = create_app()

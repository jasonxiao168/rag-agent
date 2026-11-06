"""对话接口：POST /api/chat，支持 SSE 流式与非流式两种模式。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.services.chat_service import get_chat_service
from app.services.users import list_users, resolve_user

logger = get_logger("api.chat")
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str | None = Field(None, description="会话 ID，不传则新建")
    user_id: str | None = Field(None, description="演示身份：guest / alice / bob / carol / dave")
    stream: bool = Field(True, description="是否使用 SSE 流式返回")


@router.get("/users")
def users():
    """演示用身份列表；生产环境应由 SSO 注入，不暴露此接口。"""
    return {"items": list_users()}


@router.post("/chat")
def chat(payload: ChatRequest):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    service = get_chat_service()

    if not payload.stream:
        try:
            return service.answer(payload.session_id, message, payload.user_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("非流式对话失败")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # SSE：关闭代理缓冲，避免 nginx 攒够 buffer 才下发，导致「卡住再一次性刷出」
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        service.stream(payload.session_id, message, payload.user_id),
        media_type="text/event-stream",
        headers=headers,
    )


@router.get("/sessions")
def list_sessions(limit: int = 20, user_id: str | None = None):
    """只返回当前身份的会话；生产环境 user_id 应由登录态注入。"""
    user = resolve_user(user_id)
    return {"items": get_chat_service().sessions.list_sessions(user["id"], limit=limit)}


@router.get("/sessions/{session_id}")
def get_session(session_id: str, user_id: str | None = None):
    user = resolve_user(user_id)
    try:
        memory = get_chat_service().sessions.load_for_user(session_id, user["id"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if memory is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        "session_id": memory.session_id,
        "summary": memory.summary,
        "turns": [t.as_dict() for t in memory.turns],
        "updated_at": memory.updated_at,
        "user_id": memory.user_id,
    }


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, user_id: str | None = None):
    user = resolve_user(user_id)
    try:
        deleted = get_chat_service().sessions.delete(session_id, user["id"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True, "session_id": session_id}

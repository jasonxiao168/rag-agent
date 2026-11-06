"""Agent 状态定义 + 链路追踪结构。

state 在 LangGraph 的节点之间传递；每个节点只返回自己改动的字段。
trace 是「给人看的链路日志」，前端可视化面板直接消费它。
"""
from __future__ import annotations

from typing import Any, TypedDict

from app.core.tokens import Usage


class AgentState(TypedDict, total=False):
    session_id: str
    question: str
    history_summary: str
    recent_turns: list[dict]

    # 身份与权限：检索阶段据此过滤文档
    user_id: str
    user_name: str
    departments: list[str]
    permissions: list[str]
    acl: dict

    route: str                 # kb / tool / smalltalk / clarify
    route_reason: str
    answer_style: str          # kb / tool / smalltalk

    search_query: str
    contexts: list[dict]
    sources: list[dict]
    candidates: list[dict]
    timings: dict
    best_score: float | None
    low_confidence: bool

    grade_sufficient: bool | None
    grade_reason: str
    rewritten_query: str
    retries: int

    tool_calls: list[dict]
    tool_results: list[dict]

    gen_messages: list[dict]
    answer_override: str | None
    trace: list[dict]
    usages: list[Usage]
    error: str | None


def trace_step(
    step: str,
    title: str,
    detail: str = "",
    *,
    data: dict | None = None,
    latency_ms: float | None = None,
    usage: Usage | None = None,
    status: str = "ok",
) -> dict:
    """统一的 trace 结构，前端按 step 类型渲染不同卡片。"""
    payload: dict[str, Any] = {
        "step": step,
        "title": title,
        "detail": detail,
        "status": status,
        "data": data or {},
    }
    if latency_ms is not None:
        payload["latency_ms"] = round(latency_ms, 1)
    if usage is not None:
        payload["usage"] = usage.as_dict()
    return payload

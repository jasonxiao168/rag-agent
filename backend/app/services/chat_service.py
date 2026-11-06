"""对话编排：Agent 执行 → 流式生成 → 记忆更新，并把全过程编码成 SSE。

为什么把「流式生成」放在图外面？
LangGraph 负责决策与检索（一次性完成），真正的 token 生成用流式接口单独推送，
这样既能拿到完整的 trace，又不会因为图节点里做流式而失去可观测性。
"""
from __future__ import annotations

import time
from typing import Iterator

from app.agent.graph import AgentRuntime, build_state
from app.agent.memory import SessionMemory, SessionStore
from app.config import settings
from app.core.logging import get_logger, new_trace_id
from app.core.metrics import metrics
from app.core.sse import sse_event
from app.core.tokens import UsageTracker
from app.llm.client import LLMClient, get_llm
from app.retrieval.service import KnowledgeBase, get_kb
from app.services.users import resolve_user

logger = get_logger("services.chat")

REFUSAL_MARKERS = ("抱歉", "不知道", "无法", "没有找到", "查不到")


def _is_refusal(answer: str) -> bool:
    """拒答时不应展示引用：否则会把无关文档误标成「依据」。"""
    head = answer.strip()[:30]
    return any(marker in head for marker in REFUSAL_MARKERS)


class ChatService:
    def __init__(
        self,
        kb: KnowledgeBase | None = None,
        llm: LLMClient | None = None,
        sessions: SessionStore | None = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.kb = kb
        self.sessions = sessions or SessionStore()
        self.runtime = AgentRuntime(kb=self.kb, llm=self.llm)

    @property
    def knowledge(self) -> KnowledgeBase:
        if self.kb is None:
            self.kb = get_kb()
        return self.kb

    # ------------------------------------------------------------------ #
    # 非流式：评测脚本、/api/chat?stream=false 使用
    # ------------------------------------------------------------------ #
    def answer(self, session_id: str | None, message: str, user_id: str | None = None) -> dict:
        started = time.perf_counter()
        user = resolve_user(user_id)
        memory = self.sessions.get(session_id, user["id"])
        state = build_state(message, memory, memory.session_id, user=user)
        result = self.runtime.graph.invoke(state)

        usages = list(result.get("usages") or [])
        gen_messages = result.get("gen_messages") or []
        answer = "抱歉，系统暂时无法处理这个问题。"

        override = result.get("answer_override")
        if override:
            # 配置类错误（如索引与模型维度不一致）：直接给出明确原因，不经过模型
            answer = str(override)
            usages.extend(memory.finalize_turn(message, answer))
            memory.save()
        elif gen_messages:
            try:
                response = self.llm.chat(
                    gen_messages,
                    model=settings.generation_model,
                    stage="answer",
                    max_tokens=800,
                    temperature=0.25,
                )
                answer = response.text.strip() or answer
                usages.append(response.usage)
            except Exception as exc:  # noqa: BLE001
                logger.exception("回答生成失败")
                answer = f"抱歉，生成回答时出错了：{exc}"
            usages.extend(memory.finalize_turn(message, answer))
            memory.save()
        else:
            usages.extend(memory.finalize_turn(message, answer))
            memory.save()

        tracker = UsageTracker()
        for usage in usages:
            tracker.add(usage)

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        payload = {
            "session_id": memory.session_id,
            "answer": answer,
            "route": result.get("route"),
            "route_reason": result.get("route_reason"),
            "sources": [] if _is_refusal(answer) else (result.get("sources") or []),
            "trace": result.get("trace") or [],
            "usage": tracker.as_dict(),
            "timings": result.get("timings") or {},
            "acl": result.get("acl") or {},
            "user": {"id": user["id"], "name": user["name"], "departments": user["departments"]},
            "latency_ms": latency_ms,
        }
        metrics.record(payload["route"], payload["usage"], latency_ms)
        return payload

    # ------------------------------------------------------------------ #
    # 流式：SSE
    # ------------------------------------------------------------------ #
    def stream(self, session_id: str | None, message: str, user_id: str | None = None) -> Iterator[str]:
        started = time.perf_counter()
        trace_id = new_trace_id()
        user = resolve_user(user_id)
        memory = self.sessions.get(session_id, user["id"])
        state = build_state(message, memory, memory.session_id, user=user)

        yield sse_event(
            "meta",
            {
                "session_id": memory.session_id,
                "trace_id": trace_id,
                "provider": self.llm.name,
                "mock": settings.mock_mode,
                "model": settings.generation_model,
                "history_turns": len(memory.turns),
                "summary_chars": len(memory.summary),
                "user": {
                    "id": user["id"],
                    "name": user["name"],
                    "departments": user["departments"],
                    "department_label": user.get("department_label", ""),
                },
            },
        )

        try:
            result = self.runtime.graph.invoke(state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent 执行失败")
            yield sse_event("error", {"message": f"Agent 执行失败：{exc}"})
            return

        for step in result.get("trace") or []:
            yield sse_event("trace", step)

        sources = result.get("sources") or []

        usages = list(result.get("usages") or [])
        gen_messages = result.get("gen_messages") or []
        answer_parts: list[str] = []
        override = result.get("answer_override")

        if override:
            # 配置类错误：直接按块推送明确原因，不调用模型
            text = str(override)
            for index in range(0, len(text), 12):
                answer_parts.append(text[index : index + 12])
                yield sse_event("token", {"delta": text[index : index + 12]})
        elif gen_messages:
            sink: list = []
            try:
                for delta in self.llm.stream_chat(
                    gen_messages,
                    model=settings.generation_model,
                    temperature=0.25,
                    max_tokens=800,
                    stage="answer",
                    usage_sink=sink,
                ):
                    answer_parts.append(delta)
                    yield sse_event("token", {"delta": delta})
            except Exception as exc:  # noqa: BLE001
                logger.exception("流式生成失败")
                usages.extend(sink)
                yield sse_event("error", {"message": f"生成失败：{exc}"})
                # 出错也要把已生成的内容落库，避免会话状态不一致
            usages.extend(sink)

        answer = "".join(answer_parts).strip()
        if not answer:
            answer = "抱歉，我没有找到相关信息。"

        usages.extend(memory.finalize_turn(message, answer))
        memory.save()

        tracker = UsageTracker()
        for usage in usages:
            tracker.add(usage)

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        metrics.record(result.get("route"), tracker.as_dict(), latency_ms)

        # 引用放在生成之后：先确认不是拒答，避免把无关文档标成「依据」
        if _is_refusal(answer):
            sources = []
        elif sources:
            yield sse_event("sources", {"items": sources})

        yield sse_event(
            "usage",
            {
                **tracker.as_dict(),
                "latency_ms": latency_ms,
                "chat_tokens": len(answer),
            },
        )
        yield sse_event(
            "done",
            {
                "session_id": memory.session_id,
                "trace_id": trace_id,
                "answer": answer,
                "route": result.get("route"),
                "sources": sources,
                "acl": result.get("acl") or {},
                "user": {"id": user["id"], "name": user["name"], "departments": user["departments"]},
                "latency_ms": latency_ms,
            },
        )


_APP: ChatService | None = None


def get_chat_service() -> ChatService:
    global _APP
    if _APP is None:
        _APP = ChatService()
    return _APP


def reset_chat_service() -> None:
    global _APP
    _APP = None

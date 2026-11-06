"""LangGraph 编排的 Agent 工作流。

                  ┌──────────────┐
用户问题 ────────▶│    router    │  小模型 + JSON 输出，判断意图
                  └──────┬───────┘
        ┌────────────────┼──────────────────┬─────────────┐
        ▼ kb             ▼ tool             ▼ smalltalk   ▼ clarify
   ┌──────────┐    ┌────────────┐      ┌───────────┐  ┌───────────┐
   │ retrieve │    │  tool_exec │      │  prepare  │  │  prepare  │
   └────┬─────┘    └──────┬─────┘      └───────────┘  └───────────┘
        ▼                 │
   ┌──────────┐           │
   │  grade   │◀──┐       │   检索质量不足 → 改写 query → 重新检索（有次数上限）
   └────┬─────┘   │       │   这是 LangGraph 相对 AgentExecutor 的核心优势：
        │ 不足    │       │   显式状态 + 可控循环，而不是黑盒 while。
        └──▶ rewrite ─────┘
        │ 足够
        ▼
   ┌──────────┐
   │ prepare  │──────────▶ END（真正的流式生成在 chat_service 里做，便于逐 token 推送）
   └──────────┘
"""
from __future__ import annotations

import re
import time

from app.agent.memory import SessionMemory
from app.agent.state import AgentState, trace_step
from app.agent.tools import TOOL_SCHEMAS, ToolExecutor
from app.config import settings
from app.core.logging import get_logger
from app.llm.client import LLMClient, extract_json, get_llm
from app.llm.prompts import (
    clarify_messages,
    compose_messages,
    grade_messages,
    router_messages,
    smalltalk_messages,
    tool_answer_messages,
    tool_planner_messages,
)
from app.retrieval.service import IndexDimensionMismatch, KnowledgeBase, get_kb

logger = get_logger("agent.graph")

try:  # 真库优先
    from langgraph.graph import END, StateGraph  # type: ignore

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于环境
    from app.agent.minigraph import END, StateGraph

    LANGGRAPH_AVAILABLE = False


ORDER_ID = re.compile(r"ORD[-_ ]?\d{4}[-_ ]?\d{3,4}", re.I)
# 指代/省略开头：这类问题本身不带实体，需要拼上上一轮用户问题才能检索
FOLLOWUP = re.compile(r"^(那|这|它|它们|他|她|还有|另外|上面|刚才|之前)")


class AgentRuntime:
    """把节点依赖（知识库、模型、工具、记忆）收拢在一个对象里，方便测试注入。"""

    def __init__(
        self,
        kb: KnowledgeBase | None = None,
        llm: LLMClient | None = None,
        tools: ToolExecutor | None = None,
    ) -> None:
        self.kb = kb
        self.llm = llm or get_llm()
        self.tools = tools or ToolExecutor(kb=kb)
        self.graph = self._build()

    @property
    def knowledge(self) -> KnowledgeBase:
        if self.kb is None:
            self.kb = get_kb()
        return self.kb

    @staticmethod
    def _expand_followup(
        question: str, query: str, recent_turns: list[dict]
    ) -> tuple[str, bool]:
        """模型没有改写、且问题以「那/它/还有」开头时，用上一轮用户问题补全。

        真实模型通常能自己改写；这一步是兜底，避免指代词直接拿去检索必然落空。
        """
        if query.strip() and query.strip() != question.strip():
            return query, False
        if not FOLLOWUP.search(question.strip()):
            return query, False
        previous = next(
            (turn["content"] for turn in reversed(recent_turns) if turn.get("role") == "user"),
            "",
        )
        if not previous:
            return query, False
        return f"{previous} {question}", True

    # ------------------------------------------------------------------ #
    # 节点
    # ------------------------------------------------------------------ #
    def router(self, state: AgentState) -> dict:
        started = time.perf_counter()
        question = state["question"]
        recent_turns = state.get("recent_turns") or []

        # 先做指代补全再路由：否则「它什么时候到」里没有订单号，会被判成知识问题
        routing_question, followup = self._expand_followup(question, question, recent_turns)
        messages = router_messages(
            routing_question,
            state.get("history_summary", ""),
            recent_turns,
        )

        data: dict = {}
        usage = None
        status = "ok"
        try:
            result = self.llm.chat(
                messages,
                model=settings.fast_model,
                json_mode=True,
                max_tokens=200,
                stage="route",
                temperature=0.0,
            )
            usage = result.usage
            data = extract_json(result.text)
            if not data.get("intent"):
                status = "warn"
        except Exception as exc:  # noqa: BLE001
            logger.warning("路由失败，降级为 kb：%s", exc)
            status = "error"
            data = {"reason": f"路由失败：{exc}"}

        intent = str(data.get("intent") or "kb").lower()
        if intent not in {"kb", "tool", "smalltalk", "clarify"}:
            intent = "kb"
        reason = str(data.get("reason") or "")
        search_query = str(data.get("search_query") or routing_question)
        expanded = followup

        trace = list(state.get("trace") or [])
        trace.append(
            trace_step(
                "route",
                "意图路由",
                f"{intent} · {reason}" if reason else intent,
                data={
                    "intent": intent,
                    "reason": reason,
                    "model": settings.fast_model,
                    "search_query": search_query,
                    "query_expanded": expanded,
                },
                latency_ms=(time.perf_counter() - started) * 1000,
                usage=usage,
                status=status,
            )
        )
        return {
            "route": intent,
            "route_reason": reason,
            "answer_style": intent,
            "search_query": search_query,
            "retries": state.get("retries", 0),
            "trace": trace,
            "usages": list(state.get("usages") or []) + ([usage] if usage else []),
        }

    def retrieve(self, state: AgentState) -> dict:
        started = time.perf_counter()
        trace = list(state.get("trace") or [])
        usages = list(state.get("usages") or [])
        query = state.get("search_query") or state["question"]

        if not self.knowledge.ready:
            trace.append(
                trace_step(
                    "retrieval",
                    "知识库未就绪",
                    "索引不存在，请先执行 python -m scripts.ingest",
                    status="error",
                )
            )
            return {"contexts": [], "sources": [], "answer_style": "kb", "trace": trace}

        try:
            result = self.knowledge.search(
                query,
                departments=state.get("departments"),
                permissions=state.get("permissions"),
            )
        except IndexDimensionMismatch as exc:
            # 明确的配置错误：不要伪装成“不知道”，把原因写进链路和回答
            trace.append(
                trace_step(
                    "retrieval",
                    "索引与模型不匹配",
                    str(exc),
                    data={"error": str(exc)},
                    status="error",
                )
            )
            return {
                "contexts": [],
                "sources": [],
                "answer_style": "clarify",
                "error": str(exc),
                "answer_override": f"知识库索引与当前模型不匹配，暂时无法检索。{exc}",
                "trace": trace,
            }
        usages.extend(result.usage)

        trace.append(
            trace_step(
                "retrieval",
                "混合检索 + Rerank",
                f"召回 {len(result.candidates)} 条子块 → 精排后 {len(result.contexts)} 个父块"
                + (
                    f" · 可见文档 {result.acl.get('documents_visible')}/{result.acl.get('documents_total')}"
                    if result.acl.get("restricted")
                    else ""
                ),
                data={
                    "query": query,
                    "timings": result.timings,
                    "best_score": None if result.best_score is None else round(result.best_score, 4),
                    "candidates": result.candidates[:6],
                    "acl": result.acl,
                    "contexts": [
                        {"index": c.index, "title": c.title, "section": c.section, "chars": len(c.text)}
                        for c in result.contexts
                    ],
                },
                latency_ms=(time.perf_counter() - started) * 1000,
                usage=None,
            )
        )
        return {
            "contexts": [c.as_dict() for c in result.contexts],
            "sources": result.citations,
            "candidates": result.candidates,
            "timings": result.timings,
            "best_score": result.best_score,
            "low_confidence": result.low_confidence,
            "acl": result.acl,
            "answer_style": "kb",
            "trace": trace,
            "usages": usages,
        }

    def grade(self, state: AgentState) -> dict:
        """Corrective RAG：判断检索结果够不够回答问题。

        成本优化：先用 Rerank 分数做两段判定，只有灰区才调用小模型，
        高频场景下能省掉大部分评测调用。
        """
        started = time.perf_counter()
        trace = list(state.get("trace") or [])
        usages = list(state.get("usages") or [])
        contexts = state.get("contexts") or []
        best = state.get("best_score")

        # 检索阶段已经明确报错（如索引与模型维度不一致），跳过判定与重试
        if state.get("error"):
            trace.append(
                trace_step(
                    "grade",
                    "跳过检索质量判定",
                    "检索阶段已报配置错误，重试没有意义",
                    status="error",
                )
            )
            return {
                "grade_sufficient": False,
                "grade_reason": str(state.get("error")),
                "rewritten_query": "",
                "trace": trace,
            }

        usage = None
        if not contexts:
            sufficient, reason, rewritten = False, "未检索到相关内容", state["question"]
        elif best is not None and best >= settings.grade_high_score:
            sufficient, reason, rewritten = True, f"Rerank 分数 {best:.2f} 高于阈值，跳过模型判定", ""
        elif best is not None and best < settings.grade_low_score:
            sufficient, reason, rewritten = False, f"Rerank 分数 {best:.2f} 过低", state["question"]
        else:
            try:
                result = self.llm.chat(
                    grade_messages(state["question"], contexts),
                    model=settings.fast_model,
                    json_mode=True,
                    max_tokens=200,
                    stage="grade",
                    temperature=0.0,
                )
                usage = result.usage
                data = extract_json(result.text)
                sufficient = bool(data.get("sufficient"))
                reason = str(data.get("reason") or "")
                rewritten = str(data.get("rewritten_query") or "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("检索质量判定失败，保守按不足处理：%s", exc)
                sufficient, reason, rewritten = False, f"判定失败：{exc}", ""

        retries = state.get("retries", 0)
        can_retry = (not sufficient) and retries < settings.max_rewrite_retries

        trace.append(
            trace_step(
                "grade",
                "检索质量判定（CRAG）",
                f"{'充分' if sufficient else '不足'} · {reason}",
                data={
                    "sufficient": sufficient,
                    "rewritten_query": rewritten,
                    "will_retry": can_retry,
                    "model": settings.fast_model if usage else None,
                },
                latency_ms=(time.perf_counter() - started) * 1000,
                usage=usage,
                status="ok" if sufficient else "warn",
            )
        )
        return {
            "grade_sufficient": sufficient,
            "grade_reason": reason,
            "rewritten_query": rewritten,
            "trace": trace,
            "usages": usages + ([usage] if usage else []),
        }

    def rewrite(self, state: AgentState) -> dict:
        new_query = state.get("rewritten_query") or f"{state['question']} 规定 标准"
        trace = list(state.get("trace") or [])
        trace.append(
            trace_step(
                "rewrite",
                "查询改写后重试",
                f"{state.get('search_query', '')} → {new_query}",
                data={"retry": state.get("retries", 0) + 1, "new_query": new_query},
            )
        )
        return {
            "search_query": new_query,
            "retries": state.get("retries", 0) + 1,
            "trace": trace,
        }

    def tool_exec(self, state: AgentState) -> dict:
        started = time.perf_counter()
        trace = list(state.get("trace") or [])
        usages = list(state.get("usages") or [])
        question = state["question"]
        # 工具规划同样需要指代补全：否则「它什么时候到」里没有订单号，模型只能反问
        planner_question, _ = self._expand_followup(
            question, question, state.get("recent_turns") or []
        )

        usage = None
        calls = []
        try:
            planned = self.llm.chat_with_tools(
                tool_planner_messages(planner_question, state.get("history_summary", "")),
                TOOL_SCHEMAS,
                model=settings.fast_model,
                stage="tool_plan",
            )
            usage = planned.usage
            calls = planned.tool_calls
        except Exception as exc:  # noqa: BLE001
            logger.warning("工具规划失败，尝试规则兜底：%s", exc)

        # 规则兜底：模型没给出 tool_call 时，直接用订单号正则兜住核心场景
        if not calls:
            match = ORDER_ID.search(planner_question)
            if match:
                order_id = match.group(0).replace(" ", "").replace("_", "-").upper()
                trace.append(
                    trace_step(
                        "tool_plan",
                        "工具规划兜底",
                        f"模型未返回 tool_call，按订单号直接调用 query_user_order({order_id})",
                        data={"order_id": order_id},
                        status="warn",
                    )
                )
                from app.llm.client import ToolCall  # 局部导入，避免循环依赖

                calls = [ToolCall(id="fallback_1", name="query_user_order", arguments={"order_id": order_id})]
            else:
                trace.append(
                    trace_step(
                        "tool_plan",
                        "工具规划",
                        "模型未选择任何工具",
                        status="warn",
                    )
                )

        executions = []
        for call in calls[:3]:
            execution = self.tools.execute(
                call.name,
                call.arguments,
                departments=state.get("departments"),
                permissions=state.get("permissions"),
            )
            executions.append(execution)
            trace.append(
                trace_step(
                    "tool_call",
                    f"调用工具 · {execution.name}",
                    execution.content[:160],
                    data={
                        "name": execution.name,
                        "arguments": execution.arguments,
                        "ok": execution.ok,
                        "result": execution.content,
                        "latency_ms": execution.latency_ms,
                    },
                    status="ok" if execution.ok else "warn",
                )
            )

        return {
            "tool_calls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments} for call in calls
            ],
            "tool_results": [item.as_dict() for item in executions],
            "answer_style": "tool" if executions else "clarify",
            "trace": trace,
            "usages": usages + ([usage] if usage else []),
        }

    def prepare(self, state: AgentState) -> dict:
        route = state.get("route") or "kb"
        style = state.get("answer_style") or route
        question = state["question"]
        summary = state.get("history_summary", "")
        contexts = state.get("contexts") or []

        if style == "tool":
            if state.get("tool_results"):
                messages = tool_answer_messages(question, state["tool_results"])
                detail = f"基于 {len(state['tool_results'])} 个工具结果整理回答"
            else:
                messages = clarify_messages(question, "订单号或具体诉求")
                detail = "缺少订单号，向用户追问"
        elif style in {"smalltalk", "clarify"} or route in {"smalltalk", "clarify"}:
            if route == "smalltalk" and style == "smalltalk":
                messages = smalltalk_messages(question)
                detail = "闲聊，跳过检索"
            else:
                messages = clarify_messages(question, "问题指代或关键条件")
                detail = "意图不明确，向用户追问"
        else:
            messages = compose_messages(question, contexts, summary)
            detail = f"注入 {len(contexts)} 个父块上下文"

        trace = list(state.get("trace") or [])
        trace.append(
            trace_step(
                "compose",
                "组装 Prompt",
                detail,
                data={
                    "style": style,
                    "prompt_chars": sum(len(m["content"]) for m in messages),
                    "model": settings.generation_model,
                    # 完整 Prompt 只放在 trace 里，前端可展开查看；内容已做过 ACL / PII 处理
                    "messages": [
                        {"role": message.get("role", ""), "content": message.get("content", "")}
                        for message in messages
                    ],
                },
            )
        )
        return {"gen_messages": messages, "trace": trace}

    # ------------------------------------------------------------------ #
    # 条件边
    # ------------------------------------------------------------------ #
    @staticmethod
    def route_after_router(state: AgentState) -> str:
        route = state.get("route", "kb")
        return route if route in {"kb", "tool", "smalltalk", "clarify"} else "kb"

    @staticmethod
    def route_after_grade(state: AgentState) -> str:
        if state.get("error"):
            return "prepare"
        if state.get("grade_sufficient"):
            return "prepare"
        retries = state.get("retries", 0)
        if retries < settings.max_rewrite_retries and not state.get("grade_sufficient"):
            return "rewrite"
        return "prepare"

    @staticmethod
    def route_after_tool(state: AgentState) -> str:
        # 工具没调起来时，降级去知识库找找，而不是硬编一个答案
        if not state.get("tool_results") and not ORDER_ID.search(state.get("question", "")):
            return "retrieve_fallback"
        return "prepare"

    # ------------------------------------------------------------------ #
    # 构图
    # ------------------------------------------------------------------ #
    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("router", self.router)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("grade", self.grade)
        graph.add_node("rewrite", self.rewrite)
        graph.add_node("tool_exec", self.tool_exec)
        graph.add_node("prepare", self.prepare)

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            self.route_after_router,
            {"kb": "retrieve", "tool": "tool_exec", "smalltalk": "prepare", "clarify": "prepare"},
        )
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade", self.route_after_grade, {"rewrite": "rewrite", "prepare": "prepare"}
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_conditional_edges(
            "tool_exec",
            self.route_after_tool,
            {"prepare": "prepare", "retrieve_fallback": "retrieve"},
        )
        graph.add_edge("prepare", END)
        return graph.compile()


def build_state(
    question: str,
    memory: SessionMemory | None = None,
    session_id: str = "",
    user: dict | None = None,
) -> AgentState:
    summary, recent = memory.context() if memory else ("", [])
    user = user or {"id": "guest", "name": "访客", "departments": ["public"]}
    return AgentState(
        session_id=session_id,
        question=question,
        history_summary=summary,
        recent_turns=recent,
        user_id=user.get("id", "guest"),
        user_name=user.get("name", "访客"),
        departments=user.get("departments", ["public"]),
        permissions=user.get("permissions", []),
        route="",
        route_reason="",
        search_query=question,
        contexts=[],
        sources=[],
        candidates=[],
        timings={},
        best_score=None,
        low_confidence=False,
        grade_sufficient=None,
        grade_reason="",
        rewritten_query="",
        retries=0,
        tool_calls=[],
        tool_results=[],
        gen_messages=[],
        answer_override=None,
        trace=[],
        usages=[],
        error=None,
    )

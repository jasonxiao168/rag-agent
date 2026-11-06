"""Agent 工具定义与执行器。

工具用 OpenAI Function Calling 的 JSON Schema 描述：
模型只负责「选工具 + 填参数」，真正的副作用（查库、建单）在本地执行。
这样做的好处：权限、审计、超时、降级都掌握在服务端，不依赖模型自觉。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.retrieval.service import get_kb
from app.services import mock_db

logger = get_logger("agent.tools")

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "检索公司内部知识库，包括产品资料、员工制度、客服规范、数据安全规范、经销商政策。"
                "当用户询问规定、标准、流程、成分、政策时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询，使用用户问题中的关键名词，不要添加无关修饰",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_user_order",
            "description": "根据订单号查询订单状态、物流进度、退款进度。用户提供 ORD- 开头的订单号时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，格式如 ORD-2026-1001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "为用户创建售后工单。用户要求投诉、转人工、登记售后问题，或需要人工跟进时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "关联订单号，没有则传空字符串"},
                    "reason": {"type": "string", "description": "用户的问题描述"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "description": "紧急程度，涉及过敏、舆情用 high",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    },
]

TOOL_NAMES = {schema["function"]["name"] for schema in TOOL_SCHEMAS}


@dataclass
class ToolExecution:
    name: str
    arguments: dict
    ok: bool
    content: str          # 回填给模型的可读结果
    data: dict = field(default_factory=dict)
    latency_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "result": self.content,
            "latency_ms": self.latency_ms,
        }


class ToolExecutor:
    def __init__(self, kb=None) -> None:
        self._kb = kb

    @property
    def kb(self):
        return self._kb or get_kb()

    def execute(
        self,
        name: str,
        arguments: dict,
        departments: list[str] | None = None,
        permissions: list[str] | None = None,
    ) -> ToolExecution:
        started = time.perf_counter()
        try:
            if name == "search_knowledge_base":
                # 权限必须跟着工具调用一起走，否则工具会变成绕过 ACL / 脱敏的后门
                content, data, ok = self._search(arguments, departments, permissions)
            elif name == "query_user_order":
                content, data, ok = self._order(arguments)
            elif name == "create_ticket":
                content, data, ok = self._ticket(arguments)
            else:
                content, data, ok = f"未知工具：{name}", {}, False
        except Exception as exc:  # noqa: BLE001 - 工具异常不能拖垮整条链路
            logger.exception("工具执行失败：%s", name)
            content, data, ok = f"工具执行失败：{exc}", {"error": str(exc)}, False

        return ToolExecution(
            name=name,
            arguments=arguments,
            ok=ok,
            content=content,
            data=data,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    # -- 各工具实现 -------------------------------------------------------- #
    def _search(
        self,
        arguments: dict,
        departments: list[str] | None = None,
        permissions: list[str] | None = None,
    ) -> tuple[str, dict, bool]:
        query = (arguments.get("query") or "").strip()
        if not query:
            return "缺少检索关键词。", {}, False
        kb = self.kb
        if not kb.ready:
            return "知识库暂未就绪，请稍后再试。", {"ready": False}, False
        result = kb.search(query, departments=departments, permissions=permissions)
        if not result.contexts:
            return f"未检索到与「{query}」相关的内容。", {"hits": 0}, False
        lines = []
        for ctx in result.contexts:
            snippet = ctx.text[:400].replace("\n", " ")
            lines.append(f"[{ctx.index}]《{ctx.title}》{ctx.section}：{snippet}")
        return "\n\n".join(lines), {"hits": len(result.contexts)}, True

    def _order(self, arguments: dict) -> tuple[str, dict, bool]:
        order_id = (arguments.get("order_id") or "").strip()
        order = mock_db.query_user_order(order_id)
        return mock_db.format_order(order), order, bool(order.get("ok"))

    def _ticket(self, arguments: dict) -> tuple[str, dict, bool]:
        result = mock_db.create_ticket(
            order_id=arguments.get("order_id", ""),
            reason=arguments.get("reason", "用户申请售后"),
            priority=arguments.get("priority", "normal"),
        )
        return result["message"], result, True

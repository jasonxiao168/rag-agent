"""Token 计数与成本核算。

面试点：RAG 系统的成本大头是「上下文重复注入」和「用大模型做简单判断」。
本模块统一记录每一次模型调用的 prompt/completion token 与人民币成本，
前端会实时展示每条消息花了几分钱。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from app.config import settings

_CJK = re.compile(r"[一-鿿]")
_LATIN_WORD = re.compile(r"[A-Za-z0-9_]+")
_PUNCT = re.compile(r"[，。；：、！？,.;:!?（）()\[\]【】「」\"'—\-]")


def estimate_tokens(text: str) -> int:
    """无 tiktoken 时的保守估算。

    中文按 ~0.8 token/字（Qwen 分词器对常见汉字约为 0.6~1），
    英文按 ~1.3 token/词，标点 0.4。误差通常在 10% 以内，够做成本估算。
    真实调用时以 API 返回的 usage 为准（本模块的 Usage 优先使用 API 值）。
    """
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN_WORD.findall(text))
    punct = len(_PUNCT.findall(text))
    return max(1, int(cjk * 0.8 + latin * 1.3 + punct * 0.4))


def count_messages_tokens(messages: list[dict]) -> int:
    """Chat 消息的 token 估算：每条消息额外 4 token 的角色/分隔开销。"""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(part.get("text", "")) for part in content)
        total += estimate_tokens(str(content)) + 4
    return total


@dataclass
class Usage:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    stage: str = "unknown"  # route / retrieve / rerank / compose / summarize / judge / embed

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_cny(self) -> float:
        price = settings.price_cny_per_1k.get(self.model)
        if not price:
            return 0.0
        input_price, output_price = price
        return (self.prompt_tokens / 1000) * input_price + (self.completion_tokens / 1000) * output_price

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "stage": self.stage,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_cny": round(self.cost_cny, 6),
        }


def budget_from_env() -> float | None:
    raw = os.environ.get("COST_BUDGET_CNY", "").strip()
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


class BudgetExceeded(RuntimeError):
    """累计花费超过预算时抛出，用于中止评测/批处理，避免误跑烧钱。"""


_BUDGET: dict[str, float | None] = {"limit": None, "spent": 0.0}


def set_budget(limit_cny: float | None) -> None:
    """设置全局预算（人民币）。None 或 <=0 表示不限制。"""
    _BUDGET["limit"] = limit_cny if limit_cny and limit_cny > 0 else None
    _BUDGET["spent"] = 0.0


def reset_budget() -> None:
    _BUDGET["limit"] = None
    _BUDGET["spent"] = 0.0


def budget_spent() -> float:
    return float(_BUDGET["spent"] or 0.0)


def budget_limit() -> float | None:
    return _BUDGET["limit"]


def budget_remaining() -> float | None:
    limit = _BUDGET["limit"]
    return None if limit is None else max(0.0, limit - budget_spent())


def charge(cost_cny: float) -> None:
    """记录一次花费；超预算立即抛错，让调用方停止后续请求。"""
    if cost_cny <= 0:
        return
    _BUDGET["spent"] = float(_BUDGET["spent"] or 0.0) + cost_cny
    limit = _BUDGET["limit"]
    if limit is not None and _BUDGET["spent"] > limit:
        raise BudgetExceeded(
            f"累计花费 ¥{_BUDGET['spent']:.4f} 已超过预算 ¥{limit:.2f}，已中止"
        )


@dataclass
class UsageTracker:
    """一次请求 / 一轮评测内的用量累加器。"""

    records: list[Usage] = field(default_factory=list)

    def add(self, usage: Usage | None) -> None:
        if usage is not None:
            self.records.append(usage)
            charge(usage.cost_cny)

    @property
    def prompt_tokens(self) -> int:
        return sum(u.prompt_tokens for u in self.records)

    @property
    def completion_tokens(self) -> int:
        return sum(u.completion_tokens for u in self.records)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_cny(self) -> float:
        return sum(u.cost_cny for u in self.records)

    def by_stage(self) -> dict[str, dict]:
        agg: dict[str, dict] = {}
        for u in self.records:
            item = agg.setdefault(
                u.stage, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_cny": 0.0}
            )
            item["calls"] += 1
            item["prompt_tokens"] += u.prompt_tokens
            item["completion_tokens"] += u.completion_tokens
            item["cost_cny"] = round(item["cost_cny"] + u.cost_cny, 6)
        return agg

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_cny": round(self.cost_cny, 6),
            "calls": len(self.records),
            "by_stage": self.by_stage(),
        }

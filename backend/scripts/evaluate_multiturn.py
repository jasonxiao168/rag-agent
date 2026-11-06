"""多轮对话评测：专门测「指代消解」与「会话隔离」。

单轮评测看不出的问题：
- 「那二线呢？」如果直接拿去检索，什么都找不到；
- 订单追问「它什么时候到？」里没有订单号，路由必须依赖上一轮；
- 切换身份后复用同一个 session，摘要可能带出上一位用户的内容。

用法（backend 目录下）：
    python -m scripts.evaluate_multiturn
    python -m scripts.evaluate_multiturn --limit 3
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from app.agent.memory import SessionStore
from app.config import settings
from app.core.logging import setup_logging
from app.core.tokens import BudgetExceeded, budget_from_env, budget_spent, set_budget
from app.services.chat_service import ChatService


@dataclass
class MultiTurnResult:
    id: str
    turns: list[str]
    route: str | None
    expect_route: str
    query_expanded: bool
    history_turns: int
    answer: str
    correct: bool
    must_hit: float
    route_ok: bool
    latency_ms: float
    tokens: int


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_case(service: ChatService, case: dict) -> MultiTurnResult:
    session_id = f"mt_{case['id']}"
    user_id = case.get("user", "carol")

    result = None
    for turn in case["turns"]:
        result = service.answer(session_id, turn, user_id=user_id)
    assert result is not None

    answer = result["answer"]
    must = case.get("must_contain") or []
    must_hit = (sum(1 for token in must if token in answer) / len(must)) if must else 1.0

    route_step = next((step for step in result["trace"] if step["step"] == "route"), {})
    expanded = bool((route_step.get("data") or {}).get("query_expanded"))
    route_ok = result.get("route") == case.get("expect_route")

    memory = service.sessions.get(session_id, user_id)
    usage = result.get("usage") or {}
    return MultiTurnResult(
        id=case["id"],
        turns=case["turns"],
        route=result.get("route"),
        expect_route=case.get("expect_route", ""),
        query_expanded=expanded,
        history_turns=len(memory.turns),
        answer=answer.replace("\n", " ")[:160],
        correct=bool(must_hit >= 0.5 and route_ok),
        must_hit=round(must_hit, 2),
        route_ok=route_ok,
        latency_ms=float(result.get("latency_ms") or 0),
        tokens=int(usage.get("total_tokens") or 0),
    )


def render_report(results: list[MultiTurnResult], out_path: Path) -> str:
    total = len(results)
    passed = sum(1 for item in results if item.correct)
    expanded = sum(1 for item in results if item.query_expanded)
    mode = "MOCK（规则模拟，仅验证链路）" if settings.mock_mode else settings.generation_model

    lines = [
        "# 多轮对话评测报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 运行模式：{mode}",
        f"- 用例数：{total}（每个用例至少 2 轮）",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 通过率 | {passed}/{total} = {passed / total * 100:.0f}% |",
        f"| 指代补全触发 | {expanded}/{total} |",
        f"| 平均延迟 | {sum(item.latency_ms for item in results) / total:.0f} ms |",
        "",
        "## 明细",
        "",
        "| ID | 轮次 | 路由 | 指代补全 | 命中关键事实 | 结果 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item.id} | {len(item.turns)} | {item.route} | {'是' if item.query_expanded else '否'} | "
            f"{item.must_hit:.0%} | {'通过' if item.correct else '失败'} |"
        )

    failures = [item for item in results if not item.correct]
    lines += ["", "## 失败明细", ""]
    if not failures:
        lines.append("无失败用例。")
    else:
        for item in failures:
            lines.append(f"**{item.id}**：{ ' → '.join(item.turns) }")
            lines.append("")
            lines.append(f"- 路由：{item.route}（期望 {item.expect_route}）")
            lines.append(f"- 是否补全：{item.query_expanded}")
            lines.append(f"- 回答：{item.answer}")
            lines.append("")

    lines += [
        "",
        "## 说明",
        "",
        "- 指代补全只在「模型没有改写 query」且问题以「那/它/还有」开头时触发，"
        "用上一轮用户问题拼接后再检索；真实模型通常能自己完成改写，这一步是兜底。",
        "- 会话按 `session_id + user_id` 隔离，切换身份会得到新会话，避免摘要跨用户泄露。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮对话评测")
    parser.add_argument(
        "--cases", default=str(Path(__file__).resolve().parents[1] / "eval" / "multiturn.jsonl")
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parents[1] / "eval" / "multiturn_report.md")
    )
    args = parser.parse_args()

    setup_logging()
    set_budget(budget_from_env())
    cases = load_cases(Path(args.cases))
    if args.limit:
        cases = cases[: args.limit]

    with tempfile.TemporaryDirectory() as tmp:
        service = ChatService(sessions=SessionStore(Path(tmp)))
        try:
            results = []
            for index, case in enumerate(cases, start=1):
                try:
                    result = run_case(service, case)
                except BudgetExceeded as exc:
                    print(f"\n{exc}，已处理 {len(results)} 组，停止。")
                    break
                results.append(result)
                flag = "✓" if result.correct else "✗"
                print(f"[{index}/{len(cases)}] {flag} {case['id']} {' → '.join(case['turns'])}")
        finally:
            if service.kb is not None:
                service.kb.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(results, out_path), encoding="utf-8")
    (out_path.parent / "multiturn_results.json").write_text(
        json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    passed = sum(1 for item in results if item.correct)
    print(f"\n通过 {passed}/{len(results)}")
    print(f"本次花费：¥{budget_spent():.4f}")
    print(f"报告：{out_path}")


if __name__ == "__main__":
    main()

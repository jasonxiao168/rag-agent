"""AI 效果评测：召回、正确率、幻觉率、成本，一次跑完。

用法（backend 目录下）：
    python -m scripts.evaluate                 # 完整评测
    python -m scripts.evaluate --limit 8       # 快速抽检
    python -m scripts.evaluate --compare       # 额外跑一轮「关闭 Rerank」做对照

指标口径：
- 召回命中（retrieval_hit）：回答引用的来源里是否包含期望文档；
- 回答正确（correct）：LLM-as-Judge 打分 >= 0.6 且关键事实命中；
- 幻觉（hallucination）：不可回答的问题没有拒答，或裁判标记 hallucination；
- 成本：按 app/core/tokens.py 的价目表折算人民币。
"""
from __future__ import annotations

import argparse
import re
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from app.agent.memory import SessionStore
from app.config import settings
from app.core.logging import setup_logging
from app.core.tokens import BudgetExceeded, budget_from_env, budget_spent, set_budget
from app.llm.client import extract_json, get_llm
from app.llm.prompts import judge_messages
from app.retrieval.service import KnowledgeBase, RetrievalConfig
from app.services.chat_service import ChatService

REFUSAL_MARKERS = ("不知道", "无法", "没有找到", "查不到", "暂无", "未收录", "没有相关")


def _norm(text: str) -> str:
    """比较前去掉所有空白：模型常写「30秒」，语料/标准答案写「30 秒」。"""
    return re.sub(r"\s+", "", text or "")


@dataclass
class CaseResult:
    id: str
    type: str
    question: str
    answer: str
    route: str | None
    retrieval_hit: bool
    precision: float
    correct: bool
    hallucination: bool
    must_hit: float
    score: float
    judge_reason: str
    failure: str
    latency_ms: float
    tokens: int
    cost_cny: float
    sources: list[str]


def load_testset(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def rule_judge(item: dict, answer: str) -> dict:
    """MOCK 模式判分：只看关键事实是否命中。

    mock 模型没有真正的语义能力，用它做 LLM-as-Judge 只会评估“措辞相似度”；
    这里退化为规则判分，保证评测管线本身可验证。接真实模型后自动切回 LLM 裁判。
    """
    must = item.get("must_contain") or []
    hit = [token for token in must if token in answer]
    ok = (len(hit) == len(must)) if must else bool(answer.strip())
    score = 1.0 if ok else (0.5 if hit else 0.0)
    return {
        "score": score,
        "correct": ok,
        "hallucination": False,
        "reason": f"关键事实命中 {len(hit)}/{len(must)}",
    }


def judge_answer(llm, question: str, gold: str, answer: str, item: dict) -> dict:
    if settings.mock_mode:
        return rule_judge(item, answer)
    try:
        response = llm.chat(
            judge_messages(question, gold, answer),
            model=settings.fast_model,
            json_mode=True,
            max_tokens=200,
            stage="judge",
            temperature=0.0,
        )
        data = extract_json(response.text)
        if not data:
            return {"score": 0.0, "correct": False, "hallucination": False, "reason": "裁判输出无法解析"}
        return data
    except Exception as exc:  # noqa: BLE001
        return {"score": 0.0, "correct": False, "hallucination": False, "reason": f"裁判调用失败：{exc}"}


def run_case(service: ChatService, llm, item: dict) -> CaseResult:
    user_id = item.get("user", "carol")
    result = service.answer(None, item["question"], user_id=user_id)
    answer = result["answer"]
    sources = result.get("sources") or []
    source_text = " ".join(
        f"{s['title']} {s['section']} {s.get('preview', '')}" for s in sources
    )

    must = item.get("must_contain") or []
    answer_norm = _norm(answer)
    must_hit = (
        sum(1 for token in must if _norm(token) in answer_norm) / len(must)
    ) if must else 1.0

    # 召回看「证据是否进入上下文」：文档级命中在语料少时没有区分度
    if item["type"] in {"kb", "pii"}:
        evidence = [
            s
            for s in sources
            if any(_norm(token) in _norm(f"{s.get('preview', '')} {s['section']}") for token in must)
        ]
        retrieval_hit = bool(evidence)
        precision = round(len(evidence) / len(sources), 3) if sources else 0.0
    elif item["type"] == "tool":
        retrieval_hit = result.get("route") == "tool"
        precision = 1.0 if retrieval_hit else 0.0
    else:
        retrieval_hit = True
        precision = 1.0

    hallucination = False
    reason = ""
    if item["type"] == "unanswerable":
        refused = any(marker in answer for marker in REFUSAL_MARKERS)
        clarified = result.get("route") == "clarify"
        if item.get("allow_clarify") and clarified:
            correct = True
            hallucination = False
            score = 1.0
            reason = "信息不足，主动追问而非编造"
        else:
            correct = refused
            hallucination = not refused
            score = 1.0 if refused else 0.0
            reason = "正确拒答" if refused else "未拒答，存在编造风险"
    elif item["type"] == "smalltalk":
        correct = result.get("route") == "smalltalk"
        score = 1.0 if correct else 0.4
        reason = "路由到闲聊" if correct else f"误判为 {result.get('route')}"
    elif item["type"] == "acl":
        refused = any(marker in answer for marker in REFUSAL_MARKERS)
        leaked = any(_norm(token) in answer_norm for token in item.get("forbidden", [])) or any(
            _norm(token) in _norm(source_text) for token in item.get("forbidden_sources", [])
        )
        if item.get("expect") == "refuse":
            correct = refused and not leaked
            hallucination = not refused
            score = 1.0 if correct else 0.0
            reason = "正确拒答且未泄露" if correct else ("泄露受限内容" if leaked else "未拒答")
        else:
            correct = (not refused) and must_hit >= 0.5
            score = 1.0 if correct else (0.5 if not refused else 0.0)
            reason = "有权限且答对" if correct else ("有权限但未答对" if not refused else "误拒答")
    elif item["type"] == "pii":
        leaked = any(_norm(token) in answer_norm for token in item.get("forbidden", [])) or any(
            _norm(token) in _norm(source_text) for token in item.get("forbidden", [])
        )
        correct = must_hit >= 0.5 and not leaked
        score = 1.0 if correct else 0.0
        reason = "脱敏符合预期" if correct else ("出现明文泄露" if leaked else "目标字段未命中")
    else:
        judged = judge_answer(llm, item["question"], item["gold"], answer, item)
        score = float(judged.get("score", 0) or 0)
        correct = bool(judged.get("correct")) and must_hit >= 0.5
        hallucination = bool(judged.get("hallucination"))
        reason = str(judged.get("reason", ""))

    if correct:
        failure = ""
    elif item["type"] == "unanswerable":
        failure = "幻觉：未拒答 → 加强 Prompt 的「不知道」约束，或提高检索相关度阈值"
    elif item["type"] == "acl":
        if "泄露" in reason:
            failure = "权限泄露 → 检查 ACL 是否在检索阶段生效，而不是事后过滤"
        elif item.get("expect") == "refuse":
            failure = "未拒答 → 用可见的相邻文档勉强作答；应加强「相近但不同主题也要说不知道」的约束"
        else:
            failure = "有权限但答错 → 检查检索命中与父块完整性"
    elif item["type"] == "pii":
        failure = "脱敏失败 → 检查 PII 替换是否发生在进入 Prompt / trace 之前"
    elif item["type"] == "tool":
        failure = "工具未命中 → 检查 Function Calling 参数或订单号规则兜底"
    elif not retrieval_hit:
        failure = "检索未命中 → 检查分块粒度、BM25 分词或查询改写"
    elif must_hit < 0.5:
        failure = "上下文命中但关键事实缺失 → 检查 Rerank 精度或父块是否被截断"
    else:
        failure = "生成偏离 → 检查 Prompt 约束与上下文噪声"

    usage = result.get("usage") or {}
    return CaseResult(
        id=item["id"],
        type=item["type"],
        question=item["question"],
        answer=answer.replace("\n", " ")[:160],
        route=result.get("route"),
        retrieval_hit=retrieval_hit,
        precision=precision,
        correct=correct,
        hallucination=hallucination,
        must_hit=round(must_hit, 2),
        score=round(score, 2),
        judge_reason=reason,
        failure=failure,
        latency_ms=float(result.get("latency_ms") or 0),
        tokens=int(usage.get("total_tokens") or 0),
        cost_cny=float(usage.get("cost_cny") or 0),
        sources=[f"{s['title']} / {s['section']}" for s in sources],
    )


def summarize(results: list[CaseResult]) -> dict:
    kb_cases = [r for r in results if r.type == "kb"]
    negatives = [r for r in results if r.type == "unanswerable"]
    by_type: dict[str, dict] = {}
    for type_name in ("kb", "tool", "smalltalk", "unanswerable", "acl", "pii"):
        group = [r for r in results if r.type == type_name]
        if group:
            by_type[type_name] = {
                "count": len(group),
                "accuracy": round(sum(1 for r in group if r.correct) / len(group), 3),
                "recall": (
                    round(sum(1 for r in group if r.retrieval_hit) / len(group), 3)
                    if type_name in {"kb", "tool"}
                    else None
                ),
            }

    return {
        "total": len(results),
        "accuracy": round(sum(1 for r in results if r.correct) / len(results), 3) if results else 0,
        "recall_at_k": round(sum(1 for r in kb_cases if r.retrieval_hit) / len(kb_cases), 3) if kb_cases else 0,
        "context_precision": round(sum(r.precision for r in kb_cases) / len(kb_cases), 3) if kb_cases else 0,
        "hallucination_rate": (
            round(sum(1 for r in negatives if r.hallucination) / len(negatives), 3) if negatives else 0
        ),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / len(results), 1) if results else 0,
        "total_tokens": sum(r.tokens for r in results),
        "total_cost_cny": round(sum(r.cost_cny for r in results), 6),
        "avg_cost_cny": round(sum(r.cost_cny for r in results) / len(results), 6) if results else 0,
        "by_type": by_type,
    }


def build_service(use_rerank: bool, session_dir: Path) -> ChatService:
    kb = KnowledgeBase(config=RetrievalConfig(use_rerank=use_rerank))
    kb.load()
    return ChatService(kb=kb, sessions=SessionStore(session_dir))


def render_report(
    summary: dict,
    results: list[CaseResult],
    comparison: dict | None,
    testset_path: Path,
) -> str:
    mode = "MOCK（规则模拟，仅验证链路）" if settings.mock_mode else f"在线模型 {settings.generation_model}"
    lines = [
        "# 评测报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 运行模式：{mode}",
        f"- 测试集：`{testset_path.name}`，共 {summary['total']} 题",
        f"- 检索配置：top_k={settings.retrieve_top_k}，rerank_top_n={settings.rerank_top_n}，"
        f"hybrid={settings.use_hybrid}",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 回答正确率 | {summary['accuracy'] * 100:.1f}% |",
        f"| 证据召回率（kb） | {summary['recall_at_k'] * 100:.1f}% |",
        f"| 上下文精度（kb） | {summary['context_precision'] * 100:.1f}% |",
        f"| 幻觉率（不可回答题） | {summary['hallucination_rate'] * 100:.1f}% |",
        f"| 平均延迟 | {summary['avg_latency_ms']:.0f} ms |",
        f"| 总 token | {summary['total_tokens']} |",
        f"| 总成本 | ¥{summary['total_cost_cny']:.4f} |",
        "",
        "## 分类结果",
        "",
        "| 类型 | 题数 | 正确率 | 召回率 |",
        "| --- | --- | --- | --- |",
    ]
    for type_name, item in summary["by_type"].items():
        recall = "—" if item["recall"] is None else f"{item['recall'] * 100:.1f}%"
        lines.append(f"| {type_name} | {item['count']} | {item['accuracy'] * 100:.1f}% | {recall} |")

    if comparison:
        lines += [
            "",
            "## Rerank 对照实验",
            "",
            "| 配置 | 证据召回率 | 上下文精度 | 回答正确率 | 平均成本 |",
            "| --- | --- | --- | --- | --- |",
            f"| 仅混合检索（无 Rerank） | {comparison['without']['recall_at_k'] * 100:.1f}% | "
            f"{comparison['without']['context_precision'] * 100:.1f}% | "
            f"{comparison['without']['accuracy'] * 100:.1f}% | ¥{comparison['without']['avg_cost_cny']:.4f} |",
            f"| 混合检索 + Rerank | {comparison['with']['recall_at_k'] * 100:.1f}% | "
            f"{comparison['with']['context_precision'] * 100:.1f}% | "
            f"{comparison['with']['accuracy'] * 100:.1f}% | ¥{comparison['with']['avg_cost_cny']:.4f} |",
            "",
            f"> 上下文精度提升 {comparison['precision_delta']:+.1f} 个百分点，"
            f"正确率提升 {comparison['accuracy_delta']:+.1f} 个百分点。"
            "MOCK 模式的向量是哈希替身、Rerank 是词面模拟，这组对照没有统计意义；"
            "接入真实 Embedding 与 Rerank 后用同一命令复测，再把数字写进简历。",
        ]

    failures = [r for r in results if not r.correct]
    lines += ["", "## 失败案例与归因", ""]
    if not failures:
        lines.append("无失败案例。")
    else:
        lines += ["| ID | 类型 | 问题 | 失败归因 | 裁判说明 |", "| --- | --- | --- | --- | --- |"]
        for item in failures:
            lines.append(
                f"| {item.id} | {item.type} | {item.question} | {item.failure} | {item.judge_reason[:40]} |"
            )

    lines += [
        "",
        "## 复现",
        "",
        "```bash",
        "cd backend",
        "python -m scripts.ingest",
        "python -m scripts.evaluate --compare",
        "```",
        "",
        "> 评测集包含 kb / tool / smalltalk / unanswerable / acl / pii 六类问题；"
        "`unanswerable` 测幻觉，`acl` 测文档权限，`pii` 测字段级脱敏（无权时不得出现明文）。",
        "",
        "> `acl` 用例按 `user` 字段切换身份（guest / alice / bob / carol / dave），"
        "其余用例默认以 `carol`（渠道运营）身份运行。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Agent 效果评测")
    parser.add_argument("--testset", default=str(Path(__file__).resolve().parents[1] / "eval" / "testset.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0, help="从第 N 题开始（分批跑）")
    parser.add_argument(
        "--budget",
        type=float,
        default=0.0,
        help="预算上限（人民币），超限立即中止；0 表示不限制（也可用 COST_BUDGET_CNY）",
    )
    parser.add_argument("--compare", action="store_true", help="额外跑一轮关闭 Rerank 的对照")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "eval" / "report.md"))
    args = parser.parse_args()

    setup_logging()
    limit = args.budget or budget_from_env()
    set_budget(limit)
    if limit:
        print(f"预算上限：¥{limit:.2f}（超出将立即中止）")

    llm = get_llm()
    testset_path = Path(args.testset)
    items = load_testset(testset_path)
    if args.offset:
        items = items[args.offset :]
    if args.limit:
        items = items[: args.limit]

    print(f"模式：{'MOCK' if settings.mock_mode else llm.name}，题目 {len(items)} 道")

    aborted = False
    with tempfile.TemporaryDirectory() as tmp:
        service = build_service(use_rerank=True, session_dir=Path(tmp))
        try:
            results = []
            for index, item in enumerate(items, start=1):
                try:
                    result = run_case(service, llm, item)
                except BudgetExceeded as exc:
                    print(f"\n{exc}，已处理 {len(results)} 题，停止。")
                    aborted = True
                    break
                results.append(result)
                flag = "✓" if result.correct else "✗"
                print(f"[{index:02d}/{len(items)}] {flag} {item['id']:<8} {item['question'][:28]}")

            summary = summarize(results) if results else {}
            comparison = None
            if args.compare and not aborted:
                print("\n对照实验：关闭 Rerank 重新跑一遍…")
                # 嵌入式向量库同一目录只允许一个客户端，先释放再建第二个
                service.kb.close()
                baseline = build_service(use_rerank=False, session_dir=Path(tmp))
                try:
                    base_results = []
                    for item in items:
                        try:
                            base_results.append(run_case(baseline, llm, item))
                        except BudgetExceeded as exc:
                            print(f"\n{exc}，对照实验提前结束。")
                            aborted = True
                            break
                    base_summary = summarize(base_results) if base_results else {}
                finally:
                    baseline.kb.close()
                if base_results and base_summary:
                    comparison = {
                        "with": summary,
                        "without": base_summary,
                        "recall_delta": (summary["recall_at_k"] - base_summary["recall_at_k"]) * 100,
                        "precision_delta": (summary["context_precision"] - base_summary["context_precision"]) * 100,
                        "accuracy_delta": (summary["accuracy"] - base_summary["accuracy"]) * 100,
                    }
        finally:
            if service.kb is not None:
                service.kb.close()

    report = render_report(summary, results, comparison, testset_path)
    out_path = Path(args.out)
    if aborted:
        report = report.replace(
            "# 评测报告", f"# 评测报告（预算中止，完成 {len(results)} 题）", 1
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    results_path = out_path.with_name(f"{out_path.stem}.results.json")
    results_path.write_text(
        json.dumps(
            {"summary": summary, "comparison": comparison, "cases": [asdict(r) for r in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== 汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n本次花费：¥{budget_spent():.4f}" + (f" / 预算 ¥{limit:.2f}" if limit else ""))
    print(f"报告：{out_path}")
    print(f"明细：{results_path}")


if __name__ == "__main__":
    main()

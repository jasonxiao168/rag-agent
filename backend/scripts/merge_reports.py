"""把分批跑出来的评测结果合并成一份报告。

用途：评测很慢或受执行时长限制时，可以按 --offset/--limit 分片跑，
每片输出 report_XX.md + report_XX.results.json，最后合并。

用法（backend 目录）：
    python -m scripts.merge_reports eval/report_*.results.json --out eval/report.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.core.logging import setup_logging
from scripts.evaluate import CaseResult, render_report, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="合并分批评测结果")
    parser.add_argument("results", nargs="+", help="results.json 文件列表")
    parser.add_argument("--out", default="eval/report.md")
    parser.add_argument("--testset", default="eval/testset.jsonl")
    args = parser.parse_args()

    setup_logging()
    cases: list[CaseResult] = []
    for path in args.results:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in payload.get("cases", []):
            cases.append(CaseResult(**item))

    if not cases:
        raise SystemExit("没有可合并的结果")

    summary = summarize(cases)
    report = render_report(summary, cases, None, Path(args.testset))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    out_path.with_name(f"{out_path.stem}.results.json").write_text(
        json.dumps({"summary": summary, "comparison": None, "cases": [asdict(c) for c in cases]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n合并 {len(cases)} 题 -> {out_path}")


if __name__ == "__main__":
    main()

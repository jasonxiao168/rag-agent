"""构建索引：解析文档 → 父子分块 → 向量化 → 落盘（向量库 + 父块表 + 子块表）。

用法（在 backend/ 目录下）：
    python -m scripts.ingest
    python -m scripts.ingest --raw ../data/raw --index ../data/index
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import settings
from app.core.logging import setup_logging
from app.ingestion.loader import load_documents
from app.retrieval.service import KnowledgeBase, rebuild_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 RAG 索引（默认增量 + 原子替换）")
    parser.add_argument("--raw", default=str(settings.raw_dir), help="原始文档目录")
    parser.add_argument("--index", default=str(settings.index_dir), help="索引输出目录")
    parser.add_argument("--full", action="store_true", help="忽略已有索引，全量重建")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="直接写入目标目录；默认先写 staging 再原子替换，避免读到半成品",
    )
    args = parser.parse_args()

    setup_logging()
    raw_dir = Path(args.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(raw_dir)
    if not docs:
        print(f"\n没有找到可索引的文档：{raw_dir}")
        print("可以先执行：python -m scripts.gen_corpus")
        return

    kb = KnowledgeBase(index_dir=Path(args.index))
    if args.in_place:
        manifest = kb.build(docs, incremental=not args.full)
    else:
        manifest = rebuild_atomic(Path(args.index), docs=docs, incremental=not args.full)

    print("\n索引构建完成")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    build = manifest.get("last_build", {})
    print(
        f"\n父块（喂给模型的上下文）{manifest['parents']} 个，"
        f"子块（建索引的粒度）{manifest['children']} 个"
    )
    print(
        f"本次增量：新增 {build.get('added', 0)} / 更新 {build.get('updated', 0)} / "
        f"未变 {build.get('unchanged', 0)} / 删除 {build.get('removed', 0)}，"
        f"重新向量化 {build.get('children_embedded', 0)} 条"
    )


if __name__ == "__main__":
    main()

"""预下载本地向量模型，供容器/离线环境使用。

背景：fastembed 首次运行会从 HuggingFace 下载模型。国内网络或容器内代理环境下
经常出现 `SSL: UNEXPECTED_EOF_WHILE_READING`，导致 ingest 直接失败。

用法（三种场景）：

1) 容器内直接下载（走 HF 镜像）
    docker compose run --rm \
      -e HF_ENDPOINT=https://hf-mirror.com \
      backend python -m scripts.download_models

2) 宿主机预下载，再由容器挂载（推荐，一次下载，重建镜像不重下）
    FASTEMBED_CACHE_DIR=./data/fastembed_cache HF_ENDPOINT=https://hf-mirror.com \
      python -m scripts.download_models
    # compose 已把 ./data 挂到 /data，并把缓存指向 /data/fastembed_cache

3) 完全离线：先在有网的机器上下载上面那个目录，整目录拷贝到 ./data/fastembed_cache

只下载 embedding 时加 --embedding-only，避免额外的 rerank 模型流量。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from app.config import settings


def _set_endpoint(endpoint: str | None) -> None:
    """必须在 import fastembed 之前设置，huggingface_hub 会读取该变量。"""
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"HF_ENDPOINT = {endpoint}")


def _download_embedding(model: str, cache_dir: str | None, retries: int = 3) -> None:
    from fastembed import TextEmbedding

    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            started = time.time()
            embedder = TextEmbedding(model, **kwargs)
            vector = list(embedder.embed(["预热"]))[0]
            print(f"✅ embedding 就绪：{model}（{len(vector)} 维，{time.time() - started:.1f}s）")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"⚠️  第 {attempt}/{retries} 次下载失败：{type(exc).__name__}: {exc}")
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"embedding 模型下载失败：{last_error}")


def _download_reranker(model: str, cache_dir: str | None) -> None:
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  当前 fastembed 不支持 cross-encoder，跳过 rerank 模型：{exc}")
        return

    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    try:
        encoder = TextCrossEncoder(model, **kwargs)
        list(encoder.rerank("住宿标准", ["一线城市 600 元", "期权 4 年"]))
        print(f"✅ rerank 就绪：{model}")
    except Exception as exc:  # noqa: BLE001
        # rerank 不是必需：RERANK_BACKEND=llm 时根本不用本地模型
        print(f"⚠️  rerank 模型下载失败（不影响 RERANK_BACKEND=llm）：{exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="预下载 fastembed 模型")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"), help="HF 镜像地址")
    parser.add_argument("--cache-dir", default=settings.fastembed_cache_dir or None)
    parser.add_argument("--embedding-only", action="store_true", help="只下载 embedding 模型")
    args = parser.parse_args()

    _set_endpoint(args.endpoint)
    cache_dir = str(Path(args.cache_dir).expanduser()) if args.cache_dir else None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"缓存目录：{cache_dir}")

    try:
        _download_embedding(settings.fastembed_embed_model, cache_dir)
    except RuntimeError as exc:
        print(f"\n❌ {exc}")
        print("\n可尝试：")
        print("  1) 换镜像：docker compose run --rm -e HF_ENDPOINT=https://hf-mirror.com backend \\")
        print("       python -m scripts.download_models")
        print("  2) 关掉容器代理：在 compose 的 backend.environment 里加 NO_PROXY=huggingface.co,hf-mirror.com")
        print("  3) 离线：在能上网的机器上下载 cache 目录，拷到 ./data/fastembed_cache")
        sys.exit(1)

    if not args.embedding_only:
        _download_reranker(settings.fastembed_rerank_model, cache_dir)

    print("\n完成。现在可以执行：python -m scripts.ingest --full")


if __name__ == "__main__":
    main()

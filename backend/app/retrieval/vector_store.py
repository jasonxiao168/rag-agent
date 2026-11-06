"""向量库封装：一个接口，两种后端。

- QdrantVectorStore：默认后端，支持两种部署形态
  * 嵌入式（本地演示/CI）：`QdrantClient(path=...)`，数据落在 index_dir/qdrant；
  * 服务端（生产/Docker）：设置 `QDRANT_URL`（可选 `QDRANT_API_KEY`）即可。
- NumpyVectorStore：零依赖兜底（无外网、无 qdrant-client 时），向量落盘 .npy / JSON。

两个后端都支持：
- `search(..., allowed_docs=...)`：文档级权限过滤，在检索阶段就排除无权文档；
- `delete(ids)`：增量索引更新/删除文档时清理旧向量；
- `close()`：释放客户端（嵌入式模式会持有目录锁，重建/切换前必须关闭）。
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.logging import get_logger

logger = get_logger("retrieval.vector_store")

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

# Qdrant 的 point id 只接受整数或 UUID；把 child_id 稳定映射成 UUID5
_QDRANT_NAMESPACE = uuid.UUID("6f6b6f6b-2f2f-4b8e-9c1a-2d7f0e6a8c11")


@dataclass
class VectorHit:
    child_id: str
    score: float
    metadata: dict


class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None: ...
    def search(
        self, vector: list[float], top_k: int, allowed_docs: set[str] | None = None
    ) -> list[VectorHit]: ...
    def delete(self, ids: list[str]) -> int: ...
    def count(self) -> int: ...


def _normalize(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if norm <= 0:
        return vec
    return [v / norm for v in vec]


def _point_id(child_id: str) -> str:
    return str(uuid.uuid5(_QDRANT_NAMESPACE, child_id))


class QdrantVectorStore:
    """Qdrant 适配器：同一份代码支持嵌入式与 server 两种模式。"""

    name = "qdrant"
    COLLECTION = "kb_children"
    UPSERT_BATCH = 128

    def __init__(self, index_dir: Path, collection: str = COLLECTION) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "缺少 qdrant-client，请安装或改用 VECTOR_BACKEND=local"
            ) from exc

        self.index_dir = Path(index_dir)
        self._collection = collection
        self._closed = False

        url = os.environ.get("QDRANT_URL", "").strip()
        api_key = os.environ.get("QDRANT_API_KEY", "").strip() or None
        if url:
            self._client = QdrantClient(url=url, api_key=api_key, timeout=30)
            self.location = url
        else:
            path = self.index_dir / "qdrant"
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))
            self.location = str(path)

    # -- 内部工具 ---------------------------------------------------------- #
    def _exists(self) -> bool:
        try:
            self._client.get_collection(self._collection)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _ensure_collection(self, dim: int) -> None:
        if self._exists():
            return
        from qdrant_client.models import Distance, VectorParams

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Qdrant collection 已创建：%s（dim=%d, %s）", self._collection, dim, self.location)

    @staticmethod
    def _filter(allowed_docs: set[str]):
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        return Filter(
            must=[FieldCondition(key="doc_id", match=MatchAny(any=sorted(allowed_docs)))]
        )

    # -- 写入 -------------------------------------------------------------- #
    def add(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise ValueError("ids / vectors / metadatas 长度不一致")
        if not ids:
            return

        from qdrant_client.models import PointStruct

        self._ensure_collection(len(vectors[0]))
        for start in range(0, len(ids), self.UPSERT_BATCH):
            batch_ids = ids[start : start + self.UPSERT_BATCH]
            batch_vectors = vectors[start : start + self.UPSERT_BATCH]
            batch_metas = metadatas[start : start + self.UPSERT_BATCH]
            points = [
                PointStruct(
                    id=_point_id(child_id),
                    vector=_normalize(vec),
                    payload={**meta, "child_id": child_id},
                )
                for child_id, vec, meta in zip(batch_ids, batch_vectors, batch_metas)
            ]
            self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        if not self._exists():
            return 0
        from qdrant_client.models import PointIdsList

        point_ids = [_point_id(child_id) for child_id in ids]
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=point_ids),
            wait=True,
        )
        return len(ids)

    # -- 检索 -------------------------------------------------------------- #
    def search(
        self, vector: list[float], top_k: int, allowed_docs: set[str] | None = None
    ) -> list[VectorHit]:
        if allowed_docs is not None and not allowed_docs:
            return []
        if not self._exists():
            return []

        query_filter = self._filter(allowed_docs) if allowed_docs is not None else None
        kwargs = dict(
            collection_name=self._collection,
            query=_normalize(vector),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        # qdrant-client >= 1.10 推荐 query_points；旧版本回落到 search
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(**kwargs)
            points = getattr(response, "points", response)
        else:  # pragma: no cover
            points = self._client.search(**kwargs)

        hits = []
        for point in points:
            payload = dict(point.payload or {})
            child_id = payload.get("child_id") or str(point.id)
            hits.append(VectorHit(child_id=child_id, score=float(point.score), metadata=payload))
        return hits

    def count(self) -> int:
        if not self._exists():
            return 0
        return int(self._client.count(self._collection, exact=True).count)

    # -- 生命周期 ---------------------------------------------------------- #
    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection)
        except Exception:  # noqa: BLE001
            pass

    def persist(self) -> None:
        # 嵌入式与 server 模式都由 Qdrant 自行持久化
        return None

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
        self._closed = True


class NumpyVectorStore:
    name = "numpy"

    def __init__(self, persist_dir: Path) -> None:
        self.persist_dir = Path(persist_dir)
        self.ids: list[str] = []
        self.vectors: list[list[float]] = []
        self.metadatas: list[dict] = []

    # -- 写入 -------------------------------------------------------------- #
    def add(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise ValueError("ids / vectors / metadatas 长度不一致")
        for i, vec in enumerate(vectors):
            self.ids.append(ids[i])
            self.vectors.append(_normalize(vec))
            self.metadatas.append(metadatas[i])

    def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        drop = set(ids)
        keep = [i for i, child_id in enumerate(self.ids) if child_id not in drop]
        removed = len(self.ids) - len(keep)
        if removed:
            self.ids = [self.ids[i] for i in keep]
            self.vectors = [self.vectors[i] for i in keep]
            self.metadatas = [self.metadatas[i] for i in keep]
        return removed

    # -- 检索 -------------------------------------------------------------- #
    def search(
        self, vector: list[float], top_k: int, allowed_docs: set[str] | None = None
    ) -> list[VectorHit]:
        if not self.vectors:
            return []
        if allowed_docs is not None and not allowed_docs:
            return []

        query = _normalize(vector)
        if np is not None:
            matrix = np.asarray(self.vectors, dtype=np.float32)
            scores = matrix @ np.asarray(query, dtype=np.float32)
            if allowed_docs is not None:
                mask = np.asarray(
                    [meta.get("doc_id") in allowed_docs for meta in self.metadatas], dtype=bool
                )
                if not mask.any():
                    return []
                scores = np.where(mask, scores, -np.inf)
            top_idx = np.argsort(-scores)[:top_k]
            hits = []
            for idx in top_idx:
                if not np.isfinite(scores[int(idx)]):
                    break
                hits.append(
                    VectorHit(self.ids[int(idx)], float(scores[int(idx)]), self.metadatas[int(idx)])
                )
            return hits

        scored = []
        for i, vec in enumerate(self.vectors):
            if allowed_docs is not None and self.metadatas[i].get("doc_id") not in allowed_docs:
                continue
            scored.append((sum(a * b for a, b in zip(query, vec)), i))
        scored.sort(reverse=True)
        return [
            VectorHit(self.ids[i], float(score), self.metadatas[i]) for score, i in scored[:top_k]
        ]

    def count(self) -> int:
        return len(self.ids)

    # -- 持久化 ------------------------------------------------------------ #
    def persist(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        with (self.persist_dir / "vectors_meta.jsonl").open("w", encoding="utf-8") as fh:
            for child_id, meta in zip(self.ids, self.metadatas):
                fh.write(json.dumps({"child_id": child_id, "metadata": meta}, ensure_ascii=False) + "\n")

        vector_file = self.persist_dir / ("vectors.npy" if np is not None else "vectors.json")
        if np is not None:
            np.save(vector_file, np.asarray(self.vectors, dtype=np.float32))
            stale = self.persist_dir / "vectors.json"
            if stale.exists():
                stale.unlink()
        else:
            vector_file.write_text(json.dumps(self.vectors), encoding="utf-8")
        logger.info("向量库已落盘：%s（%d 条）", self.persist_dir, len(self.ids))

    @classmethod
    def load(cls, persist_dir: Path) -> "NumpyVectorStore":
        store = cls(persist_dir)
        meta_file = persist_dir / "vectors_meta.jsonl"
        if not meta_file.exists():
            return store

        npy_file = persist_dir / "vectors.npy"
        json_file = persist_dir / "vectors.json"
        if npy_file.exists() and np is not None:
            store.vectors = np.load(npy_file).astype(float).tolist()
        elif json_file.exists():
            store.vectors = json.loads(json_file.read_text(encoding="utf-8"))

        for line in meta_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            store.ids.append(item["child_id"])
            store.metadatas.append(item["metadata"])
        return store

    def close(self) -> None:
        return None


def build_vector_store(
    index_dir: Path, backend: str = "auto", load_existing: bool = True
) -> VectorStore:
    """按配置选择后端：auto 优先 Qdrant，不可用则回落本地。

    load_existing 只对本地 numpy 后端有意义；Qdrant 是持久化服务/目录。
    """
    backend = (backend or "auto").lower()

    if backend in {"qdrant", "auto"}:
        try:
            return QdrantVectorStore(index_dir)
        except Exception as exc:  # noqa: BLE001
            if backend == "qdrant":
                raise
            logger.warning("Qdrant 不可用（%s），回落到本地向量库", exc)

    if backend not in {"qdrant", "auto", "local", "numpy"}:
        logger.warning("未知向量后端 %s，回落到本地", backend)

    return NumpyVectorStore.load(index_dir) if load_existing else NumpyVectorStore(index_dir)

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MOCK_MODE", "1")

try:
    import qdrant_client  # noqa: F401

    HAS_QDRANT = True
except Exception:  # pragma: no cover
    HAS_QDRANT = False

from app.ingestion.loader import LoadedDoc
from app.llm.client import MockLLM
from app.retrieval.service import KnowledgeBase


def make_doc(doc_id: str, title: str, text: str, acl: str = "all") -> LoadedDoc:
    return LoadedDoc(
        doc_id=doc_id, title=title, text=text, source=f"{doc_id}.md", doc_type="md", acl=acl
    )


PUBLIC = make_doc("d_pub", "公共制度", "# 公共制度\n\n## 住宿\n\n一线城市住宿上限 600 元。")
HR = make_doc("d_hr", "期权制度", "# 期权制度\n\n## 归属\n\n期权分 4 年归属。", acl="hr")


@unittest.skipUnless(HAS_QDRANT, "qdrant-client 未安装")
class QdrantBackendTest(unittest.TestCase):
    """真实 Qdrant（嵌入式模式）的读写、payload 过滤与增量删除。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.kb = KnowledgeBase(
            index_dir=Path(self.tmp.name), llm=MockLLM(), vector_backend="qdrant"
        )
        self.kb.build([PUBLIC, HR])

    def tearDown(self) -> None:
        self.kb.close()
        self.tmp.cleanup()

    def test_backend_is_qdrant(self) -> None:
        self.assertEqual(getattr(self.kb._store, "name", ""), "qdrant")
        self.assertGreaterEqual(self.kb._store.count(), 2)

    def test_query_hits(self) -> None:
        result = self.kb.search("一线城市住宿上限")
        self.assertTrue(any("600" in ctx.text for ctx in result.contexts))

    def test_acl_filter_applied(self) -> None:
        allowed = self.kb.search("期权分几年归属", departments=["hr"])
        self.assertTrue(any("4 年" in ctx.text for ctx in allowed.contexts))

        blocked = self.kb.search("期权分几年归属", departments=["marketing"])
        self.assertFalse(any("4 年" in ctx.text for ctx in blocked.contexts))

    def test_incremental_delete(self) -> None:
        before = self.kb._store.count()
        manifest = self.kb.build([PUBLIC])
        self.assertEqual(manifest["last_build"]["removed"], 1)
        self.assertLess(self.kb._store.count(), before)
        self.assertFalse(any(ctx.title == "期权制度" for ctx in self.kb.search("期权").contexts))

    def test_persistence_across_clients(self) -> None:
        self.kb.close()
        reopened = KnowledgeBase(
            index_dir=Path(self.tmp.name), llm=MockLLM(), vector_backend="qdrant"
        ).load()
        try:
            self.assertGreaterEqual(reopened._store.count(), 2)
            self.assertTrue(any("600" in ctx.text for ctx in reopened.search("住宿").contexts))
        finally:
            reopened.close()
            self.kb = reopened  # 避免 tearDown 重复关闭已释放的旧客户端


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MOCK_MODE", "1")
os.environ.setdefault("VECTOR_BACKEND", "local")

from app.ingestion.loader import LoadedDoc
from app.llm.client import MockLLM
from app.retrieval.service import KnowledgeBase


def make_doc(doc_id: str, title: str, text: str) -> LoadedDoc:
    return LoadedDoc(
        doc_id=doc_id, title=title, text=text, source=f"{doc_id}.md", doc_type="md"
    )


class FailingEmbedLLM(MockLLM):
    """模拟模型下载失败：embed 直接抛错。"""

    def embed(self, texts, *, stage="embed"):
        raise RuntimeError("模拟模型下载失败")


class IncrementalBuildTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.kb = KnowledgeBase(index_dir=Path(self.tmp.name), llm=MockLLM())
        self.doc_a = make_doc("d_a", "差旅制度", "# 差旅制度\n\n## 住宿\n\n一线城市住宿上限 600 元。")
        self.doc_b = make_doc("d_b", "客服制度", "# 客服制度\n\n## 时效\n\n工单 24 小时内闭环。")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_first_build_indexes_everything(self) -> None:
        manifest = self.kb.build([self.doc_a, self.doc_b])
        self.assertEqual(manifest["last_build"]["added"], 2)
        self.assertGreater(manifest["last_build"]["children_embedded"], 0)

    def test_rebuild_without_changes_embeds_nothing(self) -> None:
        self.kb.build([self.doc_a, self.doc_b])
        manifest = self.kb.build([self.doc_a, self.doc_b])
        self.assertEqual(manifest["last_build"]["unchanged"], 2)
        self.assertEqual(manifest["last_build"]["children_embedded"], 0)

    def test_modified_doc_is_reembedded_only(self) -> None:
        self.kb.build([self.doc_a, self.doc_b])
        updated = make_doc("d_a", "差旅制度", "# 差旅制度\n\n## 住宿\n\n一线城市住宿上限 800 元。")
        manifest = self.kb.build([updated, self.doc_b])

        self.assertEqual(manifest["last_build"]["updated"], 1)
        self.assertEqual(manifest["last_build"]["unchanged"], 1)
        self.assertGreater(manifest["last_build"]["children_embedded"], 0)

        result = self.kb.search("一线城市住宿上限")
        text = "".join(ctx.text for ctx in result.contexts)
        self.assertIn("800", text)
        self.assertNotIn("600", text)

    def test_removed_doc_is_deleted(self) -> None:
        self.kb.build([self.doc_a, self.doc_b])
        manifest = self.kb.build([self.doc_a])
        self.assertEqual(manifest["last_build"]["removed"], 1)
        self.assertEqual(len(self.kb.docs), 1)
        self.assertFalse(any(ctx.title == "客服制度" for ctx in self.kb.search("工单时效").contexts))

    def test_acl_change_triggers_reindex(self) -> None:
        """只改权限、不改正文，也必须重建（ACL 参与 content_hash）。"""
        self.kb.build([self.doc_a])
        restricted = make_doc(
            "d_a", "差旅制度", "# 差旅制度\n\n## 住宿\n\n一线城市住宿上限 600 元。"
        )
        restricted.acl = "hr"
        restricted.content_hash = ""  # 让 __post_init__ 按新 ACL 重算
        restricted.__post_init__()

        manifest = self.kb.build([restricted])
        self.assertEqual(manifest["last_build"]["updated"], 1)

    def test_failed_embedding_does_not_wipe_index(self) -> None:
        """全量重建时向量化失败，旧索引必须还能用（不能先删后算）。"""
        self.kb.build([self.doc_a, self.doc_b])
        before = len(self.kb.children)

        broken = KnowledgeBase(index_dir=Path(self.tmp.name), llm=FailingEmbedLLM())
        with self.assertRaises(RuntimeError):
            broken.build([self.doc_a, self.doc_b], incremental=False)

        reloaded = KnowledgeBase(index_dir=Path(self.tmp.name), llm=MockLLM()).load()
        self.assertEqual(len(reloaded.children), before)
        self.assertTrue(any("600" in ctx.text for ctx in reloaded.search("住宿上限").contexts))


if __name__ == "__main__":
    unittest.main()

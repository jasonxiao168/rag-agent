import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MOCK_MODE", "1")
os.environ.setdefault("VECTOR_BACKEND", "local")

from app.ingestion.loader import LoadedDoc
from app.llm.client import MockLLM
from app.retrieval.service import KnowledgeBase, rebuild_atomic


def make_doc(doc_id: str, text: str) -> LoadedDoc:
    return LoadedDoc(
        doc_id=doc_id, title=doc_id, text=text, source=f"{doc_id}.md", doc_type="md"
    )


class AtomicRebuildTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.index = Path(self.tmp.name) / "index"
        self.doc = make_doc("d_1", "# 制度\n\n## 住宿\n\n一线城市住宿上限 600 元。")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rebuild_swaps_directory(self) -> None:
        rebuild_atomic(self.index, docs=[self.doc])
        self.assertTrue((self.index / "children.jsonl").exists())
        self.assertFalse((self.index.parent / "index.staging").exists())
        self.assertFalse((self.index.parent / "index.old").exists())

        updated = make_doc("d_1", "# 制度\n\n## 住宿\n\n一线城市住宿上限 800 元。")
        rebuild_atomic(self.index, docs=[updated])

        kb = KnowledgeBase(index_dir=self.index, llm=MockLLM()).load()
        text = "".join(ctx.text for ctx in kb.search("一线城市住宿上限").contexts)
        self.assertIn("800", text)
        self.assertNotIn("600", text)

    def test_failed_build_keeps_old_index(self) -> None:
        rebuild_atomic(self.index, docs=[self.doc])

        with patch.object(KnowledgeBase, "build", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                rebuild_atomic(self.index, docs=[self.doc])

        # 正式索引没有被破坏，staging 也清理干净
        self.assertTrue((self.index / "children.jsonl").exists())
        self.assertFalse((self.index.parent / "index.staging").exists())
        kb = KnowledgeBase(index_dir=self.index, llm=MockLLM()).load()
        self.assertIn("600", "".join(ctx.text for ctx in kb.search("住宿上限").contexts))


if __name__ == "__main__":
    unittest.main()

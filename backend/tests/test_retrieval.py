import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MOCK_MODE", "1")
os.environ.setdefault("VECTOR_BACKEND", "local")

from app.ingestion.loader import LoadedDoc
from app.llm.client import MockLLM
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.service import IndexDimensionMismatch, KnowledgeBase

DOCS = [
    LoadedDoc(
        doc_id="d_travel",
        title="差旅制度",
        text="# 差旅制度\n\n## 住宿标准\n\n一线城市住宿上限 600 元每晚，含早餐。二线城市 450 元。\n\n## 餐补\n\n一线城市餐补 120 元每天。",
        source="travel.md",
        doc_type="md",
    ),
    LoadedDoc(
        doc_id="d_security",
        title="数据安全",
        text="# 数据安全\n\n## 密码\n\n密码长度不少于 12 位，每 90 天更换一次。\n\n## 事件响应\n\n安全事件须在 2 小时内上报。",
        source="security.md",
        doc_type="md",
    ),
]


class BM25Test(unittest.TestCase):
    def test_keyword_hit_beats_unrelated(self) -> None:
        index = BM25Index()
        index.build([d.doc_id for d in DOCS], [d.text for d in DOCS])
        results = index.search("住宿标准 600", top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0][0], "d_travel")

    def test_empty_corpus_is_safe(self) -> None:
        index = BM25Index()
        index.build([], [])
        self.assertEqual(index.search("任意查询"), [])


class FusionTest(unittest.TestCase):
    def test_rrf_rewards_agreement(self) -> None:
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        bm25 = [("c", 12.0), ("a", 9.0)]
        fused = reciprocal_rank_fusion(dense, bm25, k=60)
        ids = [hit.child_id for hit in fused]
        self.assertEqual(ids[0], "a")  # 两路都排第一
        self.assertIn("b", ids)
        for hit in fused:
            self.assertGreater(hit.score, 0)


class KnowledgeBaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.kb = KnowledgeBase(index_dir=Path(cls.tmp.name), llm=MockLLM())
        cls.kb.build(DOCS)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_index_ready(self) -> None:
        self.assertTrue(self.kb.ready)
        self.assertGreaterEqual(len(self.kb.parents), 2)
        self.assertGreaterEqual(len(self.kb.children), 2)

    def test_search_returns_parent_context(self) -> None:
        result = self.kb.search("一线城市住宿标准是多少？")
        self.assertTrue(result.contexts)
        text = "".join(ctx.text for ctx in result.contexts)
        self.assertIn("600", text)
        self.assertIn("住宿", text)

    def test_unrelated_query_is_low_confidence(self) -> None:
        result = self.kb.search("公司年会抽奖规则是什么？")
        self.assertIsNotNone(result.best_score)

    def test_dimension_mismatch_raises_clear_error(self) -> None:
        """索引 1024 维、查询向量 8 维时，必须报明确错误而不是 numpy 的 shape 错。"""
        from app.core.tokens import Usage

        class SmallEmbedLLM(MockLLM):
            def embed(self, texts, *, stage="embed"):
                return [[0.0] * 8 for _ in texts], Usage(model="mock", stage=stage)

        # 用独立的 KnowledgeBase，避免污染其它用例共享的类级实例
        kb = KnowledgeBase(index_dir=self.kb.index_dir, llm=SmallEmbedLLM()).load()
        with self.assertRaises(IndexDimensionMismatch) as ctx:
            kb.search("一线城市住宿标准")
        self.assertIn("ingest --full", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MOCK_MODE", "1")
os.environ.setdefault("VECTOR_BACKEND", "local")

from app.agent.tools import ToolExecutor
from app.ingestion.loader import LoadedDoc
from app.llm.client import MockLLM
from app.retrieval.service import KnowledgeBase

PUBLIC_DOC = LoadedDoc(
    doc_id="d_public",
    title="差旅制度",
    text="# 差旅制度\n\n## 住宿\n\n一线城市住宿上限 600 元每晚。",
    source="public.md",
    doc_type="md",
    acl="all",
)

HR_DOC = LoadedDoc(
    doc_id="d_hr",
    title="薪酬与期权",
    text="# 薪酬与期权\n\n## 期权\n\n期权授予价 1 元/股，分 4 年归属，首批 2026 年 3 月完成。",
    source="hr.md",
    doc_type="md",
    acl="hr",
)

CHANNEL_DOC = LoadedDoc(
    doc_id="d_channel",
    title="渠道返点",
    text="# 渠道返点\n\n## 返点\n\n季度进货 60 万元返点 3%。",
    source="channel.md",
    doc_type="md",
    acl="channel",
)


class ACLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.kb = KnowledgeBase(index_dir=Path(cls.tmp.name), llm=MockLLM())
        cls.kb.build([PUBLIC_DOC, HR_DOC, CHANNEL_DOC])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_hr_department_can_read_hr_doc(self) -> None:
        result = self.kb.search("期权分几年归属？", departments=["hr"])
        text = "".join(ctx.text for ctx in result.contexts)
        self.assertIn("4 年", text)
        self.assertTrue(any(ctx.title == "薪酬与期权" for ctx in result.contexts))

    def test_marketing_cannot_read_hr_doc(self) -> None:
        result = self.kb.search("期权分几年归属？", departments=["marketing"])
        titles = [ctx.title for ctx in result.contexts]
        self.assertNotIn("薪酬与期权", titles)
        self.assertFalse(any("4 年" in ctx.text for ctx in result.contexts))
        self.assertTrue(result.acl["restricted"])
        self.assertEqual(result.acl["documents_visible"], 1)

    def test_channel_doc_invisible_to_finance(self) -> None:
        result = self.kb.search("季度进货返点多少？", departments=["finance"])
        self.assertFalse(any("3%" in ctx.text for ctx in result.contexts))

    def test_tool_respects_departments(self) -> None:
        """工具调用不能成为绕过 ACL 的后门。"""
        executor = ToolExecutor(kb=self.kb)
        blocked = executor.execute(
            "search_knowledge_base", {"query": "期权授予价"}, departments=["marketing"]
        )
        self.assertNotIn("1 元/股", blocked.content)

        allowed = executor.execute(
            "search_knowledge_base", {"query": "期权授予价"}, departments=["hr"]
        )
        self.assertIn("1 元/股", allowed.content)

    def test_no_department_means_no_filter_for_scripts(self) -> None:
        result = self.kb.search("期权")
        self.assertTrue(any(ctx.title == "薪酬与期权" for ctx in result.contexts))


if __name__ == "__main__":
    unittest.main()

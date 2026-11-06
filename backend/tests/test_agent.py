import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MOCK_MODE", "1")
os.environ.setdefault("VECTOR_BACKEND", "local")

from app.agent.graph import AgentRuntime, build_state
from app.ingestion.loader import LoadedDoc
from app.llm.client import MockLLM
from app.retrieval.service import KnowledgeBase

DOC = LoadedDoc(
    doc_id="d_travel",
    title="差旅制度",
    text="# 差旅制度\n\n## 住宿标准\n\n一线城市住宿上限 600 元每晚。二线城市 450 元。\n\n## 报销\n\n出差结束后 15 个自然日内提交报销。",
    source="travel.md",
    doc_type="md",
)


class AgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.kb = KnowledgeBase(index_dir=Path(cls.tmp.name), llm=MockLLM())
        cls.kb.build([DOC])
        cls.runtime = AgentRuntime(kb=cls.kb, llm=MockLLM())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_knowledge_route(self) -> None:
        state = self.runtime.graph.invoke(build_state("一线城市住宿标准是多少？"))
        self.assertEqual(state["route"], "kb")
        self.assertTrue(state["contexts"])
        self.assertTrue(state["gen_messages"])
        self.assertIn("600", "".join(c["text"] for c in state["contexts"]))

    def test_tool_route_with_order_id(self) -> None:
        state = self.runtime.graph.invoke(build_state("帮我查一下订单 ORD-2026-1001 到哪了"))
        self.assertEqual(state["route"], "tool")
        self.assertTrue(state["tool_results"])
        self.assertEqual(state["tool_results"][0]["name"], "query_user_order")
        self.assertIn("已发货", state["tool_results"][0]["result"])

    def test_ticket_tool(self) -> None:
        state = self.runtime.graph.invoke(
            build_state("订单 ORD-2026-1005 一直没发货，帮我建个工单")
        )
        self.assertEqual(state["route"], "tool")
        self.assertEqual(state["tool_results"][0]["name"], "create_ticket")

    def test_smalltalk_route_skips_retrieval(self) -> None:
        state = self.runtime.graph.invoke(build_state("你好，你能做什么？"))
        self.assertEqual(state["route"], "smalltalk")
        self.assertFalse(state["contexts"])
        steps = [item["step"] for item in state["trace"]]
        self.assertNotIn("retrieval", steps)

    def test_trace_is_complete(self) -> None:
        state = self.runtime.graph.invoke(build_state("一线城市住宿标准是多少？"))
        steps = [item["step"] for item in state["trace"]]
        self.assertEqual(steps[0], "route")
        self.assertIn("retrieval", steps)
        self.assertIn("compose", steps)


if __name__ == "__main__":
    unittest.main()

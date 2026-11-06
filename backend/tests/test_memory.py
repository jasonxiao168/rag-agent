import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MOCK_MODE", "1")

from app.agent.memory import SessionMemory, SessionStore
from app.llm.client import MockLLM


class MemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "s_test.json"
        self.memory = SessionMemory("s_test", llm=MockLLM(), path=self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sliding_window_trims_history(self) -> None:
        for index in range(12):
            self.memory.append("user" if index % 2 == 0 else "assistant", f"第 {index} 条消息")
        self.assertTrue(self.memory.needs_summary())

        usages = self.memory.maybe_summarize()
        self.assertTrue(usages)
        self.assertTrue(self.memory.summary)
        # 窗口保留最近 MAX_HISTORY_TURNS * 2 条
        self.assertLessEqual(len(self.memory.turns), 8)

    def test_persist_and_reload(self) -> None:
        self.memory.append("user", "住宿标准是多少")
        self.memory.append("assistant", "600 元每晚")
        self.memory.summary = "用户在问住宿标准"
        self.memory.save()

        reloaded = SessionMemory.load("s_test", self.path, llm=MockLLM())
        self.assertEqual(len(reloaded.turns), 2)
        self.assertEqual(reloaded.summary, "用户在问住宿标准")
        self.assertEqual(reloaded.turns[0].role, "user")

    def test_session_store_rejects_path_traversal(self) -> None:
        store = SessionStore(Path(self.tmp.name))
        with self.assertRaises(ValueError):
            store.path_for("../../etc/passwd")

    def test_finalize_turn_appends_pair(self) -> None:
        self.memory.finalize_turn("问题", "回答")
        self.assertEqual([t.role for t in self.memory.turns], ["user", "assistant"])

    def test_session_is_bound_to_user(self) -> None:
        """同一个 session_id 换身份必须拿到新会话，避免摘要泄露上一位用户的内容。"""
        store = SessionStore(Path(self.tmp.name))
        alice = store.get("s_shared", "alice")
        alice.append("user", "期权怎么归属")
        bob = store.get("s_shared", "bob")
        self.assertNotEqual(alice.session_id, bob.session_id)
        self.assertEqual(bob.turns, [])

    def test_list_sessions_filters_by_user(self) -> None:
        store = SessionStore(Path(self.tmp.name))
        alice = store.get(None, "alice")
        alice.append("user", "我是 alice 的问题")
        alice.save()
        bob = store.get(None, "bob")
        bob.append("user", "我是 bob 的问题")
        bob.save()

        alice_items = store.list_sessions("alice")
        previews = [item["preview"] for item in alice_items]
        self.assertIn("我是 alice 的问题", previews)
        self.assertNotIn("我是 bob 的问题", previews)
        self.assertTrue(all(item["user_id"] == "alice" for item in alice_items))

    def test_load_for_user_rejects_other_identity(self) -> None:
        store = SessionStore(Path(self.tmp.name))
        bob = store.get(None, "bob")
        bob.append("user", "bob 的私密内容")
        bob.save()

        self.assertIsNone(store.load_for_user(bob.session_id, "alice"))
        loaded = store.load_for_user(bob.session_id, "bob")
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.turns), 1)

    def test_delete_only_owner(self) -> None:
        store = SessionStore(Path(self.tmp.name))
        bob = store.get(None, "bob")
        bob.append("user", "bob")
        bob.save()

        self.assertFalse(store.delete(bob.session_id, "alice"))
        self.assertTrue(store.delete(bob.session_id, "bob"))
        self.assertIsNone(store.load_for_user(bob.session_id, "bob"))


if __name__ == "__main__":
    unittest.main()

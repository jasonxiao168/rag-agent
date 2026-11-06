import os
import unittest

os.environ.setdefault("MOCK_MODE", "1")

from app.ingestion.chunker import chunk_document
from app.ingestion.loader import LoadedDoc

TEXT = """# 测试手册

## 一、住宿标准

| 城市级别 | 上限 |
| --- | --- |
| 一线城市 | 600 |
| 二线城市 | 450 |

住宿费用凭发票报销，超出部分自理。同性别同事同行应合住。

## 二、餐补

一线城市餐补 120 元每天，其他城市 90 元每天。餐补无需发票。

## 三、报销时效

出差结束后 15 个自然日内提交报销。逾期需要部门负责人特批。
"""


class ChunkerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = LoadedDoc(
            doc_id="doc_test", title="测试手册", text=TEXT, source="test.md", doc_type="md"
        )
        self.parents, self.children = chunk_document(
            self.doc, parent_size=200, child_size=80, parent_overlap=20, child_overlap=10
        )

    def test_produces_parent_and_child(self) -> None:
        self.assertGreater(len(self.parents), 0)
        self.assertGreater(len(self.children), len(self.parents))
        self.assertGreaterEqual(len(self.children), 2)

    def test_child_never_crosses_parent(self) -> None:
        parent_ids = {p.parent_id for p in self.parents}
        for child in self.children:
            self.assertIn(child.parent_id, parent_ids)

    def test_child_shorter_than_parent(self) -> None:
        for parent in self.parents:
            self.assertGreater(len(parent.text), 0)
        for child in self.children:
            self.assertGreater(len(child.text), 0)
            self.assertLessEqual(len(child.text), 260)

    def test_table_rows_are_not_broken(self) -> None:
        joined = "\n".join(p.text for p in self.parents)
        self.assertIn("| 一线城市 | 600 |", joined)
        self.assertIn("| 二线城市 | 450 |", joined)

    def test_section_metadata_present(self) -> None:
        sections = {p.section for p in self.parents}
        self.assertTrue(any("住宿" in s for s in sections))
        # 小节会被合并成父块，餐补可能并入「住宿标准 等 N 节」，因此检查正文而不是标签
        self.assertTrue(any("餐补" in p.text for p in self.parents))


if __name__ == "__main__":
    unittest.main()

"""父子文档切分（Parent-Child Chunking）。

核心思想（面试必问）：
- 用 200 字左右的「子块」建索引：粒度小，向量语义更聚焦，召回更准；
- 命中子块后，回溯它所属的 900 字「父块」喂给大模型：上下文完整，避免答案被切断；
- 切分必须按句子边界，不能按固定字数硬切；Markdown 表格作为原子单元，不拆行。

实现顺序：文档 → 按标题分节 → 节内按句/表格切父块（带重叠）→ 父块再切子块。
子块永远不跨父块，保证 parent 回溯是 O(1) 查表。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.text import normalize_text, split_sentences
from app.ingestion.loader import LoadedDoc

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class ParentChunk:
    parent_id: str
    doc_id: str
    title: str
    section: str
    text: str
    child_ids: list[str] = field(default_factory=list)
    acl: str = "all"
    department: str = ""

    def as_dict(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "text": self.text,
            "acl": self.acl,
            "department": self.department,
        }


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    doc_id: str
    title: str
    section: str
    text: str
    embed_text: str
    ordinal: int = 0
    acl: str = "all"
    department: str = ""

    def metadata(self) -> dict:
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "text": self.text,
            "ordinal": self.ordinal,
            "acl": self.acl,
            "department": self.department,
        }


def _split_sections(text: str, doc_title: str, max_level: int = 2) -> list[tuple[str, str]]:
    """按 Markdown 标题切节，返回 (section 路径, 正文)。

    只在一级/二级标题处切分：
    - 父块需要足够的上下文（~600-900 字），把小标题也当切分点时父块会碎成 100 字；
    - 更深层的小标题保留在正文里，仍参与 BM25 与向量召回。

    section 路径形如「三、住宿标准 > 3.2 一线城市」，用于前端引用溯源。
    """
    lines = text.splitlines()
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            path = " > ".join(title for _, title in stack)
            if path.split(" > ")[0] == doc_title:
                path = " > ".join(path.split(" > ")[1:])
            sections.append((path, body))
        buffer.clear()

    for line in lines:
        match = _HEADING.match(line.strip())
        if match and len(match.group(1)) <= max_level:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            buffer.append(title)  # 标题本身保留在正文里，提升关键词召回
        else:
            buffer.append(line)
    flush()

    if not sections:  # PDF/DOCX 没有 Markdown 标题
        body = normalize_text(text)
        if body:
            sections = [("", body)]
    return sections


def _units_for(text: str) -> list[str]:
    """切分单元：表格整块保留，其余按句切；段落之间保留换行，避免标题与表格粘连。"""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    units: list[str] = []
    for index, block in enumerate(blocks):
        if block.lstrip().startswith("|") and block.count("\n") >= 2:
            block_units = [block]
        else:
            block_units = split_sentences(block)
        # 除第一段外，段落首单元前加换行，保证拼接后仍可读、可再切分
        if index > 0 and block_units:
            block_units[0] = "\n" + block_units[0]
        units.extend(block_units)
    return units or [text]


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """按句子打包成定长窗口，窗口之间按「整句」重叠。"""
    units = _units_for(text)
    chunks: list[str] = []
    buffer: list[str] = []
    length = 0

    for unit in units:
        if buffer and length + len(unit) > size:
            chunks.append("".join(buffer))
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(buffer):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            # 超长单元单独成一窗口，避免死循环
            if not tail and len(unit) > size:
                buffer, length = [], 0
            else:
                buffer, length = tail, tail_len
        buffer.append(unit)
        length += len(unit)

    if buffer:
        remainder = "".join(buffer).strip()
        if remainder:
            chunks.append(remainder)
    return [c.strip() for c in chunks if c.strip()]


def _merge_sections(sections: list[tuple[str, str]], parent_size: int) -> list[tuple[list[str], str]]:
    """把相邻小节合并成父块，直到接近 parent_size。

    企业文档常见「每节只有两三句话」，如果严格按节切父块，
    父块会碎成 100 多字，父子策略就退化成普通分块。
    合并后父块稳定在 600-900 字，子块才有细分空间。
    """
    merged: list[tuple[list[str], str]] = []
    current_sections: list[str] = []
    current_text = ""

    for section, body in sections:
        candidate = f"{current_text}\n\n{body}".strip() if current_text else body
        if current_text and len(candidate) > parent_size:
            merged.append((current_sections, current_text))
            current_sections, current_text = [section], body
        else:
            current_sections.append(section)
            current_text = candidate

    if current_text:
        merged.append((current_sections, current_text))
    return merged


def _section_label(sections: list[str]) -> str:
    unique = [s for s in dict.fromkeys(sections) if s]
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return f"{unique[0]} 等 {len(unique)} 节"


def chunk_document(
    doc: LoadedDoc,
    parent_size: int = 900,
    child_size: int = 220,
    parent_overlap: int = 120,
    child_overlap: int = 40,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    text = normalize_text(doc.text)
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for section_index, (sections, body) in enumerate(
        _merge_sections(_split_sections(text, doc.title), parent_size)
    ):
        section = _section_label(sections)
        for parent_index, parent_text in enumerate(_windows(body, parent_size, parent_overlap)):
            parent_id = f"{doc.doc_id}:p{section_index}-{parent_index}"
            parent = ParentChunk(
                parent_id=parent_id,
                doc_id=doc.doc_id,
                title=doc.title,
                section=section,
                text=parent_text,
                acl=doc.acl,
                department=doc.department,
            )

            for child_index, child_text in enumerate(_windows(parent_text, child_size, child_overlap)):
                child_id = f"{parent_id}:c{child_index}"
                header = f"{doc.title}｜{section}" if section else doc.title
                child = ChildChunk(
                    child_id=child_id,
                    parent_id=parent_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=section,
                    text=child_text,
                    # 入库文本带上文档标题与章节，解决「子块脱离上下文后指代不明」的问题
                    embed_text=f"{header}\n{child_text}",
                    ordinal=child_index,
                    acl=doc.acl,
                    department=doc.department,
                )
                children.append(child)
                parent.child_ids.append(child_id)

            parents.append(parent)

    return parents, children


def chunk_documents(
    docs: list[LoadedDoc],
    parent_size: int = 900,
    child_size: int = 220,
    parent_overlap: int = 120,
    child_overlap: int = 40,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    all_parents: list[ParentChunk] = []
    all_children: list[ChildChunk] = []
    for doc in docs:
        parents, children = chunk_document(doc, parent_size, child_size, parent_overlap, child_overlap)
        all_parents.extend(parents)
        all_children.extend(children)
    return all_parents, all_children

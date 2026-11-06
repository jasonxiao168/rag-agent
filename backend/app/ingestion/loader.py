"""文档加载：Markdown / TXT / PDF / DOCX 统一成 LoadedDoc。

本次新增：
- frontmatter 解析（title / acl / department），用于文档级权限；
- content_hash，用于增量索引判断文档是否变化。

依赖策略：pypdf 与 python-docx 都是可选依赖，
缺失时只跳过对应格式并打日志，不会让整个索引流程崩掉。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.core.text import normalize_text

logger = get_logger("ingestion.loader")

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx"}

_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.S)


@dataclass
class LoadedDoc:
    doc_id: str
    title: str
    text: str
    source: str
    doc_type: str
    acl: str = "all"          # "all" 或 "hr|finance"（管道分隔，便于写进向量库 metadata）
    department: str = ""      # 归属部门，仅用于展示
    content_hash: str = ""    # 归一化正文的 sha1，增量索引用

    @property
    def acl_list(self) -> list[str]:
        return [item for item in self.acl.split("|") if item] or ["all"]

    def __post_init__(self) -> None:
        """没显式给 hash 时自动补一个，便于测试与手工构造文档。"""
        if not self.content_hash:
            self.content_hash = hashlib.sha1(
                f"{self.acl}|{self.department}|{self.text}".encode("utf-8")
            ).hexdigest()

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "doc_type": self.doc_type,
            "chars": len(self.text),
            "acl": self.acl,
            "acl_list": self.acl_list,
            "department": self.department,
            "content_hash": self.content_hash,
        }


def _doc_id_for(path: Path) -> str:
    digest = hashlib.sha1(str(path.name).encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def _guess_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line and not line.startswith("#"):
            return fallback
    return fallback


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """解析 `---` 包裹的键值对，不引入 PyYAML。"""
    match = _FRONTMATTER.match(raw)
    if not match:
        return {}, raw

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    return meta, raw[match.end() :]


def _normalize_acl(value: str) -> str:
    parts = {item.strip().lower() for item in re.split(r"[,|;]", value or "all") if item.strip()}
    if not parts or "all" in parts:
        return "all"
    return "|".join(sorted(parts))


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("未安装 pypdf，跳过 PDF：%s", path.name)
        return ""
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    try:
        import docx
    except ImportError:
        logger.warning("未安装 python-docx，跳过 DOCX：%s", path.name)
        return ""
    document = docx.Document(str(path))
    parts: list[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def load_file(path: Path) -> LoadedDoc | None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        logger.info("跳过不支持的文件：%s", path.name)
        return None

    if suffix == ".pdf":
        raw = _read_pdf(path)
    elif suffix == ".docx":
        raw = _read_docx(path)
    else:
        raw = path.read_text(encoding="utf-8", errors="ignore")

    text = normalize_text(raw)
    meta, text = _parse_frontmatter(text)
    text = normalize_text(text)
    if not text:
        logger.warning("文件无可用文本：%s", path.name)
        return None

    return LoadedDoc(
        doc_id=_doc_id_for(path),
        title=meta.get("title") or _guess_title(text, path.stem),
        text=text,
        source=str(path),
        doc_type=suffix.lstrip("."),
        acl=_normalize_acl(meta.get("acl", "all")),
        department=meta.get("department", ""),
        # ACL 也进 hash：只改权限不改正文时，索引必须重建
        content_hash=hashlib.sha1(
            f"{_normalize_acl(meta.get('acl', 'all'))}|{meta.get('department', '')}|{text}".encode("utf-8")
        ).hexdigest(),
    )


def load_documents(raw_dir: Path) -> list[LoadedDoc]:
    """递归加载目录下所有支持的文件，按文件名排序保证索引可复现。"""
    if not raw_dir.exists():
        logger.warning("语料目录不存在：%s", raw_dir)
        return []

    docs: list[LoadedDoc] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            doc = load_file(path)
            if doc:
                docs.append(doc)
    logger.info("加载文档 %d 份，共 %d 字", len(docs), sum(len(d.text) for d in docs))
    return docs

"""Structure-aware loading and parent-child chunking for chemical documents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Iterable
import unicodedata

from src.config import (
    DEFAULT_CHILD_MAX_TOKENS,
    DEFAULT_CHILD_OVERLAP_TOKENS,
    DEFAULT_CHILD_TARGET_TOKENS,
    DEFAULT_PARENT_TARGET_TOKENS,
)

from .models import ChildChunk, ParentChunk, SourceDocument, StructuralBlock
from .tokenizer import ChemicalTokenizer


_HEADING_RE = re.compile(r"^(?P<marker>#{1,6})\s+(?P<title>.+)$")
_CHINESE_HEADING_RE = re.compile(
    r"^第(?P<number>[一二三四五六七八九十百千〇零]+)(?P<kind>[章节部分篇])\s*(?P<title>.*)$"
)
_ARABIC_HEADING_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+){0,5})[.、]?(?:\s+|$)(?P<title>.+)$"
)
_STANDARD_CLAUSE_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)+)\s+(?P<text>.+)$")
_ARTICLE_RE = re.compile(r"^第(?P<number>[一二三四五六七八九十百千〇零]+)条\s*(?P<text>.+)$")
_CLAIM_RE = re.compile(r"^(?:权利要求|claim)\s*(?P<number>\d+)\s*[:：.、]?\s*(?P<text>.*)$", re.I)
_LIST_RE = re.compile(r"^(?:[-*•●▪]|(?:\d+|[一二三四五六七八九十])[.、)）]|[（(](?:\d+|[一二三四五六七八九十])[)）])\s*.+$")
_PROCESS_RE = re.compile(
    r"^(?:步骤|step|工序|阶段)\s*(?:[一二三四五六七八九十\d]+|S\d+)?\s*[:：.、-]?\s*.+$",
    re.I,
)
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$|^\s*[^\t]+(?:\t[^\t]+)+\s*$")
_SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.])(?=\s+[A-Z0-9])")
_UNIT_RE = re.compile(r"(?:\(([^()]+)\)|\[([^\[\]]+)\]|/\s*([%A-Za-z°μµ][\w/%°μµ·^\-]*)$)")
_DATE_COLUMN_RE = re.compile(r"date|time|日期|时间|年月|批次", re.I)


def _normalized_text(text: str) -> str:
    """Normalize text for stable hashes and compact identifiers."""

    return " ".join(unicodedata.normalize("NFKC", text).split())


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _page_range(blocks: Iterable[StructuralBlock]) -> tuple[int | None, int | None]:
    pages = [
        page
        for block in blocks
        for page in (block.page_start, block.page_end)
        if page is not None
    ]
    return (min(pages), max(pages)) if pages else (None, None)


class ChemicalDocumentLoader:
    """Load supported files into ordered, source-aware structural blocks."""

    @classmethod
    def load(cls, path: str) -> SourceDocument:
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            raw_blocks = cls._load_pdf(source_path)
        elif suffix == ".docx":
            raw_blocks = cls._load_docx(source_path)
        elif suffix == ".doc":
            raw_blocks = cls._load_doc(source_path)
        elif suffix == ".csv":
            raw_blocks = cls._load_csv(source_path)
        elif suffix in {".xlsx", ".xls"}:
            raw_blocks = cls._load_excel(source_path)
        else:
            raw_blocks = cls._text_blocks(cls._read_text(source_path), None)

        title = source_path.stem
        doc_type = cls._infer_doc_type(source_path, raw_blocks)
        blocks = cls._structure_blocks(raw_blocks, doc_type)
        normalized_source = _normalized_text(source_path.as_posix())
        normalized_content = "\n".join(_normalized_text(block.text) for block in blocks)
        return SourceDocument(
            doc_id=_digest(normalized_source),
            version_id=_digest(normalized_content),
            title=title,
            doc_type=doc_type,
            source=str(source_path),
            blocks=tuple(blocks),
            metadata={"extension": suffix.lstrip("."), "block_count": len(blocks)},
        )

    @staticmethod
    def _read_text(path: Path) -> str:
        last_error: UnicodeDecodeError | None = None
        for encoding in ("utf-8", "gbk", "gb2312", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return ""

    @classmethod
    def _load_pdf(cls, path: Path) -> list[StructuralBlock]:
        from langchain_community.document_loaders import PyPDFLoader

        pages = [document.page_content or "" for document in PyPDFLoader(str(path)).load()]
        pages = cls._without_repeated_pdf_margins(pages)
        blocks: list[StructuralBlock] = []
        for page_number, page_text in enumerate(pages, start=1):
            blocks.extend(cls._text_blocks(page_text, page_number))
        return blocks

    @classmethod
    def _load_docx(cls, path: Path) -> list[StructuralBlock]:
        from docx import Document
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = Document(str(path))
        blocks: list[StructuralBlock] = []
        for child in document.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, document)
                text = paragraph.text.strip()
                if not text:
                    continue
                style_name = paragraph.style.name.lower() if paragraph.style else ""
                block_type = "heading" if style_name.startswith("heading") else "paragraph"
                blocks.append(StructuralBlock(text, block_type, "", None, None, None))
            elif isinstance(child, CT_Tbl):
                table = Table(child, document)
                rows = [
                    "| " + " | ".join(cell.text.strip() for cell in row.cells) + " |"
                    for row in table.rows
                    if any(cell.text.strip() for cell in row.cells)
                ]
                if rows:
                    blocks.append(
                        StructuralBlock("\n".join(rows), "table", "", None, None, None)
                    )
        return blocks

    @classmethod
    def _load_doc(cls, path: Path) -> list[StructuralBlock]:
        from langchain_community.document_loaders import Docx2txtLoader

        text = "\n".join(document.page_content or "" for document in Docx2txtLoader(str(path)).load())
        return cls._text_blocks(text, None)

    @classmethod
    def _load_csv(cls, path: Path) -> list[StructuralBlock]:
        import pandas as pd

        frame = None
        last_error: Exception | None = None
        for encoding in ("utf-8", "gbk", "gb2312", "latin-1"):
            try:
                frame = pd.read_csv(path, encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError) as exc:
                last_error = exc
        if frame is None:
            if last_error is not None:
                raise last_error
            frame = pd.read_csv(path)
        return [cls._schema_block(path.name, None, frame)]

    @classmethod
    def _load_excel(cls, path: Path) -> list[StructuralBlock]:
        import pandas as pd

        workbook = pd.ExcelFile(path)
        return [
            cls._schema_block(path.name, sheet_name, pd.read_excel(workbook, sheet_name=sheet_name))
            for sheet_name in workbook.sheet_names
        ]

    @staticmethod
    def _schema_block(file_name: str, sheet_name: str | None, frame: object) -> StructuralBlock:
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Tabular loader must produce a pandas DataFrame.")
        columns = [str(column) for column in frame.columns]
        units = {
            column: next((part for part in match.groups() if part), "")
            for column in columns
            if (match := _UNIT_RE.search(column)) is not None
        }
        date_ranges: list[str] = []
        for column in columns:
            if not _DATE_COLUMN_RE.search(column):
                continue
            values = pd.to_datetime(frame[column], errors="coerce").dropna()
            if not values.empty:
                date_ranges.append(
                    f"{column}: {values.min().date().isoformat()} 至 {values.max().date().isoformat()}"
                )
        lines = [f"数据集：{file_name}"]
        if sheet_name is not None:
            lines.append(f"工作表：{sheet_name}")
        lines.append(f"行数：{len(frame)}")
        lines.append(f"列名：{', '.join(columns) if columns else '无'}")
        if units:
            lines.append("单位：" + "; ".join(f"{name}={unit}" for name, unit in units.items()))
        if date_ranges:
            lines.append("日期范围：" + "; ".join(date_ranges))
        return StructuralBlock("\n".join(lines), "table", "", None, None, None)

    @classmethod
    def _text_blocks(cls, text: str, page: int | None) -> list[StructuralBlock]:
        blocks: list[StructuralBlock] = []
        pending: list[str] = []
        table_rows: list[str] = []

        def flush_pending() -> None:
            if pending:
                value = "\n".join(pending).strip()
                if value:
                    blocks.append(StructuralBlock(value, "paragraph", "", page, page, None))
                pending.clear()

        def flush_table() -> None:
            if table_rows:
                blocks.append(
                    StructuralBlock("\n".join(table_rows), "table", "", page, page, None)
                )
                table_rows.clear()

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                flush_pending()
                flush_table()
                continue
            if _TABLE_ROW_RE.match(line):
                flush_pending()
                table_rows.append(line)
                continue
            flush_table()
            if cls._heading_details(line) is not None:
                flush_pending()
                blocks.append(StructuralBlock(line, "heading", "", page, page, None))
            elif _CLAIM_RE.match(line) or _ARTICLE_RE.match(line) or _PROCESS_RE.match(line):
                flush_pending()
                blocks.append(StructuralBlock(line, "paragraph", "", page, page, None))
            else:
                pending.append(line)
        flush_pending()
        flush_table()
        return blocks

    @staticmethod
    def _without_repeated_pdf_margins(pages: list[str]) -> list[str]:
        first_last = []
        for page in pages:
            lines = [line.strip() for line in page.splitlines() if line.strip()]
            first_last.extend({lines[0], lines[-1]} if lines else set())
        repeated = {line for line, count in Counter(first_last).items() if count >= 3}
        cleaned: list[str] = []
        for page in pages:
            lines = page.splitlines()
            nonempty = [index for index, line in enumerate(lines) if line.strip()]
            if nonempty and lines[nonempty[0]].strip() in repeated:
                lines[nonempty[0]] = ""
            if nonempty and lines[nonempty[-1]].strip() in repeated:
                lines[nonempty[-1]] = ""
            cleaned.append("\n".join(lines))
        return cleaned

    @staticmethod
    def _infer_doc_type(path: Path, blocks: list[StructuralBlock]) -> str:
        probe = f"{path.stem} " + " ".join(block.text for block in blocks[:8])
        if path.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            return "tabular"
        if re.search(r"专利|patent|权利要求", probe, re.I):
            return "patent"
        if re.search(r"GB/T|标准|standard|规范", probe, re.I):
            return "standard"
        if re.search(r"安全|SDS|MSDS|safety", probe, re.I):
            return "safety"
        if re.search(r"工艺|流程|process|procedure", probe, re.I):
            return "process"
        return "document"

    @staticmethod
    def _heading_details(text: str) -> tuple[int, str] | None:
        if match := _HEADING_RE.match(text):
            return (len(match.group("marker")), match.group("title").strip())
        if match := _CHINESE_HEADING_RE.match(text):
            return (1, text.strip())
        if match := _ARABIC_HEADING_RE.match(text):
            title = match.group("title").strip()
            if len(title) <= 160 and not title.endswith(("。", "；", ";")):
                return (match.group("number").count(".") + 1, text.strip())
        return None

    @classmethod
    def _structure_blocks(
        cls, raw_blocks: list[StructuralBlock], doc_type: str
    ) -> list[StructuralBlock]:
        stack: list[tuple[int, str]] = []
        structured: list[StructuralBlock] = []
        seen: set[tuple[str, str, int | None, int | None]] = set()
        for block in raw_blocks:
            text = block.text.strip()
            if not text:
                continue
            heading = cls._heading_details(text) if block.block_type == "heading" else None
            if doc_type in {"standard", "safety"} and _STANDARD_CLAUSE_RE.match(text):
                heading = None
            if heading is not None:
                level, title = heading
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                section_path = " > ".join(item[1] for item in stack)
                block_type, clause_no = "heading", None
            else:
                section_path = " > ".join(item[1] for item in stack)
                block_type, clause_no = cls._classify_block(text, block.block_type, doc_type)
                if clause_no:
                    section_path = " > ".join(filter(None, (section_path, clause_no)))
            identity = (text, block_type, block.page_start, block.page_end)
            if identity in seen:
                continue
            seen.add(identity)
            structured.append(
                StructuralBlock(
                    text=text,
                    block_type=block_type,
                    section_path=section_path,
                    page_start=block.page_start,
                    page_end=block.page_end,
                    clause_no=clause_no,
                )
            )
        return structured

    @staticmethod
    def _classify_block(text: str, original_type: str, doc_type: str) -> tuple[str, str | None]:
        if original_type == "table" or _TABLE_ROW_RE.match(text.splitlines()[0]):
            return "table", None
        if match := _CLAIM_RE.match(text):
            return "claim", match.group("number")
        if match := _ARTICLE_RE.match(text):
            return "clause", f"第{match.group('number')}条"
        if match := _STANDARD_CLAUSE_RE.match(text):
            if doc_type in {"standard", "safety"}:
                return "clause", match.group("number")
        if _PROCESS_RE.match(text):
            return "process_step", None
        if _LIST_RE.match(text):
            return "list_item", None
        return "paragraph", None


@dataclass(frozen=True)
class _ParentDraft:
    chunk: ParentChunk
    blocks: tuple[StructuralBlock, ...]


@dataclass(frozen=True)
class _ChildUnit:
    text: str
    is_table: bool = False


class StructureAwareChunker:
    """Create deterministic parent expansion units and embedding-ready children."""

    def __init__(
        self,
        tokenizer: ChemicalTokenizer,
        parent_target_tokens: int = DEFAULT_PARENT_TARGET_TOKENS,
        child_target_tokens: int = DEFAULT_CHILD_TARGET_TOKENS,
        child_max_tokens: int = DEFAULT_CHILD_MAX_TOKENS,
        child_overlap_tokens: int = DEFAULT_CHILD_OVERLAP_TOKENS,
    ) -> None:
        if parent_target_tokens <= 0 or child_target_tokens <= 0:
            raise ValueError("Chunk token targets must be positive integers.")
        if child_max_tokens < child_target_tokens:
            raise ValueError("child_max_tokens must be at least child_target_tokens.")
        if child_overlap_tokens < 0:
            raise ValueError("child_overlap_tokens cannot be negative.")
        self._tokenizer = tokenizer
        self._parent_target = parent_target_tokens
        self._child_target = child_target_tokens
        self._child_max = child_max_tokens
        self._child_overlap = child_overlap_tokens

    def chunk(self, document: SourceDocument) -> tuple[list[ParentChunk], list[ChildChunk]]:
        drafts = self._build_parents(document)
        parents = [draft.chunk for draft in drafts]
        children = [
            child
            for draft in drafts
            for child in self._build_children(document, draft)
        ]
        return parents, children

    def _build_parents(self, document: SourceDocument) -> list[_ParentDraft]:
        groups: list[list[StructuralBlock]] = []
        current: list[StructuralBlock] = []
        for block in document.blocks:
            if not current:
                current.append(block)
                continue
            if self._starts_parent(block, current) or self._would_exceed_target(current, block):
                groups.append(current)
                current = [block]
            else:
                current.append(block)
        if current:
            groups.append(current)

        drafts: list[_ParentDraft] = []
        for ordinal, blocks in enumerate(groups):
            section_path = next((block.section_path for block in blocks if block.section_path), "")
            clause_no = next((block.clause_no for block in blocks if block.clause_no), None)
            page_start, page_end = _page_range(blocks)
            structural_path = clause_no or section_path or blocks[0].block_type
            parent_id = _digest(
                f"{document.doc_id}\0{document.version_id}\0{structural_path}\0{ordinal}"
            )
            metadata = {
                "doc_id": document.doc_id,
                "title": document.title,
                "source": document.source,
                "doc_type": document.doc_type,
                "section_path": section_path,
                "clause_no": clause_no,
                "page_start": page_start,
                "page_end": page_end,
                "parent_ordinal": ordinal,
            }
            drafts.append(
                _ParentDraft(
                    ParentChunk(
                        parent_id=parent_id,
                        version_id=document.version_id,
                        content="\n\n".join(block.text for block in blocks),
                        metadata=metadata,
                    ),
                    tuple(blocks),
                )
            )
        return drafts

    def _starts_parent(
        self, block: StructuralBlock, current: list[StructuralBlock]
    ) -> bool:
        if block.block_type in {"heading", "clause", "claim", "process_step"}:
            return True
        return False

    def _would_exceed_target(
        self, current: list[StructuralBlock], block: StructuralBlock
    ) -> bool:
        protected = current[0].block_type in {"clause", "claim", "process_step"}
        if protected:
            return False
        candidate = "\n\n".join(item.text for item in (*current, block))
        return self._tokens(candidate) > self._parent_target

    def _build_children(
        self, document: SourceDocument, draft: _ParentDraft) -> list[ChildChunk]:
        units = [
            unit
            for block in draft.blocks
            for unit in self._units_from_block(block, document.title)
        ]
        child_units: list[list[_ChildUnit]] = []
        current: list[_ChildUnit] = []
        for unit in units:
            for piece in self._split_unit(unit):
                candidate = [*current, piece]
                if current and self._tokens(self._join_units(candidate)) > self._child_target:
                    child_units.append(current)
                    current = self._overlap_units(current, piece)
                current.append(piece)
        if current:
            child_units.append(current)

        chunks: list[ChildChunk] = []
        for ordinal, units_for_child in enumerate(child_units):
            content = self._join_units(units_for_child)
            normalized_content = _normalized_text(content)
            chunk_id = _digest(f"{draft.chunk.parent_id}\0{ordinal}\0{normalized_content}")
            metadata = dict(draft.chunk.metadata)
            metadata.update(
                {
                    "parent_id": draft.chunk.parent_id,
                    "child_ordinal": ordinal,
                    "content_hash": _digest(normalized_content),
                    "previous_sibling_id": None,
                    "next_sibling_id": None,
                }
            )
            section = str(metadata["section_path"])
            clause = str(metadata["clause_no"] or "")
            embedding_text = (
                f"文档：{document.title}\n章节：{section}\n条款：{clause}\n正文：{content}"
            )
            chunks.append(
                ChildChunk(
                    chunk_id=chunk_id,
                    parent_id=draft.chunk.parent_id,
                    version_id=document.version_id,
                    content=content,
                    embedding_text=embedding_text,
                    ordinal=ordinal,
                    metadata=metadata,
                )
            )
        return self._with_siblings(chunks)

    def _units_from_block(self, block: StructuralBlock, title: str) -> list[_ChildUnit]:
        if block.block_type == "table":
            return self._table_units(block.text, title)
        if block.block_type in {"clause", "claim", "list_item", "process_step"}:
            if self._tokens(block.text) <= self._child_max:
                return [_ChildUnit(block.text)]
        sentences = [part.strip() for part in _SENTENCE_RE.split(block.text) if part.strip()]
        return [_ChildUnit(sentence) for sentence in sentences] or [_ChildUnit(block.text)]

    def _table_units(self, text: str, title: str) -> list[_ChildUnit]:
        rows = [line.strip() for line in text.splitlines() if line.strip()]
        pipe_rows = [row for row in rows if row.startswith("|") and row.endswith("|")]
        tab_rows = [row for row in rows if "\t" in row]
        table_rows = pipe_rows or tab_rows
        if len(table_rows) < 2:
            return [_ChildUnit(text, is_table=True)]
        header = table_rows[0]
        data_rows = [
            row
            for row in table_rows[1:]
            if not re.fullmatch(r"\|?\s*[:\- ]+(?:\|\s*[:\- ]+)+\|?", row)
        ]
        context = f"表格：{title}\n表头：{header}"
        return [_ChildUnit(f"{context}\n数据行：{row}", is_table=True) for row in data_rows]

    def _split_unit(self, unit: _ChildUnit) -> list[_ChildUnit]:
        if self._tokens(unit.text) <= self._child_max:
            return [unit]
        pieces: list[_ChildUnit] = []
        remaining = unit.text.strip()
        while remaining:
            end = self._largest_prefix_within_limit(remaining, self._child_max)
            if end <= 0:
                raise ValueError("Tokenizer cannot fit a single character in a child chunk.")
            split_at = self._preferred_split(remaining, end)
            pieces.append(_ChildUnit(remaining[:split_at].strip(), unit.is_table))
            remaining = remaining[split_at:].strip()
        return pieces

    def _largest_prefix_within_limit(self, text: str, token_limit: int) -> int:
        low, high, best = 1, len(text), 0
        while low <= high:
            midpoint = (low + high) // 2
            if self._tokens(text[:midpoint]) <= token_limit:
                best = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1
        return best

    @staticmethod
    def _preferred_split(text: str, end: int) -> int:
        boundary = max(
            text.rfind(mark, 0, end + 1) + 1 for mark in ("\n", " ", "。", "；", ";", "，", ",")
        )
        return boundary if boundary > max(1, end // 2) else end

    def _overlap_units(
        self, completed: list[_ChildUnit], next_unit: _ChildUnit
    ) -> list[_ChildUnit]:
        if next_unit.is_table or any(unit.is_table for unit in completed):
            return []
        overlap: list[_ChildUnit] = []
        for unit in reversed(completed):
            candidate = [unit, *overlap]
            if self._tokens(self._join_units(candidate)) > self._child_overlap:
                break
            if self._tokens(self._join_units([*candidate, next_unit])) > self._child_max:
                break
            overlap = candidate
        return overlap

    @staticmethod
    def _join_units(units: list[_ChildUnit]) -> str:
        return "\n".join(unit.text for unit in units).strip()

    def _tokens(self, text: str) -> int:
        return self._tokenizer.model_tokens(text)

    @staticmethod
    def _with_siblings(chunks: list[ChildChunk]) -> list[ChildChunk]:
        siblings: list[ChildChunk] = []
        for index, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)
            metadata["previous_sibling_id"] = chunks[index - 1].chunk_id if index else None
            metadata["next_sibling_id"] = (
                chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
            )
            siblings.append(
                ChildChunk(
                    chunk_id=chunk.chunk_id,
                    parent_id=chunk.parent_id,
                    version_id=chunk.version_id,
                    content=chunk.content,
                    embedding_text=chunk.embedding_text,
                    ordinal=chunk.ordinal,
                    metadata=metadata,
                )
            )
        return siblings

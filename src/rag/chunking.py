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
    RAGSettings,
    get_rag_settings,
)

from .models import ChildChunk, ParentChunk, SourceDocument, StructuralBlock
from .tokenizer import ChemicalTokenizer


_HEADING_RE = re.compile(r"^(?P<marker>#{1,6})\s+(?P<title>.+)$")
_CHINESE_HEADING_RE = re.compile(
    r"^第(?P<number>[一二三四五六七八九十百千〇零]+)(?P<kind>[章节部分篇])\s*(?P<title>.*)$"
)
_ARABIC_HEADING_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+){0,5})(?:\s+|$)(?P<title>.+)$"
)
_STANDARD_CLAUSE_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+)+)[.、:：]?\s*(?P<text>.+)$"
)
_ARTICLE_RE = re.compile(r"^第(?P<number>[一二三四五六七八九十百千〇零]+)条\s*(?P<text>.+)$")
_CLAIM_RE = re.compile(r"^(?:权利要求|claim)\s*(?P<number>\d+)\s*[:：.、]?\s*(?P<text>.*)$", re.I)
_BARE_CLAIM_RE = re.compile(r"^(?P<number>\d+)[.、)）]\s*(?P<text>.+)$")
_CLAIMS_SECTION_RE = re.compile(r"^(?:权利要求(?:书)?|claims?)\b", re.I)
_LIST_RE = re.compile(r"^(?:[-*•●▪]|(?:\d+|[一二三四五六七八九十])[.、)）]|[（(](?:\d+|[一二三四五六七八九十])[)）])\s*.+$")
_PROCESS_RE = re.compile(
    r"^(?:步骤|step|工序|阶段)\s*(?:[一二三四五六七八九十\d]+|S\d+)?\s*[:：.、-]?\s*.+$",
    re.I,
)
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$|^\s*[^\t]+(?:\t[^\t]+)+\s*$")
_TABLE_TITLE_RE = re.compile(r"^(?:表\s*\d+[\s.:：、-]*|table\s*\d+[\s.:：、-]*).+", re.I)
_SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.])(?=\s+[A-Z0-9])")
_UNIT_RE = re.compile(r"(?:\(([^()]+)\)|\[([^\[\]]+)\]|/\s*([%A-Za-z°μµ][\w/%°μµ·^\-]*)$)")
_DATE_COLUMN_RE = re.compile(r"date|time|日期|时间|年月|批次", re.I)
TABLE_CONTEXT_MAX_TOKENS = 180


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
        doc_id = _digest(normalized_source)
        content_hash = _digest(normalized_content)
        return SourceDocument(
            doc_id=doc_id,
            content_hash=content_hash,
            version_id=_digest(f"{doc_id}\0{content_hash}"),
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
                heading_match = re.match(r"heading\s*(?P<level>\d+)?", style_name)
                block_type = (
                    f"heading:{heading_match.group('level') or '1'}"
                    if heading_match
                    else "paragraph"
                )
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
            if _LIST_RE.match(line):
                flush_pending()
                blocks.append(StructuralBlock(line, "list_item", "", page, page, None))
            elif cls._heading_details(line) is not None:
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
        probe = f"{path.stem} " + " ".join(block.text for block in blocks)
        if path.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            return "tabular"
        if re.search(r"专利|patent|权利要求|\bclaims?\b", probe, re.I):
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
        claim_context = False
        for block in raw_blocks:
            text = block.text.strip()
            if not text:
                continue
            is_claims_section = bool(_CLAIMS_SECTION_RE.match(text))
            block_type, clause_no = cls._classify_block(
                text, block.block_type, doc_type, claim_context
            )
            heading = None
            if block.block_type.startswith("heading") and block_type not in {
                "clause",
                "claim",
            }:
                heading = cls._heading_details(text) or (
                    cls._loader_heading_level(block.block_type),
                    text,
                )
            elif block_type not in {
                "clause",
                "claim",
                "list_item",
                "process_step",
                "table",
                "table_title",
            } and is_claims_section:
                heading = (1, text)
            if heading is not None:
                level, title = heading
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                section_path = " > ".join(item[1] for item in stack)
                block_type, clause_no = "heading", None
                claim_context = doc_type == "patent" and is_claims_section
            else:
                section_path = " > ".join(item[1] for item in stack)
                if clause_no:
                    section_path = " > ".join(filter(None, (section_path, clause_no)))
                if doc_type == "patent" and block_type == "claim":
                    claim_context = True
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
    def _loader_heading_level(block_type: str) -> int:
        match = re.fullmatch(r"heading:(?P<level>\d+)", block_type)
        return int(match.group("level")) if match else 1

    @staticmethod
    def _classify_block(
        text: str,
        original_type: str,
        doc_type: str,
        claim_context: bool = False,
    ) -> tuple[str, str | None]:
        if original_type == "table" or _TABLE_ROW_RE.match(text.splitlines()[0]):
            return "table", None
        if _TABLE_TITLE_RE.match(text):
            return "table_title", None
        if match := _CLAIM_RE.match(text):
            return "claim", match.group("number")
        if doc_type == "patent" and claim_context and (match := _BARE_CLAIM_RE.match(text)):
            return "claim", match.group("number")
        if match := _ARTICLE_RE.match(text):
            return "clause", f"第{match.group('number')}条"
        if match := _STANDARD_CLAUSE_RE.match(text):
            return "clause", match.group("number")
        if _PROCESS_RE.match(text):
            return "process_step", None
        if original_type == "list_item" or _LIST_RE.match(text):
            return "list_item", None
        return "paragraph", None


@dataclass(frozen=True)
class _ParentDraft:
    chunk: ParentChunk
    blocks: tuple[StructuralBlock, ...]


@dataclass(frozen=True)
class _SectionDraft:
    section_id: str
    structural_path: str
    ordinal: int
    table_title: str
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
        settings: RAGSettings | None = None,
    ) -> None:
        rag_settings = settings or get_rag_settings()
        if (
            rag_settings.parent_target_tokens <= 0
            or rag_settings.child_target_tokens <= 0
        ):
            raise ValueError("Chunk token targets must be positive integers.")
        if rag_settings.child_max_tokens < rag_settings.child_target_tokens:
            raise ValueError("child_max_tokens must be at least child_target_tokens.")
        if rag_settings.child_overlap_tokens < 0:
            raise ValueError("child_overlap_tokens cannot be negative.")
        if rag_settings.parent_max_tokens < rag_settings.parent_target_tokens:
            raise ValueError("parent_max_tokens must be at least parent_target_tokens.")
        self._tokenizer = tokenizer
        self._parent_target = rag_settings.parent_target_tokens
        self._parent_max = rag_settings.parent_max_tokens
        self._child_target = rag_settings.child_target_tokens
        self._child_max = rag_settings.child_max_tokens
        self._child_overlap = rag_settings.child_overlap_tokens

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
        parents: list[_ParentDraft] = []
        for section in self._build_sections(document):
            parents.extend(self._build_section_parents(document, section))
        return parents

    def _build_sections(self, document: SourceDocument) -> list[_SectionDraft]:
        sections: list[tuple[list[StructuralBlock], str]] = []
        current: list[StructuralBlock] = []
        current_table_title = document.title
        nearest_table_title = document.title
        for block in document.blocks:
            if block.block_type == "table_title" or _TABLE_TITLE_RE.match(block.text):
                nearest_table_title = block.text
            if not current:
                current.append(block)
                current_table_title = nearest_table_title
                continue
            if self._starts_section(block, current):
                sections.append((current, current_table_title))
                current = [block]
                current_table_title = nearest_table_title
            else:
                current.append(block)
        if current:
            sections.append((current, current_table_title))

        drafts: list[_SectionDraft] = []
        for ordinal, (blocks, table_title) in enumerate(sections):
            section_path = next((block.section_path for block in blocks if block.section_path), "")
            clause_no = next((block.clause_no for block in blocks if block.clause_no), None)
            structural_path = clause_no or section_path or blocks[0].block_type
            section_id = _digest(f"{document.version_id}\0{structural_path}\0{ordinal}")
            drafts.append(
                _SectionDraft(
                    section_id=section_id,
                    structural_path=structural_path,
                    ordinal=ordinal,
                    table_title=table_title,
                    blocks=tuple(blocks),
                )
            )
        return drafts

    def _build_section_parents(
        self, document: SourceDocument, section: _SectionDraft
    ) -> list[_ParentDraft]:
        groups: list[list[StructuralBlock]] = []
        current: list[StructuralBlock] = []
        for block in section.blocks:
            for fragment in self._split_block_for_parent(block, section.table_title):
                candidate = [*current, fragment]
                if current and self._tokens(self._parent_content(candidate)) > self._parent_target:
                    groups.append(current)
                    current = [fragment]
                else:
                    current.append(fragment)
        if current:
            groups.append(current)

        drafts: list[_ParentDraft] = []
        for ordinal, blocks in enumerate(groups):
            section_path = next(
                (block.section_path for block in blocks if block.section_path), ""
            )
            clause_no = next((block.clause_no for block in blocks if block.clause_no), None)
            page_start, page_end = _page_range(blocks)
            page_or_offset = (
                f"page:{page_start}" if page_start is not None else f"offset:{ordinal}"
            )
            parent_id = _digest(f"{section.section_id}\0{ordinal}\0{page_or_offset}")
            metadata = {
                "doc_id": document.doc_id,
                "title": document.title,
                "source": document.source,
                "doc_type": document.doc_type,
                "section_path": section_path,
                "clause_no": clause_no,
                "page_start": page_start,
                "page_end": page_end,
                "section_id": section.section_id,
                "section_ordinal": section.ordinal,
                "parent_ordinal": ordinal,
                "page_or_offset": page_or_offset,
                "table_title": section.table_title,
            }
            drafts.append(
                _ParentDraft(
                    ParentChunk(
                        section_id=section.section_id,
                        parent_id=parent_id,
                        version_id=document.version_id,
                        content=self._parent_content(blocks),
                        metadata=metadata,
                    ),
                    tuple(blocks),
                )
            )
        return drafts

    @staticmethod
    def _starts_section(
        block: StructuralBlock, current: list[StructuralBlock]
    ) -> bool:
        if block.block_type in {
            "heading",
            "clause",
            "claim",
            "process_step",
            "table_title",
        }:
            return True
        return block.block_type == "table" and current[-1].block_type != "table_title"

    def _split_block_for_parent(
        self, block: StructuralBlock, table_title: str
    ) -> list[StructuralBlock]:
        if self._tokens(block.text) <= self._parent_max:
            return [block]
        if block.block_type == "table":
            return self._split_table_for_parent(block, table_title)
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_RE.split(block.text)
            if sentence.strip()
        ]
        if len(sentences) > 1:
            return self._group_parent_sentences(block, sentences)
        return self._hard_prefix_parent_fragments(block, block.text)

    def _group_parent_sentences(
        self, block: StructuralBlock, sentences: list[str]
    ) -> list[StructuralBlock]:
        fragments: list[StructuralBlock] = []
        current: list[str] = []
        for sentence in sentences:
            candidate = " ".join([*current, sentence]).strip()
            if current and self._tokens(candidate) > self._parent_max:
                fragments.append(self._copy_block(block, " ".join(current)))
                current = []
            if self._tokens(sentence) > self._parent_max:
                fragments.extend(self._hard_prefix_parent_fragments(block, sentence))
            else:
                current.append(sentence)
        if current:
            fragments.append(self._copy_block(block, " ".join(current)))
        return fragments

    def _hard_prefix_parent_fragments(
        self, block: StructuralBlock, text: str
    ) -> list[StructuralBlock]:
        fragments: list[StructuralBlock] = []
        remaining = text.strip()
        while remaining:
            end = self._largest_prefix_within_limit(remaining, self._parent_max)
            if end <= 0:
                raise ValueError("Tokenizer cannot fit a single character in a parent chunk.")
            split_at = self._preferred_split(remaining, end)
            fragments.append(self._copy_block(block, remaining[:split_at]))
            remaining = remaining[split_at:].strip()
        return fragments

    def _split_table_for_parent(
        self, block: StructuralBlock, title: str
    ) -> list[StructuralBlock]:
        context, rows = self._table_parent_context(block.text, title)
        if not rows:
            return [
                self._copy_block(block, text)
                for text in self._split_table_context_texts(context, self._parent_max)
            ]

        fragments: list[StructuralBlock] = []
        current: list[str] = []
        for row in rows:
            candidate = self._table_parent_text(context, [*current, row])
            if current and self._tokens(candidate) > self._parent_max:
                fragments.append(
                    self._copy_block(block, self._table_parent_text(context, current))
                )
                current = []
            single_row = self._table_parent_text(context, [row])
            if self._tokens(single_row) > self._parent_max:
                fragments.extend(
                    self._copy_block(block, text)
                    for text in self._split_table_row_texts(
                        context, row, self._parent_max
                    )
                )
            else:
                current.append(row)
        if current:
            fragments.append(self._copy_block(block, self._table_parent_text(context, current)))
        return fragments

    def _table_parent_context(self, text: str, title: str) -> tuple[str, list[str]]:
        rows = [line.strip() for line in text.splitlines() if line.strip()]
        pipe_rows = [row for row in rows if row.startswith("|") and row.endswith("|")]
        tab_rows = [row for row in rows if "\t" in row]
        table_rows = pipe_rows or tab_rows
        if not table_rows:
            return self._schema_table_context(text, title)
        header = table_rows[0]
        data_rows = [
            row
            for row in table_rows[1:]
            if not re.fullmatch(r"\|?\s*[:\- ]+(?:\|\s*[:\- ]+)+\|?", row)
        ]
        unit_rows = [
            row for row in data_rows if re.search(r"单位|units?", row, re.I)
        ]
        content_rows = [row for row in data_rows if row not in unit_rows]
        units = "\n".join(unit_rows) if unit_rows else None
        context = self._compact_table_context(title, header, units, None)
        return context, content_rows

    def _schema_table_context(self, text: str, title: str) -> tuple[str, list[str]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        values: dict[str, str] = {}
        details: list[str] = []
        for line in lines:
            label, separator, value = line.partition("：")
            if not separator:
                details.append(line)
                continue
            if label in {"数据集", "文件"}:
                values["title"] = value.strip()
            elif label in {"工作表", "Sheet"}:
                values["sheet"] = value.strip()
            elif label in {"列名", "列", "表头"}:
                values["header"] = value.strip()
            elif label in {"单位", "Units"}:
                values["units"] = value.strip()
            else:
                details.append(line)
        context = self._compact_table_context(
            values.get("title", title),
            values.get("header", "架构摘要"),
            values.get("units"),
            values.get("sheet"),
        )
        return context, [f"详情：{detail}" for detail in details]

    def _compact_table_context(
        self,
        title: str,
        header: str,
        units: str | None,
        sheet: str | None,
    ) -> str:
        components = [("表格", title), ("表头", header)]
        if units:
            components.append(("单位", units))
        if sheet:
            components.append(("工作表", sheet))

        def render(values: list[str], truncated: bool) -> str:
            lines = [f"{label}：{value}" for (label, _), value in zip(components, values)]
            if truncated:
                lines.append("[上下文已截断]")
            return "\n".join(lines)

        values = [value.strip() for _, value in components]
        full = render(values, False)
        if self._tokens(full) <= TABLE_CONTEXT_MAX_TOKENS:
            return full

        for index in range(len(values) - 1, -1, -1):
            low, high, best = 0, len(values[index]), 0
            while low <= high:
                midpoint = (low + high) // 2
                candidate_values = list(values)
                candidate_values[index] = values[index][:midpoint].rstrip()
                if self._tokens(render(candidate_values, True)) <= TABLE_CONTEXT_MAX_TOKENS:
                    best = midpoint
                    low = midpoint + 1
                else:
                    high = midpoint - 1
            values[index] = values[index][:best].rstrip()
        compact = render(values, True)
        if self._tokens(compact) > TABLE_CONTEXT_MAX_TOKENS:
            raise ValueError("Table identity labels exceed the compact context token limit.")
        return compact

    @staticmethod
    def _table_parent_text(context: str, rows: list[str]) -> str:
        payloads = [row if row.startswith(("数据行：", "详情：")) else f"数据行：{row}" for row in rows]
        return context + "\n" + "\n".join(payloads)

    def _split_table_context_texts(self, context: str, token_limit: int) -> list[str]:
        if self._tokens(context) <= token_limit:
            return [context]
        prefix, fields = self._table_context_fields(context)
        if not fields:
            fields = ["(无列信息)"]

        fragments: list[str] = []
        current: list[str] = []
        for field in fields:
            for field_fragment in self._split_table_context_field(
                prefix, field, token_limit
            ):
                candidate = self._table_context_text(prefix, [*current, field_fragment])
                if current and self._tokens(candidate) > token_limit:
                    fragments.append(self._table_context_text(prefix, current))
                    current = [field_fragment]
                else:
                    current.append(field_fragment)
        if current:
            fragments.append(self._table_context_text(prefix, current))
        return fragments

    def _split_table_payload_texts(
        self, context: str, payload: str, token_limit: int
    ) -> list[str]:
        return [
            text.replace("数据行：", "详情：", 1)
            for text in self._split_table_cell_texts(context, payload, "", token_limit)
        ]

    @staticmethod
    def _table_context_fields(context: str) -> tuple[str, list[str]]:
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        title = next((line for line in lines if line.startswith("表格：")), "表格：未命名表格")
        remaining = [line for line in lines if line != title]
        header = next((line for line in remaining if line.startswith("表头：")), None)
        if header is None:
            return f"{title}\n架构：", remaining
        fields = [header.removeprefix("表头：").strip()]
        fields.extend(line for line in remaining if line != header)
        return f"{title}\n表头：", [field for field in fields if field]

    @staticmethod
    def _table_context_text(prefix: str, fields: list[str]) -> str:
        return prefix + ("\n" + "\n".join(fields) if fields else "")

    def _split_table_context_field(
        self, prefix: str, field: str, token_limit: int, delimiter: str | None = None
    ) -> list[str]:
        candidate = self._table_context_text(prefix, [field])
        if self._tokens(candidate) <= token_limit:
            return [field]
        cells, detected_delimiter = self._table_cells(field)
        row_delimiter = delimiter if delimiter is not None else detected_delimiter
        if cells:
            fragments: list[str] = []
            current: list[str] = []
            for cell in cells:
                row = self._format_table_row([*current, cell], row_delimiter)
                if current and self._tokens(self._table_context_text(prefix, [row])) > token_limit:
                    fragments.append(self._format_table_row(current, row_delimiter))
                    current = []
                single = self._format_table_row([cell], row_delimiter)
                if self._tokens(self._table_context_text(prefix, [single])) > token_limit:
                    fragments.extend(
                        self._split_table_context_field(
                            prefix, cell, token_limit, row_delimiter
                        )
                    )
                else:
                    current.append(cell)
            if current:
                fragments.append(self._format_table_row(current, row_delimiter))
            return fragments

        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_RE.split(field)
            if sentence.strip()
        ]
        if len(sentences) > 1:
            fragments = []
            current = []
            for sentence in sentences:
                candidate_field = " ".join([*current, sentence]).strip()
                if current and self._tokens(
                    self._table_context_text(prefix, [candidate_field])
                ) > token_limit:
                    fragments.append(" ".join(current))
                    current = []
                if self._tokens(self._table_context_text(prefix, [sentence])) > token_limit:
                    fragments.extend(
                        self._split_table_context_field(prefix, sentence, token_limit, "")
                    )
                else:
                    current.append(sentence)
            if current:
                fragments.append(" ".join(current))
            return fragments

        return self._hard_split_table_context_field(prefix, field, token_limit)

    def _hard_split_table_context_field(
        self, prefix: str, field: str, token_limit: int
    ) -> list[str]:
        fragments: list[str] = []
        remaining = field.strip()
        while remaining:
            end = self._largest_table_context_prefix(prefix, remaining, token_limit)
            if end <= 0:
                raise ValueError("Table identity prefix cannot fit a single field character.")
            split_at = self._preferred_split(remaining, end)
            fragments.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        return fragments

    def _largest_table_context_prefix(
        self, prefix: str, value: str, token_limit: int
    ) -> int:
        low, high, best = 1, len(value), 0
        while low <= high:
            midpoint = (low + high) // 2
            candidate = self._table_context_text(prefix, [value[:midpoint]])
            if self._tokens(candidate) <= token_limit:
                best = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1
        return best

    def _split_table_row_texts(
        self, context: str, row: str, token_limit: int
    ) -> list[str]:
        cells, delimiter = self._table_cells(row)
        if not cells:
            return self._split_table_cell_texts(context, row, delimiter, token_limit)

        fragments: list[str] = []
        current: list[str] = []
        for cell in cells:
            candidate = self._table_parent_text(
                context, [self._format_table_row([*current, cell], delimiter)]
            )
            if current and self._tokens(candidate) > token_limit:
                fragments.append(
                    self._table_parent_text(
                        context, [self._format_table_row(current, delimiter)]
                    )
                )
                current = []
            single_cell = self._table_parent_text(
                context, [self._format_table_row([cell], delimiter)]
            )
            if self._tokens(single_cell) > token_limit:
                fragments.extend(
                    self._split_table_cell_texts(context, cell, delimiter, token_limit)
                )
            else:
                current.append(cell)
        if current:
            fragments.append(
                self._table_parent_text(context, [self._format_table_row(current, delimiter)])
            )
        return fragments

    @staticmethod
    def _table_cells(row: str) -> tuple[list[str], str]:
        stripped = row.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            return [cell.strip() for cell in stripped.strip("|").split("|")], "|"
        if "\t" in stripped:
            return [cell.strip() for cell in stripped.split("\t")], "\t"
        return [], ""

    @staticmethod
    def _format_table_row(cells: list[str], delimiter: str) -> str:
        if delimiter == "|":
            return "| " + " | ".join(cells) + " |"
        if delimiter == "\t":
            return "\t".join(cells)
        return " ".join(cells)

    def _split_table_cell_texts(
        self, context: str, cell: str, delimiter: str, token_limit: int
    ) -> list[str]:
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_RE.split(cell)
            if sentence.strip()
        ]
        fragments: list[str] = []
        current: list[str] = []
        for sentence in sentences or [cell]:
            candidate_cell = " ".join([*current, sentence]).strip()
            candidate = self._table_parent_text(
                context, [self._format_table_row([candidate_cell], delimiter)]
            )
            if current and self._tokens(candidate) > token_limit:
                fragments.append(
                    self._table_parent_text(
                        context, [self._format_table_row([" ".join(current)], delimiter)]
                    )
                )
                current = []
            single = self._table_parent_text(
                context, [self._format_table_row([sentence], delimiter)]
            )
            if self._tokens(single) > token_limit:
                fragments.extend(
                    self._hard_split_table_cell(context, sentence, delimiter, token_limit)
                )
            else:
                current.append(sentence)
        if current:
            fragments.append(
                self._table_parent_text(
                    context, [self._format_table_row([" ".join(current)], delimiter)]
                )
            )
        return fragments

    def _hard_split_table_cell(
        self, context: str, cell: str, delimiter: str, token_limit: int
    ) -> list[str]:
        fragments: list[str] = []
        remaining = cell.strip()
        while remaining:
            end = self._largest_table_value_prefix(
                context, remaining, delimiter, token_limit
            )
            if end <= 0:
                raise ValueError("Table context cannot fit a single cell character.")
            split_at = self._preferred_split(remaining, end)
            fragments.append(
                self._table_parent_text(
                    context,
                    [self._format_table_row([remaining[:split_at].strip()], delimiter)],
                )
            )
            remaining = remaining[split_at:].strip()
        return fragments

    def _largest_table_value_prefix(
        self, context: str, value: str, delimiter: str, token_limit: int
    ) -> int:
        low, high, best = 1, len(value), 0
        while low <= high:
            midpoint = (low + high) // 2
            candidate = self._table_parent_text(
                context, [self._format_table_row([value[:midpoint]], delimiter)]
            )
            if self._tokens(candidate) <= token_limit:
                best = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1
        return best

    @staticmethod
    def _copy_block(block: StructuralBlock, text: str) -> StructuralBlock:
        return StructuralBlock(
            text=text.strip(),
            block_type=block.block_type,
            section_path=block.section_path,
            page_start=block.page_start,
            page_end=block.page_end,
            clause_no=block.clause_no,
        )

    @staticmethod
    def _parent_content(blocks: list[StructuralBlock]) -> str:
        return "\n\n".join(block.text for block in blocks)

    def _build_children(
        self, document: SourceDocument, draft: _ParentDraft) -> list[ChildChunk]:
        units: list[_ChildUnit] = []
        nearest_table_title = str(draft.chunk.metadata.get("table_title", document.title))
        for block in draft.blocks:
            if block.block_type == "table_title" or _TABLE_TITLE_RE.match(block.text):
                nearest_table_title = block.text
            units.extend(self._units_from_block(block, nearest_table_title))
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
        labeled = self._labeled_table_context(text)
        if labeled is None:
            context, data_rows = self._table_parent_context(text, title)
        else:
            context, data_rows = labeled
        if not data_rows:
            return [_ChildUnit(context, is_table=True)]
        return [_ChildUnit(self._table_parent_text(context, [row]), is_table=True) for row in data_rows]

    @staticmethod
    def _labeled_table_context(text: str) -> tuple[str, list[str]] | None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or not lines[0].startswith("表格："):
            return None
        context: list[str] = []
        payloads: list[str] = []
        for line in lines:
            if line.startswith(("数据行：", "详情：")):
                payloads.append(line)
            else:
                context.append(line)
        return "\n".join(context), payloads

    def _split_unit(self, unit: _ChildUnit) -> list[_ChildUnit]:
        if self._tokens(unit.text) <= self._child_max:
            return [unit]
        if unit.is_table:
            labeled = self._labeled_table_context(unit.text)
            if labeled is not None:
                context, rows = labeled
                if rows:
                    fragments: list[_ChildUnit] = []
                    for row in rows:
                        if row.startswith("数据行："):
                            texts = self._split_table_row_texts(
                                context, row.removeprefix("数据行：").strip(), self._child_max
                            )
                        else:
                            texts = self._split_table_payload_texts(
                                context, row.removeprefix("详情：").strip(), self._child_max
                            )
                        fragments.extend(_ChildUnit(text, is_table=True) for text in texts)
                    return fragments
                return [
                    _ChildUnit(text, is_table=True)
                    for text in self._split_table_context_texts(context, self._child_max)
                ]
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
        boundary = min(boundary, end)
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

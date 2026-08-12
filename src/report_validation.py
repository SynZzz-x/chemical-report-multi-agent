"""Deterministic validation and asset extraction for generated report sections."""

from __future__ import annotations

import re
from typing import Any


_FENCED_CODE_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
_MERMAID_RE = re.compile(r"```\s*mermaid\s*\n.*?```", re.IGNORECASE | re.DOTALL)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")
_EVIDENCE_MARKER_RE = re.compile(r"\[(E\d+)\]", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._/+%-][A-Za-z0-9]+)*")
_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


def remove_mermaid_blocks(text: str) -> str:
    """Remove Mermaid placeholders so only formal figure assets represent figures."""

    return re.sub(r"\n{3,}", "\n\n", _MERMAID_RE.sub("", str(text or ""))).strip()


def count_report_length(text: str) -> int:
    """Count Chinese characters and Latin word tokens after removing markup noise."""

    cleaned = _FENCED_CODE_RE.sub(" ", str(text or ""))
    cleaned = _MARKDOWN_IMAGE_RE.sub(" ", cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _EVIDENCE_MARKER_RE.sub(" ", cleaned)
    cjk_count = len(_CJK_RE.findall(cleaned))
    latin_only = _CJK_RE.sub(" ", cleaned)
    return cjk_count + len(_LATIN_TOKEN_RE.findall(latin_only))


def parse_length_target(description: str) -> dict[str, int | None] | None:
    """Extract the first explicit Chinese report-length constraint."""

    text = str(description or "").replace(",", "").replace("，", "")
    range_match = re.search(
        r"(\d+)\s*(?:-|–|—|~|～|至|到)\s*(\d+)\s*(?:个)?字",
        text,
    )
    if range_match:
        lower, upper = (int(range_match.group(1)), int(range_match.group(2)))
        return {"min": min(lower, upper), "max": max(lower, upper)}

    minimum_match = re.search(r"(?:不少于|至少)\s*(\d+)\s*(?:个)?字", text)
    if minimum_match:
        return {"min": int(minimum_match.group(1)), "max": None}

    maximum_match = re.search(r"(?:不超过|至多|最多)\s*(\d+)\s*(?:个)?字", text)
    if maximum_match:
        return {"min": None, "max": int(maximum_match.group(1))}

    approximate_match = re.search(
        r"(?:约\s*(\d+)\s*(?:个)?字|(\d+)\s*(?:个)?字\s*左右)",
        text,
    )
    if approximate_match:
        target = int(approximate_match.group(1) or approximate_match.group(2))
        tolerance = max(1, round(target * 0.1))
        return {"min": target - tolerance, "max": target + tolerance}
    return None


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def _is_table_separator(line: str, expected_columns: int) -> bool:
    cells = _split_table_row(line)
    return len(cells) == expected_columns and all(
        _TABLE_SEPARATOR_CELL_RE.fullmatch(cell.replace(" ", ""))
        for cell in cells
    )


def extract_markdown_tables(text: str) -> list[dict[str, Any]]:
    """Parse standard pipe tables into stable structured table assets."""

    lines = str(text or "").splitlines()
    tables: list[dict[str, Any]] = []
    index = 0
    table_number = 1
    while index + 1 < len(lines):
        header_line = lines[index]
        headers = _split_table_row(header_line)
        if "|" not in header_line or len(headers) < 2 or not _is_table_separator(
            lines[index + 1], len(headers)
        ):
            index += 1
            continue

        rows: list[list[str]] = []
        cursor = index + 2
        while cursor < len(lines) and "|" in lines[cursor]:
            row = _split_table_row(lines[cursor])
            if len(row) != len(headers):
                break
            rows.append(row)
            cursor += 1

        if rows:
            evidence_refs = sorted(
                {
                    match.group(1).upper()
                    for row in rows
                    for cell in row
                    for match in _EVIDENCE_MARKER_RE.finditer(cell)
                },
                key=lambda value: int(value[1:]),
            )
            tables.append(
                {
                    "type": "markdown",
                    "title": f"正文表格 {table_number}",
                    "headers": headers,
                    "rows": rows,
                    "data": [headers, *rows],
                    "description": "由 Worker 正文中的 Markdown 表格确定性转换。",
                    "source": "worker_markdown",
                    "evidence_refs": evidence_refs,
                }
            )
            table_number += 1
        index = max(cursor, index + 2)
    return tables

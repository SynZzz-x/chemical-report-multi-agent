"""Read-only helpers for previewing admitted Markdown report artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def first_markdown_path(paths: Iterable[Path]) -> Path | None:
    """Return the first admitted Markdown artifact without changing its order."""

    return next(
        (path for path in paths if path.suffix.lower() == ".md"),
        None,
    )


def read_markdown_preview(path: Path | None) -> str | None:
    """Read a non-empty UTF-8 Markdown artifact without breaking the UI."""

    if path is None:
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    return content if content.strip() else None

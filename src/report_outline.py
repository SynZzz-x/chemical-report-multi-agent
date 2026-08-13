"""Deterministic report-outline classification and task coverage validation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence


_ARABIC_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<section_id>\d+(?:\.\d+)*)(?:[.、]|\s)+(?P<title>.+?)\s*$"
)
_CHINESE_NUMBER = "一二三四五六七八九十百零〇两"
_CHINESE_TOP_HEADING = re.compile(
    rf"^\s*(?P<section_id>[{_CHINESE_NUMBER}]+)[、.．]\s*(?P<title>.+?)\s*$"
)
_PARENTHESIZED_HEADING = re.compile(
    rf"^\s*[（(](?P<section_id>[{_CHINESE_NUMBER}]+|\d+)[）)]\s*(?P<title>.+?)\s*$"
)
_CHAPTER_HEADING = re.compile(
    rf"^\s*第(?P<section_id>[{_CHINESE_NUMBER}]+|\d+)(?P<unit>章|节|篇|部)\s*(?P<title>.+?)\s*$"
)
_SYSTEM_SECTION_TITLES = {
    "目录",
    "参考文献",
    "参考文献与知识库引用",
    "知识库引用",
    "证据来源",
    "引用来源",
    "证据来源附录",
}
_MERGEABLE_SINGLETON_RUN = 5


@dataclass(frozen=True)
class OutlineSection:
    """One original outline entry with its deterministic execution role."""

    raw: str
    section_id: str | None
    title: str
    level: int | None
    kind: str
    ordinal: int


def _split_heading(value: str) -> tuple[str | None, str, int | None]:
    match = _ARABIC_NUMBERED_HEADING.match(value)
    if match is not None:
        section_id = match.group("section_id")
        return section_id, match.group("title").strip(), section_id.count(".") + 1

    match = _CHAPTER_HEADING.match(value)
    if match is not None:
        unit = match.group("unit")
        level = 2 if unit == "节" else 1
        return match.group("section_id"), match.group("title").strip(), level

    match = _CHINESE_TOP_HEADING.match(value)
    if match is not None:
        return match.group("section_id"), match.group("title").strip(), 1

    match = _PARENTHESIZED_HEADING.match(value)
    if match is not None:
        return match.group("section_id"), match.group("title").strip(), 2

    return None, value.strip(), None


def _is_abstract(title: str) -> bool:
    normalized = re.sub(r"[\s:：_-]+", "", title).casefold()
    return normalized.startswith(("摘要", "abstract")) or normalized.endswith(
        ("摘要", "abstract")
    )


def _is_system_generated(title: str) -> bool:
    normalized = re.sub(r"[\s:：_-]+", "", title).casefold()
    if _is_abstract(normalized):
        return True
    return normalized in {
        re.sub(r"[\s:：_-]+", "", marker).casefold()
        for marker in _SYSTEM_SECTION_TITLES
    }


def classify_outline(sections: Sequence[Any] | None) -> list[OutlineSection]:
    """Recover parent/leaf roles from the current flat numbered outline."""

    parsed: list[tuple[str, str | None, str, int | None]] = []
    for section in sections or []:
        raw = str(section or "").strip()
        if not raw:
            continue
        section_id, title, level = _split_heading(raw)
        parsed.append((raw, section_id, title, level))

    outline: list[OutlineSection] = []
    for ordinal, (raw, section_id, title, level) in enumerate(parsed):
        next_level = parsed[ordinal + 1][3] if ordinal + 1 < len(parsed) else None
        if _is_system_generated(title):
            kind = "system_generated"
        elif level is not None and next_level is not None and next_level > level:
            kind = "container"
        else:
            kind = "content"
        outline.append(
            OutlineSection(
                raw=raw,
                section_id=section_id,
                title=title,
                level=level,
                kind=kind,
                ordinal=ordinal,
            )
        )
    return outline


def content_section_names(sections: Sequence[Any] | None) -> list[str]:
    return [item.raw for item in classify_outline(sections) if item.kind == "content"]


def planner_outline(sections: Sequence[Any] | None) -> list[dict[str, str]]:
    """Return the model-facing outline without exposing implementation details."""

    return [
        {"section": item.raw, "kind": item.kind}
        for item in classify_outline(sections)
    ]


def content_container_paths(
    sections: Sequence[Any] | None,
) -> dict[str, tuple[str, ...]]:
    """Map content sections to their ordered ancestor container headings."""

    paths: dict[str, tuple[str, ...]] = {}
    stack: list[OutlineSection] = []
    for item in classify_outline(sections):
        if item.level is None:
            stack.clear()
        else:
            while stack and (
                stack[-1].level is None or stack[-1].level >= item.level
            ):
                stack.pop()
        if item.kind == "content" and stack:
            paths[item.raw] = tuple(container.raw for container in stack)
        if item.kind == "container":
            stack.append(item)
    return paths


def content_parent_map(sections: Sequence[Any] | None) -> dict[str, str]:
    """Map content sections to their nearest ancestor container heading."""

    return {
        section: path[-1]
        for section, path in content_container_paths(sections).items()
        if path
    }


def section_markdown_level(section: str) -> int:
    """Return a report-safe Markdown level for an original outline heading."""

    _, _, outline_level = _split_heading(str(section or "").strip())
    return min((outline_level or 1) + 1, 6)


def validate_task_coverage(
    tasks: Sequence[Mapping[str, Any]],
    sections: Sequence[Any] | None,
) -> None:
    """Require every content section exactly once, in original outline order."""

    outline = classify_outline(sections)
    if not outline:
        return

    by_name = {item.raw: item for item in outline}
    parent_by_section = content_parent_map(sections)
    expected = [item.raw for item in outline if item.kind == "content"]
    actual: list[str] = []
    for task in tasks:
        covered = task.get("covers_sections")
        if not isinstance(covered, list) or not covered:
            raise ValueError(
                f"task {task.get('task_id')} must cover at least one content section"
            )
        for name in covered:
            item = by_name.get(name)
            if item is None:
                raise ValueError(f"covers_sections contains unknown section: {name}")
            if item.kind == "container":
                raise ValueError(f"covers_sections cannot include container section: {name}")
            if item.kind == "system_generated":
                raise ValueError(
                    f"covers_sections cannot include system-generated section: {name}"
                )
            actual.append(name)
        covered_parents = {
            parent_by_section.get(name) for name in covered
        }
        if len(covered_parents) > 1:
            raise ValueError(
                f"task {task.get('task_id')} spans multiple containers"
            )

    if len(actual) != len(set(actual)):
        raise ValueError("each content section must be covered exactly once")
    missing = [name for name in expected if name not in actual]
    if missing:
        raise ValueError(f"content section coverage is missing: {missing}")
    if actual != expected:
        raise ValueError(
            "content sections must follow outline order: "
            f"expected={expected}, actual={actual}"
        )

    mergeable_run = 0
    previous_key: tuple[Any, ...] | None = None
    for task in tasks:
        covered = task.get("covers_sections") or []
        if len(covered) != 1:
            mergeable_run = 0
            previous_key = None
            continue
        strategy_key = (
            parent_by_section.get(covered[0]),
            task.get("task_type"),
            task.get("use_rag"),
            task.get("use_web"),
            tuple(task.get("use_resources") or []),
            task.get("generate_figure"),
            task.get("generate_table"),
            str((task.get("visualization") or {}).get("kind") or ""),
        )
        mergeable_run = mergeable_run + 1 if strategy_key == previous_key else 1
        previous_key = strategy_key
        if mergeable_run >= _MERGEABLE_SINGLETON_RUN:
            raise ValueError(
                "execution plan is over-granular; merge adjacent content sections "
                "with compatible evidence and tool strategies"
            )

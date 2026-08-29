from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from .models import EvidenceBundle, EvidenceRecord
from .identity import canonical_citation_identity


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _dedupe_key(record: EvidenceRecord) -> str:
    return canonical_citation_identity(record.model_dump(mode="json"))


def _reindex(records: Iterable[EvidenceRecord]) -> tuple[EvidenceRecord, ...]:
    unique: list[EvidenceRecord] = []
    seen: set[str] = set()
    for record in records:
        key = _dedupe_key(record)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record.model_copy(update={"evidence_id": f"E{len(unique) + 1}"}))
    return tuple(unique)


def normalize_rag_tool_calls(tool_calls: Iterable[Mapping[str, Any]]) -> EvidenceBundle:
    records: list[EvidenceRecord] = []
    warnings: list[str] = []
    for call in tool_calls:
        if call.get("tool") != "chemical_knowledge_base_tool" or not call.get("success"):
            continue
        result = _as_mapping(call.get("full_result"))
        if not result:
            result = _as_mapping(call.get("result"))
        raw_evidence = result.get("evidence")
        if not isinstance(raw_evidence, list):
            raw_data = _as_mapping(result.get("raw_data"))
            raw_evidence = raw_data.get("results", [])
        query = str(_as_mapping(call.get("parameters")).get("query", ""))
        for item in raw_evidence if isinstance(raw_evidence, list) else []:
            if not isinstance(item, Mapping):
                continue
            text = str(item.get("content") or item.get("supporting_text") or "").strip()
            file_path = str(item.get("source") or item.get("file_path") or "").strip()
            if not text or not file_path:
                warnings.append("知识库返回了缺少正文或来源路径的记录，已忽略")
                continue
            records.append(
                EvidenceRecord(
                    evidence_id="pending",
                    source_type="rag",
                    title=str(item.get("title") or item.get("source") or "知识库文档"),
                    supporting_text=text,
                    locator=str(item.get("section_path") or item.get("locator") or ""),
                    file_path=file_path,
                    chunk_ids=tuple(str(value) for value in item.get("chunk_ids", []) or ()),
                    pages=dict(item.get("pages") or {}),
                    retrieval_query=query,
                )
            )
        if result and result.get("has_evidence") is False:
            warnings.append(f"知识库查询未返回证据：{query}" if query else "知识库查询未返回证据")
    return EvidenceBundle(records=_reindex(records), warnings=tuple(dict.fromkeys(warnings)))


def merge_evidence_bundles(*bundles: EvidenceBundle) -> EvidenceBundle:
    records = (record for bundle in bundles for record in bundle.records)
    warnings = tuple(dict.fromkeys(warning for bundle in bundles for warning in bundle.warnings))
    return EvidenceBundle(records=_reindex(records), warnings=warnings)


def citation_dicts(bundle: EvidenceBundle) -> list[dict[str, Any]]:
    return [record.model_dump(mode="json") for record in bundle.records]

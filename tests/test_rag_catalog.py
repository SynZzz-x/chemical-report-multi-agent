from __future__ import annotations

from dataclasses import replace
import sqlite3

from src.rag.bm25_store import BM25Store, INDEX_FINGERPRINT_KEYS
from src.rag.catalog import build_catalog_entry, load_active_catalog
from src.rag.models import ChildChunk, ParentChunk, SourceDocument, StructuralBlock
from src.rag.service import ChemicalRAGService
from src.nodes.planner import _normalize_knowledge_catalog


class _Tokenizer:
    def bm25_terms(self, text):
        return str(text).split()


def _document(*, version_id="version-1", content_hash="hash-1"):
    return SourceDocument(
        doc_id="doc-1",
        content_hash=content_hash,
        version_id=version_id,
        title="聚乙烯生产技术手册",
        doc_type="technical_manual",
        source="/knowledge/聚乙烯生产技术手册.pdf",
        blocks=(
            StructuralBlock(
                text="反应温度和催化剂体系会影响熔融指数。",
                block_type="paragraph",
                section_path="工艺参数 > 催化剂体系",
                page_start=1,
                page_end=1,
                clause_no=None,
            ),
            StructuralBlock(
                text="熔融指数是重要质量指标。",
                block_type="paragraph",
                section_path="质量指标 > 熔融指数",
                page_start=2,
                page_end=2,
                clause_no=None,
            ),
        ),
        metadata={"extension": "pdf", "block_count": 2},
    )


def _records(document):
    parent = ParentChunk(
        section_id=f"section-{document.version_id}",
        parent_id=f"parent-{document.version_id}",
        version_id=document.version_id,
        content="父级内容",
        metadata={"title": document.title, "section_path": "工艺参数"},
    )
    chunk = ChildChunk(
        chunk_id=f"chunk-{document.version_id}",
        parent_id=parent.parent_id,
        version_id=document.version_id,
        content="反应温度影响熔融指数。",
        embedding_text="反应温度影响熔融指数。",
        ordinal=0,
        metadata={
            "doc_id": document.doc_id,
            "doc_type": document.doc_type,
            "title": document.title,
            "section_path": "工艺参数",
        },
    )
    return [parent], [chunk]


def _store(tmp_path):
    fingerprint = {key: "test" for key in INDEX_FINGERPRINT_KEYS}
    return BM25Store.open(
        tmp_path,
        _Tokenizer(),
        fingerprint=fingerprint,
    )


def _activate(store, document):
    parents, chunks = _records(document)
    store.begin_version(
        document,
        parents,
        chunks,
        build_catalog_entry(document, chunks),
    )
    store.activate_version(document.doc_id, document.version_id)


def test_catalog_builder_creates_short_structured_file_metadata():
    document = _document()
    _, chunks = _records(document)
    entry = build_catalog_entry(document, chunks)

    assert entry.resource_id == "doc-1"
    assert entry.file_name == "聚乙烯生产技术手册.pdf"
    assert entry.file_type == "pdf"
    assert entry.sha256 == "hash-1"
    assert entry.indexed is True
    assert entry.content_type == "technical_document"
    assert entry.has_structured_data is False
    assert entry.supports == ("rag", "citation", "qualitative_analysis")
    assert "催化剂体系" in entry.topics
    assert "熔融指数" in entry.topics
    assert 1 <= len(entry.summary.split("。")) <= 3
    assert len(entry.summary) <= 240
    assert entry.capabilities["parameter_mentions"] == ["反应温度", "催化剂"]
    assert entry.capabilities["metric_mentions"] == ["熔融指数"]
    assert entry.capabilities["causal_evidence"] == "detected"
    assert entry.capabilities["numeric_ranges"] == "not_detected"
    assert entry.capabilities["control_limits"] == "not_detected"
    assert entry.coverage_evidence["causal_evidence"][0]["chunk_id"] == "chunk-version-1"


def test_catalog_capability_status_distinguishes_not_detected_from_unknown():
    document = _document()
    _, chunks = _records(document)

    analyzed = build_catalog_entry(document, chunks)
    unknown = build_catalog_entry(document, [])

    assert analyzed.capabilities["numeric_ranges"] == "not_detected"
    assert unknown.capabilities["numeric_ranges"] == "unknown"
    assert unknown.capabilities["causal_evidence"] == "unknown"


def test_catalog_schema_migrates_existing_v1_table_in_place():
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            active_version_id TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE document_versions (
            version_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES documents(doc_id),
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE resource_catalog (
            version_id TEXT PRIMARY KEY REFERENCES document_versions(version_id),
            file_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            topics_json TEXT NOT NULL,
            content_type TEXT NOT NULL,
            has_structured_data INTEGER NOT NULL,
            supports_json TEXT NOT NULL,
            catalog_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )

    BM25Store._create_schema(connection)

    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(resource_catalog)")
    }
    assert "capabilities_json" in columns
    assert "coverage_evidence_json" in columns


def test_catalog_does_not_advertise_unimplemented_excel_chart_capability():
    document = replace(
        _document(),
        source="/knowledge/PE_history.xlsx",
        metadata={"extension": "xlsx", "block_count": 2},
    )

    entry = build_catalog_entry(document)

    assert entry.has_structured_data is True
    assert "structured_data" in entry.supports
    assert "chart" not in entry.supports


def test_catalog_is_persisted_and_only_active_version_is_returned(tmp_path):
    store = _store(tmp_path)
    first = _document()
    _activate(store, first)

    second = replace(
        first,
        content_hash="hash-2",
        version_id="version-2",
        blocks=(
            replace(first.blocks[0], section_path="安全控制 > 温度联锁"),
        ),
    )
    _activate(store, second)

    catalog = load_active_catalog(tmp_path)

    assert len(catalog) == 1
    assert catalog[0]["version_id"] == "version-2"
    assert catalog[0]["sha256"] == "hash-2"
    assert catalog[0]["indexed"] is True
    assert "温度联锁" in catalog[0]["topics"]
    assert catalog[0]["capabilities"]["causal_evidence"] in {
        "detected",
        "not_detected",
        "unknown",
    }
    assert "coverage_evidence" not in catalog[0]


def test_planner_catalog_receives_aggregate_capabilities_but_not_provenance():
    normalized = _normalize_knowledge_catalog(
        [
            {
                "resource_id": "doc-1",
                "file_name": "manual.pdf",
                "capabilities": {
                    "causal_evidence": "detected",
                    "numeric_ranges": "not_detected",
                },
                "coverage_evidence": {
                    "causal_evidence": [{"chunk_id": "chunk-1"}]
                },
                "path": "/secret/manual.pdf",
            }
        ]
    )

    assert normalized[0]["capabilities"]["causal_evidence"] == "detected"
    assert "coverage_evidence" not in normalized[0]
    assert "path" not in normalized[0]


def test_staged_catalog_is_hidden_and_missing_catalog_cannot_activate(tmp_path):
    store = _store(tmp_path)
    document = _document()
    parents, chunks = _records(document)
    store.begin_version(
        document,
        parents,
        chunks,
        build_catalog_entry(document, chunks),
    )

    assert load_active_catalog(tmp_path) == []

    with store._connection:
        store._connection.execute(
            "DELETE FROM resource_catalog WHERE version_id = ?",
            (document.version_id,),
        )

    try:
        store.activate_version(document.doc_id, document.version_id)
    except ValueError as exc:
        assert "catalog" in str(exc).lower()
    else:
        raise AssertionError("activation must fail when catalog metadata is missing")


def test_ready_version_can_be_reused_without_rebuilding_catalog(tmp_path):
    store = _store(tmp_path)
    document = _document()
    _activate(store, document)

    assert store.is_ready_version(document.version_id) is True
    assert store.is_ready_version("missing-version") is False


def test_ingest_skips_ready_hash_before_chunking_or_catalog_generation(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path)
    document = _document()
    _activate(store, document)

    class _FailIfCalled:
        def chunk(self, _document):
            raise AssertionError("ready versions must skip before chunking")

    service = object.__new__(ChemicalRAGService)
    service._bm25_store = store
    service._vector_store = object()
    service._chunker = _FailIfCalled()
    monkeypatch.setattr(
        "src.rag.service.ChemicalDocumentLoader.load",
        lambda _path: document,
    )
    monkeypatch.setattr(
        "src.rag.service.build_catalog_entry",
        lambda _document: (_ for _ in ()).throw(
            AssertionError("ready versions must reuse the persisted catalog")
        ),
    )

    outcome = service._ingest_one(document.source)

    assert outcome == {
        "path": document.source,
        "status": "skipped",
        "chunks": 0,
        "version_id": document.version_id,
    }


def test_ingest_backfills_catalog_for_existing_ready_index(tmp_path, monkeypatch):
    store = _store(tmp_path)
    document = _document()
    _activate(store, document)
    with store._connection:
        store._connection.execute(
            "DELETE FROM resource_catalog WHERE version_id = ?",
            (document.version_id,),
        )

    class _FailIfCalled:
        def chunk(self, _document):
            raise AssertionError("catalog backfill must not rechunk a ready version")

    service = object.__new__(ChemicalRAGService)
    service._bm25_store = store
    service._vector_store = object()
    service._chunker = _FailIfCalled()
    monkeypatch.setattr(
        "src.rag.service.ChemicalDocumentLoader.load",
        lambda _path: document,
    )

    outcome = service._ingest_one(document.source)

    assert outcome["status"] == "skipped"
    assert load_active_catalog(tmp_path)[0]["resource_id"] == document.doc_id

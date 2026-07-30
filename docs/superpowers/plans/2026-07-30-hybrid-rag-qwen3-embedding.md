# Hybrid Chemical RAG with Qwen3 Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the DashScope dense-only chemical knowledge base with a structure-aware hybrid BM25 + Qwen3-Embedding-0.6B retrieval pipeline.

**Architecture:** The Agent calls an independent TEI embedding service over HTTP, indexes stable child chunks into both SQLite FTS5 and ChromaDB, and fuses lexical and dense ranks with RRF. Parent chunks remain in SQLite and are expanded after retrieval so the Worker receives coherent evidence while retaining its existing DeepSeek answer-generation responsibility.

**Tech Stack:** Python 3, requests, jieba, SQLite FTS5, ChromaDB, pandas, python-docx, LangChain document loaders, Hugging Face TEI.

## Global Constraints

- Work only on `codex/sqlite-checkpoint-store`.
- Use `Qwen/Qwen3-Embedding-0.6B` through an independent TEI service.
- Default vector dimension is exactly `1024`.
- Child target/max/overlap are exactly `450`/`700`/`70` model tokens.
- Parent target is exactly `1200` model tokens.
- Default BM25/dense/final top-k values are exactly `20`/`20`/`5`.
- RRF constant is exactly `60`; maximum returned context is `5000` tokens.
- `src/config.py` is the only application module that reads environment variables.
- The knowledge-base tool remains retrieval-only; Worker/DeepSeek generates answers.
- Existing DashScope vectors are incompatible and must not be silently reused.
- Do not add or run automated tests, per explicit user direction.
- Run only AST parsing, SQLite capability inspection, repository searches, and Git diff checks before delivery.

---

## File Map

- Create `src/rag/__init__.py`: public RAG interfaces.
- Create `src/rag/models.py`: immutable document, block, parent, child, and ranked-hit records.
- Create `src/rag/embeddings.py`: TEI health, tokenization, and embedding HTTP client.
- Create `src/rag/tokenizer.py`: cached model-token counts and jieba BM25 terms.
- Create `src/rag/chunking.py`: source loaders and structure-aware parent-child chunking.
- Create `src/rag/bm25_store.py`: SQLite schema, version manifest, FTS5 writes/search, context lookup.
- Create `src/rag/vector_store.py`: versioned Chroma collection and dense vector operations.
- Create `src/rag/hybrid_retriever.py`: RRF fusion, active filtering, context expansion, budget.
- Create `src/rag/service.py`: ingestion/query orchestration and compatibility result contract.
- Replace `src/nodes/worker/tools/ChemicalKnowledgeBase.py`: thin compatibility façade and manual CLI.
- Modify `src/config.py`: centralized TEI and RAG settings.
- Modify `src/nodes/worker/agent/graph.py`: Worker wiring and hybrid result formatting.
- Modify `.env.example`, `requirements.txt`, and `README.md`: deployment and configuration.

---

### Task 1: Central Configuration and TEI Boundary

**Files:**
- Modify: `src/config.py`
- Create: `src/rag/__init__.py`
- Create: `src/rag/models.py`
- Create: `src/rag/embeddings.py`
- Create: `src/rag/tokenizer.py`

**Interfaces:**
- Produces: immutable `RAGSettings`.
- Produces: `get_rag_settings() -> RAGSettings`.
- Produces: `TEIEmbeddings.embed_documents(list[str]) -> list[list[float]]`.
- Produces: `TEIEmbeddings.embed_query(str) -> list[float]`.
- Produces: `TEIEmbeddings.count_tokens(str) -> int`.
- Produces: `ChemicalTokenizer.model_tokens(str) -> int`.
- Produces: `ChemicalTokenizer.bm25_terms(str) -> list[str]`.

- [ ] **Step 1: Add immutable centralized settings**

Add this shape to `src/config.py` and populate it only in `get_app_config()`:

```python
@dataclass(frozen=True)
class RAGSettings:
    embedding_base_url: str
    embedding_api_key: str | None = field(repr=False)
    embedding_model: str
    embedding_dimension: int
    embedding_timeout_seconds: float
    child_target_tokens: int
    child_max_tokens: int
    child_overlap_tokens: int
    parent_target_tokens: int
    bm25_top_k: int
    dense_top_k: int
    final_top_k: int
    rrf_k: int
    max_context_tokens: int
    storage_root: Path
```

Validate positive integers, require target `<=` maximum, and derive
`storage_root` from `get_cache_root() / "rag"`.

- [ ] **Step 2: Define shared records**

Create frozen dataclasses for:

```python
StructuralBlock(text, block_type, section_path, page_start, page_end, clause_no)
SourceDocument(doc_id, version_id, title, doc_type, source, blocks, metadata)
ParentChunk(parent_id, version_id, content, metadata)
ChildChunk(chunk_id, parent_id, version_id, content, embedding_text, ordinal, metadata)
RankedHit(chunk_id, rank, score, backend)
```

All metadata fields use JSON-compatible dictionaries.

- [ ] **Step 3: Implement the TEI client**

Use `requests.Session.post()` against:

```text
GET  {base}/health
POST {base}/tokenize       {"inputs": text}
POST {base}/v1/embeddings  {"model": model, "input": texts}
```

Normalize the base URL by removing trailing `/` and a trailing `/v1`. Parse
both TEI token-list responses and `{tokens: [...]}` responses. Sort embedding
rows by their response `index`, verify one row per input, and reject any vector
whose length differs from `embedding_dimension`. Query embeddings prepend one
fixed Chinese chemical-retrieval instruction; document embeddings do not.

- [ ] **Step 4: Implement the tokenizer boundary**

Cache model token counts with a bounded dictionary. Initialize jieba once, add
chemical units and standard identifiers such as `MPa`, `kPa`, `mol/L`,
`GB/T`, `CAS`, and `Ziegler-Natta`, and return searchable tokens with whitespace
and punctuation removed. Escape FTS query syntax at the store boundary.

- [ ] **Step 5: Static check and commit**

Run:

```bash
python -m py_compile src/config.py src/rag/models.py src/rag/embeddings.py src/rag/tokenizer.py
git diff --check
```

Expected: no output from `git diff --check` and no AST errors.

Commit:

```bash
git add src/config.py src/rag
git commit -m "feat: add TEI embedding configuration"
```

---

### Task 2: Structure-Aware Parent-Child Chunking

**Files:**
- Create: `src/rag/chunking.py`

**Interfaces:**
- Consumes: `ChemicalTokenizer.model_tokens`.
- Produces: `ChemicalDocumentLoader.load(path: str) -> SourceDocument`.
- Produces: `StructureAwareChunker.chunk(document: SourceDocument) -> tuple[list[ParentChunk], list[ChildChunk]]`.

- [ ] **Step 1: Implement structured source loading**

Use:

- `PyPDFLoader` for ordered PDF pages;
- `python-docx` for headings, paragraphs, and table rows in `.docx`;
- `Docx2txtLoader` only as the `.doc` fallback;
- pandas for CSV/Excel schema summaries;
- encoding fallback `utf-8`, `gbk`, `gb2312`, `latin-1` for text files.

Create page-aware blocks, detect repeated PDF first/last lines appearing on at
least three pages, and remove them. CSV/Excel blocks contain sheet, row count,
column names, units when detectable, and date range when detectable; raw rows
are not emitted.

- [ ] **Step 2: Detect document structure**

Recognize headings, numbered Chinese/Arabic sections, standard clauses, patent
claims, list items, process steps, and simple pipe/tabular rows. Maintain a
heading stack as `section_path`; never merge standard clauses or patent claims
across their detected boundary.

- [ ] **Step 3: Build parents**

Group coherent blocks to a 1,200-token target. Use clause/claim/process-step
boundaries before token size. Generate:

```python
parent_id = sha256(f"{doc_id}\\0{version_id}\\0{structural_path}\\0{ordinal}")
```

Store title, source, document type, section path, clause number, and page range
in metadata.

- [ ] **Step 4: Build children**

Assemble sentence/block units to a 450-token target and 700-token hard maximum.
Carry up to 70 tokens of complete trailing units into the next child only when
remaining inside the same parent. Tables repeat title/header/unit context and
use row groups without ordinary overlap. Split a single oversized sentence by
binary-searching character boundaries against `model_tokens`.

Generate:

```python
chunk_id = sha256(f"{parent_id}\\0{ordinal}\\0{normalized_child_text}")
embedding_text = f"文档：{title}\\n章节：{section}\\n条款：{clause}\\n正文：{content}"
```

Add previous/next sibling IDs after all children for a parent are created.

- [ ] **Step 5: Static check and commit**

Run:

```bash
python -m py_compile src/rag/chunking.py
git diff --check
```

Commit:

```bash
git add src/rag/chunking.py
git commit -m "feat: add structure-aware RAG chunking"
```

---

### Task 3: SQLite FTS5 and Chroma Stores

**Files:**
- Create: `src/rag/bm25_store.py`
- Create: `src/rag/vector_store.py`

**Interfaces:**
- Consumes: `SourceDocument`, `ParentChunk`, `ChildChunk`, `RankedHit`.
- Produces: `BM25Store.open()`, `begin_version()`, `activate_version()`,
  `mark_failed()`, `search()`, `active_chunk_ids()`, `get_chunks()`,
  `get_parent()`, `get_siblings()`, `stats()`.
- Produces: `VectorStore.upsert()`, `search()`, `delete()`, `count()`.

- [ ] **Step 1: Create the SQLite schema**

Open `hybrid.sqlite` with foreign keys, WAL, and busy timeout. Create:

```sql
documents(doc_id PRIMARY KEY, source, active_version_id, updated_at)
document_versions(version_id PRIMARY KEY, doc_id, content_hash, status, created_at)
parents(parent_id PRIMARY KEY, version_id, content, metadata_json)
chunks(chunk_id PRIMARY KEY, parent_id, version_id, content, metadata_json, ordinal)
chunks_fts USING fts5(chunk_id UNINDEXED, bm25_text)
```

Fail with an actionable error if FTS5 table creation is unsupported.

- [ ] **Step 2: Implement staged version activation**

`begin_version()` inserts/updates a `building` version without changing the
active version. `activate_version()` performs parents, chunks, FTS rows,
`ready` status, and active-version switch in one transaction. Return the prior
version ID and its chunk IDs so Chroma cleanup occurs after activation.

- [ ] **Step 3: Implement BM25 retrieval and context lookup**

Build an OR query from quoted jieba terms. Join FTS rows through chunks,
versions, and documents so only `documents.active_version_id` is returned.
Order by SQLite `bm25(chunks_fts)` ascending and convert rows to ranked hits.
Apply `doc_type` filtering from stored metadata in Python after over-fetching.

- [ ] **Step 4: Implement direct Chroma operations**

Use `chromadb.PersistentClient` directly, with collection:

```text
chemical_documents_v2_qwen3_1024
```

Upsert explicit IDs, documents, compact metadata, and precomputed embeddings.
Query with a precomputed query vector and optional Chroma `doc_type` filter.
Convert cosine distance to `max(0.0, 1.0 - distance)`.

- [ ] **Step 5: Static check and commit**

Run:

```bash
python -m py_compile src/rag/bm25_store.py src/rag/vector_store.py
python -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts5(x)'); print('fts5-ok')"
git diff --check
```

Expected SQLite output: `fts5-ok`.

Commit:

```bash
git add src/rag/bm25_store.py src/rag/vector_store.py
git commit -m "feat: add hybrid RAG storage"
```

---

### Task 4: RRF Retrieval and Ingestion Service

**Files:**
- Create: `src/rag/hybrid_retriever.py`
- Create: `src/rag/service.py`
- Modify: `src/rag/__init__.py`

**Interfaces:**
- Consumes: both store interfaces, TEI embeddings, tokenizer, and chunker.
- Produces: `HybridRetriever.retrieve(question, top_k, doc_type_filter, similarity_threshold) -> dict`.
- Produces: `ChemicalRAGService.ingest(file_paths) -> dict`.
- Produces: `ChemicalRAGService.query(question, top_k, doc_type_filter, similarity_threshold) -> dict`.
- Produces: `ChemicalRAGService.stats() -> dict`.

- [ ] **Step 1: Implement RRF fusion**

For each backend rank:

```python
score[chunk_id] += 1.0 / (settings.rrf_k + rank)
```

Filter dense-only candidates below `similarity_threshold`; BM25 hits remain
eligible. Validate all candidate IDs through `BM25Store.active_chunk_ids()`,
then sort by descending RRF score with best backend rank as tie-breaker.

- [ ] **Step 2: Implement degraded retrieval**

Catch BM25 and dense failures independently:

- both available -> `retrieval_mode="hybrid"`;
- only BM25 -> `retrieval_mode="bm25_only"`;
- only dense -> `retrieval_mode="dense_only"`;
- neither -> structured error.

Warnings contain backend names and sanitized exception messages.

- [ ] **Step 3: Expand and budget context**

Merge selected children from the same parent. Return the parent when it fits the
remaining 5,000-token budget; otherwise return the hit plus immediate sibling
chunks. Use TEI token counts when available and a conservative character
estimate only during TEI-degraded query mode. Preserve title, source, document
type, section, pages, child IDs, ranks, and RRF score.

- [ ] **Step 4: Implement consistent ingestion**

For each file:

1. load and hash;
2. skip when the version is already active;
3. mark version building;
4. create chunks;
5. batch embeddings and Chroma upsert;
6. activate SQLite rows;
7. delete prior-version vectors;
8. on failure delete newly written vectors and mark version failed.

Process files independently so one failed file does not discard successful
files.

- [ ] **Step 5: Static check and commit**

Run:

```bash
python -m py_compile src/rag/hybrid_retriever.py src/rag/service.py src/rag/__init__.py
git diff --check
```

Commit:

```bash
git add src/rag
git commit -m "feat: add RRF chemical retrieval service"
```

---

### Task 5: Worker Integration and Operator Documentation

**Files:**
- Replace: `src/nodes/worker/tools/ChemicalKnowledgeBase.py`
- Modify: `src/nodes/worker/agent/graph.py`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Modify: `README.md`

**Interfaces:**
- Consumes: `ChemicalRAGService`.
- Preserves: `ChemicalKnowledgeBase.load_documents()`, `.query()`, `.get_stats()`.
- Preserves: Worker operation names `query`, `load`, and `stats`.

- [ ] **Step 1: Replace the legacy knowledge-base module**

Make `ChemicalKnowledgeBase` a compatibility façade:

```python
class ChemicalKnowledgeBase:
    def __init__(self, settings: RAGSettings | None = None): ...
    def load_documents(self, file_paths: list[str]) -> dict: ...
    def query(self, question: str, top_k: int = 5,
              doc_type_filter: str | None = None,
              similarity_threshold: float = 0.3) -> dict: ...
    def get_stats(self) -> dict: ...
```

Remove import-time dependency probing, DashScope, LangChain Chroma wrappers,
the old character splitter, and test-document generation.

- [ ] **Step 2: Update Worker wiring**

Remove DashScope fields from `WorkerConfig`. Derive `KNOWLEDGE_BASE_DIR` from
the centralized RAG storage root. Initialize the façade without an API key.
Format hybrid evidence using RRF rank and source metadata, and retain full raw
retrieval diagnostics for DeepSeek.

- [ ] **Step 3: Update dependencies and environment**

Remove `dashscope` and `langchain-text-splitters` when repository search
confirms no remaining imports. Add all direct loader dependencies used by the
new implementation if not already listed. Replace DashScope variables in
`.env.example` with the exact TEI/RAG variables from the design.

- [ ] **Step 4: Document server deployment and rebuild**

README must include:

- TEI Docker command for Qwen3-Embedding-0.6B;
- `/health` check;
- Agent environment variables;
- `cache/rag/hybrid.sqlite` and `cache/rag/chroma/`;
- requirement to re-ingest old DashScope collections;
- BM25-only/dense-only query degradation;
- retrieval-only Worker responsibility.

- [ ] **Step 5: Full static verification**

Run:

```bash
python -m compileall -q src
git diff --check
rg -n "DASHSCOPE|DashScopeEmbeddings|RecursiveCharacterTextSplitter|text-embedding-v1" src .env.example README.md requirements.txt
rg -n "os\\.environ|getenv\\(" src/rag src/nodes/worker/tools/ChemicalKnowledgeBase.py
git status --short
```

Expected: compile succeeds; diff check is silent; both repository searches
return no prohibited RAG configuration or direct environment reads.

- [ ] **Step 6: Review, commit, and push**

Review the complete diff for secrets, accidental answer generation, stale
collection reuse, and unrelated changes.

Commit:

```bash
git add src .env.example requirements.txt README.md
git commit -m "feat: add hybrid chemical RAG"
git push origin codex/sqlite-checkpoint-store
```

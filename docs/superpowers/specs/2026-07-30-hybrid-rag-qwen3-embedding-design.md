# Hybrid Chemical RAG with Qwen3 Embedding Design

## Context

The chemical knowledge base is already a retrieval-only Worker tool, but its
retrieval implementation has four limitations:

- embeddings come from DashScope `text-embedding-v1`;
- documents are split by a fixed 1,000-character window with 200-character
  overlap, without preserving clauses, headings, tables, or process steps;
- Chroma dense similarity is the only retrieval path;
- retrieved child text has no parent expansion, so relevant evidence can lack
  the surrounding conditions, units, or conclusions needed by the Worker.

This change replaces the embedding backend and retrieval pipeline while keeping
answer generation in the existing DeepSeek-powered Worker.

## Decisions

1. Run `Qwen/Qwen3-Embedding-0.6B` in an independent Hugging Face Text
   Embeddings Inference (TEI) service.
2. Keep ChromaDB for dense vectors.
3. Add SQLite FTS5 for BM25 lexical retrieval.
4. Fuse BM25 and dense rankings with Reciprocal Rank Fusion (RRF).
5. Replace character windows with logical sections, bounded context parents,
   and retrieval children.
6. Index child chunks and expand final hits only inside their context-parent
   boundary before returning evidence to the Worker.
7. Keep all runtime configuration in `src/config.py`.

## Goals

- Improve recall for chemical names, model numbers, standards, clause numbers,
  units, formulas, and uncommon process terms.
- Preserve enough local context to interpret retrieved values and conclusions.
- Support Chinese and multilingual chemical documents without running a large
  embedding model inside the Agent process.
- Allow retrieval to continue in BM25-only or dense-only mode if one index is
  temporarily unavailable.
- Make ingestion idempotent and prevent partially indexed document versions
  from becoming active.

## Non-goals

- Adding a reranker.
- Moving answer generation into the knowledge-base tool.
- Replacing the existing DeepSeek Worker.
- Embedding complete CSV or Excel datasets that should be analyzed by the
  existing structured-data tools.
- Supporting multiple local embedding models in the first implementation.
- Changing LangGraph checkpoint or Store persistence.
- Adding or running automated tests, per the user's explicit direction.

## System Architecture

```text
Document
  -> structure-aware parser
  -> logical sections
  -> bounded context parents
  -> child chunks
       |-> jieba chemical tokenization -> SQLite FTS5
       `-> TEI Qwen3-Embedding-0.6B -> ChromaDB

Question
  |-> jieba chemical tokenization -> BM25 top 40
  `-> Qwen query instruction -> TEI -> dense overfetch, keep top 40 active
        -> RRF
        -> active-version filtering and deduplication
        -> top 12 child hits
        -> at most 5 parent evidence groups
        -> parent/sibling context expansion
        -> evidence budget enforcement
        -> Worker/DeepSeek
```

The embedding service is a separate process. The Agent accesses it over HTTP
through a small LangChain `Embeddings` adapter and never loads model weights.

## TEI Deployment

The default GPU deployment uses the official TEI image:

```bash
docker run --gpus all -p 8080:80 \
  -v "$PWD/data:/data" \
  ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
  --model-id Qwen/Qwen3-Embedding-0.6B \
  --revision 66e95e324bebb9453d3b5be447c898dca1ba0eb0
```

The image tag may be changed to the TEI variant matching the server's GPU
architecture, or to `cpu-1.9` for a CPU-only deployment. The Agent requires:

- `/health` for readiness;
- `/tokenize` for model-accurate chunk sizing during ingestion;
- `/v1/embeddings` for OpenAI-compatible batch embeddings.

Retrieval from an already-built BM25 index remains available while TEI is down.
Ingestion requires TEI because model-accurate token boundaries and dense vectors
must be created together.

## Central Configuration

`src/config.py` remains the only application module that reads environment
variables. `AppConfig` will replace DashScope embedding fields with:

- `EMBEDDING_BASE_URL`, default `http://127.0.0.1:8080`;
- `EMBEDDING_API_KEY`, optional and hidden from dataclass representations, for
  deployments protected by a reverse proxy;
- `EMBEDDING_MODEL`, default `Qwen/Qwen3-Embedding-0.6B`;
- `EMBEDDING_MODEL_REVISION`, default
  `66e95e324bebb9453d3b5be447c898dca1ba0eb0`;
- `EMBEDDING_DIMENSION`, default `1024`;
- `EMBEDDING_TIMEOUT_SECONDS`, default `30`;
- `RAG_CHILD_TARGET_TOKENS`, default `450`;
- `RAG_CHILD_MAX_TOKENS`, default `700`;
- `RAG_CHILD_OVERLAP_TOKENS`, default `70`;
- `RAG_PARENT_TARGET_TOKENS`, default `1200`;
- `RAG_PARENT_MAX_TOKENS`, default `1600`;
- `RAG_BM25_TOP_K`, default `40`;
- `RAG_DENSE_TOP_K`, default `40`;
- `RAG_DENSE_OVERFETCH_FACTOR`, default `3`;
- `RAG_RRF_CHILD_TOP_K`, default `12`;
- `RAG_FINAL_TOP_K`, default `5`;
- `RAG_MAX_HITS_PER_PARENT`, default `2`;
- `RAG_RRF_K`, default `60`;
- `RAG_MAX_CONTEXT_TOKENS`, default `5000`.

The RAG storage root is derived from the existing `AGENT_CACHE_ROOT`:

```text
cache/rag/
  hybrid.sqlite
  chroma/
```

The TEI adapter appends `/health`, `/tokenize`, and `/v1/embeddings` to the
configured service root. No module mutates environment variables or defines a
second provider default. DashScope embedding configuration and its direct
dependency are removed from this retrieval path.

## Component Boundaries

The current monolithic `ChemicalKnowledgeBase.py` will become a thin
retrieval-only façade. Focused modules under `src/rag/` will own the pipeline:

```text
src/rag/
  embeddings.py          TEI HTTP client and LangChain adapter
  tokenizer.py           TEI token counts and jieba chemical tokenization
  chunking.py            structural parsing and parent-child chunk creation
  bm25_store.py          SQLite schema, FTS5 writes, and BM25 queries
  vector_store.py        Chroma collection and dense retrieval
  hybrid_retriever.py    RRF, active-version filtering, expansion, budgeting
  service.py             ingestion/query orchestration and result contract
```

`ChemicalKnowledgeBaseTool` keeps its current Worker-facing responsibility:
accept a question and optional document type filter, then return sourced
evidence. It does not create a chat model or draft an answer.

## Structure-Aware Chunking

### Canonical units

Loaders first produce ordered structural blocks instead of one flattened string.
Each block carries its source location and block type:

- heading;
- paragraph;
- clause or subclause;
- list item;
- table;
- process step;
- page boundary.

Repeated page headers, footers, watermarks, and identified template boilerplate
are removed before chunk construction. Empty blocks are discarded. Identical
semantic blocks at different structural positions are preserved.

### Logical sections and context parents

Logical sections preserve the source hierarchy and may be long. Each logical
section has a stable `section_id` and is divided into one or more context
parents. Context parents are the largest units that may be returned as
evidence; they are stored for expansion and are not primary retrieval units.

- General reports and papers: one heading subtree may form a logical section;
  context parents use a 1,200-token target and 1,600-token hard maximum.
- Standards and safety documents: one clause or subclause; clauses are never
  merged across clause boundaries.
- Process documents: one process step or unit-operation section.
- Patents: abstract, individual claims, description sections, and examples stay
  separate; an individual claim is not split unless it exceeds model limits.
- Tables: table title, headers, units, and a related row group stay together.

An oversized logical section produces multiple `parent_id` values that share
one `section_id`. A single structural unit that exceeds the parent maximum is
split only at safe sentence or table-row boundaries.

### Child chunks

Only children are written to FTS5 and Chroma.

- Target size: 450 model tokens.
- Hard maximum: 700 model tokens.
- Normal overlap: 70 tokens, aligned to sentence or block boundaries.
- Minimum useful size: 120 tokens, unless the content is a complete clause,
  claim, list, or table unit.
- General reports: assemble paragraphs, then sentences, until the target size.
- Standards and safety documents: assemble within one clause only; no overlap
  crosses into another clause.
- Process documents: assemble within one process step; neighboring steps are
  linked by metadata rather than copied into overlap.
- Patent claims: keep a claim intact when it fits below the hard maximum.
- Tables: repeat title, column headers, and units in each row-group child; do
  not add ordinary text overlap.

TEI `/tokenize` supplies token counts in batches. Chunk boundaries are therefore
based on the tokenizer used by the actual embedding model, not Python character
counts.

### CSV and Excel

The RAG index stores only dataset-level evidence:

- file and sheet name;
- column names;
- units;
- row count;
- date range when detectable;
- a short schema/description summary.

Raw numeric rows remain the responsibility of `CSVTool` or the corresponding
structured-data path. This avoids flooding the vector index and producing
unreliable numeric answers from text retrieval.

### Embedding text

The vector input includes lightweight structural context:

```text
文档：<title>
章节：<section path>
条款：<clause number, when present>
正文：<child content>
```

The original child content is stored separately and is what the Worker sees.
Queries use one fixed English Qwen retrieval instruction before the user's
question because Qwen recommends English instructions for multilingual use.
The instruction is applied only to query embeddings, ensuring consistent
indexing and query behavior.

## Identity and Metadata

Every indexed item has stable identifiers:

- `doc_id`: SHA-256 of normalized source identity;
- `content_hash`: SHA-256 of normalized document content;
- `version_id`: SHA-256 of `doc_id + content_hash`;
- `section_id`: SHA-256 of `version_id + structural path + structural ordinal`;
- `parent_id`: SHA-256 of `section_id + parent ordinal + source page/offset`;
- `chunk_id`: SHA-256 of `parent_id + child ordinal + normalized child text`.

Child metadata includes:

- document title, type, and source;
- section path and clause number;
- page start and end;
- parent ID, child index, and sibling IDs;
- content hash;
- embedding model and dimension;
- index schema version.

Stable IDs make repeated ingestion idempotent and let both indexes refer to the
same canonical child.

## SQLite and Chroma Storage

SQLite is the canonical manifest and lexical store:

```text
documents(doc_id, source, active_version_id, updated_at)
document_versions(version_id, doc_id, content_hash, status, created_at)
parents(parent_id, section_id, version_id, content, metadata_json)
chunks(chunk_id, parent_id, version_id, content, metadata_json, ordinal)
chunks_fts(title, section_path, clause_number, bm25_text, chunk_id UNINDEXED)
rag_index_meta(key, value)
```

`chunks_fts` is an FTS5 virtual table. Text is pre-tokenized by jieba with a
small chemical-domain dictionary so Chinese terms, formulas, model numbers, and
standard identifiers remain searchable. Standard identifiers such as
`GB 31570-2015`, `GB31570-2015`, and `GB_31570_2015` receive a common normalized
alias. The same tokenizer normalizes queries, and user text is passed as a bound
SQL parameter after safe FTS query construction. BM25 uses column weights
`5.0, 3.0, 4.0, 1.0, 0.0` for title, section, clause, body, and unindexed ID.
SQLite's lower-is-better BM25 value is used only for rank ordering because its
scale is not comparable with cosine similarity.

Chroma stores:

- `chunk_id` as the vector ID;
- the context-prefixed embedding text;
- compact filter metadata such as `doc_type`, `doc_id`, and `version_id`.

The collection is created explicitly with cosine distance; Chroma's default L2
distance is never used. Its name and `rag_index_meta` fingerprint include index
schema version, embedding model and revision, vector dimension, distance metric,
TEI version, document embedding-text format, query-instruction version,
chunking configuration hash, normalization version, and jieba dictionary
version. Changing an index-affecting value creates a new collection and requires
a rebuild; incompatible old vectors are never silently reused.

## Ingestion Consistency

Ingestion follows a staged document-version flow:

1. Parse, split, and calculate all stable IDs without writing either index.
2. Return the existing result if the same version is already active.
3. In one SQLite transaction, create the `building` version and write its
   parents, chunks, and FTS rows without changing `active_version_id`.
4. Batch-generate vectors and upsert them to the versioned Chroma collection.
5. In a second SQLite transaction, mark the version `ready`, atomically switch
   `documents.active_version_id`, and mark the previous version inactive.
6. Attempt to remove prior-version vectors after activation.

If Chroma writing fails, the new version is marked failed and its vector IDs are
removed. The previous active version remains queryable. Startup recovery scans
`building` and `failed` versions, removes their Chroma vectors, and either
deletes incomplete rows or retains compact failure metadata for diagnostics.
Dense candidates are accepted only when their `chunk_id` belongs to an active
SQLite version, so orphaned or stale vectors cannot reach the Worker.

## Hybrid Retrieval

For each query:

1. Apply optional metadata filters.
2. Retrieve the top 40 BM25 children after active filtering in SQL.
3. Request at least 120 dense candidates, validate them against the SQLite
   active manifest, and keep the top 40 active children. If fewer than 40
   remain, retry with bounded incremental overfetch.
4. If one backend fails, continue with the available ranking and attach a
   degradation warning to diagnostics.
5. Fuse rankings with:

   ```text
   RRF score(chunk) = sum(1 / (60 + rank))
   ```

6. Remove inactive, duplicate, and near-overlapping children.
7. Keep the top 12 fused children, with at most two selected hits per parent.
8. Merge hits from the same parent into at most five evidence groups.
   `parent_score` is the maximum child RRF score; each child retains its own
   RRF score, BM25 rank, and dense rank in a `matches` array.
9. Expand each evidence group:
   - return the complete parent when it fits the remaining budget;
   - otherwise return selected children plus immediate siblings only when they
     share the same `parent_id`; expansion never crosses a clause, claim,
     process step, table, or safety boundary.
10. Stop when the total evidence reaches approximately 5,000 model tokens.

RRF uses ranks rather than adding BM25 and cosine values, avoiding
provider-specific score normalization. Result metadata exposes parent score,
per-child BM25/dense ranks, child RRF scores, and retrieval paths.

## Result Contract

The knowledge-base query result preserves the current retrieval-only shape and
adds hybrid diagnostics:

```text
{
  success,
  question,
  retrieval_mode,        # hybrid | bm25_only | dense_only
  total_results,
  warnings,
  results: [
    {
      content,
      title,
      source,
      doc_type,
      section_path,
      pages,
      chunk_ids,
      parent_id,
      parent_score,
      matches: [
        {chunk_id, rrf_score, bm25_rank, dense_rank}
      ]
    }
  ]
}
```

The Worker consumes `content` and source metadata as evidence. It does not use
RRF score as factual confidence.

## Error Handling

- TEI unavailable at startup: the RAG service opens existing SQLite and Chroma
  indexes; queries use BM25-only mode if possible.
- TEI unavailable during ingestion: ingestion stops before activation and the
  previous version remains active.
- FTS5 unavailable: dense-only retrieval remains available and diagnostics
  identify the missing lexical path.
- Chroma unavailable: BM25-only retrieval remains available.
- Both retrieval paths unavailable: return a structured tool error; do not ask
  DeepSeek to invent knowledge-base evidence.
- Dimension mismatch: reject the collection and require an explicit rebuild.
- No candidates: return an empty successful result with source count zero.
- Candidate evidence is not calibrated relevance. Every non-empty result sets
  `evidence_assessment_required=true`; the Worker states that the knowledge base
  lacks sufficient support when evidence does not directly answer the question.
- Secrets and full embedding requests are not logged.

## Operator Workflow

1. Start the TEI service and wait for `/health`.
2. Configure the Agent's embedding URL and RAG settings in `.env`.
3. Start the Agent; SQLite and Chroma directories are created beneath
   `AGENT_CACHE_ROOT`.
4. Re-ingest the source documents. The existing DashScope collection is not
   reused because its vectors and chunk boundaries are incompatible.
5. Query through the existing Worker tool.

The application will expose a rebuild operation in the knowledge-base service
or its existing manual entry point. Rebuild creates the new versioned
collection before old data is retired.

## Files in Scope

- `src/config.py`
- `src/rag/__init__.py`
- `src/rag/embeddings.py`
- `src/rag/tokenizer.py`
- `src/rag/chunking.py`
- `src/rag/bm25_store.py`
- `src/rag/vector_store.py`
- `src/rag/hybrid_retriever.py`
- `src/rag/service.py`
- `src/nodes/worker/tools/ChemicalKnowledgeBase.py`
- `.env.example`
- `requirements.txt`
- `README.md`
- directly affected operator documentation

## Verification

No automated tests will be added or run, per user direction. Before commit and
push, implementation will be checked with:

- Python AST parsing of changed Python files;
- `git diff --check`;
- repository searches confirming embedding environment reads are confined to
  `src/config.py`;
- repository searches confirming DashScope embeddings and the old
  character-based splitter are no longer used;
- SQLite schema creation and FTS5 capability inspection without running the
  application test suite;
- final diff review for credentials, stale collection reuse, and accidental
  answer generation inside the RAG tool.

## Delivery

Implementation will be made on `codex/sqlite-checkpoint-store` and pushed to the
matching GitHub branch after the agreed static verification.

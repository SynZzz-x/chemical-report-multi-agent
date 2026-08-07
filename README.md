# Auditable Multi-Agent System for Data Analysis Reports

This project is a LangGraph-based multi-agent system that turns a user's natural-language research or data-analysis request into a verified report. It plans work, executes tool-backed tasks, supports manual or automatic verification, and exports final report artifacts.

## Highlights

- Stateful `Intake -> Planner -> TaskController -> Worker -> ArtifactCommit -> QualityReview -> DecisionPolicy -> Summarizer -> Exit` workflow.
- Human-in-the-loop plan confirmation and result verification.
- Worker tools for CSV analysis, chart generation, web extraction, and domain knowledge retrieval.
- Streamlit demo interface plus CLI debugger.
- SQLite-backed checkpoint and Store persistence with historical job recovery.
- Environment-based configuration with no hardcoded secrets.
- Deterministic pytest checks for core helpers and security expectations.

## Architecture

```text
User request / CSV files
        |
        v
Intake -> Planner -> TaskController -> Worker -> ArtifactCommit
                                      -> QualityReview -> DecisionPolicy
                                      -> TaskController / Summarizer -> Exit
```

See [docs/architecture.md](docs/architecture.md) for the full workflow and state model.

## Tech Stack

- Python, LangGraph, LangChain
- Streamlit for the demo UI
- pandas, matplotlib, python-docx, reportlab
- DeepSeek official API for chat generation
- Hybrid BM25/ChromaDB retrieval with TEI-hosted Qwen3 embeddings for chemical knowledge
- pytest for deterministic checks

## Quick Start

```bash
cd agent-master
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`, then export the variables:

```bash
set -a
source .env
set +a
```

Provider and hybrid-RAG configuration is centralized in `src/config.py`:

- `DEEPSEEK_API_KEY` is required for all Agent chat generation.
- `DEEPSEEK_BASE_URL` defaults to `https://api.deepseek.com`.
- `DEEPSEEK_MODEL` defaults to `deepseek-v4-flash`.
- `EMBEDDING_BASE_URL` defaults to `http://127.0.0.1:8080` and identifies the
  independently deployed TEI service.
- `EMBEDDING_API_KEY` is optional for a reverse-proxy-protected TEI endpoint.
- `EMBEDDING_MODEL`, `EMBEDDING_MODEL_REVISION`, `EMBEDDING_DIMENSION`, and
  `EMBEDDING_TIMEOUT_SECONDS` identify the compatible embedding index.
- `RAG_CHILD_*`, `RAG_PARENT_*`, `RAG_BM25_TOP_K`, `RAG_DENSE_TOP_K`,
  `RAG_DENSE_OVERFETCH_FACTOR`, `RAG_RRF_CHILD_TOP_K`, `RAG_FINAL_TOP_K`,
  `RAG_MAX_HITS_PER_PARENT`, `RAG_RRF_K`, and `RAG_MAX_CONTEXT_TOKENS` set
  the hybrid retrieval and evidence limits.
- `AGENT_CACHE_ROOT` defaults to `cache`; hybrid RAG data lives below it.
- `CONCEPT_GRAPH_*` controls evidence-grounded causal diagrams; every rendered
  relation carries one or more `[E编号]` citations.
- `EVIDENCE_RAG_MAX_QUERIES` caps duplicate-prone Worker retrieval, while
  `EVIDENCE_WEB_MAX_QUERIES` and `EVIDENCE_WEB_FALLBACK` control RAG-first
  public-web supplementation.

Concept diagrams require the Graphviz `dot` executable and a Chinese font on
the server. On Ubuntu/Debian, install both before starting the Agent:

```bash
sudo apt-get update
sudo apt-get install -y graphviz fonts-noto-cjk
dot -V
```

The Worker does not ingest documents. Use the RAG CLI `rebuild`/`ingest`
commands first; report execution only performs retrieval. Causal graph
artifacts (`.png`, `.svg`, and validated `.json`) are written to the current
job's `charts/` directory and embedded into Markdown, DOCX, and PDF reports.

## Hybrid Chemical RAG Operations

Deploy the Qwen3 embedding model independently with Hugging Face Text
Embeddings Inference (TEI). The model revision is intentionally pinned because
the model/revision, 1,024-dimensional cosine index, chunking settings, and
other index inputs form the collection fingerprint.

```bash
docker run --gpus all -p 8080:80 \
  -v "$PWD/data:/data" \
  ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
  --model-id Qwen/Qwen3-Embedding-0.6B \
  --revision 66e95e324bebb9453d3b5be447c898dca1ba0eb0
```

For CPU-only hosts, use the compatible `cpu-1.9` TEI image. Before starting
the Agent, wait for the service to be ready:

```bash
curl http://127.0.0.1:8080/health
```

Copy `.env.example` to `.env`, set `DEEPSEEK_*` and the `EMBEDDING_*`/`RAG_*`
values for the deployment, then start the Agent. The TEI endpoint must also
provide `/tokenize` and OpenAI-compatible `/v1/embeddings`.

Hybrid RAG persists its active-manifest data in `cache/rag/hybrid.sqlite` and
its dense collections in `cache/rag/chroma/` (or the corresponding
`AGENT_CACHE_ROOT`). Existing DashScope-built collections must be re-ingested:
their vectors and character chunk boundaries are incompatible and are never
silently reused. Chroma collection names include a deterministic fingerprint
generation, so incompatible vectors cannot be opened under the current name.
A changed index fingerprint requires a safe rebuild and re-ingestion.

Stop the Agent, then provide every source document to the rebuild command:

```bash
python -m src.rag.cli rebuild \
  /srv/knowledge/process-standard.pdf \
  /srv/knowledge/safety-manual.docx
```

The command builds and validates the replacement beneath a sibling staging
directory. It switches `cache/rag` only after every source has an activated or
already-complete version. The former active directory is moved, not deleted, to
`cache/rag-archive/<timestamp>-<generation>-<id>`; the JSON result prints the
exact `archive_path`. If staging fails, `activated` is `false`, the active index
is unchanged, and `staging_path` is retained for diagnosis. To roll back after a
successful switch, stop the Agent, move the new `cache/rag` aside, then move the
reported `archive_path` back to `cache/rag`.

At startup, the service cleans incomplete `building`/`failed` versions and
idempotently retries every `cleanup_pending` retirement. If SQLite lacks FTS5,
the ordinary manifest still opens so active filtering and parent expansion keep
dense-only retrieval operational. During queries, the service degrades to
BM25-only if TEI/dense retrieval is unavailable, or dense-only if the lexical
backend is unavailable; diagnostics identify the active retrieval mode. If
neither backend is available, the tool returns an error instead of generating
an answer.

The chemical knowledge-base Worker tool is retrieval-only. It returns parent
evidence, child-match diagnostics, and source metadata to DeepSeek. Retrieval
errors and zero-result queries carry explicit fail-closed instructions: the
Worker must state that support is unavailable or insufficient and must not fill
the gap from model general knowledge. The former `similarity_threshold`
parameter is deprecated because RRF scores are rank-only, not calibrated
similarities.

Run the Streamlit app:

```bash
streamlit run app.py
```

Run the CLI debugger:

```bash
python run.py --auto-verify
```

### Local persistence

LangGraph checkpoints and job history are stored under `cache/langgraph/`.
`AGENT_USER_ID` is a single-user local adapter whose default is `local-user`.
In a shared deployment, derive a stable identity per authenticated session
instead of setting one process-wide value. Keep `LANGGRAPH_STRICT_MSGPACK=true`
enabled.

Resume a CLI thread with:

```bash
python run.py --thread-id job_<id>
```

## Demo Workflow

1. Start Streamlit and compile the workflow from the sidebar.
2. Upload `examples/sample_energy_metrics.csv`.
3. Paste the prompt from `examples/demo_prompt.md`.
4. Review the generated plan and allow the worker to execute tasks.
5. Inspect verifier checkpoints and download the final report artifacts.

## Testing

```bash
python -m pytest -q
```

The tests avoid LLM and network calls. They cover request extraction, resource mapping, verifier routing, requirements encoding, and hardcoded-secret scanning.

### Server verification for the stabilized report pipeline

Fetch the implementation branch and run the complete regression suite before
starting Streamlit:

```bash
git fetch origin
git switch codex/report-pipeline-stabilization
git pull --ff-only
python -m pytest -q
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then perform this smoke test from the browser:

1. Create a new report job and explicitly allow or deny trusted public-web
   retrieval. This choice is immutable after the job is created.
2. Confirm a nine-task plan and observe the T1-T9 ledger, attempt count,
   active Artifact, and latest review status in the work area.
3. Request one local revision (for example, T3) and confirm that the new
   attempt creates a new T3 Artifact without rerunning T1 or T2.
4. Let all tasks reach `PASSED`; Summarizer must reject any incomplete ledger.
5. Check the per-format report status. Download buttons must be shown only for
   outputs marked `SUCCEEDED`; a failed PDF must not be advertised.
6. Stop Streamlit, start it again with the same command, restore the historical
   job from the sidebar, and confirm that task, Artifact, review, and report
   status are restored from SQLite without duplicate execution.

The regression cases that reproduce the original server failures can also be
run separately:

```bash
python -m pytest \
  tests/test_pipeline_e2e.py \
  tests/test_report_generation.py \
  tests/test_pdf_tables.py -q
```

## Repository Structure

```text
agent-master/
  app.py                         # Streamlit demo
  run.py                         # CLI debugger
  src/
    graph.py                     # LangGraph workflow
    state.py                     # Shared workflow state
    nodes/                       # Intake, Planner, Worker, Verifier, Summarizer
    prompts/                     # Agent prompts
    utils/                       # Report and path helpers
  examples/                      # Demo data and interview script
  tests/                         # Deterministic test suite
  docs/                          # Architecture notes
```

## Resume Bullets

- Built a LangGraph multi-agent workflow with planning, tool execution, verification, and summarization stages.
- Integrated CSV analysis, chart generation, web extraction, and report generation tools into a Worker agent.
- Added human-in-the-loop verification and replanning paths to make long-running agent workflows auditable.
- Hardened the project for portfolio release by removing hardcoded secrets, documenting reproducible setup, and adding pytest quality gates.

## Developer Workflow

See [docs/development-workflow.md](docs/development-workflow.md) for the branch, verification, commit, push, and pull-request workflow.

Use feature branches for team development. Before starting work, pull the latest development branch, create a feature branch, test locally, and open a pull request for review. For local portfolio work in this copy, keep generated `cache/`, `logs/`, and report outputs out of version control.

## Future Improvements

- Add richer report evaluation metrics.
- Add a small CI pipeline for tests and secret scanning.
- Add a hosted demo with mocked LLM responses.

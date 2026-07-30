# Auditable Multi-Agent System for Data Analysis Reports

This project is a LangGraph-based multi-agent system that turns a user's natural-language research or data-analysis request into a verified report. It plans work, executes tool-backed tasks, supports manual or automatic verification, and exports final report artifacts.

## Highlights

- Stateful `Intake -> Planner -> Worker -> Verifier -> Summarizer -> Exit` workflow.
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
Intake -> Planner -> Worker -> Verifier -> Summarizer -> Exit
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
silently reused. A changed index fingerprint likewise requires a rebuild and
re-ingestion; the index uses cosine distance and has no compatibility guarantee
across fingerprint changes.

At startup, the service cleans incomplete `building` and `failed` versions and
their dense vectors. During queries, it degrades to BM25-only if TEI/dense
retrieval is unavailable, or dense-only if the lexical backend is unavailable;
diagnostics identify the active retrieval mode. If neither backend is available,
the tool returns an error instead of generating an answer.

The chemical knowledge-base Worker tool is retrieval-only. It returns parent
evidence, child-match diagnostics, and source metadata to DeepSeek; DeepSeek
must state that the knowledge base lacks sufficient support when the evidence
does not directly answer the question.

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
pytest
```

The tests avoid LLM and network calls. They cover request extraction, resource mapping, verifier routing, requirements encoding, and hardcoded-secret scanning.

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

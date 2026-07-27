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
- ChromaDB with DashScope embeddings for chemical knowledge retrieval
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

Provider configuration is centralized in `src/config.py`:

- `DEEPSEEK_API_KEY` is required for all Agent chat generation.
- `DEEPSEEK_BASE_URL` defaults to `https://api.deepseek.com`.
- `DEEPSEEK_MODEL` defaults to `deepseek-v4-flash`.
- `DASHSCOPE_API_KEY` is required only when loading or querying the chemical
  knowledge base.
- `DASHSCOPE_EMBEDDING_MODEL` defaults to `text-embedding-v1`.

The chemical knowledge base is retrieval-only: it uses DashScope embeddings
and ChromaDB to return sourced evidence. The Worker agent then uses the shared
DeepSeek client to interpret that evidence and generate the task result.

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

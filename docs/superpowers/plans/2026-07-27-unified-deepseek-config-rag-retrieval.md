# Unified DeepSeek Configuration and Retrieval-Only RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize chat and embedding provider configuration, route all chat generation through DeepSeek's official API, and make the chemical knowledge base retrieval-only.

**Architecture:** `src/config.py` owns an immutable cached `AppConfig` loaded from environment variables. Every chat-model consumer receives DeepSeek settings from that object, while the knowledge base receives only DashScope embedding settings and returns retrieved evidence for the Worker agent to interpret.

**Tech Stack:** Python 3.10+, dataclasses, functools cache, LangChain `ChatOpenAI`, DashScope embeddings, ChromaDB, LangGraph.

## Global Constraints

- Work on branch `codex/sqlite-checkpoint-store`.
- Use `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, and `DEEPSEEK_MODEL` for all chat-model calls.
- Default DeepSeek base URL to `https://api.deepseek.com`.
- Default DeepSeek model to `deepseek-v4-flash`.
- Use `DASHSCOPE_API_KEY` and `DASHSCOPE_EMBEDDING_MODEL` only for knowledge-base embeddings.
- Default DashScope embedding model to `text-embedding-v1`.
- Do not fall back to `OPENAI_API_KEY`, `OPENAI_BASE_URL`, or `OPENAI_MODEL`.
- Do not let runtime LangGraph configuration replace provider credentials, base URL, or default model.
- Do not add or run automated tests, per the user's explicit direction.
- Validate only with AST parsing, repository searches, `git diff --check`, and final diff review.

---

### Task 1: Establish the application configuration boundary

**Files:**
- Modify: `src/config.py`
- Modify: `src/llm.py`
- Modify: `src/nodes/verifier.py`

**Interfaces:**
- Consumes: Environment variables loaded by the application process.
- Produces: `AppConfig`, `get_app_config() -> AppConfig`, and `get_llm_settings(configurable) -> dict[str, Any]`.

- [ ] **Step 1: Add immutable provider configuration**

Add a frozen dataclass and cached accessor in `src/config.py`:

```python
@dataclass(frozen=True)
class AppConfig:
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    dashscope_api_key: str | None
    dashscope_embedding_model: str


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    return AppConfig(
        deepseek_api_key=get_env("DEEPSEEK_API_KEY"),
        deepseek_base_url=get_env("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        or DEFAULT_DEEPSEEK_BASE_URL,
        deepseek_model=get_env("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        or DEFAULT_DEEPSEEK_MODEL,
        dashscope_api_key=get_env("DASHSCOPE_API_KEY"),
        dashscope_embedding_model=get_env(
            "DASHSCOPE_EMBEDDING_MODEL", DEFAULT_DASHSCOPE_EMBEDDING_MODEL
        )
        or DEFAULT_DASHSCOPE_EMBEDDING_MODEL,
    )
```

- [ ] **Step 2: Restrict LLM runtime overrides to request parameters**

Update `get_llm_settings()` so API key, base URL, and model always come from `get_app_config()`, while `max_tokens`, `temperature`, `top_p`, and penalties continue to use runtime values.

- [ ] **Step 3: Report the correct required key**

Change `src/llm.py` to raise `missing_key_message("DEEPSEEK_API_KEY")` when the centralized chat API key is absent.

- [ ] **Step 4: Remove the verifier's direct environment lookup**

Use `get_app_config().deepseek_api_key` in `src/nodes/verifier.py` to decide whether LLM verification is configured.

- [ ] **Step 5: Commit the configuration boundary**

```bash
git add src/config.py src/llm.py src/nodes/verifier.py
git commit -m "refactor: centralize DeepSeek configuration"
```

### Task 2: Route Worker and chart generation through shared DeepSeek settings

**Files:**
- Modify: `src/nodes/worker/agent/graph.py`
- Modify: `src/nodes/worker/tools/chart_generator.py`

**Interfaces:**
- Consumes: `get_app_config() -> AppConfig`.
- Produces: `WorkerConfig` values for DeepSeek chat and DashScope embedding construction.

- [ ] **Step 1: Populate WorkerConfig from AppConfig**

Replace class-definition-time environment reads with dataclass factories:

```python
API_KEY: str = field(default_factory=lambda: get_app_config().deepseek_api_key or "")
BASE_URL: str = field(default_factory=lambda: get_app_config().deepseek_base_url)
LLM_MODEL: str = field(default_factory=lambda: get_app_config().deepseek_model)
DASHSCOPE_API_KEY: str = field(
    default_factory=lambda: get_app_config().dashscope_api_key or ""
)
DASHSCOPE_EMBEDDING_MODEL: str = field(
    default_factory=lambda: get_app_config().dashscope_embedding_model
)
```

- [ ] **Step 2: Pass embedding settings to the knowledge base**

Construct `ChemicalKnowledgeBase` with `DASHSCOPE_API_KEY` and `DASHSCOPE_EMBEDDING_MODEL`, never the DeepSeek key.

- [ ] **Step 3: Make ChartGenerator consume AppConfig**

When explicit constructor values are absent, read `deepseek_api_key`, `deepseek_base_url`, and `deepseek_model` from `get_app_config()`. Raise an error naming `DEEPSEEK_API_KEY` if unavailable.

- [ ] **Step 4: Commit Worker and chart configuration**

```bash
git add src/nodes/worker/agent/graph.py src/nodes/worker/tools/chart_generator.py
git commit -m "refactor: share provider settings with worker tools"
```

### Task 3: Make chemical RAG retrieval-only

**Files:**
- Modify: `src/nodes/worker/tools/ChemicalKnowledgeBase.py`
- Modify: `src/nodes/worker/agent/graph.py`

**Interfaces:**
- Consumes: DashScope API key, embedding model, query, top-k limit, document-type filter, and similarity threshold.
- Produces: Retrieval results containing content, score, source, title, document type, metadata, count, and average score.

- [ ] **Step 1: Remove chat-model ownership from ChemicalKnowledgeBase**

Delete the `ChatOpenAI` dependency probe, `get_llm` import, environment mutation, internal LLM initialization, and `_generate_answer()`.

- [ ] **Step 2: Parameterize only the embedding model**

Accept `embedding_model: str = "text-embedding-v1"` in the constructor and pass it to `DashScopeEmbeddings`.

- [ ] **Step 3: Return retrieval records only**

Remove `generate_answer` from `query()` and return:

```python
{
    "question": question,
    "total_results": len(results),
    "average_score": average_score,
    "results": results,
    "timestamp": datetime.now().isoformat(),
}
```

- [ ] **Step 4: Simplify the Worker tool schema and output**

Remove `generate_answer` from `KnowledgeBaseArgs`. Always serialize retrieved evidence into the tool's `content`, keep structured records in `raw_data`, and describe the tool as retrieval rather than intelligent Q&A.

- [ ] **Step 5: Update the manual knowledge-base demo**

Read DashScope settings through `get_app_config()` and call `query()` without `generate_answer`; print retrieved passages and sources only.

- [ ] **Step 6: Commit retrieval-only RAG**

```bash
git add src/nodes/worker/tools/ChemicalKnowledgeBase.py src/nodes/worker/agent/graph.py
git commit -m "refactor: make chemical RAG retrieval only"
```

### Task 4: Document provider configuration and deliver

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Consumes: The final environment variable contract.
- Produces: Operator instructions for DeepSeek chat and DashScope embeddings.

- [ ] **Step 1: Replace legacy example variables**

Document:

```dotenv
DEEPSEEK_API_KEY=replace-with-your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DASHSCOPE_API_KEY=replace-with-your-dashscope-key
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v1
```

- [ ] **Step 2: Explain provider responsibilities in README**

State that DeepSeek handles all chat generation and DashScope is used only for knowledge-base embeddings.

- [ ] **Step 3: Perform static verification**

Run:

```bash
python -m compileall -q src
git diff --check
rg -n 'OPENAI_API_KEY|OPENAI_BASE_URL|OPENAI_MODEL|qwen-max|generate_answer' src .env.example
rg -n 'os\.environ\.get\("(DEEPSEEK|DASHSCOPE)' src
```

Expected: syntax compilation succeeds; diff check is clean; no legacy chat variable, hard-coded Qwen model, or RAG answer-generation references remain; provider environment reads occur only in `src/config.py` through `get_env()`.

- [ ] **Step 4: Review, commit, and push**

```bash
git add .env.example README.md src/config.py src/llm.py src/nodes/verifier.py \
  src/nodes/worker/agent/graph.py src/nodes/worker/tools/chart_generator.py \
  src/nodes/worker/tools/ChemicalKnowledgeBase.py \
  docs/superpowers/plans/2026-07-27-unified-deepseek-config-rag-retrieval.md
git commit -m "feat: unify DeepSeek configuration and RAG retrieval"
git push origin codex/sqlite-checkpoint-store
```

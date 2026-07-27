# Unified DeepSeek Configuration and Retrieval-Only RAG Design

## Context

The project currently has a partial configuration boundary:

- `src/config.py` defines shared defaults and `get_llm_settings()`.
- `src/llm.py` consumes those shared settings.
- Worker and chart-generation modules still read environment variables and
  define provider defaults independently.
- `ChemicalKnowledgeBase` uses DashScope embeddings, overwrites
  `OPENAI_API_KEY`, and invokes a hard-coded Qwen model to generate an answer
  inside the retrieval tool.

This creates conflicting defaults, mixed API credentials, and duplicate answer
generation between the knowledge-base tool and the Worker agent.

## Goals

1. Provide one immutable application configuration object for model and
   embedding settings.
2. Route all chat-model calls through the DeepSeek official OpenAI-compatible
   API.
3. Keep DashScope only as the embedding provider for the chemical knowledge
   base.
4. Make the chemical knowledge base a retrieval-only Worker tool.
5. Remove direct environment reads and provider defaults from model consumers.
6. Preserve LangGraph runtime overrides for sampling and token settings without
   allowing modules to invent their own provider defaults.

## Non-goals

- Replacing ChromaDB.
- Re-embedding or migrating an existing vector collection.
- Introducing a separate RAG agent or reranker.
- Supporting multiple chat-model providers in this change.
- Changing prompts, workflow routing, persistence, or report generation.
- Adding or running tests, per explicit user direction.

## Configuration Model

`src/config.py` will expose an immutable `AppConfig` dataclass and a cached
`get_app_config()` accessor.

The configuration contains two explicit provider sections:

### DeepSeek chat configuration

- API key: `DEEPSEEK_API_KEY`
- Base URL: `DEEPSEEK_BASE_URL`
- Default base URL: `https://api.deepseek.com`
- Model: `DEEPSEEK_MODEL`
- Default model: `deepseek-v4-flash`

`deepseek-v4-pro` remains selectable through `DEEPSEEK_MODEL`.

### DashScope embedding configuration

- API key: `DASHSCOPE_API_KEY`
- Model: `DASHSCOPE_EMBEDDING_MODEL`
- Default model: `text-embedding-v1`

The legacy chat variables `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and
`OPENAI_MODEL` will be removed from the documented configuration and from
application model lookup. There will be no compatibility fallback because that
would preserve ambiguous ownership of credentials.

## Access Rules

- Only `src/config.py` reads model-related environment variables.
- `src/llm.py` constructs `ChatOpenAI` from `AppConfig`.
- Worker configuration receives values from `AppConfig`; it does not read
  environment variables.
- Chart generation receives the shared DeepSeek settings.
- Chemical knowledge-base construction receives the DashScope embedding
  settings explicitly.
- No module writes model credentials into `os.environ`.

LangGraph configurable values may still override request-level settings such as
temperature, token limit, and penalties. Provider identity, API key, base URL,
and default model originate from `AppConfig`.

## RAG Responsibility

`ChemicalKnowledgeBase` becomes retrieval-only:

1. Load supported source documents.
2. Split documents into chunks.
3. Create DashScope embeddings.
4. Persist vectors in ChromaDB.
5. Retrieve the top matching chunks and return:
   - chunk content,
   - title,
   - source,
   - document type,
   - relevance/rank score,
   - retrieval metadata.

The following behavior will be removed:

- internal `ChatOpenAI` construction,
- hard-coded `qwen-max`,
- mutation of `OPENAI_API_KEY`,
- `_generate_answer()`,
- the `generate_answer` tool argument,
- knowledge-base-owned final answer generation.

`ChemicalKnowledgeBaseTool` returns formatted evidence plus structured raw
retrieval results. The Worker agent uses its normal DeepSeek client to interpret
that evidence and generate the task result. This keeps generation, tool use, and
verification in the existing Agent workflow.

## Data Flow

```text
User request
    -> Planner selects RAG when relevant
    -> Worker calls ChemicalKnowledgeBaseTool
    -> DashScope embedding + Chroma retrieval
    -> Tool returns evidence with sources
    -> Worker uses global DeepSeek client
    -> Verifier reviews the Worker result
    -> Summarizer produces the report
```

## Error Handling

- Missing `DEEPSEEK_API_KEY` produces the existing actionable missing-key
  message, updated to name the DeepSeek variable.
- Missing `DASHSCOPE_API_KEY` disables knowledge-base initialization with a
  provider-specific error; non-RAG tools remain usable.
- Invalid DeepSeek base URL or model errors are surfaced by the shared LLM
  construction path.
- Retrieval errors return structured tool errors and do not fall back to an
  unrelated chat provider.
- No secret value is logged or committed.

## Files in Scope

- `src/config.py`
- `src/llm.py`
- `src/nodes/worker/agent/graph.py`
- `src/nodes/worker/tools/chart_generator.py`
- `src/nodes/worker/tools/ChemicalKnowledgeBase.py`
- `.env.example`
- `README.md`
- Any directly affected operator documentation

## Verification

No automated tests will be added or run, per user direction. Completion will be
checked with:

- Python AST parsing of changed Python files.
- `git diff --check`.
- Repository search confirming model-related environment reads are confined to
  `src/config.py`.
- Repository search confirming no hard-coded DashScope chat URL, `qwen-max`,
  or mutation of `OPENAI_API_KEY` remains in application code.
- Repository search confirming the RAG tool no longer generates answers.
- Final diff review before commit and push.

## Delivery

Implementation commits will be made on
`codex/sqlite-checkpoint-store` and pushed to the matching GitHub branch after
static verification.

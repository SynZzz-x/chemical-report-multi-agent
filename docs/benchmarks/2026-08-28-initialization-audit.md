# Initialization noise audit

Date: 2026-08-29

Scope: Task 8 of the grounding/performance plan

Conclusion: **audit-only; no production or test code changed**

## Method and exit criterion

The audit reviewed every production match from the command below. The findings
table groups related call sites and explicitly identifies legacy code, existing
safe caches, and state-registry false positives rather than listing test fixture
matches individually.

```text
rg -n "makedirs|mkdir|register|registry|pkg_resources|jieba" src tests
```

It then inspected construction call sites and ran the dependency probes below
with the repository's parent virtual environment. No network or provider was
used.

```text
/Users/synzzz/Documents/work_space/agent/agent-master/.venv/bin/python \
  -W default -c "import jieba"

/Users/synzzz/Documents/work_space/agent/agent-master/.venv/bin/python - <<'PY'
import inspect, jieba, sys
print(sys.version.split()[0])
print(jieba.__version__)
print(inspect.getfile(jieba))
try:
    import pkg_resources
except Exception as exc:
    print(type(exc).__name__, str(exc))
PY
```

The exit criterion was deliberately strict: change production code only for an
object proven both immutable and process-global, and only after a failing test
demonstrated duplicate registration. Job/task objects, path state, RAG state,
mutable registries, and LLM clients were ineligible for caching.

## Findings

| Candidate | Exact source and frequency | Scope / mutability | Safety assessment |
| --- | --- | --- | --- |
| Cache, job, and upload directories | `src/config.py:96-112`, `src/utils/path_manager.py:91-135,146-157`, `src/nodes/summarizer_v2.py:763-782` | `get_cache_root()` checks/creates the configured root per call. `get_session_cache_dir()` creates a path derived from the current user, conversation, and job; `manage_session_files()` creates that session's `uploads` directory per resource-management call. Summarizer resolves the job path for the report and again for math images. These paths are job-scoped and environment/config dependent. | Retain `mkdir(..., exist_ok=True)`. Caching the result across jobs could route artifacts into another job; caching the creation side effect could also skip a directory removed during a long-running process. The repeated syscall is idempotent and safer than cross-job state. |
| Worker directories | `src/nodes/worker/agent/graph.py:157-177`, `1225-1241`, `1424-1429`, `3001-3007`, `3044-3052`, `3316-3345` | Every `WorkerConfig` construction ensures its configured chart/report/output/spider/log/knowledge directories. `AutonomousToolNode` and its config are built for each Worker tool-loop invocation. Chart relocation also ensures the current config's target directory. `WorkerAgent` checks directories when explicitly constructed. Config includes mutable/job-local output locations. | Do not cache `WorkerConfig`, `AutonomousToolNode`, `WorkerAgent`, their paths, or their LLM/tool objects. The visible repeated messages are logging noise, but eliminating them through a process singleton would violate job isolation. Creation at chart relocation remains a safe write-boundary check. |
| Tool registration | `src/nodes/worker/agent/graph.py:1266-1329` (including custom-tools directory creation at `1295-1297`) | Each `ToolManager` owns a fresh mutable `tool_classes` dict. Built-ins are filtered by that invocation's `ENABLED_TOOLS`; custom modules depend on `CUSTOM_TOOL_MODULES` and `TOOLS_DIR`. Registration therefore repeats once per `AutonomousToolNode`, but it is not duplicate mutation of a shared registry. The custom-tools directory is created only when configured modules exist and the configured directory is absent. Tool instances may own RAG services, caches, or clients. | No process-global registry. A shared dict would make one job's enabled/custom tools authoritative for another and could retain mutable tool state. There is no failing duplicate-registration behavior to fix; the config-specific directory check also must remain local. |
| RAG storage directories | `src/rag/bm25_store.py:109-123`, `src/rag/vector_store.py:39-46`, `src/rag/service.py:137-156` | Created when a `ChemicalRAGService`/store is opened and during rebuild activation. The storage root selects persistent SQLite/Chroma state; rebuild paths and archives are generation-specific mutable state. | Do not cache service/store construction or directory readiness. Re-checking existence is part of safe open/rebuild behavior, and the stores hold live connections and mutable indexes. |
| Persistence directory | `src/config.py:121-131`, `src/persistence.py:49-79` | `SQLitePersistence.open()` checks/creates the selected root per open, then creates live SQLite connections and applies file permissions. A caller may pass a different root. | Do not cache the path side effect or persistence object. Connections and stores are mutable resources with explicit `close()` lifetime. |
| Renderer/artifact directories | `src/utils/md_to_pdf.py:548-562,665-674`, `src/utils/md_to_docx.py:142,208,288,332,621`, `src/nodes/worker/tools/chart_generator.py:147,722`, `src/nodes/worker/tools/spider_final.py:308,921`, `src/concept_graph/renderer.py:68`, `src/concept_graph/service.py:72-81` | Created per output operation. PDF inline-math creation checks its content-hash image and creates the supplied job math-image directory only on a cache miss. Concept graph rendering creates its output directory; the service repeats that check only on renderer failure before persisting a fallback JSON spec. Other output directories likewise come from the current report/job/tool invocation and may be deleted or changed between calls. | Keep idempotent creation at the write/fallback boundary. A cached "directory exists" bit would be stale and can cross job boundaries; the math content-hash file cache is already bounded by the caller-provided directory identity and needs no process singleton. |
| Legacy summarizer shared directory | `src/nodes/summarizer.py:181-183,234-241`; active graph wiring is `src/graph.py:19,219` -> `src/nodes/summarizer_v2.py` | The legacy summarizer writes `cache/report/report.docx` or `final_report.md` once per direct legacy invocation. It is not the Summarizer node wired into the current production graph, which imports `summarizer_v2`. Its hard-coded relative path is shared mutable filesystem state if a caller invokes this legacy module directly. | Do not cache or optimize this path. A singleton would preserve, not fix, its lack of job isolation. Migrating/removing a legacy public entry point is outside this narrow audit and requires compatibility analysis. |
| Verifier log directory | `src/nodes/verifier.py:32-35` | Module-import scoped `logs` path. Import runs once per module object in a normal interpreter, but reload/subinterpreters can repeat it. The directory is filesystem state, not an immutable object. | No change. The repeated call is already bounded by Python import caching in the normal path. A second cache adds no material benefit and could become stale. |
| ReportLab font registration | `src/utils/md_to_pdf.py:22-125`, called at `665-670` | Font discovery and registration run per PDF. A two-call probe returned `ChineseFont` both times and ReportLab returned the same registered font object, but every call rechecks font files and logs registration. The registry is process-global and mutable; the selected file is filesystem/platform dependent. | Not proven immutable, so no cache was added. A cache could freeze a transient Helvetica fallback or ignore a font installed/replaced after the first render. Existing repeated registration is idempotent in the observed environment, not a correctness defect demonstrated by a red test. |
| Dedicated BM25 tokenizer | `src/rag/tokenizer.py:9-27,30-97`; fingerprint references at `src/rag/bm25_store.py:40,74-88` | `_JIEBA_TOKENIZER` is already a module-global dedicated `jieba.Tokenizer`, initialized once per module and augmented with the fixed `CHEMICAL_TERMS`. The BM25 store only hashes those fixed terms into index compatibility metadata. Each `ChemicalTokenizer` keeps a separate mutable model-token LRU because its TEI embedding dependency and lifetime are instance-specific. | No further caching. The dictionary initialization is already process/module scoped, and the fingerprint is a pure per-open compatibility value. Sharing the model-token LRU or embeddings across jobs would retain mutable data/client state. |
| Spider jieba state | `src/nodes/worker/tools/spider_final.py:16-44,47-119,301` | Module import calls `jieba.initialize()` on jieba's default global tokenizer. Every `ChemicalKeywordExtractor` then calls `jieba.add_word()` for its large term set. This is repeated per scraper and mutates a dependency-owned process-global dictionary. | No change in this scoped pass. Although repeated additions look idempotent, the object is explicitly mutable and shared with `jieba.analyse`/`posseg`; caching the extractor would also retain mutable stopword/term sets and scraper state. A safe redesign would require dedicated-tokenizer compatibility tests, not a singleton shortcut. |
| State/contract "registry" terminology | `src/state.py:134-144`, `src/requirements.py`, `src/blocker_registry.py`, `src/failure_registry.py`, `src/failure_semantics.py`, `src/recovery/policy.py`, plus registry reads/writes in `src/nodes/{intake,planner,verifier,recovery,summarizer_v2}.py` and registry docstrings in `src/evidence/{citations,projection}.py` | These `rg` matches are checkpoint/job records, requirement construction, copy-on-update helpers, or evidence-registry parameters—not initialization registries. Their contents and revisions change during execution. | Explicitly excluded. No caching or deduplication was attempted. |
| Existing warning controls | `src/nodes/worker/tools/spider_final.py:31`; `src/utils/md_to_pdf.py:19`; `src/utils/md_to_docx.py:17` | Spider import globally disables urllib3 `InsecureRequestWarning`; both renderer modules call `logging.basicConfig` at import. These are process-global side effects, not immutable registrations. | No new global suppression was introduced. Changing these legacy side effects is outside the narrow initialization-performance change and requires separate behavior/security review; they are recorded here rather than expanded into an unrelated refactor. |
| Existing immutable capability cache | `src/tool_capabilities.py:5-17` | `public_web_runtime_available()` is already protected by `@lru_cache(maxsize=1)`. It returns only whether four import specs (`requests`, `bs4`, `jieba`, `playwright.sync_api`) exist. This is a process-level dependency-capability probe, not a job registry, client, or result cache. | Already-safe/non-candidate. Installed module availability is immutable for the intended process lifetime, and the cached boolean owns no job or mutable dependency state. No additional cache or change is needed. |

## Warning probes

The required jieba probe produced no stdout/stderr and exited `0` under Python
`3.14.6` with `jieba==0.42.1`. The installed package is:

```text
/Users/synzzz/Documents/work_space/agent/agent-master/.venv/lib/python3.14/site-packages/jieba/__init__.py
```

`requirements.txt` pins `jieba==0.42.1`. The installed jieba tree contains no
`pkg_resources` reference, and `setuptools`/`pkg_resources` is not installed in
this virtual environment (`ModuleNotFoundError: No module named
'pkg_resources'`). Therefore the previously reported `jieba/pkg_resources`
warning is not reproducible in the current dependency environment. There is no
dependency-safe repository change to make, and the warning was not suppressed.

Importing the PDF renderer without a writable Matplotlib cache produced a
different environment warning: Matplotlib created a temporary cache because
`/Users/synzzz/.matplotlib` was not writable in the sandbox. Repeating the probe
with an explicitly writable temporary `MPLCONFIGDIR` removed that warning. This
is host/runtime cache configuration, not evidence for a production singleton;
the audit does not hard-code a host-specific cache path.

## Decision

No candidate met all three requirements: proven immutable process-global
identity, meaningful repeated cost, and a failing duplicate-registration
behavior that could be fixed without changing job isolation or resource
lifetime. Consequently:

- production code is unchanged;
- no behavior test was added, because there is no behavior change;
- repeated job-scoped `mkdir`/`makedirs(..., exist_ok=True)` remains intentional;
- no job paths, task/checkpoint state, RAG results/services, mutable registries,
  tool instances, font selection, tokenizer caches, or LLM clients were cached;
- no warning was globally suppressed as part of this task.

This audit-only outcome is the planned safe exit, not an incomplete
optimization.

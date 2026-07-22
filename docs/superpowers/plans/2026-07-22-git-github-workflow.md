# Git and GitHub Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an audited baseline commit and publish `main` to the private `SynZzz-x/chemical-report-multi-agent` GitHub repository.

**Architecture:** The current `agent-master` directory remains the independent repository root. A precise `.gitignore` excludes credentials and runtime state, while source, tests, docs, examples, dependency metadata, fonts, and the small demo image are committed. GitHub is added as `origin` only after local safety checks pass.

**Tech Stack:** Git, GitHub private repositories, Python 3, pytest, zsh.

## Global Constraints

- Repository root: `/Users/synzzz/Documents/work_space/agent/agent-master`.
- GitHub repository: `SynZzz-x/chemical-report-multi-agent`, visibility Private.
- Default branch: `main`; feature branches use descriptive names such as `codex/sqlite-checkpoint`.
- Local identity: user `SynZzz-x`, email `synzzz979@gmail.com`.
- Do not alter `/Users/synzzz/Documents/work_space/.git`.
- Do not commit secrets, caches, generated reports, Chroma indexes, or LangGraph SQLite files.
- Keep `requirements.txt` tracked.

---

### Task 1: Harden repository ignore rules

**Files:**
- Modify: `/Users/synzzz/Documents/work_space/agent/agent-master/.gitignore`
- Test: `git check-ignore`

**Interfaces:**
- Consumes: Existing project layout.
- Produces: The ignore policy used by every later staging step.

- [ ] **Step 1: Reproduce the current requirements exclusion**

Run: `git check-ignore -v requirements.txt`

Expected: output identifies the broad `*.txt` rule.

- [ ] **Step 2: Replace `.gitignore` with this complete content**

```gitignore
# macOS and editor state
.DS_Store
.idea/
.vscode/
.trae/

# Python environments and caches
.venv/
venv/
env/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Local secrets and credentials
.env
.env.*
!.env.example
secrets/
*.pem
*.key
*.p12

# Generated application state and artifacts
cache/
logs/
output/
outputs/
report/
reports/
charts/
worker_scrape_results/
*.log

# Runtime databases and indexes
data/chemical_kb/
*.sqlite
*.sqlite3
*.db

# Temporary files
tmp/
*.tmp
*.swp
```

- [ ] **Step 3: Verify required and forbidden paths**

Run:

```bash
if git check-ignore -q requirements.txt; then exit 1; fi
git check-ignore -v .env .DS_Store data/chemical_kb/chroma.sqlite3 __pycache__/app.cpython-312.pyc
```

Expected: `requirements.txt` is not ignored; every runtime path is ignored.

- [ ] **Step 4: Review the rule diff**

Run: `git diff -- .gitignore`

Expected: `*.txt` is removed and only scoped rules remain.

---

### Task 2: Document the contributor workflow

**Files:**
- Create: `/Users/synzzz/Documents/work_space/agent/agent-master/docs/development-workflow.md`
- Modify: `/Users/synzzz/Documents/work_space/agent/agent-master/README.md`
- Test: Markdown path and command inspection

**Interfaces:**
- Consumes: `main` and the `codex/` branch convention.
- Produces: A self-contained branch, test, commit, push, and PR guide.

- [ ] **Step 1: Create `docs/development-workflow.md` with these sections and commands**

````markdown
# Development Workflow

## First-time setup

Clone `https://github.com/SynZzz-x/chemical-report-multi-agent.git`, create `.venv`, install `requirements.txt`, and copy `.env.example` to `.env`. Keep `.env` local.

## Start a change

```bash
git switch main
git pull --ff-only origin main
git switch -c codex/sqlite-checkpoint
```

## Verify and commit

```bash
PYTHONPYCACHEPREFIX=/private/tmp/chemical-report-pycache python3 -m compileall -q app.py src
PYTHONPATH=. python -m pytest -q
git status --short
git diff
git add src/persistence.py tests/test_persistence.py
git diff --cached
git commit -m "feat: add SQLite checkpoint persistence"
```

Use `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, or `chore:` commit types.

## Push and review

```bash
git push -u origin codex/sqlite-checkpoint
```

Open a pull request into `main`; merge after review and checks pass.

## After merge

```bash
git switch main
git pull --ff-only origin main
git branch -d codex/sqlite-checkpoint
```
````

- [ ] **Step 2: Link the workflow from README**

Add immediately after `## Developer Workflow`:

```markdown
See [docs/development-workflow.md](docs/development-workflow.md) for the branch, verification, commit, push, and pull-request workflow.
```

- [ ] **Step 3: Verify references**

Run:

```bash
test -f docs/development-workflow.md
test -f requirements.txt
rg -n "development-workflow.md|codex/sqlite-checkpoint|git push -u origin" README.md docs/development-workflow.md
```

Expected: both files exist and all workflow references are found.

---

### Task 3: Audit, stage, and commit the baseline

**Files:**
- Stage: `.gitignore`, `.env.example`, `README.md`, `DOC.md`, `app.py`, `run.py`, `requirements.txt`
- Stage: `src/`, `tests/`, `docs/`, `examples/`, `data/fonts/`
- Stage: `202512160202.mp4_000010.980.jpg`

**Interfaces:**
- Consumes: Task 1 ignore policy and Task 2 documentation.
- Produces: An audited `chore: establish project baseline` commit.

- [ ] **Step 1: Compile Python syntax**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/chemical-report-pycache python3 -m compileall -q app.py src
```

Expected: exit code 0 with no output.

- [ ] **Step 2: Run tests when dependencies are available**

Run: `PYTHONPATH=. python -m pytest -q`

Expected: seven tests pass. If dependencies are unavailable, record that fact and do not claim the tests passed.

- [ ] **Step 3: Scan for obvious secrets**

Run:

```bash
rg -n --hidden -g '!.git/**' -g '!data/chemical_kb/**' \
  -e 'sk-[A-Za-z0-9_-]{16,}' \
  -e 'lsv2_[A-Za-z0-9_-]+' \
  -e 'OPENAI_API_KEY[[:space:]]*=[[:space:]]*[^[:space:]]+' \
  -e 'LANGSMITH_API_KEY[[:space:]]*=[[:space:]]*[^[:space:]]+' .
```

Expected: no real secret values; `.env.example` placeholders are reviewed and retained.

- [ ] **Step 4: Check GitHub file-size safety**

Run: `find . -type f -not -path './.git/*' -size +90M -print`

Expected: no output.

- [ ] **Step 5: Stage and inspect all non-ignored files**

Run:

```bash
git add .
git status --short
git diff --cached --stat
git diff --cached --name-only
```

Expected: source, tests, docs, examples, fonts, and `requirements.txt` are staged. Secrets, caches, Chroma data, and SQLite files are absent.

- [ ] **Step 6: Commit the baseline**

Run: `git commit -m "chore: establish project baseline"`

Expected: a second commit is created on `main`.

---

### Task 4: Create the private GitHub remote and publish main

**Files:**
- Modify Git metadata: `/Users/synzzz/Documents/work_space/agent/agent-master/.git/config`
- External resource: `https://github.com/SynZzz-x/chemical-report-multi-agent`

**Interfaces:**
- Consumes: The audited local commits and an authenticated `SynZzz-x` GitHub session.
- Produces: Private remote repository, `origin`, and upstream-tracked `main`.

- [ ] **Step 1: Create an empty private repository**

Open `https://github.com/new`, select owner `SynZzz-x`, enter `chemical-report-multi-agent`, select Private, and leave README, `.gitignore`, and license initialization disabled.

Expected: GitHub creates an empty private repository at `SynZzz-x/chemical-report-multi-agent`.

- [ ] **Step 2: Add and verify origin**

Run:

```bash
git remote add origin https://github.com/SynZzz-x/chemical-report-multi-agent.git
git remote -v
```

Expected: fetch and push URLs both match the private repository.

- [ ] **Step 3: Push main**

Run: `git push -u origin main`

Expected: push succeeds using GitHub browser, credential-manager, or token authentication; local `main` tracks `origin/main`.

- [ ] **Step 4: Verify local and remote state**

Run:

```bash
git status --short --branch
git remote -v
git log --oneline --decorate -n 5
```

Expected: the working tree is clean and `main...origin/main` is shown. On GitHub, verify Private visibility and confirm ignored local/runtime files are absent.

---

## Plan Self-Review

- Spec coverage: repository boundary, visibility, naming, branch policy, ignore rules, identity, baseline audit, remote creation, push, and contributor workflow are covered.
- Placeholder scan: repository names, paths, commands, file contents, and expected outcomes are explicit, using the concrete `codex/sqlite-checkpoint` branch as the workflow example.
- Interface consistency: every task uses the same repository root, GitHub owner, repository name, `main` branch, and `codex/` prefix.

# Development Workflow

## First-time setup

Clone the private repository and create an isolated Python environment:

```bash
git clone https://github.com/SynZzz-x/chemical-report-multi-agent.git
cd chemical-report-multi-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` locally. Never commit `.env` or generated runtime data.

## Start a change

Update `main`, then create a focused feature branch. The SQLite checkpoint work is shown as a concrete example:

```bash
git switch main
git pull --ff-only origin main
git switch -c codex/sqlite-checkpoint-store
```

Use the `codex/` prefix followed by a short description for later branches.

## Verify and commit

Run syntax and unit checks before staging changes:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/chemical-report-pycache python3 -m compileall -q app.py src
PYTHONPATH=. python -m pytest -q
git status --short
git diff
```

Stage only the files belonging to the current change, inspect the staged diff, and commit:

```bash
git add src/persistence.py tests/test_persistence.py
git diff --cached
git commit -m "feat: add SQLite checkpoint persistence"
```

Use `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, or `chore:` as the commit type.

## Push and review

```bash
git push -u origin codex/sqlite-checkpoint-store
```

Open a pull request into `main`. Merge only after the diff is reviewed and relevant checks pass.

## After merge

```bash
git switch main
git pull --ff-only origin main
git branch -d codex/sqlite-checkpoint-store
```

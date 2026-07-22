# Git and GitHub Workflow Design

## Goal

Turn the existing `agent-master` folder into an independent Git repository and prepare it for a private GitHub repository named `chemical-report-multi-agent`.

## Repository Boundary

The Git repository root is the current `agent-master` directory. The unrelated, empty Git repository in the parent `work_space` directory is not used for this project. A nested `.git` directory in `agent-master` makes Git commands executed inside the project resolve to the project repository.

## Branching Model

- `main` is the stable default branch.
- New work uses short-lived branches named `codex/<feature-name>`.
- Feature branches are pushed to GitHub and merged through pull requests.
- Direct feature development on `main` is avoided after the baseline commit.

## Initial Commit Scope

The baseline commit includes:

- Application entry points: `app.py` and `run.py`.
- Source code under `src/`.
- Tests and test fixtures under `tests/`.
- Documentation under `docs/` and `DOC.md`.
- Examples under `examples/`.
- Dependency and environment templates: `requirements.txt` and `.env.example`.
- Required font assets under `data/fonts/`.
- Existing small project assets that are needed to understand or demonstrate the repository.

The baseline commit excludes:

- Local secrets: `.env`, private keys, tokens, and credential files.
- Generated outputs: `cache/`, `logs/`, `output/`, `reports/`, and report artifacts.
- Python and test caches: `__pycache__/`, `.pytest_cache/`, `*.pyc`, and related files.
- OS/editor files such as `.DS_Store`.
- Runtime databases and indexes under `data/chemical_kb/`.
- Future LangGraph checkpoint and Store SQLite files.

`requirements.txt` must remain tracked; the current broad `*.txt` ignore pattern must be removed or narrowed.

## Private GitHub Remote

The GitHub repository is named `chemical-report-multi-agent` and is created as Private. The remote name is `origin`, and the local `main` branch tracks `origin/main` after the first push.

The local setup is completed before remote creation. Because GitHub CLI is not currently installed, remote creation uses an authenticated GitHub web session or is completed after installing and authenticating `gh`. No repository is created under an assumed GitHub account.

## Commit Identity

Git author name and email are configured locally for this repository before the first commit. Global Git identity is not modified unless explicitly requested. The email should be either the user's GitHub-verified email or GitHub-provided private noreply email.

## Validation and Safety Gates

Before the baseline commit:

1. Inspect `git status --short` and the complete staged file list.
2. Confirm `requirements.txt` is tracked.
3. Confirm `.env`, caches, runtime databases, and generated outputs are not staged.
4. Scan staged text files for obvious hard-coded secret patterns.
5. Confirm no staged file exceeds GitHub's 100 MB per-file limit.
6. Run the available syntax and test checks; if the environment lacks dependencies, record that limitation without claiming tests passed.

Before the first push:

1. Confirm the remote URL points to the intended private repository.
2. Confirm the local branch is `main`.
3. Push with upstream tracking using `git push -u origin main`.
4. Verify the GitHub repository remains Private and the expected files are present.

## Routine Development Flow

For each feature:

1. Update local `main` from `origin/main`.
2. Create `codex/<feature-name>` from current `main`.
3. Make one focused change with tests.
4. Review `git diff` and run relevant checks.
5. Commit with a concise conventional message.
6. Push the feature branch and open a pull request.
7. Merge after review and checks pass.
8. Delete the merged local and remote feature branch.

## Failure Handling

- If Git author identity is missing, stop before committing and request the correct name and email.
- If GitHub authentication is unavailable, keep the verified local repository and provide the exact remote commands; do not create a repository under another account.
- If a secret or runtime database is staged, unstage it, update `.gitignore`, and repeat the staged-file audit before committing.
- If the parent repository interferes with Git resolution, verify that `git rev-parse --show-toplevel` returns the `agent-master` directory before any commit or push.


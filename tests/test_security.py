import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"lsv2_[A-Za-z0-9_-]+"),
    re.compile(r"OPENAI_API_KEY\s*=\s*[\"'][^\"']+[\"']"),
    re.compile(r"LANGSMITH_API_KEY\s*=\s*[\"'][^\"']+[\"']"),
]
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".example",
    ".gitignore",
}
IGNORED_DIRS = {
    ".git",
    ".venv",
    ".worktrees",
    "__pycache__",
    ".pytest_cache",
    "cache",
    "logs",
    "output",
    "data",
}
ALLOWED_PLACEHOLDERS = {
    "replace-with-your-openai-compatible-key",
    "replace-with-your-langsmith-key",
    "replace-with-your-dashscope-key",
}


def iter_text_files():
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name == "requirements.txt":
            yield path


def test_no_hardcoded_secret_values():
    findings = []
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(0)
                if any(placeholder in value for placeholder in ALLOWED_PLACEHOLDERS):
                    continue
                findings.append(f"{path.relative_to(PROJECT_ROOT)}: {value[:24]}")

    assert findings == []

"""Runtime capability probes used before offering or exposing optional tools."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec


@lru_cache(maxsize=1)
def public_web_runtime_available() -> bool:
    """Return whether every dependency required by the public-web tool exists."""

    required_modules = ("requests", "bs4", "jieba", "playwright.sync_api")
    try:
        return all(find_spec(module_name) is not None for module_name in required_modules)
    except (ImportError, ModuleNotFoundError, ValueError):
        return False

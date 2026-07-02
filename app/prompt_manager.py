"""
PromptManager: loads prompt files from disk once at startup.

Design rationale:
  - Prompts as files → iterate without touching Python source
  - Loaded at startup → zero per-request I/O
  - Versioning via filename suffix (e.g., recommend_v2.txt) without code change
  - Missing prompt at startup = fail fast (don't discover at request time)

Alternative considered: prompts in a database → adds infrastructure dependency
for no benefit at this scale.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

_REQUIRED_PROMPTS = [
    "guardrail",
    "intent",
    "clarify",
    "recommend",
    "compare",
    "refine",
    "refuse",
]


class PromptManager:
    """Loads and caches all agent prompts from a directory."""

    def __init__(self, prompts_dir: Path) -> None:
        self._cache: Dict[str, str] = {}
        self._dir = prompts_dir

        if not prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        for name in _REQUIRED_PROMPTS:
            path = prompts_dir / f"{name}.txt"
            if not path.exists():
                raise FileNotFoundError(f"Required prompt file missing: {path}")
            self._cache[name] = path.read_text(encoding="utf-8").strip()

        logger.info(
            "PromptManager loaded %d prompts from %s",
            len(self._cache),
            prompts_dir,
        )

    def get(self, name: str) -> str:
        """Return prompt text by name. Raises KeyError if not found."""
        try:
            return self._cache[name]
        except KeyError:
            raise KeyError(f"Prompt '{name}' not loaded. Available: {list(self._cache)}")

    @property
    def loaded(self) -> list[str]:
        return list(self._cache)

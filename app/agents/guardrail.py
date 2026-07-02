"""
GuardrailAgent: classifies user input as safe or unsafe before any retrieval.

Two-pass design:
  1. Keyword fast-path (microseconds, no LLM) — catches obvious injection patterns
  2. LLM classifier — handles nuanced cases (encoded attacks, indirect instructions)

Fail-open on LLM error: a broken guardrail shouldn't block legitimate users.
The keyword fast-path still catches the most dangerous patterns even if LLM fails.

Alternative considered: single LLM-only pass → slower, and fails completely if
LLM is unavailable. Keyword pre-filter adds defense in depth.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from groq import Groq

from app.agents.base import BaseAgent
from app.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

# Hard-coded injection signatures — zero-LLM fast path
# These patterns are clear enough that no LLM call is needed
_INJECTION_PATTERNS = [
    "ignore previous",
    "ignore all previous",
    "forget your",
    "you are now",
    "act as dan",
    "jailbreak",
    "system: override",
    "reveal your prompt",
    "print your instructions",
    "bypass",
    "disregard",
]


class GuardrailAgent(BaseAgent):
    """Fast content safety classifier."""

    name = "guardrail"

    def __init__(self, client: Groq, prompt_manager: PromptManager) -> None:
        super().__init__(client)
        self._system = prompt_manager.get("guardrail")

    def is_safe(self, messages: List[Dict[str, str]]) -> Tuple[bool, str]:
        """
        Returns (is_safe, reason).

        Fast path: O(1) keyword check before any LLM call.
        LLM path: nuanced classification for edge cases.
        """
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # Fast-path: keyword injection check
        lower = last_user.lower()
        for pattern in _INJECTION_PATTERNS:
            if pattern in lower:
                logger.warning("guardrail fast-path blocked pattern=%r", pattern)
                return False, f"Prompt injection pattern detected: '{pattern}'"

        # LLM path: nuanced classifier
        try:
            result, _ = self._call_llm(
                system=self._system,
                messages=[{"role": "user", "content": last_user}],
                max_tokens=64,
            )
            safe = bool(result.get("safe", True))
            reason = result.get("reason", "")
            if not safe:
                logger.warning("guardrail LLM blocked: %s", reason)
            return safe, reason
        except Exception as e:
            # Fail open: don't block legitimate users on infrastructure failures
            logger.error("guardrail LLM error, defaulting safe: %s", e)
            return True, "guardrail error — defaulting safe"

"""
BaseAgent: shared infrastructure for all pipeline agents.

Provides:
  - LLM call with exponential backoff retry
  - JSON response extraction (bare JSON, markdown fences, embedded JSON)
  - Structured per-call logging (agent, latency, tokens)
  - No business logic — pure infrastructure

LLM Backend: Groq (free tier)
  Model: llama-3.3-70b-versatile
  Why Groq: free tier with generous rate limits, extremely fast inference
  (~300 tokens/sec), OpenAI-compatible API so the client is a one-line swap.

Retry strategy: exponential backoff on RateLimitError (429) only.
Other errors (auth, bad request) are permanent — no point retrying.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq, RateLimitError, APIError

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_RETRY_BASE_SECONDS = 1.5

# Groq free-tier model — best quality available on free tier.
# Alternative: "mixtral-8x7b-32768" for longer context needs.
# Alternative: "llama-3.1-8b-instant" for lower latency on simple tasks.
_DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _parse_json(raw: str) -> Dict[str, Any]:
    """
    Robust JSON extraction from LLM output.

    Handles three common patterns:
      1. Bare JSON object: {"key": "value"}
      2. Markdown fenced: ```json\n{...}\n```
      3. JSON embedded in prose: "Here you go: {...} Hope that helps!"

    Returns empty dict on complete parse failure (caller handles fallback).
    """
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: find first {...} block in arbitrary text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


class BaseAgent:
    """
    Abstract base for all pipeline agents.

    Dependency injection: receives Groq client at construction.
    Thread-safe: no mutable state after __init__.
    """

    name: str = "base"

    def __init__(self, client: Groq) -> None:
        self._client = client

    def _call_llm(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Call Groq LLM with retry logic.

        Groq uses the OpenAI chat-completions format:
          system prompt is a message with role="system", prepended to messages.

        Returns:
            (parsed_json_dict, token_usage_dict)

        Raises:
            groq.APIError after _MAX_RETRIES failures.
        """
        last_error: Optional[Exception] = None

        # Groq uses OpenAI-style messages — system is the first message
        full_messages = [{"role": "system", "content": system}] + messages

        for attempt in range(_MAX_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=_DEFAULT_MODEL,
                    max_tokens=max_tokens,
                    messages=full_messages,
                    temperature=0.1,   # low temperature for consistent JSON output
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                raw = response.choices[0].message.content or ""
                usage = {
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                }
                parsed = _parse_json(raw)
                logger.info(
                    "llm_call agent=%s attempt=%d latency_ms=%.0f "
                    "tokens_in=%d tokens_out=%d",
                    self.name,
                    attempt,
                    latency_ms,
                    usage["input_tokens"],
                    usage["output_tokens"],
                )
                return parsed, usage

            except RateLimitError as e:
                last_error = e
                wait = _RETRY_BASE_SECONDS * (2 ** attempt)
                logger.warning(
                    "agent=%s RateLimitError attempt=%d, retrying in %.1fs",
                    self.name,
                    attempt,
                    wait,
                )
                time.sleep(wait)

            except APIError as e:
                last_error = e
                logger.error("agent=%s APIError: %s", self.name, e)
                if attempt < _MAX_RETRIES:
                    time.sleep(1)

        raise last_error or RuntimeError(f"LLM call failed after {_MAX_RETRIES} retries")

"""
IntentAgent: extracts structured HiringIntent from conversation history.

Why structured extraction (vs. routing purely on LLM response)?
  - Routing becomes a deterministic switch on intent.action
  - Every downstream agent sees the same typed interface
  - Full conversation → typed struct is independently testable
  - retrieval_query is reproducible: same intent → same query → same results

Falls back to CLARIFY on parse failure — safer than guessing RECOMMEND
with incomplete intent.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from groq import Groq

from app.agents.base import BaseAgent
from app.models.intent import AgentAction, HiringIntent
from app.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class IntentAgent(BaseAgent):
    """Extracts HiringIntent from the full conversation."""

    name = "intent"

    def __init__(self, client: Groq, prompt_manager: PromptManager) -> None:
        super().__init__(client)
        self._system = prompt_manager.get("intent")

    def extract(self, messages: List[Dict[str, str]]) -> HiringIntent:
        """
        Analyze conversation and return typed HiringIntent.

        Returns default CLARIFY intent on any parse failure.
        """
        try:
            result, _ = self._call_llm(
                system=self._system,
                messages=messages,
                max_tokens=512,
            )
            # Only pass fields that HiringIntent actually declares
            valid_fields = {k: v for k, v in result.items() if k in HiringIntent.model_fields}
            intent = HiringIntent(**valid_fields)
            logger.info(
                "intent extracted action=%s role=%r seniority=%r skills=%s",
                intent.action,
                intent.role,
                intent.seniority,
                intent.skills[:3],
            )
            return intent
        except Exception as e:
            logger.error("IntentAgent failed: %s — defaulting to CLARIFY", e)
            return HiringIntent(action=AgentAction.CLARIFY)

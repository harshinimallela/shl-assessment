"""
ClarificationAgent: asks one focused question when intent is underspecified.

Design: ask ONE question, not a list. Multiple questions overwhelm users and
reduce response rate. The prompt enforces a single, highest-priority question.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from groq import Groq

from app.agents.base import BaseAgent
from app.models.api import ChatResponse
from app.models.intent import HiringIntent
from app.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class ClarificationAgent(BaseAgent):
    name = "clarify"

    def __init__(self, client: Groq, prompt_manager: PromptManager) -> None:
        super().__init__(client)
        self._system = prompt_manager.get("clarify")

    def run(self, messages: List[Dict[str, str]], intent: HiringIntent) -> ChatResponse:
        system = (
            f"{self._system}\n\n"
            f"## Current understanding\n{intent.model_dump_json(indent=2)}"
        )
        try:
            result, _ = self._call_llm(system=system, messages=messages, max_tokens=256)
        except Exception as e:
            logger.error("ClarificationAgent failed: %s", e)
            return ChatResponse(
                reply="Could you tell me more about the role you're hiring for?",
                recommendations=[],
                end_of_conversation=False,
            )

        return ChatResponse(
            reply=result.get("reply", "Could you tell me more about the role?"),
            recommendations=[],
            end_of_conversation=False,
        )

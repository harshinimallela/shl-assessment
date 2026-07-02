"""
RefinementAgent: merges new constraints into the existing recommendation set.

The key difference from RecommendationAgent: the conversation history
already contains prior recommendations, so the agent merges the user's
refinement request into the existing context rather than starting fresh.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from groq import Groq

from app.agents.base import BaseAgent
from app.agents.recommend import _format_entries, _validate
from app.models.api import ChatResponse
from app.models.catalog import CatalogEntry
from app.models.intent import HiringIntent
from app.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class RefinementAgent(BaseAgent):
    name = "refine"

    def __init__(self, client: Groq, prompt_manager: PromptManager) -> None:
        super().__init__(client)
        self._prompt = prompt_manager.get("refine")

    def run(
        self,
        messages: List[Dict[str, str]],
        intent: HiringIntent,
        retrieved: List[CatalogEntry],
    ) -> ChatResponse:
        retrieved_text = _format_entries(retrieved)
        valid_urls = {e.url for e in retrieved}
        valid_names = {e.name.lower(): e for e in retrieved}

        system = (
            f"{self._prompt}\n\n"
            f"## Updated Hiring Intent\n{intent.model_dump_json(indent=2)}\n\n"
            f"## Available Assessments (your ONLY candidate pool)\n{retrieved_text}"
        )

        try:
            result, _ = self._call_llm(system=system, messages=messages)
        except Exception as e:
            logger.error("RefinementAgent failed: %s", e)
            return ChatResponse(
                reply="I couldn't refine the recommendations. Please try again.",
                recommendations=[],
                end_of_conversation=False,
            )

        validated = _validate(result.get("recommendations", []), valid_urls, valid_names)

        return ChatResponse(
            reply=result.get("reply", "Here is the refined recommendation set."),
            recommendations=validated,
            end_of_conversation=bool(result.get("end_of_conversation", False)),
        )

"""
ComparisonAgent: provides catalog-grounded comparison of assessments.

Grounding: only the retrieved entries appear in context.
Even for comparison queries, the LLM cannot hallucinate facts about
assessments that weren't retrieved.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from groq import Groq

from app.agents.base import BaseAgent
from app.agents.recommend import _format_entries, _validate
from app.models.api import ChatResponse
from app.models.catalog import CatalogEntry
from app.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class ComparisonAgent(BaseAgent):
    name = "compare"

    def __init__(self, client: Groq, prompt_manager: PromptManager) -> None:
        super().__init__(client)
        self._prompt = prompt_manager.get("compare")

    def run(
        self,
        messages: List[Dict[str, str]],
        entries: List[CatalogEntry],
    ) -> ChatResponse:
        retrieved_text = _format_entries(entries)
        valid_urls = {e.url for e in entries}
        valid_names = {e.name.lower(): e for e in entries}

        system = (
            f"{self._prompt}\n\n"
            f"## Assessments to compare\n{retrieved_text}"
        )

        try:
            result, _ = self._call_llm(system=system, messages=messages)
        except Exception as e:
            logger.error("ComparisonAgent failed: %s", e)
            return ChatResponse(
                reply="I couldn't complete the comparison. Please try again.",
                recommendations=[],
                end_of_conversation=False,
            )

        validated = _validate(result.get("recommendations", []), valid_urls, valid_names)

        return ChatResponse(
            reply=result.get("reply", "Here is a comparison of the requested assessments."),
            recommendations=validated,
            end_of_conversation=bool(result.get("end_of_conversation", False)),
        )

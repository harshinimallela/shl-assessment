"""
RecommendationAgent: selects from retrieved entries and generates a shortlist.

Key design:
  - Only retrieved entries appear in the prompt — the LLM acts as a reranker
    + explainer, not as a catalog
  - Hallucination guard: _validate() cross-checks every returned URL against
    the retrieved set. Anything not retrieved is structurally impossible to
    recommend (prompt layer) and is silently dropped (code layer).
  - Two-layer defense: prompt says "only from retrieved list"; code enforces it.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Set

from groq import Groq

from app.agents.base import BaseAgent
from app.models.api import ChatResponse, Recommendation
from app.models.catalog import CatalogEntry
from app.models.intent import HiringIntent
from app.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


def _format_entries(entries: List[CatalogEntry]) -> str:
    lines = []
    for e in entries:
        lines.append(
            f'- name: "{e.name}" | url: "{e.url}" | type: {e.test_type} '
            f"({e.test_type_label}) | levels: {', '.join(e.job_levels)} "
            f"| duration: {e.duration_minutes}min | desc: {e.description}"
        )
    return "\n".join(lines)


class RecommendationAgent(BaseAgent):
    name = "recommend"

    def __init__(self, client: Groq, prompt_manager: PromptManager) -> None:
        super().__init__(client)
        self._prompt = prompt_manager.get("recommend")

    def run(
        self,
        messages: List[Dict[str, str]],
        intent: HiringIntent,
        retrieved: List[CatalogEntry],
    ) -> ChatResponse:
        retrieved_text = _format_entries(retrieved)
        valid_urls: Set[str] = {e.url for e in retrieved}
        valid_names: Dict[str, CatalogEntry] = {e.name.lower(): e for e in retrieved}

        system = (
            f"{self._prompt}\n\n"
            f"## Hiring Intent\n{intent.model_dump_json(indent=2)}\n\n"
            f"## Retrieved Assessments (your ONLY candidate pool)\n{retrieved_text}"
        )

        try:
            result, _ = self._call_llm(system=system, messages=messages)
        except Exception as e:
            logger.error("RecommendationAgent LLM failed: %s", e)
            return ChatResponse(
                reply="I encountered an issue generating recommendations. Please try again.",
                recommendations=[],
                end_of_conversation=False,
            )

        validated = _validate(result.get("recommendations", []), valid_urls, valid_names)

        return ChatResponse(
            reply=result.get("reply", "Here are my recommendations."),
            recommendations=validated,
            end_of_conversation=bool(result.get("end_of_conversation", False)),
        )


def _validate(
    raw: list,
    valid_urls: Set[str],
    valid_names: Dict[str, CatalogEntry],
) -> List[Recommendation]:
    """
    Two-pass validation: URL exact match, then name fuzzy match.

    Drops anything not in the retrieved set — hard API-boundary guarantee.
    """
    seen_urls: Set[str] = set()
    result: List[Recommendation] = []

    for rec in raw:
        url = rec.get("url", "")
        name = rec.get("name", "")

        if url in valid_urls and url not in seen_urls:
            seen_urls.add(url)
            result.append(
                Recommendation(name=name, url=url, test_type=rec.get("test_type", ""))
            )
        elif name.lower() in valid_names and url not in seen_urls:
            entry = valid_names[name.lower()]
            seen_urls.add(entry.url)
            result.append(
                Recommendation(name=entry.name, url=entry.url, test_type=entry.test_type)
            )
        else:
            logger.warning("Dropped hallucinated rec: name=%r url=%r", name, url)

        if len(result) >= 10:
            break

    return result

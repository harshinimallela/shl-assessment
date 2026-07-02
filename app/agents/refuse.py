"""
RefuseAgent: polite redirect for off-topic or unsafe requests.

Design: always returns valid schema (empty recommendations, end_of_conversation=False).
Returning end_of_conversation=True on refusal would close the session — users
should be able to ask a valid question next.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from groq import Groq

from app.agents.base import BaseAgent
from app.models.api import ChatResponse
from app.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

_FALLBACK_REPLY = (
    "I'm here to help you find the right SHL assessments for your hiring needs. "
    "Could you describe the role you're assessing candidates for?"
)


class RefuseAgent(BaseAgent):
    name = "refuse"

    def __init__(self, client: Groq, prompt_manager: PromptManager) -> None:
        super().__init__(client)
        self._system = prompt_manager.get("refuse")

    def run(
        self,
        messages: List[Dict[str, str]],
        reason: Optional[str] = None,
    ) -> ChatResponse:
        system = self._system
        if reason:
            system += f"\n\n## Refusal reason\n{reason}"

        try:
            result, _ = self._call_llm(
                system=system, messages=messages, max_tokens=256
            )
        except Exception as e:
            logger.error("RefuseAgent failed: %s", e)
            return ChatResponse(
                reply=_FALLBACK_REPLY,
                recommendations=[],
                end_of_conversation=False,
            )

        return ChatResponse(
            reply=result.get("reply", _FALLBACK_REPLY),
            recommendations=[],
            end_of_conversation=bool(result.get("end_of_conversation", False)),
        )

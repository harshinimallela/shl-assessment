"""
HiringIntent: the structured output of IntentAgent.

Why a typed struct instead of raw dict?
  - Downstream agents consume the same interface regardless of LLM output variations
  - Routing is a switch on intent.action — explicit, testable, no LLM magic
  - The retrieval_query property is deterministic and reproducible
  - Logging/debugging shows exactly what was understood from the conversation
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AgentAction(str, Enum):
    CLARIFY = "CLARIFY"
    RECOMMEND = "RECOMMEND"
    COMPARE = "COMPARE"
    REFINE = "REFINE"
    REFUSE = "REFUSE"


class HiringIntent(BaseModel):
    action: AgentAction = AgentAction.CLARIFY
    role: Optional[str] = None
    seniority: Optional[str] = None          # "entry" | "mid" | "senior" | "lead" | "director" | "executive" | "graduate"
    skills: List[str] = Field(default_factory=list)
    personality_needed: bool = False
    cognitive_needed: bool = False
    knowledge_needed: bool = False
    simulation_needed: bool = False
    comparison_targets: List[str] = Field(default_factory=list)

    @property
    def retrieval_query(self) -> str:
        """
        Synthesize a search string from structured fields.

        Using structured fields rather than raw conversation text ensures
        retrieval is grounded in what was actually understood, not noise words.
        """
        parts: List[str] = []
        if self.role:
            parts.append(self.role)
        if self.seniority:
            parts.append(self.seniority)
        parts.extend(self.skills[:5])  # cap to avoid diluting the vector
        if self.personality_needed:
            parts.append("personality questionnaire behavior")
        if self.cognitive_needed:
            parts.append("cognitive ability reasoning")
        if self.knowledge_needed:
            parts.append("knowledge skills test")
        if self.simulation_needed:
            parts.append("simulation situational judgment")
        return " ".join(parts) if parts else "assessment"

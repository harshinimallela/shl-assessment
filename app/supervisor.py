"""
Supervisor: routes conversations to the appropriate agent pipeline.

The Supervisor is the composition root of the multi-agent system.
It knows WHICH agent to call and WHAT data to pass.
It does NOT generate replies, craft prompts, or touch the LLM directly.

Routing logic:
  GuardrailAgent → unsafe? → RefuseAgent
                 ↓ safe
  IntentAgent → extract HiringIntent
                 ↓
  action=REFUSE   → RefuseAgent
  action=CLARIFY  → ClarificationAgent
  action=COMPARE  → store.get_by_names → ComparisonAgent
  action=REFINE   → store.search → RefinementAgent
  action=RECOMMEND→ store.search → RecommendationAgent

Thread-safety: all request state is in local variables. No shared mutable state.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple

from groq import Groq

from app.agents.clarify import ClarificationAgent
from app.agents.compare import ComparisonAgent
from app.agents.guardrail import GuardrailAgent
from app.agents.intent import IntentAgent
from app.agents.recommend import RecommendationAgent
from app.agents.refine import RefinementAgent
from app.agents.refuse import RefuseAgent
from app.models.api import ChatResponse
from app.models.intent import AgentAction
from app.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class Supervisor:
    """
    Orchestrates the multi-agent pipeline.

    Constructed once at startup via DI; thread-safe across requests.
    """

    def __init__(
        self,
        client: Groq,
        catalog_store,   # Retriever protocol — TFIDFCatalogStore or FAISSCatalogStore
        prompt_manager: PromptManager,
    ) -> None:
        self._store = catalog_store
        self._guardrail = GuardrailAgent(client, prompt_manager)
        self._intent = IntentAgent(client, prompt_manager)
        self._clarify = ClarificationAgent(client, prompt_manager)
        self._recommend = RecommendationAgent(client, prompt_manager)
        self._compare = ComparisonAgent(client, prompt_manager)
        self._refine = RefinementAgent(client, prompt_manager)
        self._refuse = RefuseAgent(client, prompt_manager)

    def handle(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[ChatResponse, Dict]:
        """
        Process a conversation and return (response, trace).

        trace contains structured telemetry:
          request_id, steps[], routed_to, total_ms, rec_count
        """
        request_id = str(uuid.uuid4())[:8]
        t_start = time.perf_counter()
        trace: Dict = {"request_id": request_id, "steps": []}

        # ── Step 1: Safety guardrail ──────────────────────────────────────────
        t0 = time.perf_counter()
        safe, reason = self._guardrail.is_safe(messages)
        trace["steps"].append({
            "agent": "guardrail",
            "safe": safe,
            "reason": reason,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        })

        if not safe:
            logger.info("[%s] guardrail blocked: %s", request_id, reason)
            response = self._refuse.run(messages, reason)
            trace["routed_to"] = "refuse(guardrail)"
            trace["total_ms"] = round((time.perf_counter() - t_start) * 1000)
            trace["rec_count"] = 0
            return response, trace

        # ── Step 2: Intent extraction ─────────────────────────────────────────
        t0 = time.perf_counter()
        intent = self._intent.extract(messages)
        trace["steps"].append({
            "agent": "intent",
            "action": intent.action,
            "role": intent.role,
            "seniority": intent.seniority,
            "skills": intent.skills,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        })
        logger.info(
            "[%s] intent action=%s role=%r seniority=%r",
            request_id, intent.action, intent.role, intent.seniority,
        )

        # ── Step 3: Route to specialized agent ───────────────────────────────
        action = intent.action

        if action == AgentAction.REFUSE:
            trace["routed_to"] = "refuse(intent)"
            response = self._refuse.run(messages)

        elif action == AgentAction.CLARIFY:
            trace["routed_to"] = "clarify"
            response = self._clarify.run(messages, intent)

        elif action == AgentAction.COMPARE:
            t0 = time.perf_counter()
            entries = self._store.get_by_names(intent.comparison_targets)
            if not entries:
                entries = self._store.search(" ".join(intent.comparison_targets), top_k=5)
            trace["steps"].append({
                "agent": "retrieval",
                "query": str(intent.comparison_targets),
                "retrieved_count": len(entries),
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            })
            trace["routed_to"] = "compare"
            response = self._compare.run(messages, entries)

        elif action == AgentAction.REFINE:
            t0 = time.perf_counter()
            retrieved = self._store.search(
                intent.retrieval_query,
                top_k=15,
                test_types=_build_type_filter(intent),
                job_level=_normalize_level(intent.seniority),
            )
            trace["steps"].append({
                "agent": "retrieval",
                "query": intent.retrieval_query,
                "retrieved_count": len(retrieved),
                "top_names": [e.name for e in retrieved[:5]],
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            })
            trace["routed_to"] = "refine"
            response = self._refine.run(messages, intent, retrieved)

        else:  # RECOMMEND (default)
            t0 = time.perf_counter()
            retrieved = self._store.search(
                intent.retrieval_query,
                top_k=15,
                test_types=_build_type_filter(intent),
                job_level=_normalize_level(intent.seniority),
            )
            trace["steps"].append({
                "agent": "retrieval",
                "query": intent.retrieval_query,
                "retrieved_count": len(retrieved),
                "top_names": [e.name for e in retrieved[:5]],
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            })
            trace["routed_to"] = "recommend"
            response = self._recommend.run(messages, intent, retrieved)

        trace["total_ms"] = round((time.perf_counter() - t_start) * 1000)
        trace["rec_count"] = len(response.recommendations)
        logger.info(
            "[%s] routed=%s recs=%d total_ms=%d",
            request_id,
            trace["routed_to"],
            trace["rec_count"],
            trace["total_ms"],
        )
        return response, trace


def _build_type_filter(intent) -> Optional[List[str]]:
    types = []
    if intent.personality_needed:
        types.append("P")
    if intent.cognitive_needed:
        types.append("A")
    if intent.knowledge_needed:
        types.append("K")
    if intent.simulation_needed:
        types.append("S")
    return types or None


def _normalize_level(seniority: Optional[str]) -> Optional[str]:
    if not seniority:
        return None
    mapping = {
        "entry": "Entry Level",
        "junior": "Entry Level",
        "mid": "Professional",
        "senior": "Professional",
        "lead": "Manager",
        "director": "Director",
        "executive": "Executive",
        "graduate": "Graduate",
    }
    return mapping.get(seniority.lower())

"""
Unit tests for models, API validation, and PromptManager.

All tests run without LLM or network.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.api import ChatRequest, ChatResponse, Recommendation
from app.models.catalog import CatalogEntry
from app.models.intent import AgentAction, HiringIntent
from app.prompt_manager import PromptManager


# ── ChatRequest validation ─────────────────────────────────────────────────────

class TestChatRequest:
    def test_valid_request(self):
        req = ChatRequest(messages=[{"role": "user", "content": "hello"}])
        assert req.messages[0].role == "user"

    def test_empty_messages_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])

    def test_first_message_must_be_user(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[{"role": "assistant", "content": "hi"}])

    def test_multi_turn_valid(self):
        req = ChatRequest(messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "follow up"},
        ])
        assert len(req.messages) == 3

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[{"role": "system", "content": "hack"}])


# ── ChatResponse ───────────────────────────────────────────────────────────────

class TestChatResponse:
    def test_valid_response(self):
        resp = ChatResponse(
            reply="Here are recommendations.",
            recommendations=[
                Recommendation(name="Test A", url="https://shl.com/a", test_type="K")
            ],
            end_of_conversation=False,
        )
        assert resp.reply
        assert len(resp.recommendations) == 1

    def test_empty_recommendations(self):
        resp = ChatResponse(reply="Ask me more.", recommendations=[], end_of_conversation=False)
        assert resp.recommendations == []


# ── CatalogEntry ───────────────────────────────────────────────────────────────

class TestCatalogEntry:
    def test_searchable_text_combines_fields(self):
        entry = CatalogEntry(
            name="Java Test",
            url="https://shl.com/java",
            description="Tests Java skills",
            test_type="K",
            test_type_label="Knowledge",
            keywords=["java", "oop"],
        )
        text = entry.searchable_text
        assert "Java Test" in text
        assert "Tests Java skills" in text
        assert "java" in text

    def test_valid_test_types(self):
        for t in ["A", "P", "K", "S", "B"]:
            entry = CatalogEntry(
                name="X", url="https://shl.com/x",
                description="d", test_type=t,
            )
            assert entry.test_type == t

    def test_default_empty_lists(self):
        entry = CatalogEntry(
            name="X", url="https://shl.com/x",
            description="desc", test_type="A",
        )
        assert entry.job_levels == []
        assert entry.keywords == []
        assert entry.languages == []


# ── HiringIntent ───────────────────────────────────────────────────────────────

class TestHiringIntent:
    def test_default_action_is_clarify(self):
        intent = HiringIntent()
        assert intent.action == AgentAction.CLARIFY

    def test_retrieval_query_with_role(self):
        intent = HiringIntent(role="Java developer", seniority="senior")
        q = intent.retrieval_query
        assert "Java developer" in q
        assert "senior" in q

    def test_retrieval_query_with_personality_flag(self):
        intent = HiringIntent(role="manager", personality_needed=True)
        q = intent.retrieval_query
        assert "personality" in q.lower()

    def test_retrieval_query_with_cognitive_flag(self):
        intent = HiringIntent(role="analyst", cognitive_needed=True)
        q = intent.retrieval_query
        assert "cognitive" in q.lower() or "reasoning" in q.lower()

    def test_retrieval_query_empty_intent(self):
        intent = HiringIntent()
        q = intent.retrieval_query
        assert q == "assessment"

    def test_retrieval_query_caps_skills(self):
        intent = HiringIntent(skills=["a", "b", "c", "d", "e", "f", "g", "h"])
        q = intent.retrieval_query
        # Should only include first 5 skills
        words = q.split()
        skill_words = [w for w in words if w in ["a", "b", "c", "d", "e", "f", "g", "h"]]
        assert len(skill_words) <= 5

    def test_all_actions_valid(self):
        for action in AgentAction:
            intent = HiringIntent(action=action)
            assert intent.action == action


# ── PromptManager ──────────────────────────────────────────────────────────────

class TestPromptManager:
    def _make_prompts_dir(self, tmp_path: Path, names: list) -> Path:
        d = tmp_path / "prompts"
        d.mkdir()
        for name in names:
            (d / f"{name}.txt").write_text(f"prompt for {name}")
        return d

    def test_loads_all_required_prompts(self, tmp_path):
        required = ["guardrail", "intent", "clarify", "recommend", "compare", "refine", "refuse"]
        d = self._make_prompts_dir(tmp_path, required)
        pm = PromptManager(d)
        for name in required:
            assert pm.get(name) == f"prompt for {name}"

    def test_missing_prompt_raises_at_init(self, tmp_path):
        # Only provide some prompts — should fail at init
        d = self._make_prompts_dir(tmp_path, ["guardrail", "intent"])
        with pytest.raises(FileNotFoundError):
            PromptManager(d)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PromptManager(tmp_path / "nonexistent")

    def test_unknown_prompt_raises_key_error(self, tmp_path):
        required = ["guardrail", "intent", "clarify", "recommend", "compare", "refine", "refuse"]
        d = self._make_prompts_dir(tmp_path, required)
        pm = PromptManager(d)
        with pytest.raises(KeyError):
            pm.get("nonexistent_prompt")

    def test_loaded_property(self, tmp_path):
        required = ["guardrail", "intent", "clarify", "recommend", "compare", "refine", "refuse"]
        d = self._make_prompts_dir(tmp_path, required)
        pm = PromptManager(d)
        assert set(pm.loaded) == set(required)


# ── Recommendation validation (hallucination guard) ───────────────────────────

class TestRecommendationValidation:
    def test_valid_url_passes(self):
        from app.agents.recommend import _validate
        from app.models.catalog import CatalogEntry

        entry = CatalogEntry(
            name="Test A", url="https://shl.com/a",
            description="d", test_type="K",
        )
        valid_urls = {"https://shl.com/a"}
        valid_names = {"test a": entry}

        raw = [{"name": "Test A", "url": "https://shl.com/a", "test_type": "K"}]
        result = _validate(raw, valid_urls, valid_names)
        assert len(result) == 1
        assert result[0].url == "https://shl.com/a"

    def test_hallucinated_url_dropped(self):
        from app.agents.recommend import _validate

        valid_urls = {"https://shl.com/real"}
        valid_names = {}
        raw = [{"name": "Fake", "url": "https://shl.com/fake", "test_type": "K"}]
        result = _validate(raw, valid_urls, valid_names)
        assert len(result) == 0

    def test_name_fallback_match(self):
        from app.agents.recommend import _validate
        from app.models.catalog import CatalogEntry

        entry = CatalogEntry(
            name="Real Test", url="https://shl.com/real",
            description="d", test_type="A",
        )
        valid_urls = set()
        valid_names = {"real test": entry}

        raw = [{"name": "Real Test", "url": "https://shl.com/wrong-url", "test_type": "A"}]
        result = _validate(raw, valid_urls, valid_names)
        assert len(result) == 1
        assert result[0].url == "https://shl.com/real"  # corrected to canonical URL

    def test_deduplication(self):
        from app.agents.recommend import _validate
        from app.models.catalog import CatalogEntry

        entry = CatalogEntry(
            name="Test A", url="https://shl.com/a",
            description="d", test_type="K",
        )
        valid_urls = {"https://shl.com/a"}
        valid_names = {"test a": entry}

        raw = [
            {"name": "Test A", "url": "https://shl.com/a", "test_type": "K"},
            {"name": "Test A", "url": "https://shl.com/a", "test_type": "K"},  # duplicate
        ]
        result = _validate(raw, valid_urls, valid_names)
        assert len(result) == 1  # deduplicated

    def test_max_10_cap(self):
        from app.agents.recommend import _validate
        from app.models.catalog import CatalogEntry

        entries = [
            CatalogEntry(name=f"Test {i}", url=f"https://shl.com/{i}",
                        description="d", test_type="K")
            for i in range(15)
        ]
        valid_urls = {e.url for e in entries}
        valid_names = {e.name.lower(): e for e in entries}
        raw = [{"name": e.name, "url": e.url, "test_type": "K"} for e in entries]

        result = _validate(raw, valid_urls, valid_names)
        assert len(result) <= 10


# ── JSON parsing (base agent) ─────────────────────────────────────────────────

class TestJsonParsing:
    def test_bare_json(self):
        from app.agents.base import _parse_json
        result = _parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced(self):
        from app.agents.base import _parse_json
        result = _parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_embedded_json(self):
        from app.agents.base import _parse_json
        result = _parse_json('Here is the result: {"key": "value"} done.')
        assert result == {"key": "value"}

    def test_invalid_returns_empty(self):
        from app.agents.base import _parse_json
        result = _parse_json("not json at all")
        assert result == {}

    def test_empty_string(self):
        from app.agents.base import _parse_json
        result = _parse_json("")
        assert result == {}

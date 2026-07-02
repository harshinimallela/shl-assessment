"""
Unit tests for TFIDFCatalogStore.

All tests run without a live LLM or network connection.
Uses a minimal in-memory catalog for fast, deterministic tests.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from app.retrieval.tfidf_store import TFIDFCatalogStore, _tokenize


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def minimal_catalog_path(tmp_path_factory) -> Path:
    """Minimal catalog with known entries for deterministic assertions."""
    catalog = [
        {
            "name": "Java Programming Test",
            "url": "https://www.shl.com/java-test",
            "description": "Measures Java programming skills and object-oriented design",
            "test_type": "K",
            "test_type_label": "Knowledge",
            "job_levels": ["Professional", "Manager"],
            "duration_minutes": 45,
            "languages": ["English"],
            "keywords": ["java", "programming", "oop"],
        },
        {
            "name": "OPQ32r Personality Questionnaire",
            "url": "https://www.shl.com/opq32r",
            "description": "Comprehensive occupational personality questionnaire",
            "test_type": "P",
            "test_type_label": "Personality",
            "job_levels": ["Professional", "Manager", "Director"],
            "duration_minutes": 25,
            "languages": ["English", "French"],
            "keywords": ["personality", "behavior", "traits"],
        },
        {
            "name": "Verbal Reasoning Test",
            "url": "https://www.shl.com/verbal",
            "description": "Measures ability to understand and evaluate written information",
            "test_type": "A",
            "test_type_label": "Ability",
            "job_levels": ["Graduate", "Professional"],
            "duration_minutes": 19,
            "languages": ["English"],
            "keywords": ["verbal", "reasoning", "cognitive"],
        },
        {
            "name": "Python Developer Assessment",
            "url": "https://www.shl.com/python",
            "description": "Python programming skills test for software developers",
            "test_type": "K",
            "test_type_label": "Knowledge",
            "job_levels": ["Professional"],
            "duration_minutes": 40,
            "languages": ["English"],
            "keywords": ["python", "programming", "developer"],
        },
    ]
    p = tmp_path_factory.mktemp("catalog") / "catalog.json"
    p.write_text(json.dumps(catalog))
    return p


@pytest.fixture(scope="module")
def store(minimal_catalog_path) -> TFIDFCatalogStore:
    return TFIDFCatalogStore(minimal_catalog_path)


# ── Tokenizer ──────────────────────────────────────────────────────────────────

class TestTokenizer:
    def test_lowercases(self):
        assert "java" in _tokenize("Java Developer")

    def test_preserves_c_plus_plus(self):
        assert "c++" in _tokenize("C++ developer")

    def test_preserves_c_sharp(self):
        assert "c#" in _tokenize("C# engineer")

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numbers(self):
        assert "java8" in _tokenize("Java8 developer") or "java" in _tokenize("Java8 developer")


# ── Initialization ─────────────────────────────────────────────────────────────

class TestInitialization:
    def test_loads_all_entries(self, store):
        assert len(store.entries) == 4

    def test_url_set_populated(self, store):
        assert len(store.all_urls) == 4

    def test_all_entries_have_urls(self, store):
        for e in store.entries:
            assert e.url.startswith("https://")

    def test_rejects_missing_catalog(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TFIDFCatalogStore(tmp_path / "nonexistent.json")

    def test_rejects_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        with pytest.raises(Exception):
            TFIDFCatalogStore(bad)


# ── Search ─────────────────────────────────────────────────────────────────────

class TestSearch:
    def test_returns_results(self, store):
        results = store.search("java developer")
        assert len(results) > 0

    def test_java_query_ranks_java_first(self, store):
        results = store.search("java programming test")
        assert results[0].name == "Java Programming Test"

    def test_python_query_ranks_python_first(self, store):
        results = store.search("python developer assessment")
        assert results[0].name == "Python Developer Assessment"

    def test_personality_query_returns_personality_type(self, store):
        results = store.search("personality questionnaire behavior traits")
        types = {r.test_type for r in results[:2]}
        assert "P" in types

    def test_top_k_respected(self, store):
        results = store.search("developer", top_k=2)
        assert len(results) <= 2

    def test_type_filter_knowledge_only(self, store):
        results = store.search("assessment", test_types=["K"])
        assert all(r.test_type == "K" for r in results)

    def test_type_filter_personality_only(self, store):
        results = store.search("assessment", test_types=["P"])
        assert all(r.test_type == "P" for r in results)

    def test_language_filter(self, store):
        results = store.search("assessment", language="French")
        assert all(
            any("french" in lang.lower() for lang in r.languages)
            for r in results
        )

    def test_job_level_filter(self, store):
        results = store.search("leadership test", job_level="Director")
        for r in results:
            assert any("director" in lvl.lower() for lvl in r.job_levels)

    def test_empty_query_returns_list(self, store):
        results = store.search("")
        assert isinstance(results, list)

    def test_results_are_catalog_entries(self, store):
        from app.models.catalog import CatalogEntry
        results = store.search("test")
        for r in results:
            assert isinstance(r, CatalogEntry)


# ── Validation ─────────────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_url_returns_true(self, store):
        url = store.entries[0].url
        assert store.is_valid_url(url)

    def test_fake_url_returns_false(self, store):
        assert not store.is_valid_url("https://www.shl.com/fake-assessment-xyz")

    def test_valid_name_exact(self, store):
        assert store.is_valid_name("Java Programming Test")

    def test_valid_name_case_insensitive(self, store):
        assert store.is_valid_name("java programming test")

    def test_invalid_name_returns_false(self, store):
        assert not store.is_valid_name("Completely Fake Assessment 9000")


# ── get_by_names ──────────────────────────────────────────────────────────────

class TestGetByNames:
    def test_exact_match(self, store):
        result = store.get_by_names(["Java Programming Test"])
        assert len(result) == 1
        assert result[0].name == "Java Programming Test"

    def test_partial_match(self, store):
        result = store.get_by_names(["OPQ32r"])
        assert len(result) >= 1
        assert "OPQ" in result[0].name or "opq" in result[0].name.lower()

    def test_missing_name_returns_empty(self, store):
        result = store.get_by_names(["Nonexistent Assessment XYZ"])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_multiple_names(self, store):
        result = store.get_by_names(["Java Programming Test", "Verbal Reasoning Test"])
        assert len(result) == 2


# ── Searchable text ───────────────────────────────────────────────────────────

class TestSearchableText:
    def test_contains_name(self, store):
        e = store.entries[0]
        assert e.name in e.searchable_text

    def test_contains_description_fragment(self, store):
        e = store.entries[0]
        assert e.description[:20] in e.searchable_text

    def test_contains_keywords(self, store):
        e = store.entries[0]
        for kw in e.keywords:
            assert kw in e.searchable_text

.PHONY: help install install-dev run test test-unit test-integration lint format typecheck eval docker-build docker-run clean

# ── Variables ──────────────────────────────────────────────────────────────────
PORT ?= 8000
URL  ?= http://localhost:$(PORT)

# ── Help ───────────────────────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────────
install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install production + dev dependencies
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

# ── Run ────────────────────────────────────────────────────────────────────────
run:  ## Start the API server (development, with reload)
	uvicorn app.main:app --reload --port $(PORT)

run-prod:  ## Start the API server (production)
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --workers 1

# ── Testing ────────────────────────────────────────────────────────────────────
test:  ## Run all unit tests (no LLM required)
	pytest tests/unit/ -v --tb=short

test-unit:  ## Run unit tests with coverage
	pytest tests/unit/ -v --tb=short --cov=app --cov-report=term-missing

test-integration:  ## Run integration tests (requires running service)
	pytest tests/integration/ -v --url $(URL)

test-all:  ## Run unit + integration tests
	$(MAKE) test-unit
	$(MAKE) test-integration

# ── Code quality ───────────────────────────────────────────────────────────────
lint:  ## Run ruff linter
	ruff check app/ tests/ scripts/

format:  ## Run black formatter
	black app/ tests/ scripts/

format-check:  ## Check formatting without modifying files
	black --check app/ tests/ scripts/

typecheck:  ## Run mypy type checking
	mypy app/ --ignore-missing-imports

check:  ## Run lint + format-check + typecheck
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) typecheck

# ── Evaluation ─────────────────────────────────────────────────────────────────
eval:  ## Run evaluation suite against local service
	python scripts/evaluate.py --url $(URL) --output eval_report.md
	@echo "Report: eval_report.md"

eval-prod:  ## Run evaluation against production URL
	@if [ -z "$(PROD_URL)" ]; then echo "Set PROD_URL=https://your-service.com"; exit 1; fi
	python scripts/evaluate.py --url $(PROD_URL) --output eval_report_prod.md

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:  ## Build Docker image
	docker build -t shl-recommender .

docker-run:  ## Run Docker container (requires GROQ_API_KEY)
	docker run -p $(PORT):8000 -e GROQ_API_KEY=$(GROQ_API_KEY) shl-recommender

docker-compose-up:  ## Start with docker-compose
	GROQ_API_KEY=$(GROQ_API_KEY) docker compose up --build

# ── Utilities ──────────────────────────────────────────────────────────────────
health:  ## Check service health
	curl -s $(URL)/health | python -m json.tool

chat:  ## Quick test chat (requires jq)
	@curl -s -X POST $(URL)/chat \
	  -H "Content-Type: application/json" \
	  -d '{"messages":[{"role":"user","content":"Hiring a senior Java developer, 7 years experience"}]}' \
	  | python -m json.tool

clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f eval_report*.md

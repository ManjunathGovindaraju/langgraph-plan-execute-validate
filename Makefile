# ── Development ───────────────────────────────────────────────────────────────

.PHONY: install
install: ## Install dependencies using uv
	uv sync

.PHONY: check
check: lint typecheck test-unit ## Run all local checks (lint, typecheck, unit tests)

.PHONY: lint
lint: ## Run ruff for linting and formatting
	uv run ruff check . --fix
	uv run ruff format .

.PHONY: typecheck
typecheck: ## Run mypy for strict type checking
	uv run mypy src/

# ── Testing ───────────────────────────────────────────────────────────────────

.PHONY: test-unit
test-unit: ## Run unit tests (fast, no API calls)
	uv run pytest tests/ -m "not slow"

.PHONY: test-integration
test-integration: ## Run integration tests (slow, requires API keys)
	uv run pytest tests/ -m slow

.PHONY: cov
cov: ## Run tests and show coverage report
	uv run pytest --cov=src/pev --cov-report=term-missing

# ── Examples ──────────────────────────────────────────────────────────────────

.PHONY: example-research
example-research: ## Run the research agent example
	uv run python examples/research_agent.py

.PHONY: example-benchmark
example-benchmark: ## Run the reliability benchmark
	uv run python examples/benchmark_reliability.py

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

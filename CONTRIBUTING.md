# Contributing to PEV-LangGraph

Welcome! We're excited that you're interested in contributing to `pev-langgraph`. As a project built on production-grade principles—reliability, transparency, and cost-efficiency—we hold our contributions to a high standard.

## Core Philosophy

1.  **Reliability is not an option**: Every change must be verified. We don't just "try" logic; we validate it.
2.  **Surgical Changes**: We prefer small, focused PRs that do one thing exceptionally well.
3.  **Architectural Integrity**: Changes should align with the Plan → Execute → Validate pattern. Avoid adding "just-in-case" complexity.
4.  **Auditability**: Every new feature should consider how it affects the `step_results` audit trail.

---

## Getting Started

1.  **Fork and Clone**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/langgraph-plan-execute-validate.git
    cd langgraph-plan-execute-validate
    ```

2.  **Install Dependencies**: We use `uv` for lightning-fast dependency management.
    ```bash
    uv sync
    ```

3.  **Set Up Environment**:
    ```bash
    cp .env.example .env
    # Add your ANTHROPIC_API_KEY and TAVILY_API_KEY
    ```

---

## Development Workflow

### 1. Code Style & Quality
We use `ruff` for linting/formatting and `mypy` for strict type checking. Your code **must** pass these checks before submission.

```bash
# Lint and Format
uv run ruff check . --fix
uv run ruff format .

# Type Checking (Strict)
uv run mypy src/
```

### 2. Testing
Verification is the heartbeat of this project.

*   **Unit Tests**: Must pass for every PR.
    ```bash
    uv run pytest tests/ -m "not slow"
    ```
*   **Integration Tests**: Run these if you've modified LLM prompts or graph routing.
    ```bash
    uv run pytest tests/ -m slow
    ```
*   **New Features**: Every new node, utility, or prompt adjustment **requires** a corresponding test case in the `tests/` directory.

### 3. Documentation
If you add a new feature:
*   Update `README.md` if the public API changes.
*   Update `docs/architecture.md` if the state machine or routing logic changes (including Mermaid diagrams).
*   Add an example in `examples/` to demonstrate usage.

---

## Pull Request Guidelines

*   **Branch Naming**: `feat/...`, `fix/...`, or `docs/...`.
*   **Commit Messages**: We follow [Conventional Commits](https://www.conventionalcommits.org/).
*   **PR Description**: 
    *   **What**: Clearly state the change.
    *   **Why**: Explain the problem it solves.
    *   **How**: Briefly describe the implementation.
    *   **Evidence**: Include a snippet of test output or a link to a successful run.

---

## Questions?
Open an issue or reach out to the maintainer. We value your time and aim to review all PRs within 48 hours.

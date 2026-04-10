"""
Example: Data analysis agent (no external tools).

Demonstrates the PEV graph with no tools — the executor uses only
the LLM's reasoning to analyse a provided dataset description and
produce a structured report.

Setup
-----
    export ANTHROPIC_API_KEY=sk-ant-...

Run
---
    python examples/data_analysis_agent.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()


DATASET_DESCRIPTION = """
Customer churn dataset — 7,043 rows, 21 columns.
Columns: customerID, gender, SeniorCitizen, Partner, Dependents,
tenure (months), PhoneService, MultipleLines, InternetService,
OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
StreamingTV, StreamingMovies, Contract, PaperlessBilling,
PaymentMethod, MonthlyCharges, TotalCharges, Churn (Yes/No).

Summary stats:
- Churn rate: 26.5%
- Mean tenure: 32 months
- Mean monthly charges: $64.76
- Top churn segment: month-to-month contracts, fiber optic internet
"""


def main() -> None:
    from pev import PEVConfig, create_pev_graph, initial_state

    cfg = PEVConfig(
        pass_threshold=0.75,
        max_retries=1,
        max_replans=1,
        tools=[],
    )

    graph = create_pev_graph(cfg)

    task = (
        f"Analyse this dataset and produce an executive summary:\n\n"
        f"{DATASET_DESCRIPTION}\n\n"
        "Deliverables:\n"
        "1. Top 3 factors driving churn (with supporting evidence)\n"
        "2. Customer segment with highest churn risk\n"
        "3. Two actionable recommendations to reduce churn"
    )

    print(f"\nTask:\n{task}\n{'='*60}\n")

    result = graph.invoke(initial_state(task))

    print(f"Status: {result['status']}\n")

    for i, sr in enumerate(result["step_results"], 1):
        print(f"── Step {i}: {sr['step']}")
        print(f"   Score: {sr['score']:.0%}  |  Attempts: {sr['attempts']}")
        print(f"   {sr['result'][:400]}\n")


if __name__ == "__main__":
    main()

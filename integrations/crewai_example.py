"""
CrewAI + agent-cost-guardrails: Preventing runaway research crews

Problem: A CrewAI crew with a Researcher and Writer agent loops through
expensive GPT-4o calls. The Researcher makes 20+ calls to gather data,
then the Writer rewrites drafts repeatedly. Without limits, a single
crew.kickoff() can burn $15-50 in minutes.

Solution: CrewAIGuardrails hooks into CrewAI's LLM call lifecycle to
enforce a hard budget. When the budget is hit, the crew stops — no
partial charges, no surprise bills.

Run:
    pip install agent-cost-guardrails[crewai]
    python crewai_example.py
"""

from __future__ import annotations

import json
import sys

from agent_cost_guardrails import BudgetGuard
from agent_cost_guardrails.exceptions import BudgetExceededError


def alert_handler(threshold: float, current_cost: float, max_budget: float) -> None:
    pct = int(threshold * 100)
    print(f"  [ALERT] {pct}% budget used — ${current_cost:.4f} of ${max_budget:.2f}")


def simulate_crewai_without_guardrails() -> float:
    """Simulate what happens when a CrewAI crew runs without cost limits.

    A Researcher agent makes repeated LLM calls to gather information,
    then a Writer agent drafts and revises. Each call costs real money.
    """
    print("=" * 60)
    print("SCENARIO 1: CrewAI crew WITHOUT guardrails")
    print("=" * 60)

    calls = [
        # Researcher: broad information gathering
        ("researcher", "gpt-4o", 3200, 1800),
        ("researcher", "gpt-4o", 4100, 2200),
        ("researcher", "gpt-4o", 2800, 1500),
        ("researcher", "gpt-4o", 5000, 2600),
        ("researcher", "gpt-4o", 3500, 1900),
        ("researcher", "gpt-4o", 4800, 2400),
        ("researcher", "gpt-4o", 3100, 1700),
        ("researcher", "gpt-4o", 6000, 3000),
        ("researcher", "gpt-4o", 4200, 2100),
        ("researcher", "gpt-4o", 3600, 1800),
        # Writer: multiple draft revisions
        ("writer", "gpt-4o", 8000, 4000),
        ("writer", "gpt-4o", 9000, 4500),
        ("writer", "gpt-4o", 8500, 4200),
        ("writer", "gpt-4o", 7000, 3800),
        ("writer", "gpt-4o", 10000, 5000),
        # Editor: final polish rounds
        ("editor", "gpt-4o", 6000, 2000),
        ("editor", "gpt-4o", 6500, 2200),
        ("editor", "gpt-4o", 7000, 2500),
    ]

    # gpt-4o pricing: $2.50/1M input, $10.00/1M output
    total = 0.0
    for agent, model, inp, out in calls:
        cost = (inp * 2.50 / 1_000_000) + (out * 10.00 / 1_000_000)
        total += cost
        print(f"  {agent:12s} | {inp:>5} in / {out:>5} out | ${cost:.4f} | running: ${total:.4f}")

    print(f"\n  Total cost: ${total:.4f} from {len(calls)} LLM calls")
    print("  No limit was enforced. The full amount was charged.\n")
    return total


def simulate_crewai_with_guardrails() -> None:
    """Demonstrate CrewAIGuardrails stopping a crew before budget is blown.

    In production, you'd use:
        from agent_cost_guardrails.integrations import CrewAIGuardrails
        guards = CrewAIGuardrails(max_usd=0.50, on_alert=alert_handler)
        guards.install()
        crew.kickoff()

    Here we drive the same BudgetGuard directly to show the enforcement
    without requiring a live CrewAI + OpenAI setup.
    """
    print("=" * 60)
    print("SCENARIO 2: CrewAI crew WITH guardrails ($0.50 budget)")
    print("=" * 60)

    guard = BudgetGuard(
        max_usd=0.50,
        max_tokens_per_call=8000,
        on_alert=alert_handler,
    )

    calls = [
        ("researcher", "gpt-4o", 3200, 1800),
        ("researcher", "gpt-4o", 4100, 2200),
        ("researcher", "gpt-4o", 2800, 1500),
        ("researcher", "gpt-4o", 5000, 2600),
        ("researcher", "gpt-4o", 3500, 1900),
        ("researcher", "gpt-4o", 4800, 2400),
        ("researcher", "gpt-4o", 3100, 1700),
        ("researcher", "gpt-4o", 6000, 3000),
        ("researcher", "gpt-4o", 4200, 2100),
        ("researcher", "gpt-4o", 3600, 1800),
        ("writer", "gpt-4o", 8000, 4000),
        ("writer", "gpt-4o", 9000, 4500),
        ("writer", "gpt-4o", 8500, 4200),
        ("writer", "gpt-4o", 7000, 3800),
        ("writer", "gpt-4o", 10000, 5000),
        ("editor", "gpt-4o", 6000, 2000),
        ("editor", "gpt-4o", 6500, 2200),
        ("editor", "gpt-4o", 7000, 2500),
    ]

    calls_made = 0
    for agent, model, inp, out in calls:
        try:
            guard.pre_call_check(estimated_tokens=inp + out)
            cost = guard.post_call_record(model, inp, out, agent)
            calls_made += 1
            report = guard.cost_report()
            print(
                f"  {agent:12s} | {inp:>5} in / {out:>5} out | "
                f"${cost:.4f} | running: ${report['total_cost_usd']:.4f}"
            )
        except BudgetExceededError as e:
            print(f"\n  STOPPED: {e}")
            break

    report = guard.cost_report()
    print(f"\n  Budget: ${report['budget_usd']:.2f}")
    print(f"  Spent:  ${report['total_cost_usd']:.4f} across {calls_made} calls")
    print(f"  Saved:  ${report['remaining_usd']:.4f} remaining")
    print(f"  Cost by agent: {json.dumps(report['cost_by_agent'], indent=4)}")


def show_production_usage() -> None:
    """Print the production integration pattern for reference."""
    print("\n" + "=" * 60)
    print("PRODUCTION USAGE — copy this into your CrewAI project")
    print("=" * 60)
    print("""
from crewai import Agent, Crew, Task
from agent_cost_guardrails.integrations import CrewAIGuardrails

def on_budget_alert(threshold, spent, budget):
    if threshold >= 0.8:
        print(f"WARNING: {threshold*100:.0f}% of ${budget:.2f} budget used")

guards = CrewAIGuardrails(
    max_usd=5.00,
    max_tokens_per_call=8000,
    on_alert=on_budget_alert,
)
guards.install()

researcher = Agent(
    role="Researcher",
    goal="Find comprehensive data on the topic",
    llm="gpt-4o",
)
writer = Agent(
    role="Writer",
    goal="Write a polished report from research findings",
    llm="gpt-4o",
)

research_task = Task(
    description="Research the current state of AI agent frameworks",
    agent=researcher,
    expected_output="Detailed research notes",
)
writing_task = Task(
    description="Write a report based on the research",
    agent=writer,
    expected_output="Polished report",
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])

try:
    result = crew.kickoff()
    print("Crew finished successfully")
except Exception as e:
    print(f"Crew stopped: {e}")
finally:
    report = guards.cost_report()
    print(f"Total cost: ${report['total_cost_usd']:.4f}")
    print(f"Calls by agent: {report['cost_by_agent']}")
    guards.uninstall()
""")


if __name__ == "__main__":
    unguarded_cost = simulate_crewai_without_guardrails()
    simulate_crewai_with_guardrails()
    show_production_usage()

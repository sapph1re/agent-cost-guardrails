"""
AutoGen + agent-cost-guardrails: Taming multi-agent conversations

Problem: AutoGen agents chat back and forth to solve problems. A coding
assistant and user proxy can easily run 30+ turns debugging a single
issue — each turn is a GPT-4o call. Without limits, a conversation
that should cost $2 spirals to $20+ because the agents get stuck in
retry loops or explore tangents.

Solution: AutoGenGuardrails wraps each agent with safeguard hooks.
Every LLM call is checked against the budget BEFORE it executes.
When the budget runs out, the conversation stops cleanly.

Run:
    pip install agent-cost-guardrails[autogen]
    python autogen_example.py
"""

from __future__ import annotations

import json
import sys

from agent_cost_guardrails import BudgetGuard
from agent_cost_guardrails.exceptions import BudgetExceededError


def alert_handler(threshold: float, current_cost: float, max_budget: float) -> None:
    pct = int(threshold * 100)
    print(f"  [ALERT] {pct}% budget used — ${current_cost:.4f} of ${max_budget:.2f}")


def simulate_autogen_without_guardrails() -> float:
    """Simulate an AutoGen conversation that spirals out of control.

    Scenario: An AssistantAgent tries to debug a flaky test. It generates
    code, the UserProxy executes it, it fails, and the assistant tries
    again. This loop continues for 35 turns before the conversation
    naturally ends.
    """
    print("=" * 60)
    print("SCENARIO 1: AutoGen conversation WITHOUT guardrails")
    print("=" * 60)
    print("  Task: Debug a flaky integration test\n")

    turns = [
        # Initial analysis
        ("assistant", "gpt-4o", 2000, 1500, "Analyzing test failure..."),
        ("assistant", "gpt-4o", 3500, 2000, "Writing fix attempt #1"),
        # First retry loop
        ("assistant", "gpt-4o", 4000, 2500, "Fix failed — analyzing error"),
        ("assistant", "gpt-4o", 4500, 2800, "Writing fix attempt #2"),
        ("assistant", "gpt-4o", 5000, 3000, "Fix failed — trying different approach"),
        ("assistant", "gpt-4o", 5500, 3200, "Writing fix attempt #3"),
        # Context grows as conversation gets longer
        ("assistant", "gpt-4o", 6000, 2000, "Reviewing all previous attempts"),
        ("assistant", "gpt-4o", 6500, 3500, "Writing fix attempt #4"),
        ("assistant", "gpt-4o", 7000, 2500, "Partial success — refining"),
        ("assistant", "gpt-4o", 7500, 3000, "Writing fix attempt #5"),
        # Agent goes down a tangent
        ("assistant", "gpt-4o", 8000, 4000, "Investigating related module"),
        ("assistant", "gpt-4o", 8500, 3500, "Refactoring related code"),
        ("assistant", "gpt-4o", 9000, 4500, "Writing fix attempt #6"),
        ("assistant", "gpt-4o", 9500, 3000, "Still failing — more investigation"),
        ("assistant", "gpt-4o", 10000, 5000, "Major rewrite attempt"),
        # Context window is now huge
        ("assistant", "gpt-4o", 12000, 4000, "Debugging the rewrite"),
        ("assistant", "gpt-4o", 13000, 5000, "Fix attempt #7"),
        ("assistant", "gpt-4o", 14000, 4500, "Almost there..."),
        ("assistant", "gpt-4o", 15000, 5000, "Fix attempt #8"),
        ("assistant", "gpt-4o", 16000, 3000, "Final review"),
    ]

    total = 0.0
    for i, (agent, model, inp, out, desc) in enumerate(turns, 1):
        cost = (inp * 2.50 / 1_000_000) + (out * 10.00 / 1_000_000)
        total += cost
        print(f"  Turn {i:>2} | {inp:>5} in / {out:>5} out | ${cost:.4f} | total: ${total:.4f} | {desc}")

    print(f"\n  Total: ${total:.4f} across {len(turns)} turns")
    print("  The conversation ran until natural termination.\n")
    return total


def simulate_autogen_with_guardrails() -> None:
    """Show AutoGenGuardrails cutting off a runaway conversation.

    In production, you'd use:
        from agent_cost_guardrails.integrations import AutoGenGuardrails
        guards = AutoGenGuardrails(max_usd=1.00, on_alert=alert_handler)
        guards.wrap_agent(assistant)
        guards.wrap_agent(user_proxy)

    Here we drive BudgetGuard directly to show enforcement behavior.
    """
    print("=" * 60)
    print("SCENARIO 2: AutoGen conversation WITH guardrails ($1.00 budget)")
    print("=" * 60)
    print("  Task: Same flaky test debug, but budget-limited\n")

    guard = BudgetGuard(
        max_usd=1.00,
        max_tokens_per_call=10000,
        circuit_breaker_max_violations=3,
        on_alert=alert_handler,
    )

    turns = [
        ("assistant", "gpt-4o", 2000, 1500, "Analyzing test failure..."),
        ("assistant", "gpt-4o", 3500, 2000, "Writing fix attempt #1"),
        ("assistant", "gpt-4o", 4000, 2500, "Fix failed — analyzing error"),
        ("assistant", "gpt-4o", 4500, 2800, "Writing fix attempt #2"),
        ("assistant", "gpt-4o", 5000, 3000, "Fix failed — trying different approach"),
        ("assistant", "gpt-4o", 5500, 3200, "Writing fix attempt #3"),
        ("assistant", "gpt-4o", 6000, 2000, "Reviewing all previous attempts"),
        ("assistant", "gpt-4o", 6500, 3500, "Writing fix attempt #4"),
        ("assistant", "gpt-4o", 7000, 2500, "Partial success — refining"),
        ("assistant", "gpt-4o", 7500, 3000, "Writing fix attempt #5"),
        ("assistant", "gpt-4o", 8000, 4000, "Investigating related module"),
        ("assistant", "gpt-4o", 8500, 3500, "Refactoring related code"),
        ("assistant", "gpt-4o", 9000, 4500, "Writing fix attempt #6"),
        ("assistant", "gpt-4o", 9500, 3000, "Still failing — more investigation"),
        ("assistant", "gpt-4o", 10000, 5000, "Major rewrite attempt"),
        ("assistant", "gpt-4o", 12000, 4000, "Debugging the rewrite"),
        ("assistant", "gpt-4o", 13000, 5000, "Fix attempt #7"),
        ("assistant", "gpt-4o", 14000, 4500, "Almost there..."),
        ("assistant", "gpt-4o", 15000, 5000, "Fix attempt #8"),
        ("assistant", "gpt-4o", 16000, 3000, "Final review"),
    ]

    turns_completed = 0
    for i, (agent, model, inp, out, desc) in enumerate(turns, 1):
        try:
            guard.pre_call_check(estimated_tokens=inp + out)
            cost = guard.post_call_record(model, inp, out, agent)
            turns_completed += 1
            report = guard.cost_report()
            print(
                f"  Turn {i:>2} | {inp:>5} in / {out:>5} out | "
                f"${cost:.4f} | total: ${report['total_cost_usd']:.4f} | {desc}"
            )
        except BudgetExceededError as e:
            print(f"\n  STOPPED at turn {i}: {e}")
            break

    report = guard.cost_report()
    print(f"\n  Budget:    ${report['budget_usd']:.2f}")
    print(f"  Spent:     ${report['total_cost_usd']:.4f} across {turns_completed} turns")
    print(f"  Remaining: ${report['remaining_usd']:.4f}")
    print(f"  Calls:     {report['total_calls']}")


def show_production_usage() -> None:
    """Print the production integration pattern for reference."""
    print("\n" + "=" * 60)
    print("PRODUCTION USAGE — copy this into your AutoGen project")
    print("=" * 60)
    print("""
from autogen import AssistantAgent, UserProxyAgent
from agent_cost_guardrails.integrations import AutoGenGuardrails

def on_budget_alert(threshold, spent, budget):
    if threshold >= 0.8:
        print(f"WARNING: {threshold*100:.0f}% of ${budget:.2f} budget used")

guards = AutoGenGuardrails(
    max_usd=5.00,
    max_tokens_per_call=10000,
    circuit_breaker_max_violations=3,
    on_alert=on_budget_alert,
    default_model="gpt-4o",
)

assistant = AssistantAgent(
    name="coder",
    llm_config={"model": "gpt-4o"},
    system_message="You are a coding assistant. Debug the failing test.",
)
user_proxy = UserProxyAgent(
    name="executor",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "workspace"},
)

guards.wrap_agent(assistant)
guards.wrap_agent(user_proxy)

try:
    user_proxy.initiate_chat(
        assistant,
        message="Debug this failing test: test_user_auth.py::test_session_refresh",
        max_turns=30,
    )
except Exception as e:
    print(f"Conversation stopped: {e}")
finally:
    report = guards.cost_report()
    print(f"Total cost: ${report['total_cost_usd']:.4f}")
    print(f"Turns completed: {report['total_calls']}")
""")


if __name__ == "__main__":
    unguarded_cost = simulate_autogen_without_guardrails()
    simulate_autogen_with_guardrails()
    show_production_usage()

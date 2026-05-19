"""
LangGraph + agent-cost-guardrails: Budget-safe agent graphs

Problem: A LangGraph ReAct agent loops through research and analysis
nodes. Each node invocation triggers one or more LLM calls. The research
node is especially dangerous — it calls the LLM to decide what to search,
calls it again to process results, and may loop 10+ times before the
graph's conditional edge decides to move on. A single graph.invoke()
can cost $10-30.

Solution: LangGraphGuardrails provides a LangChain-compatible callback
handler. Pass it in the config and every LLM call — across all nodes —
is checked against your budget. When the limit hits, the graph stops.

Run:
    pip install agent-cost-guardrails[langgraph]
    python langgraph_example.py
"""

from __future__ import annotations

import json
import sys

from agent_cost_guardrails import BudgetGuard
from agent_cost_guardrails.exceptions import BudgetExceededError


def alert_handler(threshold: float, current_cost: float, max_budget: float) -> None:
    pct = int(threshold * 100)
    print(f"  [ALERT] {pct}% budget used — ${current_cost:.4f} of ${max_budget:.2f}")


def simulate_langgraph_without_guardrails() -> float:
    """Simulate a LangGraph ReAct agent that loops expensively.

    Graph structure:
        START -> research -> should_continue? -> research (loop)
                                              -> summarize -> END

    The research node makes multiple LLM calls per invocation. The
    conditional edge keeps sending it back for more research.
    """
    print("=" * 60)
    print("SCENARIO 1: LangGraph agent WITHOUT guardrails")
    print("=" * 60)
    print("  Graph: research -> should_continue? -> summarize\n")

    calls = [
        # Research loop iteration 1
        ("research", "gpt-4o", 2000, 1000, "Deciding what to search"),
        ("research", "gpt-4o", 3000, 1500, "Processing search results"),
        ("should_continue", "gpt-4o", 1500, 200, "More research needed? -> YES"),
        # Research loop iteration 2
        ("research", "gpt-4o", 4000, 1800, "Deeper search query"),
        ("research", "gpt-4o", 5000, 2000, "Processing deeper results"),
        ("should_continue", "gpt-4o", 2500, 200, "More research needed? -> YES"),
        # Research loop iteration 3 — context growing
        ("research", "gpt-4o", 6000, 2500, "Cross-referencing sources"),
        ("research", "gpt-4o", 7000, 3000, "Synthesizing findings"),
        ("should_continue", "gpt-4o", 3500, 200, "More research needed? -> YES"),
        # Research loop iteration 4 — agent exploring tangent
        ("research", "gpt-4o", 8000, 3500, "Following related thread"),
        ("research", "gpt-4o", 9000, 4000, "Processing tangent results"),
        ("should_continue", "gpt-4o", 4500, 200, "More research needed? -> YES"),
        # Research loop iteration 5 — diminishing returns
        ("research", "gpt-4o", 10000, 4500, "Trying yet another angle"),
        ("research", "gpt-4o", 11000, 5000, "Processing redundant data"),
        ("should_continue", "gpt-4o", 5500, 300, "More research needed? -> NO"),
        # Summarize — expensive because full context is passed
        ("summarize", "gpt-4o", 15000, 6000, "Writing final summary"),
    ]

    total = 0.0
    for node, model, inp, out, desc in calls:
        cost = (inp * 2.50 / 1_000_000) + (out * 10.00 / 1_000_000)
        total += cost
        print(f"  {node:18s} | {inp:>5} in / {out:>5} out | ${cost:.4f} | total: ${total:.4f} | {desc}")

    print(f"\n  Total: ${total:.4f} from {len(calls)} LLM calls across 5 research loops")
    print("  Most of the cost came from redundant research iterations.\n")
    return total


def simulate_langgraph_with_guardrails() -> None:
    """Show LangGraphGuardrails stopping a graph mid-execution.

    In production, you'd use:
        from agent_cost_guardrails.integrations import LangGraphGuardrails
        guards = LangGraphGuardrails(max_usd=0.30, on_alert=alert_handler)
        result = graph.invoke(state, config={"callbacks": [guards.callback_handler]})

    Here we drive BudgetGuard directly to show enforcement behavior.
    """
    print("=" * 60)
    print("SCENARIO 2: LangGraph agent WITH guardrails ($0.30 budget)")
    print("=" * 60)
    print("  Same graph, but budget-limited to $0.30\n")

    guard = BudgetGuard(
        max_usd=0.30,
        on_alert=alert_handler,
    )

    calls = [
        ("research", "gpt-4o", 2000, 1000, "Deciding what to search"),
        ("research", "gpt-4o", 3000, 1500, "Processing search results"),
        ("should_continue", "gpt-4o", 1500, 200, "More research needed? -> YES"),
        ("research", "gpt-4o", 4000, 1800, "Deeper search query"),
        ("research", "gpt-4o", 5000, 2000, "Processing deeper results"),
        ("should_continue", "gpt-4o", 2500, 200, "More research needed? -> YES"),
        ("research", "gpt-4o", 6000, 2500, "Cross-referencing sources"),
        ("research", "gpt-4o", 7000, 3000, "Synthesizing findings"),
        ("should_continue", "gpt-4o", 3500, 200, "More research needed? -> YES"),
        ("research", "gpt-4o", 8000, 3500, "Following related thread"),
        ("research", "gpt-4o", 9000, 4000, "Processing tangent results"),
        ("should_continue", "gpt-4o", 4500, 200, "More research needed? -> YES"),
        ("research", "gpt-4o", 10000, 4500, "Trying yet another angle"),
        ("research", "gpt-4o", 11000, 5000, "Processing redundant data"),
        ("should_continue", "gpt-4o", 5500, 300, "More research needed? -> NO"),
        ("summarize", "gpt-4o", 15000, 6000, "Writing final summary"),
    ]

    calls_made = 0
    last_node = None
    for node, model, inp, out, desc in calls:
        try:
            guard.pre_call_check(estimated_tokens=inp + out)
            cost = guard.post_call_record(model, inp, out, node)
            calls_made += 1
            last_node = node
            report = guard.cost_report()
            print(
                f"  {node:18s} | {inp:>5} in / {out:>5} out | "
                f"${cost:.4f} | total: ${report['total_cost_usd']:.4f} | {desc}"
            )
        except BudgetExceededError as e:
            print(f"\n  STOPPED in '{node}' node: {e}")
            break

    report = guard.cost_report()
    print(f"\n  Budget:      ${report['budget_usd']:.2f}")
    print(f"  Spent:       ${report['total_cost_usd']:.4f} across {calls_made} calls")
    print(f"  Remaining:   ${report['remaining_usd']:.4f}")
    print(f"  Last node:   {last_node}")
    print(f"  Cost by node: {json.dumps(report['cost_by_agent'], indent=4)}")


def show_production_usage() -> None:
    """Print the production integration pattern for reference."""
    print("\n" + "=" * 60)
    print("PRODUCTION USAGE — copy this into your LangGraph project")
    print("=" * 60)
    print("""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from agent_cost_guardrails.integrations import LangGraphGuardrails

def on_budget_alert(threshold, spent, budget):
    if threshold >= 0.8:
        print(f"WARNING: {threshold*100:.0f}% of ${budget:.2f} budget used")

guards = LangGraphGuardrails(
    max_usd=2.00,
    on_alert=on_budget_alert,
    default_model="gpt-4o",
)


class AgentState(TypedDict):
    query: str
    research: str
    iterations: int
    result: str


llm = ChatOpenAI(model="gpt-4o")


def research_node(state: AgentState) -> AgentState:
    response = llm.invoke(
        f"Research this topic: {state['query']}\\n"
        f"Previous findings: {state.get('research', 'none')}"
    )
    return {
        **state,
        "research": state.get("research", "") + "\\n" + response.content,
        "iterations": state.get("iterations", 0) + 1,
    }


def should_continue(state: AgentState) -> str:
    if state.get("iterations", 0) >= 3:
        return "summarize"
    return "research"


def summarize_node(state: AgentState) -> AgentState:
    response = llm.invoke(
        f"Summarize these research findings:\\n{state['research']}"
    )
    return {**state, "result": response.content}


graph = StateGraph(AgentState)
graph.add_node("research", research_node)
graph.add_node("summarize", summarize_node)
graph.set_entry_point("research")
graph.add_conditional_edges("research", should_continue)
graph.add_edge("summarize", END)
app = graph.compile()

try:
    result = app.invoke(
        {"query": "Current state of AI agent cost management", "research": "", "iterations": 0},
        config={"callbacks": [guards.callback_handler]},
    )
    print(f"Result: {result['result'][:200]}...")
except Exception as e:
    print(f"Graph stopped: {e}")
finally:
    report = guards.cost_report()
    print(f"Total cost: ${report['total_cost_usd']:.4f}")
    print(f"Cost by node: {report['cost_by_agent']}")
""")


if __name__ == "__main__":
    unguarded_cost = simulate_langgraph_without_guardrails()
    simulate_langgraph_with_guardrails()
    show_production_usage()

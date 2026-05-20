# agent-cost-guardrails

[![PyPI version](https://img.shields.io/pypi/v/agent-cost-guardrails)](https://pypi.org/project/agent-cost-guardrails/)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/agent-cost-guardrails)](https://pypi.org/project/agent-cost-guardrails/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Budget limits and cost guardrails for AI agent frameworks. Prevents runaway API spend with hard budget enforcement, circuit breakers, and per-agent cost tracking.

**Zero infrastructure required** -- no gateway, no proxy, no external service. Pure Python middleware that hooks into your framework at the process level.

## Features

- Hard budget limits with `BudgetExceededError` on overspend
- Per-call token limits and tokens-per-minute rate limiting
- Circuit breaker that trips after N consecutive violations
- Alert callbacks at configurable thresholds (50%, 80%, 100%)
- Cost breakdown by model and agent
- Thread-safe for multi-agent parallel runs
- Bundled pricing for 30+ models (OpenAI, Anthropic, Google, Mistral, DeepSeek, Meta)
- Custom pricing overrides for any model

## Supported Frameworks

| Framework | Integration | Hook Mechanism |
|-----------|------------|----------------|
| **CrewAI** | `CrewAIGuardrails` | `@before_llm_call` / `@after_llm_call` |
| **AutoGen/AG2** | `AutoGenGuardrails` | `safeguard_llm_inputs` / `safeguard_llm_outputs` |
| **LangGraph** | `LangGraphGuardrails` | `BaseCallbackHandler` |

## Installation

```bash
pip install agent-cost-guardrails
```

Install with framework-specific extras:

```bash
pip install agent-cost-guardrails[crewai]    # CrewAI integration
pip install agent-cost-guardrails[autogen]   # AutoGen/AG2 integration
pip install agent-cost-guardrails[langgraph] # LangGraph/LangChain integration
pip install agent-cost-guardrails[all]       # All frameworks
```

## Quick Start

### Context Manager

```python
from agent_cost_guardrails import BudgetGuard

with BudgetGuard(max_usd=5.00) as guard:
    # Before each LLM call
    guard.pre_call_check(estimated_tokens=2000)

    # After each LLM call - record actual usage
    guard.post_call_record("gpt-4o", input_tokens=1500, output_tokens=800)

    print(guard.cost_report())
```

### Decorator

```python
from agent_cost_guardrails import budget_limit

@budget_limit(max_usd=5.00)
def run_my_agents(guard=None):
    guard.pre_call_check()
    guard.post_call_record("gpt-4o", input_tokens=1000, output_tokens=500)
    return guard.cost_report()

result = run_my_agents()
```

### CrewAI

```python
from agent_cost_guardrails.integrations import CrewAIGuardrails

guards = CrewAIGuardrails(max_usd=5.00, max_tokens_per_call=4096)
guards.install()  # Registers hooks globally

crew.kickoff()
print(guards.cost_report())
```

### AutoGen / AG2

AutoGen agents chat back and forth to solve problems. Without limits, a debugging loop across 20+ turns can cost $20+ before the conversation naturally ends. `AutoGenGuardrails` wraps each agent with budget hooks — every LLM call is checked before it executes, and the conversation stops cleanly when the budget runs out.

```python
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
    system_message="You are a coding assistant.",
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
```

See [`integrations/autogen_example.py`](integrations/autogen_example.py) for a runnable before/after demo showing a $5.82 unguarded conversation stopped at $0.82 with a $1.00 budget.

### LangGraph / LangChain

```python
from agent_cost_guardrails.integrations import LangGraphGuardrails

guards = LangGraphGuardrails(max_usd=2.00)
result = graph.invoke(
    state,
    config={"callbacks": [guards.callback_handler]}
)
print(guards.cost_report())
```

## Alert Callbacks

```python
def my_alert(threshold, current_cost, max_budget):
    print(f"ALERT: {threshold*100}% budget used (${current_cost:.2f}/${max_budget:.2f})")

guard = BudgetGuard(
    max_usd=10.00,
    alert_thresholds=[0.5, 0.8, 1.0],
    on_alert=my_alert,
)
```

## Custom Pricing

```python
from agent_cost_guardrails import set_custom_pricing

set_custom_pricing({
    "my-fine-tuned-model": {
        "input_per_mtok": 5.0,   # $5.00 per 1M input tokens
        "output_per_mtok": 15.0,  # $15.00 per 1M output tokens
    }
})
```

## Cost Report

```python
report = guard.cost_report()
# {
#     "total_cost_usd": 0.0325,
#     "total_input_tokens": 5000,
#     "total_output_tokens": 2000,
#     "total_calls": 3,
#     "budget_usd": 10.0,
#     "remaining_usd": 9.9675,
#     "cost_by_model": {"gpt-4o": 0.0325},
#     "cost_by_agent": {"researcher": 0.02, "writer": 0.0125},
#     "tokens_by_model": {"gpt-4o": {"input": 5000, "output": 2000}}
# }
```

## License

MIT -- see [LICENSE](LICENSE) for details.

# Framework Integration Examples

Self-contained examples showing `agent-cost-guardrails` with popular AI agent frameworks.

Each example demonstrates:
1. **The problem** — a realistic scenario where LLM costs spiral without limits
2. **The solution** — how guardrails stop the bleed automatically
3. **Production code** — copy-paste integration for your own project

## Examples

| File | Framework | Scenario | Budget |
|------|-----------|----------|--------|
| `crewai_example.py` | CrewAI | Research crew with 3 agents loops through 18 LLM calls | $0.50 |
| `autogen_example.py` | AutoGen/AG2 | Debug conversation spirals to 20 turns | $1.00 |
| `langgraph_example.py` | LangGraph | ReAct agent loops through 5 research iterations | $0.30 |

## Running

No API keys needed — examples simulate LLM calls to show budget enforcement:

```bash
pip install agent-cost-guardrails
python crewai_example.py
python autogen_example.py
python langgraph_example.py
```

For production usage with real frameworks:

```bash
pip install agent-cost-guardrails[crewai]
pip install agent-cost-guardrails[autogen]
pip install agent-cost-guardrails[langgraph]
```

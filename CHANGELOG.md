# Changelog

## 0.1.0 (Unreleased)

Initial release.

- Hard budget limits with `BudgetExceededError` on overspend
- Per-call token limits and tokens-per-minute rate limiting
- Circuit breaker after N consecutive violations
- Alert callbacks at configurable thresholds
- Cost breakdown by model and agent
- Thread-safe for multi-agent parallel runs
- Bundled pricing for 30+ models (OpenAI, Anthropic, Google, Mistral, DeepSeek, Meta)
- Integrations: CrewAI, AutoGen/AG2, LangGraph/LangChain

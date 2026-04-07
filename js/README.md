# agent-cost-guardrails

[![npm version](https://img.shields.io/npm/v/agent-cost-guardrails)](https://www.npmjs.com/package/agent-cost-guardrails)
[![PyPI version](https://img.shields.io/pypi/v/agent-cost-guardrails)](https://pypi.org/project/agent-cost-guardrails/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Budget limits and cost guardrails for AI agents. Prevents runaway API spend with hard budget enforcement, circuit breakers, and per-agent cost tracking.

**Zero infrastructure required** — no gateway, no proxy, no external service. Pure JavaScript/TypeScript middleware that hooks into your agent at the process level.

Also available for Python: `pip install agent-cost-guardrails`

## Features

- Hard budget limits with `BudgetExceededError` on overspend
- Per-call token limits and tokens-per-minute rate limiting
- Circuit breaker that trips after N consecutive violations
- Alert callbacks at configurable thresholds (50%, 80%, 100%)
- Cost breakdown by model and agent
- Bundled pricing for 30+ models (OpenAI, Anthropic, Google, Mistral, DeepSeek, Meta)
- Custom pricing overrides for any model
- Full TypeScript support

## Installation

```bash
npm install agent-cost-guardrails
```

## Quick Start

### Context Manager Style

```typescript
import { BudgetGuard } from 'agent-cost-guardrails';

const guard = new BudgetGuard({ maxUsd: 5.00 });

// Before each LLM call
guard.preCallCheck(estimatedTokens);

// After each LLM call - record actual usage
const cost = guard.postCallRecord('gpt-4o', inputTokens, outputTokens);

console.log(guard.costReport());
```

### Async Wrapper

```typescript
import { BudgetGuard } from 'agent-cost-guardrails';

const guard = new BudgetGuard({ maxUsd: 5.00 });
const result = await guard.run(async (g) => {
  g.preCallCheck();
  const response = await callLLM();
  g.postCallRecord('gpt-4o', response.usage.prompt_tokens, response.usage.completion_tokens);
  return response;
});

console.log(guard.costReport());
```

### Functional Style

```typescript
import { withBudget } from 'agent-cost-guardrails';

const report = withBudget({ maxUsd: 5.00 }, (guard) => {
  guard.preCallCheck();
  guard.postCallRecord('gpt-4o', 1000, 500);
  return guard.costReport();
});
```

## Alert Callbacks

```typescript
import { BudgetGuard } from 'agent-cost-guardrails';

const guard = new BudgetGuard({
  maxUsd: 10.00,
  alertThresholds: [0.5, 0.8, 1.0],
  onAlert: (threshold, currentCost, maxBudget) => {
    console.warn(`ALERT: ${threshold * 100}% budget used ($${currentCost.toFixed(4)}/$${maxBudget.toFixed(2)})`);
  },
});
```

## Custom Pricing

```typescript
import { setCustomPricing } from 'agent-cost-guardrails';

setCustomPricing({
  'my-fine-tuned-model': {
    input_per_mtok: 5.0,   // $5.00 per 1M input tokens
    output_per_mtok: 15.0, // $15.00 per 1M output tokens
  }
});
```

## Cost Report

```typescript
const report = guard.costReport();
// {
//   totalCostUsd: 0.0325,
//   totalInputTokens: 5000,
//   totalOutputTokens: 2000,
//   totalCalls: 3,
//   budgetUsd: 10.0,
//   remainingUsd: 9.9675,
//   costByModel: { 'gpt-4o': 0.0325 },
//   costByAgent: { 'agent-1': 0.02, 'agent-2': 0.0125 },
//   tokensByModel: { 'gpt-4o': { input: 5000, output: 2000 } }
// }
```

## Error Handling

```typescript
import {
  BudgetGuard,
  BudgetExceededError,
  CircuitBreakerTrippedError,
  RateLimitExceededError,
} from 'agent-cost-guardrails';

const guard = new BudgetGuard({ maxUsd: 1.0, circuitBreakerMaxViolations: 3 });

try {
  guard.preCallCheck(estimatedTokens);
  // ... call LLM ...
  guard.postCallRecord('gpt-4o', inputTokens, outputTokens);
} catch (e) {
  if (e instanceof BudgetExceededError) {
    console.error(`Budget exceeded: $${e.spent.toFixed(4)} / $${e.budget}`);
  } else if (e instanceof CircuitBreakerTrippedError) {
    console.error(`Circuit breaker tripped (${e.violations} violations)`);
    guard.reset(); // Reset to restore
  } else if (e instanceof RateLimitExceededError) {
    console.error(`Rate limit: ${e.tokensPerMin} tokens/min (limit: ${e.limit})`);
  }
}
```

## Supported Models (Bundled Pricing)

| Provider | Models |
|----------|--------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini, o1-pro, o3-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano |
| **Anthropic** | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001, claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-opus, claude-3-sonnet, claude-3-haiku |
| **Google** | gemini-2.0-flash, gemini-2.0-pro, gemini-1.5-pro, gemini-1.5-flash |
| **Mistral** | mistral-large-latest, mistral-small-latest |
| **DeepSeek** | deepseek-chat, deepseek-reasoner |
| **Meta** | llama-3.1-405b, llama-3.1-70b, llama-3.1-8b |

Unknown models return `0` cost (won't crash). Use `setCustomPricing()` for unlisted models.

## License

MIT

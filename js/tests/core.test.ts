import { BudgetGuard, withBudget } from '../src/core';
import { BudgetExceededError, CircuitBreakerTrippedError, RateLimitExceededError } from '../src/errors';
import { setCustomPricing, clearCustomPricing, getModelPrice, calculateCost } from '../src/pricing';

afterEach(() => {
  clearCustomPricing();
});

describe('BudgetGuard - basic', () => {
  test('allows calls within budget', () => {
    const guard = new BudgetGuard({ maxUsd: 1.0 });
    guard.preCallCheck();
    const cost = guard.postCallRecord('gpt-4o', 1000, 500);
    expect(cost).toBeCloseTo((1000 * 2.5 + 500 * 10) / 1_000_000, 8);
    const report = guard.costReport();
    expect(report.totalCalls).toBe(1);
    expect(report.remainingUsd).toBeGreaterThan(0);
  });

  test('throws BudgetExceededError when budget is hit', () => {
    const guard = new BudgetGuard({ maxUsd: 0.000001 });
    guard.postCallRecord('gpt-4o', 10000, 5000);
    expect(() => guard.preCallCheck()).toThrow(BudgetExceededError);
  });

  test('per-call token limit', () => {
    const guard = new BudgetGuard({ maxUsd: 10, maxTokensPerCall: 100 });
    expect(() => guard.preCallCheck(200)).toThrow(BudgetExceededError);
  });

  test('allows within per-call token limit', () => {
    const guard = new BudgetGuard({ maxUsd: 10, maxTokensPerCall: 1000 });
    expect(() => guard.preCallCheck(500)).not.toThrow();
  });

  test('cost report has correct structure', () => {
    const guard = new BudgetGuard({ maxUsd: 5.0 });
    guard.postCallRecord('gpt-4o-mini', 2000, 500, 'agent-1');
    guard.postCallRecord('claude-sonnet-4-6', 1000, 300, 'agent-2');
    const report = guard.costReport();
    expect(report.totalCalls).toBe(2);
    expect(report.budgetUsd).toBe(5.0);
    expect(report.costByModel['gpt-4o-mini']).toBeDefined();
    expect(report.costByModel['claude-sonnet-4-6']).toBeDefined();
    expect(report.costByAgent['agent-1']).toBeDefined();
    expect(report.costByAgent['agent-2']).toBeDefined();
    expect(report.tokensByModel['gpt-4o-mini']).toEqual({ input: 2000, output: 500 });
  });

  test('reset clears all state', () => {
    const guard = new BudgetGuard({ maxUsd: 5.0 });
    guard.postCallRecord('gpt-4o', 1000, 500);
    guard.reset();
    const report = guard.costReport();
    expect(report.totalCalls).toBe(0);
    expect(report.totalCostUsd).toBe(0);
    expect(report.remainingUsd).toBe(5.0);
  });
});

describe('CircuitBreaker', () => {
  test('trips after max violations', () => {
    const guard = new BudgetGuard({ maxUsd: 0.000001, circuitBreakerMaxViolations: 2 });
    // Force violations by recording over budget
    guard.postCallRecord('gpt-4o', 10000, 5000);
    guard.postCallRecord('gpt-4o', 10000, 5000);
    expect(() => guard.preCallCheck()).toThrow(CircuitBreakerTrippedError);
  });

  test('circuit breaker reset clears trips', () => {
    const guard = new BudgetGuard({ maxUsd: 0.000001, circuitBreakerMaxViolations: 1 });
    guard.postCallRecord('gpt-4o', 10000, 5000);
    guard.reset();
    expect(() => guard.circuitBreaker.check()).not.toThrow();
  });
});

describe('Alert callbacks', () => {
  test('fires alert at threshold', () => {
    const alerts: number[] = [];
    const guard = new BudgetGuard({
      maxUsd: 0.01,
      alertThresholds: [0.5],
      onAlert: (threshold) => alerts.push(threshold),
    });
    // Spend >50% of $0.01
    guard.postCallRecord('gpt-4o', 500, 500); // ~$0.006
    expect(alerts).toContain(0.5);
  });

  test('fires alert only once per threshold', () => {
    const alerts: number[] = [];
    const guard = new BudgetGuard({
      maxUsd: 0.01,
      alertThresholds: [0.5],
      onAlert: (threshold) => alerts.push(threshold),
    });
    guard.postCallRecord('gpt-4o', 500, 500);
    guard.postCallRecord('gpt-4o', 500, 500);
    expect(alerts.filter(t => t === 0.5).length).toBe(1);
  });
});

describe('withBudget helper', () => {
  test('passes guard to function', () => {
    const report = withBudget({ maxUsd: 5 }, (guard) => {
      guard.postCallRecord('gpt-4o-mini', 1000, 200);
      return guard.costReport();
    });
    expect(report.totalCalls).toBe(1);
  });
});

describe('Pricing', () => {
  test('calculates cost for known model', () => {
    const cost = calculateCost('gpt-4o', 1_000_000, 1_000_000);
    expect(cost).toBeCloseTo(12.50, 2); // $2.50 input + $10 output per 1M
  });

  test('returns 0 for unknown model', () => {
    expect(calculateCost('unknown-model-xyz', 1000, 1000)).toBe(0);
  });

  test('prefix matching works', () => {
    const price = getModelPrice('gpt-4o-2024-05-13');
    expect(price).not.toBeNull();
    expect(price!.inputPerMtok).toBe(2.50);
  });

  test('custom pricing overrides built-in', () => {
    setCustomPricing({ 'gpt-4o': { input_per_mtok: 99, output_per_mtok: 99 } });
    const cost = calculateCost('gpt-4o', 1_000_000, 0);
    expect(cost).toBe(99);
  });

  test('clearCustomPricing restores original', () => {
    setCustomPricing({ 'gpt-4o': { input_per_mtok: 99, output_per_mtok: 99 } });
    clearCustomPricing();
    const cost = calculateCost('gpt-4o', 1_000_000, 0);
    expect(cost).toBe(2.50);
  });
});

describe('Rate limiting', () => {
  test('throws when tokens/min exceeded', () => {
    const guard = new BudgetGuard({ maxUsd: 100, maxTokensPerMinute: 1000 });
    expect(() => guard.preCallCheck(1500)).toThrow(RateLimitExceededError);
  });

  test('allows within rate limit', () => {
    const guard = new BudgetGuard({ maxUsd: 100, maxTokensPerMinute: 10000 });
    expect(() => guard.preCallCheck(500)).not.toThrow();
  });
});

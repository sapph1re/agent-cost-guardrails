import { BudgetExceededError, CircuitBreakerTrippedError, RateLimitExceededError } from './errors';
import { calculateCost, CustomPricing } from './pricing';

export type AlertCallback = (threshold: number, currentCost: number, maxBudget: number) => void;

export interface BudgetConfig {
  maxUsd: number;
  maxTokensPerCall?: number;
  maxTokensPerMinute?: number;
  alertThresholds: number[];
  circuitBreakerMaxViolations: number;
  onAlert?: AlertCallback;
  customPricing?: CustomPricing;
}

export interface CostReport {
  totalCostUsd: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCalls: number;
  budgetUsd: number;
  remainingUsd: number;
  costByModel: Record<string, number>;
  costByAgent: Record<string, number>;
  tokensByModel: Record<string, { input: number; output: number }>;
}

export class CostTracker {
  private config: BudgetConfig;
  private totalCost = 0;
  private totalInputTokens = 0;
  private totalOutputTokens = 0;
  private calls = 0;
  private costByModel: Record<string, number> = {};
  private costByAgent: Record<string, number> = {};
  private tokensByModel: Record<string, { input: number; output: number }> = {};
  private tokenLog: Array<[number, number]> = []; // [timestamp_ms, tokens]
  private alertedThresholds = new Set<number>();

  constructor(config: BudgetConfig) {
    this.config = config;
  }

  get currentCost(): number {
    return this.totalCost;
  }

  get currentCalls(): number {
    return this.calls;
  }

  get remainingBudget(): number {
    return Math.max(0, this.config.maxUsd - this.totalCost);
  }

  checkBudget(): void {
    if (this.totalCost >= this.config.maxUsd) {
      throw new BudgetExceededError(
        `Budget exceeded: $${this.totalCost.toFixed(4)} >= $${this.config.maxUsd.toFixed(2)}`,
        this.totalCost,
        this.config.maxUsd,
      );
    }
  }

  checkRateLimit(tokens: number): void {
    if (!this.config.maxTokensPerMinute) return;
    const now = Date.now();
    const cutoff = now - 60_000;
    this.tokenLog = this.tokenLog.filter(([t]) => t > cutoff);
    const recentTokens = this.tokenLog.reduce((sum, [, c]) => sum + c, 0) + tokens;
    if (recentTokens > this.config.maxTokensPerMinute) {
      throw new RateLimitExceededError(
        `Rate limit: ${recentTokens} tokens/min exceeds ${this.config.maxTokensPerMinute}`,
        recentTokens,
        this.config.maxTokensPerMinute,
      );
    }
  }

  record(model: string, inputTokens: number, outputTokens: number, agentId = 'default'): number {
    const cost = calculateCost(model, inputTokens, outputTokens, this.config.customPricing);

    this.totalCost += cost;
    this.totalInputTokens += inputTokens;
    this.totalOutputTokens += outputTokens;
    this.calls += 1;
    this.costByModel[model] = (this.costByModel[model] ?? 0) + cost;
    this.costByAgent[agentId] = (this.costByAgent[agentId] ?? 0) + cost;
    if (!this.tokensByModel[model]) this.tokensByModel[model] = { input: 0, output: 0 };
    this.tokensByModel[model].input += inputTokens;
    this.tokensByModel[model].output += outputTokens;
    this.tokenLog.push([Date.now(), inputTokens + outputTokens]);

    this.checkAlerts(this.totalCost, this.config.maxUsd);
    return cost;
  }

  private checkAlerts(currentCost: number, maxUsd: number): void {
    if (!this.config.onAlert) return;
    const ratio = maxUsd > 0 ? currentCost / maxUsd : 0;
    for (const threshold of this.config.alertThresholds) {
      if (ratio >= threshold && !this.alertedThresholds.has(threshold)) {
        this.alertedThresholds.add(threshold);
        this.config.onAlert(threshold, currentCost, maxUsd);
      }
    }
  }

  costReport(): CostReport {
    return {
      totalCostUsd: parseFloat(this.totalCost.toFixed(6)),
      totalInputTokens: this.totalInputTokens,
      totalOutputTokens: this.totalOutputTokens,
      totalCalls: this.calls,
      budgetUsd: this.config.maxUsd,
      remainingUsd: parseFloat(Math.max(0, this.config.maxUsd - this.totalCost).toFixed(6)),
      costByModel: { ...this.costByModel },
      costByAgent: { ...this.costByAgent },
      tokensByModel: Object.fromEntries(
        Object.entries(this.tokensByModel).map(([k, v]) => [k, { ...v }]),
      ),
    };
  }

  reset(): void {
    this.totalCost = 0;
    this.totalInputTokens = 0;
    this.totalOutputTokens = 0;
    this.calls = 0;
    this.costByModel = {};
    this.costByAgent = {};
    this.tokensByModel = {};
    this.tokenLog = [];
    this.alertedThresholds.clear();
  }
}

export class CircuitBreaker {
  private maxViolations: number;
  private violationCount = 0;
  private tripped = false;

  constructor(maxViolations = 3) {
    this.maxViolations = maxViolations;
  }

  get isTripped(): boolean {
    return this.tripped;
  }

  get violations(): number {
    return this.violationCount;
  }

  recordViolation(): void {
    this.violationCount += 1;
    if (this.violationCount >= this.maxViolations) {
      this.tripped = true;
    }
  }

  recordSuccess(): void {
    this.violationCount = 0;
  }

  check(): void {
    if (this.tripped) {
      throw new CircuitBreakerTrippedError(
        `Circuit breaker tripped after ${this.violationCount} consecutive violations. Call reset() to restore.`,
        this.violationCount,
      );
    }
  }

  reset(): void {
    this.violationCount = 0;
    this.tripped = false;
  }
}

export interface BudgetGuardOptions {
  maxUsd?: number;
  maxTokensPerCall?: number;
  maxTokensPerMinute?: number;
  alertThresholds?: number[];
  circuitBreakerMaxViolations?: number;
  onAlert?: AlertCallback;
  customPricing?: CustomPricing;
}

export class BudgetGuard {
  config: BudgetConfig;
  tracker: CostTracker;
  circuitBreaker: CircuitBreaker;

  constructor(options: BudgetGuardOptions = {}) {
    this.config = {
      maxUsd: options.maxUsd ?? 10.0,
      maxTokensPerCall: options.maxTokensPerCall,
      maxTokensPerMinute: options.maxTokensPerMinute,
      alertThresholds: options.alertThresholds ?? [0.5, 0.8, 1.0],
      circuitBreakerMaxViolations: options.circuitBreakerMaxViolations ?? 3,
      onAlert: options.onAlert,
      customPricing: options.customPricing,
    };
    this.tracker = new CostTracker(this.config);
    this.circuitBreaker = new CircuitBreaker(this.config.circuitBreakerMaxViolations);
  }

  preCallCheck(estimatedTokens?: number): void {
    this.circuitBreaker.check();
    this.tracker.checkBudget();
    if (
      estimatedTokens !== undefined &&
      this.config.maxTokensPerCall !== undefined &&
      estimatedTokens > this.config.maxTokensPerCall
    ) {
      this.circuitBreaker.recordViolation();
      throw new BudgetExceededError(
        `Call token estimate ${estimatedTokens} exceeds per-call limit ${this.config.maxTokensPerCall}`,
        this.tracker.currentCost,
        this.config.maxUsd,
      );
    }
    if (estimatedTokens !== undefined) {
      this.tracker.checkRateLimit(estimatedTokens);
    }
  }

  postCallRecord(model: string, inputTokens: number, outputTokens: number, agentId = 'default'): number {
    const cost = this.tracker.record(model, inputTokens, outputTokens, agentId);
    if (this.tracker.currentCost >= this.config.maxUsd) {
      this.circuitBreaker.recordViolation();
    } else {
      this.circuitBreaker.recordSuccess();
    }
    return cost;
  }

  costReport(): CostReport {
    return this.tracker.costReport();
  }

  reset(): void {
    this.tracker.reset();
    this.circuitBreaker.reset();
  }

  async run<T>(fn: (guard: BudgetGuard) => Promise<T>): Promise<T> {
    return fn(this);
  }
}

export function withBudget<T>(
  options: BudgetGuardOptions,
  fn: (guard: BudgetGuard) => T,
): T {
  const guard = new BudgetGuard(options);
  return fn(guard);
}

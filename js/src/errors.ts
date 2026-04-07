export class GuardrailError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GuardrailError';
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class BudgetExceededError extends GuardrailError {
  spent: number;
  budget: number;

  constructor(message: string, spent = 0, budget = 0) {
    super(message);
    this.name = 'BudgetExceededError';
    this.spent = spent;
    this.budget = budget;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class CircuitBreakerTrippedError extends GuardrailError {
  violations: number;

  constructor(message: string, violations = 0) {
    super(message);
    this.name = 'CircuitBreakerTrippedError';
    this.violations = violations;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class RateLimitExceededError extends GuardrailError {
  tokensPerMin: number;
  limit: number;

  constructor(message: string, tokensPerMin = 0, limit = 0) {
    super(message);
    this.name = 'RateLimitExceededError';
    this.tokensPerMin = tokensPerMin;
    this.limit = limit;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

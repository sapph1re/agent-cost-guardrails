"""Exception types for agent cost guardrails."""


class GuardrailError(Exception):
    """Base exception for all guardrail errors."""
    pass


class BudgetExceededError(GuardrailError):
    """Raised when a budget limit is exceeded."""

    def __init__(self, message: str, spent: float = 0.0, budget: float = 0.0):
        self.spent = spent
        self.budget = budget
        super().__init__(message)


class CircuitBreakerTrippedError(GuardrailError):
    """Raised when the circuit breaker has tripped."""

    def __init__(self, message: str, violations: int = 0):
        self.violations = violations
        super().__init__(message)


class RateLimitExceededError(GuardrailError):
    """Raised when token rate limit is exceeded."""

    def __init__(self, message: str, tokens_per_min: int = 0, limit: int = 0):
        self.tokens_per_min = tokens_per_min
        self.limit = limit
        super().__init__(message)

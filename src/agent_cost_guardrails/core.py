"""Core guardrail components: BudgetConfig, CostTracker, CircuitBreaker, BudgetGuard."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent_cost_guardrails.exceptions import (
    BudgetExceededError,
    CircuitBreakerTrippedError,
    RateLimitExceededError,
)
from agent_cost_guardrails.pricing import calculate_cost


@dataclass
class BudgetConfig:
    """Configuration for budget guardrails."""

    max_usd: float = 10.0
    max_tokens_per_call: Optional[int] = None
    max_tokens_per_minute: Optional[int] = None
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 1.0])
    circuit_breaker_max_violations: int = 3
    on_alert: Optional[Callable[[float, float, float], None]] = None
    custom_pricing: Optional[Dict[str, Dict[str, float]]] = None


class CostTracker:
    """Thread-safe cost accumulator with per-agent/model breakdown."""

    def __init__(self, config: BudgetConfig):
        self.config = config
        self._lock = threading.Lock()
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._calls: int = 0
        self._cost_by_model: Dict[str, float] = defaultdict(float)
        self._cost_by_agent: Dict[str, float] = defaultdict(float)
        self._tokens_by_model: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0}
        )
        # Rate limiting: sliding window of (timestamp, token_count) entries
        self._token_log: List[tuple] = []
        self._alerted_thresholds: set = set()

    @property
    def total_cost(self) -> float:
        with self._lock:
            return self._total_cost

    @property
    def total_calls(self) -> int:
        with self._lock:
            return self._calls

    @property
    def remaining_budget(self) -> float:
        with self._lock:
            return max(0.0, self.config.max_usd - self._total_cost)

    def check_budget(self) -> None:
        """Raise BudgetExceededError if budget is already exceeded."""
        with self._lock:
            if self._total_cost >= self.config.max_usd:
                raise BudgetExceededError(
                    f"Budget exceeded: ${self._total_cost:.4f} >= ${self.config.max_usd:.2f}",
                    spent=self._total_cost,
                    budget=self.config.max_usd,
                )

    def check_rate_limit(self, tokens: int) -> None:
        """Check tokens-per-minute rate limit."""
        if self.config.max_tokens_per_minute is None:
            return
        with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            self._token_log = [
                (t, c) for t, c in self._token_log if t > cutoff
            ]
            recent_tokens = sum(c for _, c in self._token_log) + tokens
            if recent_tokens > self.config.max_tokens_per_minute:
                raise RateLimitExceededError(
                    f"Rate limit: {recent_tokens} tokens/min exceeds {self.config.max_tokens_per_minute}",
                    tokens_per_min=recent_tokens,
                    limit=self.config.max_tokens_per_minute,
                )

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent_id: str = "default",
    ) -> float:
        """Record a completed LLM call. Returns the cost of this call."""
        cost = calculate_cost(
            model, input_tokens, output_tokens, self.config.custom_pricing
        )
        with self._lock:
            self._total_cost += cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._calls += 1
            self._cost_by_model[model] += cost
            self._cost_by_agent[agent_id] += cost
            self._tokens_by_model[model]["input"] += input_tokens
            self._tokens_by_model[model]["output"] += output_tokens
            total_tokens = input_tokens + output_tokens
            self._token_log.append((time.monotonic(), total_tokens))
            current_cost = self._total_cost
            max_usd = self.config.max_usd

        # Fire alert callbacks outside lock
        self._check_alerts(current_cost, max_usd)
        return cost

    def _check_alerts(self, current_cost: float, max_usd: float) -> None:
        if self.config.on_alert is None:
            return
        ratio = current_cost / max_usd if max_usd > 0 else 0.0
        for threshold in self.config.alert_thresholds:
            if ratio >= threshold and threshold not in self._alerted_thresholds:
                self._alerted_thresholds.add(threshold)
                self.config.on_alert(threshold, current_cost, max_usd)

    def cost_report(self) -> Dict[str, Any]:
        """Return a breakdown of costs."""
        with self._lock:
            return {
                "total_cost_usd": round(self._total_cost, 6),
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_calls": self._calls,
                "budget_usd": self.config.max_usd,
                "remaining_usd": round(
                    max(0.0, self.config.max_usd - self._total_cost), 6
                ),
                "cost_by_model": dict(self._cost_by_model),
                "cost_by_agent": dict(self._cost_by_agent),
                "tokens_by_model": {
                    k: dict(v) for k, v in self._tokens_by_model.items()
                },
            }

    def reset(self) -> None:
        """Reset all tracked state."""
        with self._lock:
            self._total_cost = 0.0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._calls = 0
            self._cost_by_model.clear()
            self._cost_by_agent.clear()
            self._tokens_by_model.clear()
            self._token_log.clear()
            self._alerted_thresholds.clear()


class CircuitBreaker:
    """Trips after N consecutive budget violations, requiring manual reset."""

    def __init__(self, max_violations: int = 3):
        self.max_violations = max_violations
        self._violations = 0
        self._tripped = False
        self._lock = threading.Lock()

    @property
    def is_tripped(self) -> bool:
        with self._lock:
            return self._tripped

    @property
    def violations(self) -> int:
        with self._lock:
            return self._violations

    def record_violation(self) -> None:
        """Record a budget violation. Trips after max_violations consecutive."""
        with self._lock:
            self._violations += 1
            if self._violations >= self.max_violations:
                self._tripped = True

    def record_success(self) -> None:
        """Record a successful call — resets consecutive violation counter."""
        with self._lock:
            self._violations = 0

    def check(self) -> None:
        """Raise CircuitBreakerTrippedError if tripped."""
        with self._lock:
            if self._tripped:
                raise CircuitBreakerTrippedError(
                    f"Circuit breaker tripped after {self._violations} consecutive violations. "
                    f"Call reset() to restore.",
                    violations=self._violations,
                )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._violations = 0
            self._tripped = False


class BudgetGuard:
    """Main guardrail object — combines CostTracker + CircuitBreaker.

    Can be used as a context manager:
        with BudgetGuard(max_usd=5.00) as guard:
            # ... run agents ...
            print(guard.cost_report())
    """

    def __init__(
        self,
        max_usd: float = 10.0,
        max_tokens_per_call: Optional[int] = None,
        max_tokens_per_minute: Optional[int] = None,
        alert_thresholds: Optional[List[float]] = None,
        circuit_breaker_max_violations: int = 3,
        on_alert: Optional[Callable[[float, float, float], None]] = None,
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.config = BudgetConfig(
            max_usd=max_usd,
            max_tokens_per_call=max_tokens_per_call,
            max_tokens_per_minute=max_tokens_per_minute,
            alert_thresholds=alert_thresholds or [0.5, 0.8, 1.0],
            circuit_breaker_max_violations=circuit_breaker_max_violations,
            on_alert=on_alert,
            custom_pricing=custom_pricing,
        )
        self.tracker = CostTracker(self.config)
        self.circuit_breaker = CircuitBreaker(circuit_breaker_max_violations)

    def pre_call_check(self, estimated_tokens: Optional[int] = None) -> None:
        """Run all pre-call checks. Raises on any violation."""
        self.circuit_breaker.check()
        self.tracker.check_budget()
        if (
            estimated_tokens is not None
            and self.config.max_tokens_per_call is not None
            and estimated_tokens > self.config.max_tokens_per_call
        ):
            self.circuit_breaker.record_violation()
            raise BudgetExceededError(
                f"Call token estimate {estimated_tokens} exceeds per-call limit "
                f"{self.config.max_tokens_per_call}",
                spent=self.tracker.total_cost,
                budget=self.config.max_usd,
            )
        if estimated_tokens is not None:
            self.tracker.check_rate_limit(estimated_tokens)

    def post_call_record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent_id: str = "default",
    ) -> float:
        """Record a completed call and run post-call checks."""
        cost = self.tracker.record(model, input_tokens, output_tokens, agent_id)
        if self.tracker.total_cost >= self.config.max_usd:
            self.circuit_breaker.record_violation()
        else:
            self.circuit_breaker.record_success()
        return cost

    def cost_report(self) -> Dict[str, Any]:
        return self.tracker.cost_report()

    def reset(self) -> None:
        self.tracker.reset()
        self.circuit_breaker.reset()

    def __enter__(self) -> "BudgetGuard":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

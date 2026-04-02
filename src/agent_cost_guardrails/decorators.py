"""Decorator API for budget guardrails."""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional

from agent_cost_guardrails.core import BudgetGuard


def budget_limit(
    max_usd: float = 10.0,
    max_tokens_per_call: Optional[int] = None,
    max_tokens_per_minute: Optional[int] = None,
    on_alert: Optional[Callable[[float, float, float], None]] = None,
    custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> Callable:
    """Decorator that wraps a function with budget guardrails.

    The decorated function receives a `guard` keyword argument (BudgetGuard instance)
    that it can use to record costs and check budgets.

    Usage:
        @budget_limit(max_usd=5.00)
        def run_agents(guard=None):
            # Use guard.pre_call_check() before LLM calls
            # Use guard.post_call_record() after LLM calls
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            guard = BudgetGuard(
                max_usd=max_usd,
                max_tokens_per_call=max_tokens_per_call,
                max_tokens_per_minute=max_tokens_per_minute,
                on_alert=on_alert,
                custom_pricing=custom_pricing,
            )
            kwargs["guard"] = guard
            try:
                return func(*args, **kwargs)
            finally:
                # Store report on the wrapper for post-execution access
                wrapper.last_cost_report = guard.cost_report()  # type: ignore[attr-defined]

        wrapper.last_cost_report = None  # type: ignore[attr-defined]
        return wrapper

    return decorator

"""Tests for decorator API."""

import pytest

from agent_cost_guardrails.decorators import budget_limit
from agent_cost_guardrails.exceptions import BudgetExceededError


class TestBudgetLimitDecorator:
    def test_basic_usage(self):
        @budget_limit(max_usd=10.0)
        def my_func(guard=None):
            guard.post_call_record("gpt-4o", 1000, 500)
            return guard.cost_report()

        result = my_func()
        assert result["total_calls"] == 1
        assert result["total_cost_usd"] > 0

    def test_last_cost_report(self):
        @budget_limit(max_usd=10.0)
        def my_func(guard=None):
            guard.post_call_record("gpt-4o", 1000, 500)

        my_func()
        report = my_func.last_cost_report
        assert report is not None
        assert report["total_calls"] == 1

    def test_budget_exceeded_in_decorated(self):
        @budget_limit(max_usd=0.001)
        def my_func(guard=None):
            guard.post_call_record("gpt-4o", 1_000_000, 1_000_000)
            guard.pre_call_check()

        with pytest.raises(BudgetExceededError):
            my_func()

    def test_custom_alert_callback(self):
        alerts = []

        @budget_limit(max_usd=0.01, on_alert=lambda t, c, m: alerts.append(t))
        def my_func(guard=None):
            guard.post_call_record("gpt-4o", 1_000_000, 1_000_000)

        my_func()
        assert len(alerts) > 0

    def test_preserves_function_name(self):
        @budget_limit(max_usd=10.0)
        def my_special_func(guard=None):
            pass

        assert my_special_func.__name__ == "my_special_func"

    def test_passes_through_args(self):
        @budget_limit(max_usd=10.0)
        def add(a, b, guard=None):
            return a + b

        assert add(3, 4) == 7

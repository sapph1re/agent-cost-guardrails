"""Tests for core module: CostTracker, CircuitBreaker, BudgetGuard."""

import threading
import time

import pytest

from agent_cost_guardrails.core import (
    BudgetConfig,
    BudgetGuard,
    CircuitBreaker,
    CostTracker,
)
from agent_cost_guardrails.exceptions import (
    BudgetExceededError,
    CircuitBreakerTrippedError,
    RateLimitExceededError,
)


class TestCostTracker:
    def test_record_accumulates(self):
        config = BudgetConfig(max_usd=1.0)
        tracker = CostTracker(config)
        tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
        assert tracker.total_cost > 0
        assert tracker.total_calls == 1

    def test_check_budget_raises_when_exceeded(self):
        config = BudgetConfig(max_usd=0.001)
        tracker = CostTracker(config)
        tracker.record("gpt-4o", input_tokens=100_000, output_tokens=50_000)
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()

    def test_check_budget_ok_when_under(self):
        config = BudgetConfig(max_usd=100.0)
        tracker = CostTracker(config)
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.check_budget()  # Should not raise

    def test_remaining_budget(self):
        config = BudgetConfig(max_usd=10.0)
        tracker = CostTracker(config)
        assert tracker.remaining_budget == 10.0
        tracker.record("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        assert tracker.remaining_budget < 10.0

    def test_cost_by_model(self):
        config = BudgetConfig(max_usd=100.0)
        tracker = CostTracker(config)
        tracker.record("gpt-4o", 1000, 500)
        tracker.record("gpt-3.5-turbo", 1000, 500)
        report = tracker.cost_report()
        assert "gpt-4o" in report["cost_by_model"]
        assert "gpt-3.5-turbo" in report["cost_by_model"]

    def test_cost_by_agent(self):
        config = BudgetConfig(max_usd=100.0)
        tracker = CostTracker(config)
        tracker.record("gpt-4o", 1000, 500, agent_id="agent-a")
        tracker.record("gpt-4o", 1000, 500, agent_id="agent-b")
        report = tracker.cost_report()
        assert "agent-a" in report["cost_by_agent"]
        assert "agent-b" in report["cost_by_agent"]

    def test_alert_callbacks(self):
        alerts = []

        def on_alert(threshold, current, maximum):
            alerts.append((threshold, current, maximum))

        config = BudgetConfig(
            max_usd=0.01,
            alert_thresholds=[0.5, 1.0],
            on_alert=on_alert,
        )
        tracker = CostTracker(config)
        # Record enough to exceed budget
        tracker.record("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
        assert len(alerts) >= 1  # At least the 1.0 threshold

    def test_alert_fires_once_per_threshold(self):
        alerts = []

        def on_alert(threshold, current, maximum):
            alerts.append(threshold)

        config = BudgetConfig(max_usd=0.01, alert_thresholds=[0.5, 1.0], on_alert=on_alert)
        tracker = CostTracker(config)
        tracker.record("gpt-4o", 1_000_000, 1_000_000)
        tracker.record("gpt-4o", 1_000_000, 1_000_000)
        # Each threshold should fire at most once
        assert alerts.count(0.5) <= 1
        assert alerts.count(1.0) <= 1

    def test_rate_limit_raises(self):
        config = BudgetConfig(max_usd=100.0, max_tokens_per_minute=100)
        tracker = CostTracker(config)
        with pytest.raises(RateLimitExceededError):
            tracker.check_rate_limit(200)

    def test_rate_limit_ok_under_limit(self):
        config = BudgetConfig(max_usd=100.0, max_tokens_per_minute=10000)
        tracker = CostTracker(config)
        tracker.check_rate_limit(500)  # Should not raise

    def test_reset(self):
        config = BudgetConfig(max_usd=100.0)
        tracker = CostTracker(config)
        tracker.record("gpt-4o", 1000, 500)
        tracker.reset()
        assert tracker.total_cost == 0.0
        assert tracker.total_calls == 0

    def test_cost_report_structure(self):
        config = BudgetConfig(max_usd=10.0)
        tracker = CostTracker(config)
        tracker.record("gpt-4o", 1000, 500)
        report = tracker.cost_report()
        assert "total_cost_usd" in report
        assert "total_input_tokens" in report
        assert "total_output_tokens" in report
        assert "total_calls" in report
        assert "budget_usd" in report
        assert "remaining_usd" in report
        assert "cost_by_model" in report
        assert "cost_by_agent" in report
        assert "tokens_by_model" in report

    def test_thread_safety(self):
        config = BudgetConfig(max_usd=1000.0)
        tracker = CostTracker(config)
        errors = []

        def record_many():
            try:
                for _ in range(100):
                    tracker.record("gpt-4o", 100, 50)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.total_calls == 400


class TestCircuitBreaker:
    def test_not_tripped_initially(self):
        cb = CircuitBreaker(max_violations=3)
        assert not cb.is_tripped
        cb.check()  # Should not raise

    def test_trips_after_max_violations(self):
        cb = CircuitBreaker(max_violations=3)
        cb.record_violation()
        cb.record_violation()
        cb.record_violation()
        assert cb.is_tripped
        with pytest.raises(CircuitBreakerTrippedError):
            cb.check()

    def test_success_resets_violations(self):
        cb = CircuitBreaker(max_violations=3)
        cb.record_violation()
        cb.record_violation()
        cb.record_success()
        assert cb.violations == 0
        assert not cb.is_tripped

    def test_manual_reset(self):
        cb = CircuitBreaker(max_violations=1)
        cb.record_violation()
        assert cb.is_tripped
        cb.reset()
        assert not cb.is_tripped
        cb.check()  # Should not raise


class TestBudgetGuard:
    def test_context_manager(self):
        with BudgetGuard(max_usd=5.0) as guard:
            guard.post_call_record("gpt-4o", 1000, 500)
            report = guard.cost_report()
            assert report["total_calls"] == 1

    def test_pre_call_check_passes(self):
        guard = BudgetGuard(max_usd=100.0)
        guard.pre_call_check(estimated_tokens=100)

    def test_pre_call_check_budget_exceeded(self):
        guard = BudgetGuard(max_usd=0.001)
        guard.post_call_record("gpt-4o", 1_000_000, 1_000_000)
        with pytest.raises(BudgetExceededError):
            guard.pre_call_check()

    def test_pre_call_check_per_call_limit(self):
        guard = BudgetGuard(max_usd=100.0, max_tokens_per_call=1000)
        with pytest.raises(BudgetExceededError):
            guard.pre_call_check(estimated_tokens=5000)

    def test_circuit_breaker_integration(self):
        guard = BudgetGuard(max_usd=0.001, circuit_breaker_max_violations=2)
        # Exceed budget to trigger violations
        guard.post_call_record("gpt-4o", 1_000_000, 1_000_000)
        guard.post_call_record("gpt-4o", 1_000_000, 1_000_000)
        with pytest.raises(CircuitBreakerTrippedError):
            guard.pre_call_check()

    def test_post_call_record_returns_cost(self):
        guard = BudgetGuard(max_usd=100.0)
        cost = guard.post_call_record("gpt-4o", 1000, 500)
        assert cost > 0

    def test_reset(self):
        guard = BudgetGuard(max_usd=100.0)
        guard.post_call_record("gpt-4o", 1000, 500)
        guard.reset()
        assert guard.cost_report()["total_calls"] == 0

    def test_full_lifecycle(self):
        alerts = []
        guard = BudgetGuard(
            max_usd=0.02,
            on_alert=lambda t, c, m: alerts.append(t),
        )
        # Make calls until budget exceeded
        for _ in range(10):
            try:
                guard.pre_call_check()
                guard.post_call_record("gpt-4o", 100_000, 50_000)
            except (BudgetExceededError, CircuitBreakerTrippedError):
                break
        assert guard.cost_report()["total_cost_usd"] > 0

"""Tests for pricing module."""

import pytest

from agent_cost_guardrails.pricing import (
    ModelPrice,
    calculate_cost,
    clear_custom_pricing,
    get_model_price,
    set_custom_pricing,
)


class TestGetModelPrice:
    def test_known_model(self):
        price = get_model_price("gpt-4o")
        assert price is not None
        assert price.input_per_mtok == 2.50
        assert price.output_per_mtok == 10.00

    def test_unknown_model(self):
        assert get_model_price("nonexistent-model-xyz") is None

    def test_prefix_matching(self):
        price = get_model_price("gpt-4o-2024-05-13")
        assert price is not None
        assert price.input_per_mtok == 2.50

    def test_claude_model(self):
        price = get_model_price("claude-opus-4-6")
        assert price is not None
        assert price.input_per_mtok == 15.00
        assert price.output_per_mtok == 75.00

    def test_gemini_model(self):
        price = get_model_price("gemini-2.0-flash")
        assert price is not None


class TestCustomPricing:
    def setup_method(self):
        clear_custom_pricing()

    def teardown_method(self):
        clear_custom_pricing()

    def test_set_custom_pricing(self):
        set_custom_pricing({
            "my-model": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}
        })
        price = get_model_price("my-model")
        assert price is not None
        assert price.input_per_mtok == 1.0
        assert price.output_per_mtok == 2.0

    def test_custom_overrides_bundled(self):
        set_custom_pricing({
            "gpt-4o": {"input_per_mtok": 99.0, "output_per_mtok": 99.0}
        })
        price = get_model_price("gpt-4o")
        assert price.input_per_mtok == 99.0

    def test_clear_custom_pricing(self):
        set_custom_pricing({
            "my-model": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}
        })
        clear_custom_pricing()
        assert get_model_price("my-model") is None


class TestCalculateCost:
    def test_basic_cost(self):
        # gpt-4o: input=2.50/M, output=10.00/M
        cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        expected = (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
        assert abs(cost - expected) < 1e-10

    def test_unknown_model_returns_zero(self):
        cost = calculate_cost("unknown-model", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_custom_pricing_in_calculate(self):
        cost = calculate_cost(
            "my-custom",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            custom_pricing={"my-custom": {"input_per_mtok": 5.0, "output_per_mtok": 10.0}},
        )
        assert cost == 15.0

    def test_zero_tokens(self):
        cost = calculate_cost("gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_large_token_count(self):
        cost = calculate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == 2.50 + 10.00

"""Model pricing tables and cost calculation.

Prices are per 1M tokens in USD. Sources: LiteLLM, tokencost, provider docs.
Last updated: 2026-03-06.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float   # USD per 1M input tokens
    output_per_mtok: float  # USD per 1M output tokens


# Bundled pricing table — covers major models.
# Users can override/extend via set_custom_pricing().
_PRICING: Dict[str, ModelPrice] = {
    # OpenAI
    "gpt-4o": ModelPrice(2.50, 10.00),
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    "gpt-4-turbo": ModelPrice(10.00, 30.00),
    "gpt-4": ModelPrice(30.00, 60.00),
    "gpt-3.5-turbo": ModelPrice(0.50, 1.50),
    "o1": ModelPrice(15.00, 60.00),
    "o1-mini": ModelPrice(3.00, 12.00),
    "o1-pro": ModelPrice(150.00, 600.00),
    "o3-mini": ModelPrice(1.10, 4.40),
    "gpt-4.1": ModelPrice(2.00, 8.00),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60),
    "gpt-4.1-nano": ModelPrice(0.10, 0.40),
    # Anthropic
    "claude-opus-4-6": ModelPrice(15.00, 75.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5-20251001": ModelPrice(0.80, 4.00),
    "claude-3-5-sonnet-20241022": ModelPrice(3.00, 15.00),
    "claude-3-5-haiku-20241022": ModelPrice(0.80, 4.00),
    "claude-3-opus-20240229": ModelPrice(15.00, 75.00),
    "claude-3-sonnet-20240229": ModelPrice(3.00, 15.00),
    "claude-3-haiku-20240307": ModelPrice(0.25, 1.25),
    # Google
    "gemini-2.0-flash": ModelPrice(0.10, 0.40),
    "gemini-2.0-pro": ModelPrice(1.25, 10.00),
    "gemini-1.5-pro": ModelPrice(1.25, 5.00),
    "gemini-1.5-flash": ModelPrice(0.075, 0.30),
    # Mistral
    "mistral-large-latest": ModelPrice(2.00, 6.00),
    "mistral-small-latest": ModelPrice(0.10, 0.30),
    # DeepSeek
    "deepseek-chat": ModelPrice(0.27, 1.10),
    "deepseek-reasoner": ModelPrice(0.55, 2.19),
    # Meta (via providers)
    "llama-3.1-405b": ModelPrice(3.00, 3.00),
    "llama-3.1-70b": ModelPrice(0.88, 0.88),
    "llama-3.1-8b": ModelPrice(0.18, 0.18),
}

# User-provided custom/override pricing
_CUSTOM_PRICING: Dict[str, ModelPrice] = {}


def set_custom_pricing(prices: Dict[str, Dict[str, float]]) -> None:
    """Set custom pricing for models.

    Args:
        prices: Dict mapping model name to {"input_per_mtok": float, "output_per_mtok": float}
    """
    for model, rates in prices.items():
        _CUSTOM_PRICING[model] = ModelPrice(
            input_per_mtok=rates["input_per_mtok"],
            output_per_mtok=rates["output_per_mtok"],
        )


def clear_custom_pricing() -> None:
    """Clear all custom pricing overrides."""
    _CUSTOM_PRICING.clear()


def get_model_price(model: str) -> Optional[ModelPrice]:
    """Get pricing for a model. Custom pricing takes priority over bundled."""
    if model in _CUSTOM_PRICING:
        return _CUSTOM_PRICING[model]
    if model in _PRICING:
        return _PRICING[model]
    # Try prefix matching (e.g., "gpt-4o-2024-05-13" -> "gpt-4o")
    for key in sorted(_PRICING.keys(), key=len, reverse=True):
        if model.startswith(key):
            return _PRICING[key]
    return None


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Calculate cost in USD for a model call.

    Returns 0.0 if model pricing is unknown (logs warning).
    """
    if custom_pricing and model in custom_pricing:
        rates = custom_pricing[model]
        price = ModelPrice(rates["input_per_mtok"], rates["output_per_mtok"])
    else:
        price = get_model_price(model)

    if price is None:
        return 0.0

    cost = (input_tokens * price.input_per_mtok / 1_000_000) + (
        output_tokens * price.output_per_mtok / 1_000_000
    )
    return cost

"""LangGraph/LangChain integration — custom BaseCallbackHandler."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from agent_cost_guardrails.core import BudgetGuard
from agent_cost_guardrails.exceptions import BudgetExceededError

try:
    import tiktoken

    _encoder = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _encoder = None


def _count_tokens(text: str) -> int:
    if _encoder:
        return len(_encoder.encode(text))
    return len(text) // 4


class BudgetCallbackHandler:
    """LangChain-compatible callback handler that enforces budget limits.

    Implements the BaseCallbackHandler interface methods used for cost tracking.
    Does not inherit from BaseCallbackHandler to avoid requiring langchain-core
    as a hard dependency — but is fully compatible when passed as a callback.
    """

    def __init__(self, guard: BudgetGuard, default_model: str = "gpt-4o"):
        self.guard = guard
        self.default_model = default_model
        self._current_model: str = default_model
        self._current_input_tokens: int = 0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Called before LLM call. Checks budget."""
        # Try to extract model name
        model = serialized.get("kwargs", {}).get("model_name", self.default_model)
        if not model:
            model = serialized.get("id", [""]).pop() if serialized.get("id") else self.default_model
        self._current_model = model

        estimated = sum(_count_tokens(p) for p in prompts)
        self._current_input_tokens = estimated
        self.guard.pre_call_check(estimated_tokens=estimated)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Called before chat model call."""
        model = serialized.get("kwargs", {}).get("model_name", self.default_model)
        self._current_model = model

        estimated = 0
        for msg_list in messages:
            for msg in msg_list:
                content = getattr(msg, "content", str(msg))
                estimated += _count_tokens(str(content))
        self._current_input_tokens = estimated
        self.guard.pre_call_check(estimated_tokens=estimated)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called after LLM response. Records cost."""
        output_tokens = 0
        input_tokens = self._current_input_tokens

        # Try to get actual token usage from response
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage", {})
            if usage:
                input_tokens = usage.get("prompt_tokens", input_tokens)
                output_tokens = usage.get("completion_tokens", 0)

        if output_tokens == 0:
            # Estimate from response text
            generations = getattr(response, "generations", [])
            for gen_list in generations:
                if isinstance(gen_list, list):
                    for gen in gen_list:
                        text = getattr(gen, "text", str(gen))
                        output_tokens += _count_tokens(str(text))
                else:
                    text = getattr(gen_list, "text", str(gen_list))
                    output_tokens += _count_tokens(str(text))

        self.guard.post_call_record(
            self._current_model, input_tokens, output_tokens
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Called on LLM error — no cost recorded."""
        pass


class LangGraphGuardrails:
    """Budget guardrails for LangGraph/LangChain.

    Usage:
        guards = LangGraphGuardrails(max_usd=2.00)
        result = graph.invoke(state, config={"callbacks": [guards.callback_handler]})
        print(guards.cost_report())
    """

    def __init__(
        self,
        max_usd: float = 10.0,
        max_tokens_per_call: Optional[int] = None,
        max_tokens_per_minute: Optional[int] = None,
        on_alert: Optional[Callable[[float, float, float], None]] = None,
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
        default_model: str = "gpt-4o",
    ):
        self.guard = BudgetGuard(
            max_usd=max_usd,
            max_tokens_per_call=max_tokens_per_call,
            max_tokens_per_minute=max_tokens_per_minute,
            on_alert=on_alert,
            custom_pricing=custom_pricing,
        )
        self.callback_handler = BudgetCallbackHandler(self.guard, default_model)

    def cost_report(self) -> dict:
        return self.guard.cost_report()

    def reset(self) -> None:
        self.guard.reset()

"""CrewAI integration — hooks into @before_llm_call / @after_llm_call."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from agent_cost_guardrails.core import BudgetGuard
from agent_cost_guardrails.exceptions import BudgetExceededError

# tiktoken for token estimation
try:
    import tiktoken

    _encoder = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _encoder = None


def _estimate_tokens(messages: list) -> int:
    """Estimate token count from a list of message dicts."""
    if _encoder is None:
        # Rough fallback: ~4 chars per token
        return sum(len(str(m.get("content", ""))) // 4 for m in messages)
    total = 0
    for m in messages:
        content = m.get("content", "") if isinstance(m, dict) else str(m)
        total += len(_encoder.encode(str(content)))
    return total


class CrewAIGuardrails:
    """Budget guardrails for CrewAI via LLM call hooks.

    Usage:
        guards = CrewAIGuardrails(max_usd=5.00)
        guards.install()
        crew.kickoff()
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
        self.default_model = default_model
        self._installed = False
        self._before_hook = None
        self._after_hook = None

    def install(self) -> None:
        """Register before/after LLM call hooks with CrewAI."""
        try:
            from crewai.hooks import before_llm_call, after_llm_call
        except ImportError:
            raise ImportError(
                "crewai is required for CrewAIGuardrails. "
                "Install with: pip install agent-cost-guardrails[crewai]"
            )

        guard = self.guard
        default_model = self.default_model

        @before_llm_call
        def _budget_before_hook(context: Any) -> bool:
            messages = getattr(context, "messages", []) or []
            estimated = _estimate_tokens(messages)
            try:
                guard.pre_call_check(estimated_tokens=estimated)
                return True
            except (BudgetExceededError, Exception):
                return False

        @after_llm_call
        def _budget_after_hook(context: Any) -> None:
            messages = getattr(context, "messages", []) or []
            response = getattr(context, "response", "") or ""
            model = default_model
            llm = getattr(context, "llm", None)
            if llm and hasattr(llm, "model"):
                model = llm.model
            input_tokens = _estimate_tokens(messages)
            output_tokens = (
                len(_encoder.encode(str(response)))
                if _encoder
                else len(str(response)) // 4
            )
            agent = getattr(context, "agent", None)
            agent_id = getattr(agent, "role", "default") if agent else "default"
            guard.post_call_record(model, input_tokens, output_tokens, agent_id)

        self._before_hook = _budget_before_hook
        self._after_hook = _budget_after_hook
        self._installed = True

    def uninstall(self) -> None:
        """Remove hooks from CrewAI."""
        if not self._installed:
            return
        try:
            from crewai.hooks import clear_all_llm_call_hooks

            clear_all_llm_call_hooks()
        except ImportError:
            pass
        self._installed = False

    def cost_report(self) -> dict:
        return self.guard.cost_report()

    def reset(self) -> None:
        self.guard.reset()

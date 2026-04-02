"""AutoGen/AG2 integration — hooks into ConversableAgent safeguard hooks."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

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


class AutoGenGuardrails:
    """Budget guardrails for AutoGen/AG2 agents.

    Usage:
        guards = AutoGenGuardrails(max_usd=10.00)
        guards.wrap_agent(assistant)
        guards.wrap_agent(user_proxy)
        # run chat...
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
        self._wrapped_agents: List[Any] = []

    def wrap_agent(self, agent: Any) -> None:
        """Register safeguard hooks on an AutoGen ConversableAgent."""
        guard = self.guard
        default_model = self.default_model

        def _budget_safeguard_inputs(messages: Any) -> Any:
            """Hook for safeguard_llm_inputs or process_all_messages_before_reply."""
            text = str(messages) if not isinstance(messages, str) else messages
            estimated = _count_tokens(text)
            guard.pre_call_check(estimated_tokens=estimated)
            return messages

        def _budget_safeguard_outputs(response: Any) -> Any:
            """Hook for safeguard_llm_outputs or process_message_before_send."""
            text = str(response) if not isinstance(response, str) else response
            output_tokens = _count_tokens(text)
            model = default_model
            # Try to get model from agent config
            if hasattr(agent, "llm_config") and agent.llm_config:
                config = agent.llm_config
                if isinstance(config, dict):
                    model = config.get("model", default_model)
                    config_list = config.get("config_list", [])
                    if config_list and isinstance(config_list[0], dict):
                        model = config_list[0].get("model", model)
            agent_name = getattr(agent, "name", "default")
            # Estimate input tokens as ~2x output (rough heuristic for chat)
            guard.post_call_record(model, output_tokens * 2, output_tokens, agent_name)
            return response

        # Try newer AG2 hook names first, fall back to legacy
        if hasattr(agent, "register_hook"):
            try:
                agent.register_hook("safeguard_llm_inputs", _budget_safeguard_inputs)
                agent.register_hook("safeguard_llm_outputs", _budget_safeguard_outputs)
            except (ValueError, KeyError):
                # Fall back to legacy hook names
                agent.register_hook(
                    "process_all_messages_before_reply", _budget_safeguard_inputs
                )
                agent.register_hook(
                    "process_message_before_send", _budget_safeguard_outputs
                )
        else:
            raise TypeError(
                f"Agent {type(agent).__name__} does not support register_hook(). "
                f"Expected an AutoGen ConversableAgent."
            )

        self._wrapped_agents.append(agent)

    def cost_report(self) -> dict:
        return self.guard.cost_report()

    def reset(self) -> None:
        self.guard.reset()

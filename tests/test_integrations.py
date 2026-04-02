"""Tests for framework integrations (mocked — no actual framework deps)."""

import pytest

from agent_cost_guardrails.core import BudgetGuard
from agent_cost_guardrails.exceptions import BudgetExceededError
from agent_cost_guardrails.integrations.langgraph import BudgetCallbackHandler


class TestBudgetCallbackHandler:
    """Test the LangGraph callback handler with mock objects."""

    def test_on_llm_start_checks_budget(self):
        guard = BudgetGuard(max_usd=100.0)
        handler = BudgetCallbackHandler(guard)
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            prompts=["Hello world"],
        )
        # Should not raise

    def test_on_llm_start_budget_exceeded(self):
        guard = BudgetGuard(max_usd=0.001)
        guard.post_call_record("gpt-4o", 1_000_000, 1_000_000)
        handler = BudgetCallbackHandler(guard)
        with pytest.raises(BudgetExceededError):
            handler.on_llm_start(
                serialized={"kwargs": {"model_name": "gpt-4o"}},
                prompts=["Hello"],
            )

    def test_on_llm_end_records_cost(self):
        guard = BudgetGuard(max_usd=100.0)
        handler = BudgetCallbackHandler(guard)
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            prompts=["Tell me a story"],
        )

        # Mock a response object
        class MockGeneration:
            def __init__(self, text):
                self.text = text

        class MockResponse:
            llm_output = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}
            generations = [[MockGeneration("Once upon a time...")]]

        handler.on_llm_end(MockResponse())
        report = guard.cost_report()
        assert report["total_calls"] == 1
        assert report["total_cost_usd"] > 0

    def test_on_llm_end_estimates_from_text(self):
        guard = BudgetGuard(max_usd=100.0)
        handler = BudgetCallbackHandler(guard)
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            prompts=["Hello"],
        )

        class MockGeneration:
            def __init__(self, text):
                self.text = text

        class MockResponse:
            llm_output = None  # No token usage info
            generations = [[MockGeneration("A long response text " * 20)]]

        handler.on_llm_end(MockResponse())
        report = guard.cost_report()
        assert report["total_calls"] == 1

    def test_on_chat_model_start(self):
        guard = BudgetGuard(max_usd=100.0)
        handler = BudgetCallbackHandler(guard)

        class MockMessage:
            content = "Hello there"

        handler.on_chat_model_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            messages=[[MockMessage()]],
        )
        # Should not raise

    def test_on_llm_error_no_crash(self):
        guard = BudgetGuard(max_usd=100.0)
        handler = BudgetCallbackHandler(guard)
        handler.on_llm_error(RuntimeError("test error"))
        assert guard.cost_report()["total_calls"] == 0


class TestAutoGenGuardrailsMocked:
    """Test AutoGen integration with a mock agent."""

    def test_wrap_agent_registers_hooks(self):
        from agent_cost_guardrails.integrations.autogen import AutoGenGuardrails

        hooks = {}

        class MockAgent:
            name = "test-agent"
            llm_config = {"model": "gpt-4o"}

            def register_hook(self, name, fn):
                hooks[name] = fn

        guards = AutoGenGuardrails(max_usd=10.0)
        agent = MockAgent()
        guards.wrap_agent(agent)
        assert len(hooks) == 2

    def test_wrap_agent_safeguard_hooks(self):
        from agent_cost_guardrails.integrations.autogen import AutoGenGuardrails

        hooks = {}

        class MockAgent:
            name = "assistant"
            llm_config = {"model": "gpt-4o"}

            def register_hook(self, name, fn):
                hooks[name] = fn

        guards = AutoGenGuardrails(max_usd=10.0)
        guards.wrap_agent(MockAgent())

        # Test input hook
        result = hooks["safeguard_llm_inputs"]("Hello world")
        assert result == "Hello world"

        # Test output hook records cost
        result = hooks["safeguard_llm_outputs"]("Response text")
        assert result == "Response text"
        assert guards.cost_report()["total_calls"] == 1

    def test_wrap_agent_budget_exceeded(self):
        from agent_cost_guardrails.integrations.autogen import AutoGenGuardrails

        hooks = {}

        class MockAgent:
            name = "assistant"
            llm_config = {"model": "gpt-4o"}

            def register_hook(self, name, fn):
                hooks[name] = fn

        guards = AutoGenGuardrails(max_usd=0.001)
        guards.wrap_agent(MockAgent())

        # Burn the budget
        hooks["safeguard_llm_outputs"]("x" * 100000)

        with pytest.raises(BudgetExceededError):
            hooks["safeguard_llm_inputs"]("Another call")

    def test_wrap_agent_no_hook_support(self):
        from agent_cost_guardrails.integrations.autogen import AutoGenGuardrails

        class BadAgent:
            pass

        guards = AutoGenGuardrails(max_usd=10.0)
        with pytest.raises(TypeError):
            guards.wrap_agent(BadAgent())

    def test_wrap_agent_legacy_hooks(self):
        from agent_cost_guardrails.integrations.autogen import AutoGenGuardrails

        hooks = {}
        call_count = 0

        class LegacyAgent:
            name = "legacy"
            llm_config = {"model": "gpt-4o"}

            def register_hook(self, name, fn):
                nonlocal call_count
                call_count += 1
                if name.startswith("safeguard_") and call_count <= 2:
                    raise ValueError(f"Unknown hook: {name}")
                hooks[name] = fn

        guards = AutoGenGuardrails(max_usd=10.0)
        guards.wrap_agent(LegacyAgent())
        assert "process_all_messages_before_reply" in hooks
        assert "process_message_before_send" in hooks


class TestCrewAIGuardrailsMocked:
    """Test CrewAI integration — just import check since hooks need crewai."""

    def test_import_error_on_install(self):
        from agent_cost_guardrails.integrations.crewai import CrewAIGuardrails

        guards = CrewAIGuardrails(max_usd=5.0)
        with pytest.raises(ImportError, match="crewai is required"):
            guards.install()

    def test_token_estimation(self):
        from agent_cost_guardrails.integrations.crewai import _estimate_tokens

        tokens = _estimate_tokens([{"content": "Hello world, how are you?"}])
        assert tokens > 0

    def test_token_estimation_empty(self):
        from agent_cost_guardrails.integrations.crewai import _estimate_tokens

        tokens = _estimate_tokens([])
        assert tokens == 0

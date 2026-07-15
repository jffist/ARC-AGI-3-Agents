"""Unit tests for the OpenRouter-backed guided Locksmith agent."""

from unittest.mock import MagicMock

import pytest
from arcengine import FrameData, GameAction, GameState

from agents import AVAILABLE_AGENTS
from agents.templates import llm_agents as llm_module


def _tool_response(name: str, arguments: str = "{}") -> MagicMock:
    tool_call = MagicMock()
    tool_call.id = "call_12345"
    tool_call.function.name = name
    tool_call.function.arguments = arguments

    message = MagicMock()
    message.tool_calls = [tool_call]
    message.content = None

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(total_tokens=17)
    return response


def _frame(state: GameState = GameState.NOT_FINISHED) -> FrameData:
    return FrameData(
        game_id="ls20-test",
        state=state,
        available_actions=[1, 2, 3, 4, 5, 6],
        frame=[[[0] * 4 for _ in range(4)]],
    )


def _make_agent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")

    client = MagicMock()
    client.chat.completions.create.return_value = _tool_response("ACTION3")
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(llm_module, "OpenAIClient", client_factory)

    agent = llm_module.GuidedLLMls20OpenRouter(
        card_id="card-abc",
        game_id="ls20-test",
        agent_name="guided-openrouter-test",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )
    return agent, client, client_factory


@pytest.mark.unit
class TestGuidedOpenRouterValidation:
    def test_requires_openrouter_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            llm_module.GuidedLLMls20OpenRouter(
                card_id="card-abc",
                game_id="ls20-test",
                agent_name="guided-openrouter-test",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=MagicMock(),
            )

    def test_requires_openrouter_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

        with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
            llm_module.GuidedLLMls20OpenRouter(
                card_id="card-abc",
                game_id="ls20-test",
                agent_name="guided-openrouter-test",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=MagicMock(),
            )

    def test_empty_openrouter_model_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
        monkeypatch.setenv("OPENROUTER_MODEL", "")

        with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
            llm_module.GuidedLLMls20OpenRouter(
                card_id="card-abc",
                game_id="ls20-test",
                agent_name="guided-openrouter-test",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=MagicMock(),
            )


@pytest.mark.unit
class TestGuidedOpenRouterBehavior:
    def test_first_turn_returns_reset_and_builds_openrouter_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, client, client_factory = _make_agent(monkeypatch)

        action = agent.choose_action([], _frame())

        assert action is GameAction.RESET
        client_factory.assert_called_once_with(
            api_key="test-openrouter-key",
            base_url="https://openrouter.ai/api/v1",
        )
        client.chat.completions.create.assert_not_called()
        assert agent.name.endswith("openai-gpt-4.1-mini.with-observe.high")

    def test_second_turn_uses_openrouter_model_and_required_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, client, _ = _make_agent(monkeypatch)

        agent.choose_action([], _frame())
        action = agent.choose_action([_frame()], _frame())

        assert action is GameAction.ACTION3
        assert client.chat.completions.create.call_count == 2

        observation_call = client.chat.completions.create.call_args_list[0]
        assert observation_call.kwargs["model"] == "openai/gpt-4.1-mini"
        assert observation_call.kwargs["reasoning_effort"] == "high"
        assert "tools" not in observation_call.kwargs

        action_call = client.chat.completions.create.call_args_list[1]
        assert action_call.kwargs["model"] == "openai/gpt-4.1-mini"
        assert action_call.kwargs["tool_choice"] == "required"
        assert action_call.kwargs["reasoning_effort"] == "high"
        assert action_call.kwargs["tools"] == agent.build_tools()

    def test_reasoning_metadata_uses_runtime_openrouter_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _, _ = _make_agent(monkeypatch)

        agent.choose_action([], _frame())
        action = agent.choose_action([_frame()], _frame())

        assert action.reasoning["model"] == "openai/gpt-4.1-mini"


@pytest.mark.unit
class TestGuidedOpenRouterRegistration:
    def test_agent_is_registered(self) -> None:
        assert AVAILABLE_AGENTS["guidedllmls20openrouter"] is llm_module.GuidedLLMls20OpenRouter

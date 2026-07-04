"""Unit tests for the OpenRouter-backed guided game-state Locksmith agent."""

import json
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


def _observation_response(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(total_tokens=11)
    return response


def _frame(state: GameState = GameState.NOT_FINISHED) -> FrameData:
    return FrameData(
        game_id="ls20-test",
        state=state,
        available_actions=[1, 2, 3, 4, 5, 6],
        frame=[[[0] * 4 for _ in range(4)]],
    )


def _belief_state(
    action: str = "ACTION3",
    reason: str = "Move left toward the key corridor.",
) -> dict[str, object]:
    return {
        "game_mechanics": "Collect keys, open matching doors, and navigate a tile grid.",
        "game_goal": "Reach WIN without entering a losing state.",
        "current_state": "The agent is at the starting room with one visible corridor.",
        "next_recommended_action": {
            "action": action,
            "input_data": {},
            "reason": reason,
        },
    }


def _make_agent(
    monkeypatch: pytest.MonkeyPatch,
    *,
    observation_content: str | None = None,
    action_name: str = "ACTION3",
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")

    observation_payload = observation_content or json.dumps(_belief_state())

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _observation_response(observation_payload),
        _tool_response(action_name),
    ]
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(llm_module, "OpenAIClient", client_factory)

    agent = llm_module.GuidedGameStateLLMOpenRouter(
        card_id="card-abc",
        game_id="ls20-test",
        agent_name="guided-game-state-openrouter-test",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=MagicMock(),
    )
    return agent, client, client_factory


@pytest.mark.unit
class TestGuidedGameStateOpenRouterValidation:
    def test_requires_openrouter_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            llm_module.GuidedGameStateLLMOpenRouter(
                card_id="card-abc",
                game_id="ls20-test",
                agent_name="guided-game-state-openrouter-test",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=MagicMock(),
            )

    def test_requires_openrouter_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

        with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
            llm_module.GuidedGameStateLLMOpenRouter(
                card_id="card-abc",
                game_id="ls20-test",
                agent_name="guided-game-state-openrouter-test",
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
            llm_module.GuidedGameStateLLMOpenRouter(
                card_id="card-abc",
                game_id="ls20-test",
                agent_name="guided-game-state-openrouter-test",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=MagicMock(),
            )


@pytest.mark.unit
class TestGuidedGameStateOpenRouterBehavior:
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

    def test_observation_response_is_parsed_into_canonical_belief_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _, _ = _make_agent(monkeypatch)

        agent.choose_action([], _frame())
        action = agent.choose_action([_frame()], _frame())

        expected = _belief_state()

        assert action is GameAction.ACTION3
        assert agent.belief_state == expected
        assert action.reasoning["belief_state"] == expected
        assert json.loads(action.reasoning["belief_state_raw"]) == expected

    def test_next_action_call_includes_latest_canonical_belief_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        belief_state = _belief_state(action="ACTION4")
        agent, client, _ = _make_agent(
            monkeypatch, observation_content=json.dumps(belief_state), action_name="ACTION4"
        )

        agent.choose_action([], _frame())
        agent.choose_action([_frame()], _frame())

        action_call = client.chat.completions.create.call_args_list[1]
        user_messages = [
            message["content"]
            for message in action_call.kwargs["messages"]
            if isinstance(message, dict) and message.get("role") == "user"
        ]

        assert any('"game_mechanics": "Collect keys, open matching doors, and navigate a tile grid."' in message for message in user_messages)
        assert any('"action": "ACTION4"' in message for message in user_messages)

    def test_invalid_observation_json_records_error_and_uses_placeholder_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _, _ = _make_agent(monkeypatch, observation_content="not-json")

        agent.choose_action([], _frame())
        action = agent.choose_action([_frame()], _frame())

        assert action is GameAction.ACTION3
        assert action.reasoning["belief_state_error"]
        assert agent.belief_state["next_recommended_action"]["action"] == "ACTION5"
        assert "Fallback placeholder" in agent.belief_state["next_recommended_action"]["reason"]

    def test_invalid_observation_keeps_last_valid_belief_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
        monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _observation_response(json.dumps(_belief_state(action="ACTION2"))),
            _tool_response("ACTION2"),
            _observation_response("definitely-not-json"),
            _tool_response("ACTION6"),
        ]
        monkeypatch.setattr(llm_module, "OpenAIClient", MagicMock(return_value=client))

        agent = llm_module.GuidedGameStateLLMOpenRouter(
            card_id="card-abc",
            game_id="ls20-test",
            agent_name="guided-game-state-openrouter-test",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=MagicMock(),
        )

        agent.choose_action([], _frame())
        first_action = agent.choose_action([_frame()], _frame())
        second_action = agent.choose_action([_frame(), _frame()], _frame())

        assert first_action is GameAction.ACTION2
        assert second_action is GameAction.ACTION6
        assert agent.belief_state == _belief_state(action="ACTION2")
        assert second_action.reasoning["belief_state"] == _belief_state(action="ACTION2")
        assert second_action.reasoning["belief_state_error"]


@pytest.mark.unit
class TestGuidedGameStateOpenRouterRegistration:
    def test_agent_is_registered(self) -> None:
        assert AVAILABLE_AGENTS["guidedgamestatellmopenrouter"] is llm_module.GuidedGameStateLLMOpenRouter

import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main


class DummyAgent:
    MAX_ACTIONS = 10


@pytest.mark.unit
class TestMainHelpers:
    def test_positive_int_accepts_positive_values(self):
        assert main.positive_int("25") == 25

    def test_positive_int_rejects_zero_and_negative_values(self):
        with pytest.raises(argparse.ArgumentTypeError):
            main.positive_int("0")

        with pytest.raises(argparse.ArgumentTypeError):
            main.positive_int("-1")

    def test_apply_steps_override_updates_selected_agent_class_only(self):
        other_agent = type("OtherAgent", (), {"MAX_ACTIONS": 80})
        available_agents = {
            "dummyagent": DummyAgent,
            "otheragent": other_agent,
        }
        original_dummy_max_actions = DummyAgent.MAX_ACTIONS
        original_other_max_actions = other_agent.MAX_ACTIONS

        try:
            selected_class = main.apply_steps_override(
                available_agents=available_agents,
                agent_name="dummyagent",
                steps=25,
            )

            assert selected_class is DummyAgent
            assert DummyAgent.MAX_ACTIONS == 25
            assert other_agent.MAX_ACTIONS == 80
        finally:
            DummyAgent.MAX_ACTIONS = original_dummy_max_actions
            other_agent.MAX_ACTIONS = original_other_max_actions

    def test_main_applies_steps_override_before_swarm_starts(self, monkeypatch):
        session = Mock()
        response = Mock()
        response.status_code = 200
        response.json.return_value = [{"game_id": "ls20"}]
        session.get.return_value = response
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)

        dummy_swarm = Mock()
        dummy_thread = Mock()
        dummy_thread.is_alive.return_value = False

        monkeypatch.setattr(main.requests, "Session", Mock(return_value=session))
        monkeypatch.setattr(main, "init_agentops", Mock())
        monkeypatch.setattr(main.threading, "Thread", Mock(return_value=dummy_thread))
        monkeypatch.setattr(main.signal, "signal", Mock())
        monkeypatch.setattr(main, "Swarm", Mock(return_value=dummy_swarm))
        monkeypatch.setattr(
            main,
            "AVAILABLE_AGENTS",
            {"dummyagent": DummyAgent},
        )
        monkeypatch.setattr(
            main.argparse.ArgumentParser,
            "parse_args",
            Mock(
                return_value=SimpleNamespace(
                    agent="dummyagent",
                    game="ls20",
                    tags=None,
                    steps=25,
                )
            ),
        )

        original_dummy_max_actions = DummyAgent.MAX_ACTIONS

        try:
            main.main()
            assert DummyAgent.MAX_ACTIONS == 25
            main.Swarm.assert_called_once_with(
                "dummyagent",
                main.ROOT_URL,
                ["ls20"],
                tags=[],
            )
        finally:
            DummyAgent.MAX_ACTIONS = original_dummy_max_actions

    def test_apply_steps_override_leaves_default_when_steps_not_provided(self):
        available_agents = {"dummyagent": DummyAgent}
        original_dummy_max_actions = DummyAgent.MAX_ACTIONS

        try:
            selected_class = main.apply_steps_override(
                available_agents=available_agents,
                agent_name="dummyagent",
                steps=None,
            )

            assert selected_class is DummyAgent
            assert DummyAgent.MAX_ACTIONS == 10
        finally:
            DummyAgent.MAX_ACTIONS = original_dummy_max_actions

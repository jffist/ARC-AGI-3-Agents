import numpy as np
import pytest
from arcengine import ActionInput, FrameData, FrameDataRaw, GameAction, GameState

from agents.agent import Agent


class MinimalAgent(Agent):
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        return GameAction.RESET


@pytest.mark.unit
def test_raw_frame_conversion_preserves_action_input() -> None:
    raw = FrameDataRaw(
        game_id="test-game",
        state=GameState.NOT_FINISHED,
        levels_completed=2,
        win_levels=7,
        action_input=ActionInput(
            id=GameAction.ACTION6,
            data={"game_id": "test-game", "x": 12, "y": 34},
            reasoning={"text": "chosen by test"},
        ),
        guid="test-guid",
        available_actions=[1, 2, 6],
    )
    raw.frame = [np.array([[1, 2], [3, 4]])]
    agent = MinimalAgent(
        card_id="test-card",
        game_id="test-game",
        agent_name="minimal",
        ROOT_URL="https://example.com",
        record=False,
        arc_env=None,
    )

    frame = agent._convert_raw_frame_data(raw)

    assert frame.action_input.id is GameAction.ACTION6
    assert frame.action_input.data == {"game_id": "test-game", "x": 12, "y": 34}
    assert frame.action_input.reasoning == {"text": "chosen by test"}

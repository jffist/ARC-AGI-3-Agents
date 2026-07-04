import json
import logging
import os
import textwrap
from typing import Any, Optional

import openai
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

from ..agent import Agent

logger = logging.getLogger()


class LLM(Agent):
    """An agent that uses a base LLM model to play games."""

    MAX_ACTIONS: int = 10
    DO_OBSERVATION: bool = True
    REASONING_EFFORT: Optional[str] = None
    MODEL_REQUIRES_TOOLS: bool = False

    MESSAGE_LIMIT: int = 10
    MODEL: str = "gpt-4o-mini"
    messages: list[dict[str, Any]]
    token_counter: int

    _latest_tool_call_id: str = "call_12345"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages = []
        self.token_counter = 0

    def _build_client(self) -> OpenAIClient:
        return OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))

    def _resolve_model(self) -> str:
        return self.MODEL

    def _postprocess_observation_response(self, content: str) -> str:
        """Normalize the observation response before storing it in the transcript."""
        return content

    @property
    def name(self) -> str:
        obs = "with-observe" if self.DO_OBSERVATION else "no-observe"
        sanitized_model_name = self._resolve_model().replace("/", "-").replace(":", "-")
        name = f"{super().name}.{sanitized_model_name}.{obs}"
        if self.REASONING_EFFORT:
            name += f".{self.REASONING_EFFORT}"
        return name

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment below to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""

        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        client = self._build_client()
        model = self._resolve_model()

        functions = self.build_functions()
        tools = self.build_tools()

        # if latest_frame.state in [GameState.NOT_PLAYED]:
        if len(self.messages) == 0:
            # have to manually trigger the first reset to kick off agent
            user_prompt = self.build_user_prompt(latest_frame)
            message0 = {"role": "user", "content": user_prompt}
            self.push_message(message0)
            if self.MODEL_REQUIRES_TOOLS:
                message1 = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": self._latest_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": GameAction.RESET.name,
                                "arguments": json.dumps({}),
                            },
                        }
                    ],
                }
            else:
                message1 = {
                    "role": "assistant",
                    "function_call": {"name": "RESET", "arguments": json.dumps({})},  # type: ignore
                }
            self.push_message(message1)
            action = GameAction.RESET
            return action

        # let the agent comment observations before choosing action
        # on the first turn, this will be in response to RESET action
        function_name = latest_frame.action_input.id.name
        function_response = self.build_func_resp_prompt(latest_frame)
        if self.MODEL_REQUIRES_TOOLS:
            message2 = {
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": str(function_response),
            }
        else:
            message2 = {
                "role": "function",
                "name": function_name,
                "content": str(function_response),
            }
        self.push_message(message2)

        if self.DO_OBSERVATION:
            logger.info("Sending to Assistant for observation...")
            try:
                create_kwargs = {
                    "model": model,
                    "messages": self.messages,
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                response = client.chat.completions.create(**create_kwargs)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e
            observation_content = self._postprocess_observation_response(
                response.choices[0].message.content or ""
            )
            self.track_tokens(response.usage.total_tokens, observation_content)
            message3 = {
                "role": "assistant",
                "content": observation_content,
            }
            logger.info(f"Assistant: {observation_content}")
            self.push_message(message3)

        # now ask for the next action
        user_prompt = self.build_user_prompt(latest_frame)
        message4 = {"role": "user", "content": user_prompt}
        self.push_message(message4)

        name = GameAction.ACTION5.name  # default action if LLM doesnt call one
        arguments = None
        message5 = None

        if self.MODEL_REQUIRES_TOOLS:
            logger.info("Sending to Assistant for action...")
            try:
                create_kwargs = {
                    "model": model,
                    "messages": self.messages,
                    "tools": tools,
                    "tool_choice": "required",
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                response = client.chat.completions.create(**create_kwargs)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e
            self.track_tokens(response.usage.total_tokens)
            message5 = response.choices[0].message
            logger.debug(f"... got response {message5}")
            tool_call = message5.tool_calls[0]
            self._latest_tool_call_id = tool_call.id
            logger.debug(
                f"Assistant: {tool_call.function.name} ({tool_call.id}) {tool_call.function.arguments}"
            )
            name = tool_call.function.name
            arguments = tool_call.function.arguments

            # sometimes the model will call multiple tools which isnt allowed
            extra_tools = message5.tool_calls[1:]
            for tc in extra_tools:
                logger.info(
                    "Error: assistant called more than one action, only using the first."
                )
                message_extra = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Error: assistant can only call one action (tool) at a time. default to only the first chosen action.",
                }
                self.push_message(message_extra)
        else:
            logger.info("Sending to Assistant for action...")
            try:
                create_kwargs = {
                    "model": model,
                    "messages": self.messages,
                    "functions": functions,
                    "function_call": "auto",
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                response = client.chat.completions.create(**create_kwargs)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e
            self.track_tokens(response.usage.total_tokens)
            message5 = response.choices[0].message
            function_call = message5.function_call
            logger.debug(f"Assistant: {function_call.name} {function_call.arguments}")
            name = function_call.name
            arguments = function_call.arguments

        if message5:
            self.push_message(message5)
        action_id = name
        if arguments:
            try:
                data = json.loads(arguments) or {}
            except Exception as e:
                data = {}
                logger.warning(f"JSON parsing error on LLM function response: {e}")
        else:
            data = {}

        action = GameAction.from_name(action_id)
        action.set_data(data)
        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        self.token_counter += tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "tokens": tokens,
                    "total_tokens": self.token_counter,
                    "assistant": message,
                }
            )
        logger.info(f"Received {tokens} tokens, new total {self.token_counter}")
        # handle tool to debug messages:
        # with open("messages.json", "w") as f:
        #     json.dump(
        #         [
        #             msg if isinstance(msg, dict) else msg.model_dump()
        #             for msg in self.messages
        #         ],
        #         f,
        #         indent=2,
        #     )

    def push_message(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Push a message onto stack, store up to MESSAGE_LIMIT with FIFO."""
        self.messages.append(message)
        if len(self.messages) > self.MESSAGE_LIMIT:
            self.messages = self.messages[-self.MESSAGE_LIMIT :]
        if self.MODEL_REQUIRES_TOOLS:
            # cant clip the message list between tool
            # and tool_call else llm will error
            while (
                self.messages[0].get("role")
                if isinstance(self.messages[0], dict)
                else getattr(self.messages[0], "role", None)
            ) == "tool":
                self.messages.pop(0)
        return self.messages

    def build_functions(self) -> list[dict[str, Any]]:
        """Build JSON function description of game actions for LLM."""
        empty_params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        functions: list[dict[str, Any]] = [
            {
                "name": GameAction.RESET.name,
                "description": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION1.name,
                "description": "Send this simple input action (1, W, Up).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION2.name,
                "description": "Send this simple input action (2, S, Down).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION3.name,
                "description": "Send this simple input action (3, A, Left).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION4.name,
                "description": "Send this simple input action (4, D, Right).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION5.name,
                "description": "Send this simple input action (5, Enter, Spacebar, Delete).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION6.name,
                "description": "Send this complex input action (6, Click, Point).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "string",
                            "description": "Coordinate X which must be Int<0,63>",
                        },
                        "y": {
                            "type": "string",
                            "description": "Coordinate Y which must be Int<0,63>",
                        },
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            },
        ]
        return functions

    def build_tools(self) -> list[dict[str, Any]]:
        """Support models that expect tool_call format."""
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                        "strict": True,
                    },
                }
            )
        return tools

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# State:
{state}

# Score:
{score}

# Frame:
{latest_frame}

# TURN:
Reply with a few sentences of plain-text strategy observation about the frame to inform your next action.
        """.format(
                latest_frame=self.pretty_print_3d(latest_frame.frame),
                score=latest_frame.levels_completed,
                state=latest_frame.state.name,
            )
        )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Build the user prompt for the LLM. Override this method to customize the prompt."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
        """.format()
        )

    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
            lines.append("")
        return "\n".join(lines)

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup:
            if hasattr(self, "recorder") and not self.is_playback:
                meta = {
                    "llm_user_prompt": self.build_user_prompt(self.frames[-1]),
                    "llm_tools": self.build_tools()
                    if self.MODEL_REQUIRES_TOOLS
                    else self.build_functions(),
                    "llm_tool_resp_prompt": self.build_func_resp_prompt(
                        self.frames[-1]
                    ),
                }
                self.recorder.record(meta)
        super().cleanup(*args, **kwargs)


class ReasoningLLM(LLM, Agent):
    """An LLM agent that uses o4-mini and captures reasoning metadata in the action.reasoning field."""

    MAX_ACTIONS = 10
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MODEL = "o4-mini"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self._resolve_model(),
            "action_chosen": action.name,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.levels_completed,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "response_preview": self._last_response_content[:200] + "..."
            if len(self._last_response_content) > 200
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from reasoning models."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    def capture_reasoning_from_response(self, response: Any) -> None:
        """Helper method to capture reasoning tokens from OpenAI API response.

        This should be called from the parent class if we have access to the raw response.
        For reasoning models, reasoning tokens are in response.usage.completion_tokens_details.reasoning_tokens
        """
        if hasattr(response, "usage") and hasattr(
            response.usage, "completion_tokens_details"
        ):
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                self._last_reasoning_tokens = (
                    response.usage.completion_tokens_details.reasoning_tokens
                )
                self._total_reasoning_tokens += self._last_reasoning_tokens
                logger.debug(
                    f"Captured {self._last_reasoning_tokens} reasoning tokens from {self._resolve_model()} response"
                )


class FastLLM(LLM, Agent):
    """Similar to LLM, but skips observations."""

    MAX_ACTIONS = 10
    DO_OBSERVATION = False
    MODEL = "gpt-4o-mini"

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
        """.format()
        )

class GuidedLLMls20(LLM, Agent):
    """Similar to LLM, with explicit human-provided rules in the user prompt to increase success rate."""

    MAX_ACTIONS = 10
    DO_OBSERVATION = True
    #MODEL = "o3"
    #MODEL = "gpt-5.4" # Function tools with reasoning_effort are not supported for gpt-5.4 in /v1/chat/completions. Please use /v1/responses instead.', 'type': 'invalid_request_error
    MODEL = "gpt-5.2"
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 10
    REASONING_EFFORT = "high"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self._resolve_model(),
            "action_chosen": action.name,
            "reasoning_effort": self.REASONING_EFFORT,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.levels_completed,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "agent_type": "guided_llm",
            "game_rules": "locksmith",
            "response_preview": self._last_response_content[:2000] + "..."
            if len(self._last_response_content) > 2000
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from o3 models."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    def capture_reasoning_from_response(self, response: Any) -> None:
        """Helper method to capture reasoning tokens from OpenAI API response.

        This should be called from the parent class if we have access to the raw response.
        For o3 models, reasoning tokens are in response.usage.completion_tokens_details.reasoning_tokens
        """
        if hasattr(response, "usage") and hasattr(
            response.usage, "completion_tokens_details"
        ):
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                self._last_reasoning_tokens = (
                    response.usage.completion_tokens_details.reasoning_tokens
                )
                self._total_reasoning_tokens += self._last_reasoning_tokens
                logger.debug(
                    f"Captured {self._last_reasoning_tokens} reasoning tokens from {self._resolve_model()} response"
                )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

At each stage 
* analyse trace of previous actions and environment state changes
* produce hypothesis about game goal, mechanics, constraints and your current state
* choose most probable next action

# TURN:
Call exactly one action.
        """.format()
        )


class GuidedLLMls20OpenRouter(GuidedLLMls20, Agent):
    """Guided Locksmith agent that routes requests through OpenRouter."""

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self._openrouter_model = os.environ.get("OPENROUTER_MODEL", "").strip()
        if not self._openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY must be set for GuidedLLMls20OpenRouter")
        if not self._openrouter_model:
            raise ValueError("OPENROUTER_MODEL must be set for GuidedLLMls20OpenRouter")
        super().__init__(*args, **kwargs)

    def _build_client(self) -> OpenAIClient:
        return OpenAIClient(
            api_key=self._openrouter_api_key,
            base_url=self.OPENROUTER_BASE_URL,
        )

    def _resolve_model(self) -> str:
        return self._openrouter_model


class GuidedGameStateLLMOpenRouter(GuidedLLMls20OpenRouter, Agent):
    """OpenRouter-guided Locksmith agent with a persisted canonical belief state."""

    FALLBACK_BELIEF_STATE: dict[str, Any] = {
        "game_mechanics": "Unknown; infer from frame transitions.",
        "game_goal": "Reach WIN and avoid GAME_OVER while minimizing actions.",
        "current_state": "No validated belief state yet; rely on the latest frame and action trace.",
        "next_recommended_action": {
            "action": "ACTION5",
            "input_data": {},
            "reason": "Fallback placeholder because the observation output was invalid.",
        },
    }
    VALID_ACTION_NAMES = {action.name for action in GameAction}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.belief_state: dict[str, Any] | None = None
        self.belief_state_raw = ""
        self.belief_state_error: str | None = None

    def _canonical_belief_state_json(self) -> str:
        belief_state = self.belief_state or self.FALLBACK_BELIEF_STATE
        return json.dumps(belief_state, indent=2)

    def _validate_belief_state(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("belief state must be a JSON object")

        sections = (
            "game_mechanics",
            "game_goal",
            "current_state",
            "next_recommended_action",
        )
        for section in sections[:3]:
            value = payload.get(section)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{section} must be a non-empty string")

        next_action = payload.get("next_recommended_action")
        if not isinstance(next_action, dict):
            raise ValueError("next_recommended_action must be an object")

        action_name = next_action.get("action")
        if action_name not in self.VALID_ACTION_NAMES:
            raise ValueError("next_recommended_action.action must be a valid GameAction")

        input_data = next_action.get("input_data")
        if not isinstance(input_data, dict):
            raise ValueError("next_recommended_action.input_data must be an object")

        reason = next_action.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("next_recommended_action.reason must be a non-empty string")

        return {
            "game_mechanics": payload["game_mechanics"],
            "game_goal": payload["game_goal"],
            "current_state": payload["current_state"],
            "next_recommended_action": {
                "action": action_name,
                "input_data": input_data,
                "reason": reason,
            },
        }

    def _postprocess_observation_response(self, content: str) -> str:
        self.belief_state_raw = content
        self.belief_state_error = None
        try:
            parsed = json.loads(content)
            self.belief_state = self._validate_belief_state(parsed)
        except Exception as exc:
            self.belief_state_error = str(exc)
            if self.belief_state is None:
                self.belief_state = json.loads(json.dumps(self.FALLBACK_BELIEF_STATE))
            logger.warning("Failed to normalize belief state: %s", self.belief_state_error)
        return self._canonical_belief_state_json()

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# CONTEXT:
You are building a belief state for the game from the latest transition.
Return JSON only. Do not include markdown, prose outside JSON, or extra keys.

# REQUIRED_SCHEMA:
{{
  "game_mechanics": "string",
  "game_goal": "string",
  "current_state": "string",
  "next_recommended_action": {{
    "action": "RESET|ACTION1|ACTION2|ACTION3|ACTION4|ACTION5|ACTION6",
    "input_data": {{}},
    "reason": "string"
  }}
}}

# PRIOR_BELIEF_STATE:
{belief_state}

# LATEST_TRANSITION:
State: {state}
Score: {score}
Frame:
{latest_frame}

# TURN:
Infer the most likely rules, goal, current game situation, and the single best next action.
Return JSON only and ensure next_recommended_action.input_data is an object.
        """.format(
                belief_state=self._canonical_belief_state_json()
                if self.belief_state is not None
                else "null",
                latest_frame=self.pretty_print_3d(latest_frame.frame),
                score=latest_frame.levels_completed,
                state=latest_frame.state.name,
            )
        )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        belief_state_json = self._canonical_belief_state_json()
        recommended_action = (self.belief_state or self.FALLBACK_BELIEF_STATE)[
            "next_recommended_action"
        ]
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a game. Use the latest validated belief state as the default plan for this turn.
Call exactly one tool and prefer the recommended action unless the current frame clearly contradicts it.

# BELIEF_STATE:
{belief_state}

# DEFAULT_PLAN:
Action: {action}
Input: {input_data}
Reason: {reason}

# LIVE_STATE:
State: {state}
Score: {score}
Frame:
{latest_frame}

# TURN:
Call exactly one action tool. Do not call multiple tools.
        """.format(
                belief_state=belief_state_json,
                action=recommended_action["action"],
                input_data=json.dumps(recommended_action["input_data"]),
                reason=recommended_action["reason"],
                latest_frame=self.pretty_print_3d(latest_frame.frame),
                score=latest_frame.levels_completed,
                state=latest_frame.state.name,
            )
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        action = super().choose_action(frames, latest_frame)
        if action is GameAction.RESET:
            return action

        action.reasoning = {
            "model": self._resolve_model(),
            "action_chosen": action.name,
            "reasoning_effort": self.REASONING_EFFORT,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.levels_completed,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "agent_type": "guided_game_state_llm",
            "game_rules": "locksmith",
            "response_preview": self._last_response_content[:2000] + "..."
            if len(self._last_response_content) > 2000
            else self._last_response_content,
            "belief_state": self.belief_state,
            "belief_state_raw": self.belief_state_raw,
        }
        if self.belief_state_error:
            action.reasoning["belief_state_error"] = self.belief_state_error
        return action

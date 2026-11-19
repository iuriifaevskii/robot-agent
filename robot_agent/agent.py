"""Send prompts to the OpenAI model with bounded history.

Terminal input and output stay in cli.py. Saved history stays in storage.py.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai import (
    Agent,
    AgentRunError,
    ModelRequest,
    ModelSettings,
    UsageLimitExceeded,
    UsageLimits,
    UserError,
    UserPromptPart,
)
from pydantic_ai.messages import ModelMessage

from robot_agent.tools import ClockReadError, current_local_time


class AgentRequestError(Exception):
    """The model request failed. The message is safe to print."""


@dataclass(frozen=True)
class ModelLimits:
    """Per-request history and token bounds supplied by the CLI."""

    max_history_turns: int
    max_request_input_tokens: int
    max_output_tokens: int


def run_model(
    model_name: str,
    prompt: str,
    user_id: str,
    message_history: Sequence[ModelMessage],
    limits: ModelLimits,
) -> tuple[str, list[ModelMessage], list[ModelMessage]]:
    """Call the OpenAI model for the resolved user.

    The current local date and time tool is available for the model to call.
    This function does not call that tool itself. Tests replace the agent so
    checks never call the OpenAI API or its token-counting endpoint. user_id
    is the application user. It is not part of the prompt or history.
    message_history is prior completed turns. The current prompt is sent
    separately, once. The second list keeps at most
    limits.max_history_turns complete turns after this successful run.
    The third list is only the messages produced by this run.
    """
    outgoing = _recent_turns(message_history, limits.max_history_turns)
    # The openai: provider reads OPENAI_API_KEY from the environment.
    # count_tokens_before_request uses the OpenAI Responses input-token endpoint
    # and rejects an oversized request before generation.
    model_agent = Agent(f"openai:{model_name}", tools=[current_local_time])
    settings: ModelSettings = {"max_tokens": limits.max_output_tokens}
    usage_limits = UsageLimits(
        per_request_input_tokens_limit=limits.max_request_input_tokens,
        count_tokens_before_request=True,
    )
    result = model_agent.run_sync(
        prompt,
        message_history=list(outgoing),
        model_settings=settings,
        usage_limits=usage_limits,
    )
    if not isinstance(result.output, str):
        raise AgentRequestError("The model returned a non-text response.")
    produced = list(result.new_messages())
    kept = _recent_turns(result.all_messages(), limits.max_history_turns)
    return result.output, kept, produced


def request_reply(
    model_name: str,
    user_id: str,
    prompt: str,
    message_history: Sequence[ModelMessage],
    limits: ModelLimits,
) -> tuple[str, list[ModelMessage], list[ModelMessage]]:
    """Return the reply, bounded history, and messages produced by this run."""
    try:
        return run_model(model_name, prompt, user_id, message_history, limits)
    except UsageLimitExceeded as exc:
        raise AgentRequestError(_input_limit_message(exc, limits)) from exc
    except ClockReadError as exc:
        raise AgentRequestError(_failure_message(exc)) from exc
    except (AgentRunError, UserError) as exc:
        # One line for the terminal. cli.py prints it and exits.
        raise AgentRequestError(_failure_message(exc)) from exc


def _recent_turns(messages: Sequence[ModelMessage], limit: int) -> list[ModelMessage]:
    """Keep the newest complete turns, including each turn's tool messages."""
    if limit <= 0:
        return []
    starts = [index for index, message in enumerate(messages) if _starts_turn(message)]
    if len(starts) <= limit:
        return list(messages)
    return list(messages[starts[-limit] :])


def _starts_turn(message: ModelMessage) -> bool:
    if not isinstance(message, ModelRequest):
        return False
    return any(isinstance(part, UserPromptPart) for part in message.parts)


def _input_limit_message(exc: UsageLimitExceeded, limits: ModelLimits) -> str:
    if "per_request_input_tokens_limit" in str(exc):
        return (
            "The request exceeds the input token limit of "
            f"{limits.max_request_input_tokens}."
        )
    return _failure_message(exc)


def _failure_message(exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return detail
    return exc.__class__.__name__

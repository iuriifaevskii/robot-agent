"""Send prompts to the OpenAI model with bounded history.

Terminal input and output stay in cli.py. Saved history stays in storage.py.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

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

from robot_agent.context import RecalledPrompt, reference_prompt
from robot_agent.tools import ClockReadError, current_local_time


class AgentRequestError(Exception):
    """The model request failed. The message is safe to print."""


@dataclass(frozen=True)
class ModelLimits:
    """Per-request history, recall, and token bounds supplied by the CLI."""

    max_history_turns: int
    max_request_input_tokens: int
    max_output_tokens: int
    max_recalled_memories: int = 0


def run_model(
    model_name: str,
    prompt: str | RecalledPrompt,
    user_id: str,
    message_history: Sequence[ModelMessage],
    limits: ModelLimits,
) -> tuple[str, list[ModelMessage], list[ModelMessage]]:
    """Call the OpenAI model for the resolved user.

    The current local date and time tool is available for the model to call.
    This function does not call that tool itself. Tests replace the agent so
    checks never call the OpenAI API or its token-counting endpoint. user_id
    is the application user. It is not part of the prompt or history.
    message_history is prior completed turns. The current question is sent
    once. Recalled fact text is untrusted reference data for this request,
    then removed from both returned message lists. The second list keeps at
    most limits.max_history_turns complete turns after this successful run.
    The third list is only the messages produced by this run.
    """
    question, sent = _outgoing_prompt(prompt)
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
        sent,
        message_history=list(outgoing),
        model_settings=settings,
        usage_limits=usage_limits,
    )
    if not isinstance(result.output, str):
        raise AgentRequestError("The model returned a non-text response.")
    produced = _drop_reference(list(result.new_messages()), sent, question)
    kept = _recent_turns(
        _drop_reference(list(result.all_messages()), sent, question),
        limits.max_history_turns,
    )
    return result.output, kept, produced


def request_reply(
    model_name: str,
    user_id: str,
    prompt: str | RecalledPrompt,
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


def _outgoing_prompt(prompt: str | RecalledPrompt) -> tuple[str, str]:
    if isinstance(prompt, str):
        return prompt, prompt
    return prompt.text, reference_prompt(prompt.text, prompt.facts)


def _drop_reference(
    messages: list[ModelMessage],
    sent: str,
    question: str,
) -> list[ModelMessage]:
    """Replace this run's reference prompt with the original question."""
    if sent == question:
        return messages
    index = _reference_index(messages, sent)
    if index is None:
        return messages
    rewritten = _rewrite_prompt(messages[index], sent, question)
    return [*messages[:index], rewritten, *messages[index + 1 :]]


def _reference_index(messages: Sequence[ModelMessage], sent: str) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if _contains_prompt(messages[index], sent):
            return index
    return None


def _contains_prompt(message: ModelMessage, sent: str) -> bool:
    if not isinstance(message, ModelRequest):
        return False
    return any(
        isinstance(part, UserPromptPart) and part.content == sent
        for part in message.parts
    )


def _rewrite_prompt(message: ModelMessage, sent: str, question: str) -> ModelMessage:
    if not isinstance(message, ModelRequest):
        return message
    parts = []
    replaced = False
    for part in message.parts:
        if not replaced and isinstance(part, UserPromptPart) and part.content == sent:
            parts.append(replace(part, content=question))
            replaced = True
        else:
            parts.append(part)
    return replace(message, parts=parts)


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

"""Send prompts to the OpenAI model with in-memory history.

Terminal input and output stay in cli.py. History lives with the active chat.
"""

from collections.abc import Sequence

from pydantic_ai import Agent, AgentRunError, UserError
from pydantic_ai.messages import ModelMessage


class AgentRequestError(Exception):
    """The model request failed. The message is safe to print."""


def run_model(
    model_name: str,
    prompt: str,
    user_id: str,
    message_history: Sequence[ModelMessage],
) -> tuple[str, list[ModelMessage]]:
    """Call the OpenAI model for the resolved user.

    Tests replace the agent so checks never call the OpenAI API.
    user_id is the application user. It is not part of the prompt or history.
    message_history is the completed conversation so far. The returned list is
    the history to keep after this successful run, each turn included once.
    """
    # The openai: provider reads OPENAI_API_KEY from the environment.
    model_agent = Agent(f"openai:{model_name}")
    result = model_agent.run_sync(prompt, message_history=list(message_history))
    if not isinstance(result.output, str):
        raise AgentRequestError("The model returned a non-text response.")
    return result.output, list(result.all_messages())


def request_reply(
    model_name: str,
    user_id: str,
    prompt: str,
    message_history: Sequence[ModelMessage],
) -> tuple[str, list[ModelMessage]]:
    """Return the model's text reply and the history after a successful run."""
    try:
        return run_model(model_name, prompt, user_id, message_history)
    except (AgentRunError, UserError) as exc:
        # One line for the terminal. cli.py prints it and exits.
        raise AgentRequestError(_failure_message(exc)) from exc


def _failure_message(exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return detail
    return exc.__class__.__name__

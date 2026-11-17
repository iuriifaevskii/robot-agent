"""Send one prompt to the OpenAI model.

Terminal input and output stay in cli.py.
"""

from pydantic_ai import Agent, AgentRunError, UserError


class AgentRequestError(Exception):
    """The model request failed. The message is safe to print."""


def run_model(model_name: str, prompt: str, user_id: str) -> str:
    """Call the OpenAI model for the resolved user.

    Tests replace this function so checks never call the OpenAI API.
    user_id is the application user. It is not part of the prompt.
    """
    # The openai: provider reads OPENAI_API_KEY from the environment.
    model_agent = Agent(f"openai:{model_name}")
    result = model_agent.run_sync(prompt)
    if not isinstance(result.output, str):
        raise AgentRequestError("The model returned a non-text response.")
    return result.output


def request_reply(model_name: str, user_id: str, prompt: str) -> str:
    """Return the model's text reply for one user line."""
    try:
        return run_model(model_name, prompt, user_id)
    except (AgentRunError, UserError) as exc:
        # One line for the terminal. cli.py prints it and exits.
        raise AgentRequestError(_failure_message(exc)) from exc


def _failure_message(exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return detail
    return exc.__class__.__name__

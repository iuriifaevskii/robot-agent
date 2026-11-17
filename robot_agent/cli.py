import os
import sys

from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessage

from robot_agent.agent import AgentRequestError, request_reply
from robot_agent.identity import resolve_user

REQUIRED_ENV = ("OPENAI_API_KEY", "OPENAI_MODEL")
EXIT_COMMAND = "/exit"


def main() -> None:
    load_dotenv()
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        print(
            f"Missing configuration: {joined}. "
            "Copy .env.example to .env and set the values.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    model_name = os.environ["OPENAI_MODEL"]
    profile = resolve_user()
    run_chat(model_name, profile.user_id)


def run_chat(model_name: str, user_id: str) -> None:
    """Read lines until /exit or end of input.

    Successful turns stay in memory until this invocation returns.
    """
    history: list[ModelMessage] = []
    while True:
        line = _read_line()
        if line is None:
            return
        history, finished = _turn(model_name, user_id, line, history)
        if finished:
            return


def _read_line() -> str | None:
    try:
        return input()
    except EOFError:
        return None


def _turn(
    model_name: str,
    user_id: str,
    line: str,
    history: list[ModelMessage],
) -> tuple[list[ModelMessage], bool]:
    """Handle one line. Return history and whether the chat should stop."""
    text = line.strip()
    if not text:
        return history, False
    if text == EXIT_COMMAND:
        return history, True
    reply, history = _reply(model_name, user_id, text, history)
    print(reply)
    return history, False


def _reply(
    model_name: str,
    user_id: str,
    prompt: str,
    history: list[ModelMessage],
) -> tuple[str, list[ModelMessage]]:
    try:
        return request_reply(model_name, user_id, prompt, history)
    except AgentRequestError as exc:
        print(f"Agent request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

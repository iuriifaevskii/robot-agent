import os
import sys

from dotenv import load_dotenv

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
    """Read lines until /exit or end of input. Each line is one request."""
    while True:
        line = _read_line()
        if line is None:
            return
        if _turn(model_name, user_id, line):
            return


def _read_line() -> str | None:
    try:
        return input()
    except EOFError:
        return None


def _turn(model_name: str, user_id: str, line: str) -> bool:
    """Handle one line. Return True when the chat should stop."""
    text = line.strip()
    if not text:
        return False
    if text == EXIT_COMMAND:
        return True
    print(_reply(model_name, user_id, text))
    return False


def _reply(model_name: str, user_id: str, prompt: str) -> str:
    try:
        return request_reply(model_name, user_id, prompt)
    except AgentRequestError as exc:
        print(f"Agent request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

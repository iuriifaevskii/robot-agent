import os
import sys

from dotenv import load_dotenv

from robot_agent.agent import AgentRequestError, request_reply
from robot_agent.identity import resolve_user

REQUIRED_ENV = ("OPENAI_API_KEY", "OPENAI_MODEL")


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
    try:
        reply = request_reply(model_name, profile.user_id)
    except AgentRequestError as exc:
        print(f"Agent request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(reply)

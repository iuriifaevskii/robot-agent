import os
import sys

from dotenv import load_dotenv

from agent import AgentRequestError, request_reply

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
    try:
        reply = request_reply(model_name)
    except AgentRequestError as exc:
        print(f"Agent request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(reply)


if __name__ == "__main__":
    main()

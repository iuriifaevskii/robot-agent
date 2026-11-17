import os
import sys
from typing import NamedTuple

from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessage

from robot_agent.agent import AgentRequestError, ModelLimits, request_reply
from robot_agent.identity import resolve_user

REQUIRED_ENV = ("OPENAI_API_KEY", "OPENAI_MODEL")
EXIT_COMMAND = "/exit"


class _LimitSpec(NamedTuple):
    name: str
    default: int
    minimum: int
    description: str


_LIMIT_SPECS = (
    _LimitSpec("AGENT_MAX_HISTORY_TURNS", 6, 0, "a non-negative integer"),
    _LimitSpec("AGENT_MAX_REQUEST_INPUT_TOKENS", 4096, 1, "a positive integer"),
    _LimitSpec("AGENT_MAX_OUTPUT_TOKENS", 256, 1, "a positive integer"),
)


def main() -> None:
    load_dotenv()
    _require_env()
    limits = _request_limits()
    model_name = os.environ["OPENAI_MODEL"]
    profile = resolve_user()
    run_chat(model_name, profile.user_id, limits)


def run_chat(model_name: str, user_id: str, limits: ModelLimits) -> None:
    """Read lines until /exit or end of input.

    Successful turns stay in memory until this invocation returns.
    At most limits.max_history_turns complete turns are kept.
    """
    history: list[ModelMessage] = []
    while True:
        line = _read_line()
        if line is None:
            return
        history, finished = _turn(model_name, user_id, line, history, limits)
        if finished:
            return


def _require_env() -> None:
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if not missing:
        return
    joined = ", ".join(missing)
    print(
        f"Missing configuration: {joined}. "
        "Copy .env.example to .env and set the values.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _request_limits() -> ModelLimits:
    values = {spec.name: _parse_limit(spec) for spec in _LIMIT_SPECS}
    return ModelLimits(
        max_history_turns=values["AGENT_MAX_HISTORY_TURNS"],
        max_request_input_tokens=values["AGENT_MAX_REQUEST_INPUT_TOKENS"],
        max_output_tokens=values["AGENT_MAX_OUTPUT_TOKENS"],
    )


def _parse_limit(spec: _LimitSpec) -> int:
    raw = os.getenv(spec.name)
    if raw is None:
        return spec.default
    value = _integer_setting(raw, spec)
    if value < spec.minimum:
        _invalid_limit(spec)
    return value


def _integer_setting(raw: str, spec: _LimitSpec) -> int:
    try:
        return int(raw.strip())
    except ValueError:
        _invalid_limit(spec)


def _invalid_limit(spec: _LimitSpec) -> None:
    print(
        f"Invalid configuration: {spec.name} must be {spec.description}.",
        file=sys.stderr,
    )
    raise SystemExit(1)


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
    limits: ModelLimits,
) -> tuple[list[ModelMessage], bool]:
    """Handle one line. Return history and whether the chat should stop."""
    text = line.strip()
    if not text:
        return history, False
    if text == EXIT_COMMAND:
        return history, True
    reply, history = _reply(model_name, user_id, text, history, limits)
    print(reply)
    return history, False


def _reply(
    model_name: str,
    user_id: str,
    prompt: str,
    history: list[ModelMessage],
    limits: ModelLimits,
) -> tuple[str, list[ModelMessage]]:
    try:
        return request_reply(model_name, user_id, prompt, history, limits)
    except AgentRequestError as exc:
        print(f"Agent request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

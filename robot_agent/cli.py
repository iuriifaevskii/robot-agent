import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessage

from robot_agent.agent import AgentRequestError, ModelLimits, request_reply
from robot_agent.identity import UserProfile, resolve_user
from robot_agent.storage import ConversationStore, StorageError, connect

REQUIRED_ENV = ("OPENAI_API_KEY", "OPENAI_MODEL")
DEFAULT_DATABASE_PATH = "robot-agent.sqlite3"
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


@dataclass
class ActiveChat:
    """The current user's session and the turns loaded for this process."""

    user_id: str
    session_id: str
    history: list[ModelMessage]
    store: ConversationStore


def main() -> None:
    load_dotenv()
    _require_env()
    limits = _request_limits()
    profile = resolve_user()
    try:
        chat = _open_chat(profile, limits.max_history_turns)
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    run_chat(os.environ["OPENAI_MODEL"], limits, chat)


def run_chat(model_name: str, limits: ModelLimits, chat: ActiveChat) -> None:
    """Read lines until /exit or end of input.

    Each successful turn is stored before its reply is printed.
    At most limits.max_history_turns complete turns are sent next.
    """
    while True:
        line = _read_line()
        if line is None:
            return
        if _turn(model_name, line, limits, chat):
            return


def _open_chat(profile: UserProfile, history_limit: int) -> ActiveChat:
    store = connect(_database_path())
    session_id = store.ensure_session(profile.user_id, profile.display_name)
    history = store.load_recent(profile.user_id, session_id, history_limit)
    return ActiveChat(profile.user_id, session_id, history, store)


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


def _database_path() -> Path:
    raw = os.getenv("AGENT_DB_PATH", DEFAULT_DATABASE_PATH).strip()
    if not raw:
        raw = DEFAULT_DATABASE_PATH
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _read_line() -> str | None:
    try:
        return input()
    except EOFError:
        return None


def _turn(
    model_name: str,
    line: str,
    limits: ModelLimits,
    chat: ActiveChat,
) -> bool:
    """Handle one line. Return whether the chat should stop."""
    text = line.strip()
    if not text:
        return False
    if text == EXIT_COMMAND:
        return True
    reply, history, produced = _reply(
        model_name,
        chat.user_id,
        text,
        chat.history,
        limits,
    )
    _save_turn(chat, produced)
    chat.history = history
    print(reply)
    return False


def _save_turn(chat: ActiveChat, messages: list[ModelMessage]) -> None:
    try:
        chat.store.append_turn(chat.user_id, chat.session_id, messages)
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def _reply(
    model_name: str,
    user_id: str,
    prompt: str,
    history: list[ModelMessage],
    limits: ModelLimits,
) -> tuple[str, list[ModelMessage], list[ModelMessage]]:
    try:
        return request_reply(model_name, user_id, prompt, history, limits)
    except AgentRequestError as exc:
        print(f"Agent request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

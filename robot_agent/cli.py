import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessage

from robot_agent.agent import AgentRequestError, ModelLimits, request_reply
from robot_agent.identity import UserProfile, resolve_user
from robot_agent.sessions import (
    create_session,
    list_sessions,
    open_session,
    resume_session,
)
from robot_agent.storage import ConversationStore, StorageError, connect

REQUIRED_ENV = ("OPENAI_API_KEY", "OPENAI_MODEL")
DEFAULT_DATABASE_PATH = "robot-agent.sqlite3"
EXIT_COMMAND = "/exit"
NEW_COMMAND = "/new"
SESSIONS_COMMAND = "/sessions"
RESUME_COMMAND = "/resume"
NEW_USAGE = "Usage: /new"
SESSIONS_USAGE = "Usage: /sessions"
RESUME_USAGE = "Usage: /resume <session_id>"


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
    display_name: str
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

    Session commands run locally. Each successful model turn is stored
    before its reply is printed. At most limits.max_history_turns
    complete turns are sent next.
    """
    while True:
        line = _read_line()
        if line is None:
            return
        if _turn(model_name, line, limits, chat):
            return


def _open_chat(profile: UserProfile, history_limit: int) -> ActiveChat:
    store = connect(_database_path())
    session_id = open_session(store, profile.user_id, profile.display_name)
    history = store.load_recent(profile.user_id, session_id, history_limit)
    return ActiveChat(
        profile.user_id,
        profile.display_name,
        session_id,
        history,
        store,
    )


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
    if _session_command(text, limits, chat):
        return False
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


def _session_command(text: str, limits: ModelLimits, chat: ActiveChat) -> bool:
    """Run a local session command. Return whether the line was one.

    Invalid arguments print usage and leave the current session active.
    These commands do not call the model or record a turn.
    """
    name, args = _command_parts(text)
    if name == NEW_COMMAND:
        _new_session(args, chat)
        return True
    if name == SESSIONS_COMMAND:
        _list_sessions(args, chat)
        return True
    if name == RESUME_COMMAND:
        _resume_session(args, limits, chat)
        return True
    return False


def _command_parts(text: str) -> tuple[str, list[str]]:
    parts = text.split()
    return parts[0], parts[1:]


def _new_session(args: list[str], chat: ActiveChat) -> None:
    if args:
        print(NEW_USAGE, file=sys.stderr)
        return
    try:
        session_id = create_session(chat.store, chat.user_id, chat.display_name)
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        return
    chat.session_id = session_id
    chat.history = []
    print(session_id)


def _list_sessions(args: list[str], chat: ActiveChat) -> None:
    if args:
        print(SESSIONS_USAGE, file=sys.stderr)
        return
    try:
        found = list_sessions(chat.store, chat.user_id)
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        return
    for record in found:
        print(_session_line(record.session_id, record.created_at, chat))


def _resume_session(
    args: list[str],
    limits: ModelLimits,
    chat: ActiveChat,
) -> None:
    if len(args) != 1:
        print(RESUME_USAGE, file=sys.stderr)
        return
    try:
        history = resume_session(
            chat.store,
            chat.user_id,
            args[0],
            limits.max_history_turns,
        )
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        return
    chat.session_id = args[0]
    chat.history = history
    print(args[0])


def _session_line(session_id: str, created_at: str, chat: ActiveChat) -> str:
    if session_id == chat.session_id:
        return f"{session_id} {created_at} active"
    return f"{session_id} {created_at}"


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

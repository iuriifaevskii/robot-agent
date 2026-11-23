import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessage

from robot_agent import tools
from robot_agent.agent import (
    DEFAULT_MAX_MODEL_REQUESTS,
    REQUEST_LIMIT_SETTING,
    AgentRequestError,
    ModelLimits,
    ModelReply,
    request_reply,
    request_summary,
)
from robot_agent.context import (
    RecalledPrompt,
    recent_turns,
    select_fact_texts,
    sendable_turns,
    summary_span,
    summary_transcript,
)
from robot_agent.identity import UserProfile, resolve_user
from robot_agent.memory import (
    MemoryRecord,
    delete_memory,
    list_memories,
    save_memory,
    search_memories,
    update_memory,
)
from robot_agent.sessions import (
    create_session,
    list_sessions,
    open_session,
    resume_session,
)
from robot_agent.storage import ConversationStore, StorageError, connect
from robot_agent.usage import (
    MAX_REQUESTS_SETTING,
    MAX_TOKENS_SETTING,
    NO_BUDGET,
    UNTRUSTED_MESSAGE,
    DailyBudget,
    UsageCounts,
    blocking_limit,
    refusal_message,
)

REQUIRED_ENV = ("OPENAI_API_KEY", "OPENAI_MODEL")
DEFAULT_DATABASE_PATH = "robot-agent.sqlite3"
EXIT_COMMAND = "/exit"
NEW_COMMAND = "/new"
SESSIONS_COMMAND = "/sessions"
RESUME_COMMAND = "/resume"
TIME_COMMAND = "/time"
NEW_USAGE = "Usage: /new"
TIME_USAGE = "Usage: /time"
SESSIONS_USAGE = "Usage: /sessions"
RESUME_USAGE = "Usage: /resume <session_id>"
MEMORY_COMMAND = "/memory"
MEMORY_USAGE = (
    "Usage: /memory save <text> | list | search <query> | "
    "update <id> <text> | delete <id>"
)
NO_MEMORIES = "No saved facts."
NO_MEMORY_MATCHES = "No matching facts."


class _LimitSpec(NamedTuple):
    name: str
    default: int
    minimum: int
    maximum: int | None
    description: str


_LIMIT_SPECS = (
    _LimitSpec("AGENT_MAX_HISTORY_TURNS", 6, 0, None, "a non-negative integer"),
    _LimitSpec(
        "AGENT_MAX_REQUEST_INPUT_TOKENS",
        4096,
        1,
        None,
        "a positive integer",
    ),
    _LimitSpec("AGENT_MAX_OUTPUT_TOKENS", 256, 1, None, "a positive integer"),
    _LimitSpec(
        "AGENT_MAX_RECALLED_MEMORIES",
        0,
        0,
        3,
        "an integer from 0 through 3",
    ),
    _LimitSpec("AGENT_SUMMARY_AFTER_TURNS", 0, 0, None, "a non-negative integer"),
    _LimitSpec(
        REQUEST_LIMIT_SETTING,
        DEFAULT_MAX_MODEL_REQUESTS,
        1,
        None,
        "a positive integer",
    ),
)

_BUDGET_SPECS = (
    _LimitSpec(MAX_REQUESTS_SETTING, 0, 0, None, "a non-negative integer"),
    _LimitSpec(MAX_TOKENS_SETTING, 0, 0, None, "a non-negative integer"),
)


class _SummaryState(NamedTuple):
    """The stored summary text, its covered turn position and the saved turns."""

    text: str
    covered: int
    total: int


@dataclass
class ActiveChat:
    """The current user's session and the turns loaded for this process."""

    user_id: str
    display_name: str
    session_id: str
    history: list[ModelMessage]
    store: ConversationStore
    summary_failed: bool = False
    budget: DailyBudget = NO_BUDGET
    usage_untrusted: bool = False


def main() -> None:
    load_dotenv()
    _require_env()
    limits = _request_limits()
    budget = _daily_budget()
    profile = resolve_user()
    try:
        chat = _open_chat(profile, limits.max_history_turns, budget)
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    run_chat(os.environ["OPENAI_MODEL"], limits, chat)


def run_chat(model_name: str, limits: ModelLimits, chat: ActiveChat) -> None:
    """Read lines until /exit or end of input.

    Session, memory and /time commands run locally. Each successful model turn
    is stored before its reply is printed. At most limits.max_history_turns
    complete turns are sent next. Matching saved facts and a session summary
    may be attached to an ordinary request, and the stored turn keeps the
    original line.
    """
    while True:
        line = _read_line()
        if line is None:
            return
        if _turn(model_name, line, limits, chat):
            return


def _open_chat(
    profile: UserProfile,
    history_limit: int,
    budget: DailyBudget = NO_BUDGET,
) -> ActiveChat:
    store = connect(_database_path())
    session_id = open_session(store, profile.user_id, profile.display_name)
    history = store.load_recent(profile.user_id, session_id, history_limit)
    return ActiveChat(
        profile.user_id,
        profile.display_name,
        session_id,
        history,
        store,
        budget=budget,
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
        max_recalled_memories=values["AGENT_MAX_RECALLED_MEMORIES"],
        summary_after_turns=values["AGENT_SUMMARY_AFTER_TURNS"],
        max_model_requests=values[REQUEST_LIMIT_SETTING],
    )


def _daily_budget() -> DailyBudget:
    max_requests, max_tokens = (_parse_limit(spec) for spec in _BUDGET_SPECS)
    return DailyBudget(max_requests, max_tokens)


def _parse_limit(spec: _LimitSpec) -> int:
    raw = os.getenv(spec.name)
    if raw is None:
        return spec.default
    value = _integer_setting(raw, spec)
    if not _in_range(value, spec):
        _invalid_limit(spec)
    return value


def _in_range(value: int, spec: _LimitSpec) -> bool:
    if value < spec.minimum:
        return False
    return spec.maximum is None or value <= spec.maximum


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
    if _local_command(text, limits, chat):
        return False
    _paid_turn(model_name, text, limits, chat)
    return False


def _local_command(text: str, limits: ModelLimits, chat: ActiveChat) -> bool:
    """Run a local command. Return whether the line was one."""
    return (
        _session_command(text, limits, chat)
        or _memory_command(text, chat)
        or _time_command(text)
    )


def _paid_turn(
    model_name: str,
    text: str,
    limits: ModelLimits,
    chat: ActiveChat,
) -> None:
    """Send an ordinary line to the model unless the daily budget is reached.

    The budget is checked before the summarizer and again before the chat
    request. A refused line saves nothing and is not retried.
    """
    if not _budget_open(chat):
        return
    summary, outgoing = _summary_context(model_name, limits, chat)
    if not _budget_open(chat):
        return
    reply = _reply(
        model_name, _recall_prompt(text, summary, limits, chat), outgoing, limits, chat
    )
    _record_usage(chat, reply.usage)
    _save_turn(chat, reply.produced)
    chat.history = reply.history
    print(reply.text)


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


def _time_command(text: str) -> bool:
    """Print the local time. Return whether the line was /time.

    This does not call the model, read storage or usage, or record a turn.
    """
    name, args = _command_parts(text)
    if name != TIME_COMMAND:
        return False
    if args:
        print(TIME_USAGE, file=sys.stderr)
        return True
    try:
        timestamp = tools.current_local_time()
    except tools.ClockReadError as exc:
        print(str(exc), file=sys.stderr)
        return True
    print(timestamp)
    return True


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


def _memory_command(text: str, chat: ActiveChat) -> bool:
    """Run a local memory command. Return whether the line was one.

    Invalid forms print usage and leave saved facts unchanged.
    These commands do not call the model or record a turn.
    """
    if text.split(maxsplit=1)[0] != MEMORY_COMMAND:
        return False
    _run_memory(text[len(MEMORY_COMMAND) :].lstrip(), chat)
    return True


def _run_memory(remainder: str, chat: ActiveChat) -> None:
    if not remainder:
        print(MEMORY_USAGE, file=sys.stderr)
        return
    name, payload = _split_payload(remainder)
    handler = _MEMORY_HANDLERS.get(name)
    if handler is None:
        print(MEMORY_USAGE, file=sys.stderr)
        return
    handler(payload, chat)


def _split_payload(remainder: str) -> tuple[str, str]:
    parts = remainder.split(maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _save_memory(payload: str, chat: ActiveChat) -> None:
    if not payload:
        print(MEMORY_USAGE, file=sys.stderr)
        return
    try:
        memory_id = save_memory(
            chat.store,
            chat.user_id,
            chat.display_name,
            payload,
        )
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        return
    print(memory_id)


def _list_memories(payload: str, chat: ActiveChat) -> None:
    if payload:
        print(MEMORY_USAGE, file=sys.stderr)
        return
    try:
        found = list_memories(chat.store, chat.user_id)
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        return
    _print_memories(found, NO_MEMORIES)


def _search_memories(payload: str, chat: ActiveChat) -> None:
    if not payload:
        print(MEMORY_USAGE, file=sys.stderr)
        return
    try:
        found = search_memories(chat.store, chat.user_id, payload)
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        return
    _print_memories(found, NO_MEMORY_MATCHES)


def _update_memory(payload: str, chat: ActiveChat) -> None:
    target = _update_target(payload)
    if target is None:
        print(MEMORY_USAGE, file=sys.stderr)
        return
    memory_id, text = target
    try:
        update_memory(chat.store, chat.user_id, memory_id, text)
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        return
    print(f"Updated {memory_id}.")


def _delete_memory(payload: str, chat: ActiveChat) -> None:
    if len(payload.split()) != 1:
        print(MEMORY_USAGE, file=sys.stderr)
        return
    try:
        delete_memory(chat.store, chat.user_id, payload)
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        return
    print(f"Deleted {payload}.")


def _update_target(payload: str) -> tuple[str, str] | None:
    parts = payload.split(maxsplit=1)
    if len(parts) != 2 or not parts[1]:
        return None
    return parts[0], parts[1]


def _print_memories(found: list[MemoryRecord], empty_message: str) -> None:
    if not found:
        print(empty_message)
        return
    for record in found:
        print(f"{record.memory_id} {record.text}")


_MEMORY_HANDLERS = {
    "save": _save_memory,
    "list": _list_memories,
    "search": _search_memories,
    "update": _update_memory,
    "delete": _delete_memory,
}


def _save_turn(chat: ActiveChat, messages: list[ModelMessage]) -> None:
    try:
        chat.store.append_turn(chat.user_id, chat.session_id, messages)
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def _summary_context(
    model_name: str,
    limits: ModelLimits,
    chat: ActiveChat,
) -> tuple[str, list[ModelMessage]]:
    """Return the summary text and the turns to send, refreshing it when due.

    With summaries off, nothing is read and the history is unchanged.
    """
    if limits.summary_after_turns <= 0:
        return "", chat.history
    try:
        state = _summary_state(chat)
    except StorageError as exc:
        print(f"Warning: {exc}", file=sys.stderr)
        return "", chat.history
    state = _refresh_summary(model_name, limits, chat, state)
    count = sendable_turns(state.total, limits.max_history_turns, state.covered)
    return state.text, recent_turns(chat.history, count)


def _summary_state(chat: ActiveChat) -> _SummaryState:
    stored = chat.store.load_summary(chat.user_id, chat.session_id)
    total = chat.store.turn_count(chat.user_id, chat.session_id)
    if stored is None:
        return _SummaryState("", 0, total)
    return _SummaryState(stored[0], stored[1], total)


def _refresh_summary(
    model_name: str,
    limits: ModelLimits,
    chat: ActiveChat,
    state: _SummaryState,
) -> _SummaryState:
    """Make at most one summarizer request when enough older turns are uncovered."""
    span = summary_span(
        state.total,
        limits.max_history_turns,
        state.covered,
        limits.summary_after_turns,
    )
    if chat.summary_failed or span is None:
        return state
    try:
        text = _summarize(model_name, limits, chat, state, span)
    except AgentRequestError as exc:
        _record_usage(chat, exc.usage)
        return _summary_skipped(chat, state, exc)
    except StorageError as exc:
        return _summary_skipped(chat, state, exc)
    count = span[1] - span[0] + 1
    noun = "turn" if count == 1 else "turns"
    print(f"Summarized {count} earlier {noun}.", file=sys.stderr)
    return _SummaryState(text, span[1], state.total)


def _summary_skipped(
    chat: ActiveChat, state: _SummaryState, exc: Exception
) -> _SummaryState:
    chat.summary_failed = True
    print(f"Warning: summary skipped. {exc}", file=sys.stderr)
    return state


def _summarize(
    model_name: str,
    limits: ModelLimits,
    chat: ActiveChat,
    state: _SummaryState,
    span: tuple[int, int],
) -> str:
    older = chat.store.load_turns(chat.user_id, chat.session_id, span[0], span[1])
    text, usage = request_summary(
        model_name, state.text, summary_transcript(older), limits
    )
    _record_usage(chat, usage)
    chat.store.save_summary(chat.user_id, chat.session_id, text, span[1])
    return text


def _recall_prompt(
    text: str,
    summary: str,
    limits: ModelLimits,
    chat: ActiveChat,
) -> RecalledPrompt:
    """Attach the summary and current matching facts to the question."""
    return RecalledPrompt(text, _recalled_facts(text, limits, chat), summary)


def _recalled_facts(
    text: str, limits: ModelLimits, chat: ActiveChat
) -> tuple[str, ...]:
    if limits.max_recalled_memories <= 0:
        return ()
    try:
        records = list_memories(chat.store, chat.user_id)
    except StorageError as exc:
        print(str(exc), file=sys.stderr)
        return ()
    return tuple(select_fact_texts(text, records, limits.max_recalled_memories))


def _reply(
    model_name: str,
    prompt: RecalledPrompt,
    history: list[ModelMessage],
    limits: ModelLimits,
    chat: ActiveChat,
) -> ModelReply:
    try:
        return request_reply(model_name, chat.user_id, prompt, history, limits)
    except AgentRequestError as exc:
        _record_usage(chat, exc.usage)
        print(f"Agent request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _today() -> str:
    return tools.read_local_clock().date().isoformat()


def _budget_open(chat: ActiveChat) -> bool:
    """Return whether a paid request may be sent. Print the reason when not.

    With no limit enabled nothing is read. If today's counters cannot be
    determined, or an earlier recording failed, no request is sent.
    """
    if not chat.budget.enabled:
        return True
    if chat.usage_untrusted:
        print(UNTRUSTED_MESSAGE, file=sys.stderr)
        return False
    try:
        used = UsageCounts(*chat.store.daily_usage(_today()))
    except (StorageError, tools.ClockReadError) as exc:
        print(str(exc), file=sys.stderr)
        return False
    limit = blocking_limit(used, chat.budget)
    if limit is None:
        return True
    print(refusal_message(limit, used, chat.budget), file=sys.stderr)
    return False


def _record_usage(chat: ActiveChat, usage: UsageCounts | None) -> None:
    """Add a completed run's reported usage to today's counters."""
    if usage is None or not chat.budget.enabled:
        return
    try:
        chat.store.add_daily_usage(_today(), *usage)
    except (StorageError, tools.ClockReadError) as exc:
        chat.usage_untrusted = True
        print(f"Warning: daily usage was not recorded. {exc}", file=sys.stderr)

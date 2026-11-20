import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage

from robot_agent import cli, identity, storage
from robot_agent.memory import MEMORY_UNAVAILABLE
from tests.test_cli import (
    _exchange,
    _message_rows,
    _ModelScript,
    _scripted_input,
)
from tests.test_sessions import _body, _ready, _run, _seed_turns, _session_ids, _user_id

_FACT = "любить   чай"
_COLOR = "Улюблений колір — синій"
_PERCENT = "100% готово"
_UNDERSCORE = "оцінка_ризику"
_SECRET = "секретний факт"
_PRE_MEMORY_SCHEMA = """
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL
);
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users (user_id),
    created_at TEXT NOT NULL
);
CREATE TABLE messages (
    session_id TEXT NOT NULL REFERENCES sessions (session_id),
    position INTEGER NOT NULL,
    body TEXT NOT NULL,
    PRIMARY KEY (session_id, position)
);
CREATE TABLE active_sessions (
    user_id TEXT PRIMARY KEY REFERENCES users (user_id),
    session_id TEXT NOT NULL REFERENCES sessions (session_id)
);
"""


@dataclass(frozen=True)
class _StoredMemory:
    memory_id: str
    user_id: str
    text: str
    created_at: str
    updated_at: str
    source: str


def _memory_line(memory_id: str, text: str) -> str:
    return f"{memory_id} {text}"


def _usage_error() -> str:
    return f"{cli.MEMORY_USAGE}\n"


def _memory_rows(path: Path) -> list[_StoredMemory]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            """
            SELECT memory_id, user_id, text, created_at, updated_at, source
            FROM memories
            ORDER BY created_at, memory_id
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        _StoredMemory(
            str(memory_id),
            str(user_id),
            str(text),
            str(created_at),
            str(updated_at),
            str(source),
        )
        for memory_id, user_id, text, created_at, updated_at, source in rows
    ]


def _insert_memory(
    path: Path,
    user_id: str,
    memory_id: str,
    text: str,
    created_at: str,
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO memories (
                memory_id, user_id, text, created_at, updated_at, source
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                user_id,
                text,
                created_at,
                created_at,
                storage.CLI_MEMORY_SOURCE,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _has_table(path: Path, name: str) -> bool:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (name,),
        ).fetchone()
    finally:
        connection.close()
    return row is not None


def _write_pre_memory(path: Path, session_id: str, turn: list[ModelMessage]) -> None:
    profile = identity.resolve_user()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(_PRE_MEMORY_SCHEMA)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO users (user_id, display_name) VALUES (?, ?)",
            (profile.user_id, profile.display_name),
        )
        connection.execute(
            """
            INSERT INTO sessions (session_id, user_id, created_at)
            VALUES (?, ?, ?)
            """,
            (session_id, profile.user_id, "2026-04-01T00:00:00+00:00"),
        )
        connection.execute(
            """
            INSERT INTO messages (session_id, position, body)
            VALUES (?, 1, ?)
            """,
            (session_id, _body(turn)),
        )
        connection.execute(
            """
            INSERT INTO active_sessions (user_id, session_id)
            VALUES (?, ?)
            """,
            (profile.user_id, session_id),
        )
        connection.commit()
    finally:
        connection.close()


def _fail_disk(*_args: object) -> None:
    raise sqlite3.OperationalError("disk full")


def _visible_text(messages: list[ModelMessage]) -> str:
    chunks: list[str] = []
    for message in messages:
        for part in message.parts:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                chunks.append(content)
    return "\n".join(chunks)


def test_saved_fact_survives_new_session_and_restart(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    calls = _run(
        monkeypatch,
        [
            "  /memory save   любить   чай  ",
            "/memory save любить   чай",
            "/memory list",
            "/new",
            "/memory list",
            "/exit",
        ],
        _ModelScript(),
    )

    assert calls == []
    assert calls.token_counts == []
    printed = capsys.readouterr()
    assert printed.err == ""
    lines = printed.out.splitlines()
    older_id, newer_id, session_id = lines[0], lines[1], lines[4]
    listed = [
        _memory_line(newer_id, _FACT),
        _memory_line(older_id, _FACT),
    ]
    assert lines == [older_id, newer_id, *listed, session_id, *listed]
    assert _message_rows(database) == []
    assert len(_session_ids(database, _user_id())) == 2
    stored = _memory_rows(database)
    assert [row.memory_id for row in stored] == [older_id, newer_id]
    assert {row.text for row in stored} == {_FACT}
    assert {row.user_id for row in stored} == {_user_id()}
    assert {row.source for row in stored} == {storage.CLI_MEMORY_SOURCE}
    _scripted_input(monkeypatch, ["/memory list", "/exit"])
    cli.main()

    assert calls == []
    assert calls.token_counts == []
    assert capsys.readouterr().out == f"{listed[0]}\n{listed[1]}\n"
    assert _message_rows(database) == []


def test_search_ignores_case_and_treats_wildcards_literally(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    calls = _run(
        monkeypatch,
        [
            "/memory list",
            "/memory search київ",
            "/memory save Улюблений колір — синій",
            "/memory save 100% готово",
            "/memory save оцінка_ризику",
            "/memory search уЛЮблений Колір",
            "/memory search %",
            "/memory search _",
            "/memory search немає",
            "/memory list",
            "/exit",
        ],
        _ModelScript(),
    )

    assert calls == []
    assert calls.token_counts == []
    printed = capsys.readouterr()
    assert printed.err == ""
    lines = printed.out.splitlines()
    color_id, percent_id, underscore_id = lines[2], lines[3], lines[4]
    assert lines == [
        cli.NO_MEMORIES,
        cli.NO_MEMORY_MATCHES,
        color_id,
        percent_id,
        underscore_id,
        _memory_line(color_id, _COLOR),
        _memory_line(percent_id, _PERCENT),
        _memory_line(underscore_id, _UNDERSCORE),
        cli.NO_MEMORY_MATCHES,
        _memory_line(underscore_id, _UNDERSCORE),
        _memory_line(percent_id, _PERCENT),
        _memory_line(color_id, _COLOR),
    ]
    assert _message_rows(database) == []


def test_update_keeps_id_and_delete_removes_the_fact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    profile = identity.resolve_user()
    seeded = storage.connect(database)
    session_id = seeded.ensure_session(profile.user_id, profile.display_name)
    seeded.append_turn(profile.user_id, session_id, _exchange("hello", "saved"))
    before = _message_rows(database)
    created_at = "2026-05-02T00:00:00+00:00"
    updated_at = "2026-05-03T00:00:00+00:00"
    times = iter((created_at, updated_at))
    monkeypatch.setattr(storage, "_timestamp", lambda: next(times))
    calls = _run(
        monkeypatch,
        ["/memory save перший запис", "/exit"],
        _ModelScript(),
    )
    memory_id = capsys.readouterr().out.strip()
    saved = _memory_rows(database)[0]

    assert calls == []
    assert saved.memory_id == memory_id
    assert saved.created_at == created_at
    assert saved.updated_at == created_at
    calls = _run(
        monkeypatch,
        [f"/memory update {memory_id} оновлений   запис", "/memory list", "/exit"],
        _ModelScript(),
    )
    changed = _memory_rows(database)[0]

    assert calls == []
    assert calls.token_counts == []
    printed = capsys.readouterr()
    assert printed.err == ""
    assert printed.out == (
        f"Updated {memory_id}.\n{_memory_line(memory_id, 'оновлений   запис')}\n"
    )
    assert changed.memory_id == memory_id
    assert changed.created_at == created_at
    assert changed.updated_at == updated_at
    assert changed.text == "оновлений   запис"
    calls = _run(
        monkeypatch,
        [
            f"/memory delete {memory_id}",
            "/memory list",
            "/memory search запис",
            "/exit",
        ],
        _ModelScript(),
    )

    assert calls == []
    assert capsys.readouterr().out == (
        f"Deleted {memory_id}.\n{cli.NO_MEMORIES}\n{cli.NO_MEMORY_MATCHES}\n"
    )
    assert _memory_rows(database) == []
    _scripted_input(monkeypatch, ["/memory list", "/memory search запис", "/exit"])
    cli.main()

    assert calls == []
    assert calls.token_counts == []
    assert capsys.readouterr().out == f"{cli.NO_MEMORIES}\n{cli.NO_MEMORY_MATCHES}\n"
    assert _message_rows(database) == before
    assert _session_ids(database, profile.user_id) == [session_id]


def test_memory_commands_stay_with_the_current_user(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    profile = identity.resolve_user()
    store = storage.connect(database)
    store.ensure_session(profile.user_id, profile.display_name)
    other_id = store.save_memory("other-user", "Other", "чужий факт")
    other_before = _memory_rows(database)[0]
    _insert_memory(
        database, profile.user_id, "mem-b", "beta", "2026-01-02T00:00:00+00:00"
    )
    _insert_memory(
        database, profile.user_id, "mem-a", "alpha", "2026-01-02T00:00:00+00:00"
    )
    _insert_memory(
        database, profile.user_id, "mem-old", "old", "2026-01-01T00:00:00+00:00"
    )
    calls = _run(
        monkeypatch,
        [
            "/memory save мій факт",
            "/memory list",
            "/memory search a",
            "/memory search факт",
            "/memory search чужий",
            f"/memory update {other_id} stolen",
            f"/memory delete {other_id}",
            "/memory update missing-memory stolen",
            "/memory delete missing-memory",
            "/exit",
        ],
        _ModelScript(),
    )

    assert calls == []
    assert calls.token_counts == []
    printed = capsys.readouterr()
    lines = printed.out.splitlines()
    mine = lines[0]
    assert lines == [
        mine,
        _memory_line(mine, "мій факт"),
        _memory_line("mem-a", "alpha"),
        _memory_line("mem-b", "beta"),
        _memory_line("mem-old", "old"),
        _memory_line("mem-a", "alpha"),
        _memory_line("mem-b", "beta"),
        _memory_line(mine, "мій факт"),
        cli.NO_MEMORY_MATCHES,
    ]
    assert "чужий факт" not in printed.out
    assert printed.err == f"{MEMORY_UNAVAILABLE}\n" * 4
    stored = {row.memory_id: row for row in _memory_rows(database)}
    assert stored[other_id] == other_before
    assert _message_rows(database) == []


def test_invalid_memory_commands_keep_records_and_skip_the_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    calls = _run(
        monkeypatch,
        [
            "/memory",
            "/memory save",
            "/memory save   ",
            "/memory list now",
            "/memory search",
            "/memory update",
            "/memory update missing-id",
            "/memory delete",
            "/memory delete missing-id extra",
            "/memory unknown",
            "  /memory save   kept fact  ",
            "/memory list",
            "/exit",
        ],
        _ModelScript(replies=["should not be called"]),
    )

    assert calls == []
    assert calls.token_counts == []
    printed = capsys.readouterr()
    memory_id, listed = printed.out.splitlines()
    assert listed == _memory_line(memory_id, "kept fact")
    assert printed.err == _usage_error() * 10
    assert [row.text for row in _memory_rows(database)] == ["kept fact"]
    assert _message_rows(database) == []


def test_failed_memory_writes_keep_existing_records(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    calls = _run(
        monkeypatch,
        ["/memory save kept fact", "/exit"],
        _ModelScript(),
    )
    memory_id = capsys.readouterr().out.strip()
    before = _memory_rows(database)
    monkeypatch.setattr(storage.ConversationStore, "_insert_memory", _fail_disk)
    monkeypatch.setattr(storage.ConversationStore, "_update_memory_text", _fail_disk)
    monkeypatch.setattr(storage.ConversationStore, "_delete_memory_row", _fail_disk)
    calls = _run(
        monkeypatch,
        [
            "/memory save another",
            f"/memory update {memory_id} changed",
            f"/memory delete {memory_id}",
            "/memory list",
            "/exit",
        ],
        _ModelScript(replies=["should not be called"]),
    )

    assert calls == []
    assert calls.token_counts == []
    printed = capsys.readouterr()
    assert printed.out == f"{_memory_line(memory_id, 'kept fact')}\n"
    assert "Updated" not in printed.out
    assert "Deleted" not in printed.out
    assert printed.err == (
        f"{storage.SAVE_MEMORY_FAILED}\n"
        f"{storage.UPDATE_MEMORY_FAILED}\n"
        f"{storage.DELETE_MEMORY_FAILED}\n"
    )
    assert _memory_rows(database) == before
    assert _message_rows(database) == []


def test_model_request_omits_saved_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ready(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_HISTORY_TURNS", "1")
    turn_1 = _exchange("one", "r1")
    turn_2 = _exchange("two", "r2")
    follow = _exchange("next", "r3")
    _seed_turns(database, [turn_1, turn_2])
    calls = _run(
        monkeypatch,
        ["/memory save секретний факт", "next", "/exit"],
        _ModelScript(
            replies=["r3"], histories=[[*turn_2, *follow]], new_turns=[follow]
        ),
    )

    assert [call.prompt for call in calls] == ["next"]
    assert calls[0].message_history == turn_2
    assert _SECRET not in calls[0].prompt
    assert _SECRET not in _visible_text(calls[0].message_history)
    assert len(calls.token_counts) == 1
    assert calls.token_counts[0].prompt == "next"
    session_id = _session_ids(database, _user_id())[0]
    loaded = storage.connect(database).load_recent(_user_id(), session_id, 10)
    assert loaded == [*turn_1, *turn_2, *follow]
    assert [row.text for row in _memory_rows(database)] == [_SECRET]


def test_legacy_database_keeps_history_and_accepts_memory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    turn = _exchange("earlier", "saved reply")
    follow = _exchange("next question", "next reply")
    _write_pre_memory(database, "session-kept", turn)
    assert not _has_table(database, "memories")
    calls = _run(
        monkeypatch,
        [
            "next question",
            "/memory save новий факт",
            "/memory list",
            "/exit",
        ],
        _ModelScript(replies=["next reply"], new_turns=[follow]),
    )

    assert [call.prompt for call in calls] == ["next question"]
    assert calls[0].message_history == turn
    assert "новий факт" not in calls[0].prompt
    assert len(calls.token_counts) == 1
    printed = capsys.readouterr()
    assert printed.err == ""
    reply, memory_id, listed = printed.out.splitlines()
    assert reply == "next reply"
    assert listed == _memory_line(memory_id, "новий факт")
    assert _session_ids(database, _user_id()) == ["session-kept"]
    assert [row.text for row in _memory_rows(database)] == ["новий факт"]
    loaded = storage.connect(database).load_recent(_user_id(), "session-kept", 10)
    assert loaded == [*turn, *follow]

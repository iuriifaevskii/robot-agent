import sqlite3
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ToolCallPart,
    ToolReturnPart,
)

from robot_agent import cli, identity, storage
from robot_agent.sessions import SESSION_UNAVAILABLE
from tests.test_cli import (
    _configure,
    _exchange,
    _isolate_env,
    _message_rows,
    _ModelScript,
    _replace_model,
    _scripted_input,
    _tool_turn,
)

_LEGACY_SCHEMA = """
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
"""


def _ready(monkeypatch: pytest.MonkeyPatch) -> Path:
    database = _isolate_env(monkeypatch)
    _configure(monkeypatch)
    return database


def _body(messages: list[ModelMessage]) -> str:
    return ModelMessagesTypeAdapter.dump_json(list(messages)).decode()


def _user_id() -> str:
    return identity.resolve_user().user_id


def _session_ids(path: Path, user_id: str) -> list[str]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            """
            SELECT session_id FROM sessions
            WHERE user_id = ?
            ORDER BY created_at, session_id
            """,
            (user_id,),
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def _active_session(path: Path, user_id: str) -> str | None:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            """
            SELECT session_id FROM active_sessions
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return str(row[0])


def _loaded(path: Path, user_id: str) -> dict[str, list[ModelMessage]]:
    store = storage.connect(path)
    return {
        session_id: store.load_recent(user_id, session_id, 20)
        for session_id in _session_ids(path, user_id)
    }


def _seed_turns(path: Path, turns: list[list[ModelMessage]]) -> str:
    profile = identity.resolve_user()
    store = storage.connect(path)
    session_id = store.ensure_session(profile.user_id, profile.display_name)
    for turn in turns:
        store.append_turn(profile.user_id, session_id, turn)
    return session_id


def _insert_user(path: Path, user_id: str, display_name: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            INSERT OR IGNORE INTO users (user_id, display_name)
            VALUES (?, ?)
            """,
            (user_id, display_name),
        )
        connection.commit()
    finally:
        connection.close()


def _insert_session(
    path: Path,
    user_id: str,
    session_id: str,
    created_at: str,
    body: str | None = None,
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO sessions (session_id, user_id, created_at)
            VALUES (?, ?, ?)
            """,
            (session_id, user_id, created_at),
        )
        if body is not None:
            connection.execute(
                """
                INSERT INTO messages (session_id, position, body)
                VALUES (?, 1, ?)
                """,
                (session_id, body),
            )
        connection.commit()
    finally:
        connection.close()


def _write_legacy(
    path: Path,
    sessions: list[tuple[str, str, list[ModelMessage]]],
) -> None:
    profile = identity.resolve_user()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(_LEGACY_SCHEMA)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO users (user_id, display_name) VALUES (?, ?)",
            (profile.user_id, profile.display_name),
        )
        for session_id, created_at, turn in sessions:
            connection.execute(
                """
                INSERT INTO sessions (session_id, user_id, created_at)
                VALUES (?, ?, ?)
                """,
                (session_id, profile.user_id, created_at),
            )
            connection.execute(
                """
                INSERT INTO messages (session_id, position, body)
                VALUES (?, 1, ?)
                """,
                (session_id, _body(turn)),
            )
        connection.commit()
    finally:
        connection.close()


def _run(monkeypatch: pytest.MonkeyPatch, lines: list[str], script: _ModelScript):
    calls = _replace_model(monkeypatch, script)
    _scripted_input(monkeypatch, lines)
    cli.main()
    return calls


def _tool_ids(messages: list[ModelMessage]) -> list[str]:
    ids: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart | ToolReturnPart):
                ids.append(part.tool_call_id)
    return ids


def test_upgraded_database_opens_the_previous_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ready(monkeypatch)
    old_turn = _exchange("hello", "old reply")
    newer_turn = _exchange("other", "newer reply")
    follow = _exchange("continue", "next")
    _write_legacy(
        database,
        [
            ("session-old", "2024-01-01T00:00:00+00:00", old_turn),
            ("session-new", "2024-06-01T00:00:00+00:00", newer_turn),
        ],
    )
    before = _message_rows(database)
    calls = _run(
        monkeypatch,
        ["continue", "/exit"],
        _ModelScript(replies=["next"], new_turns=[follow]),
    )

    assert calls[0].message_history == old_turn
    assert len(calls.token_counts) == 1
    assert _session_ids(database, _user_id()) == ["session-old", "session-new"]
    assert _active_session(database, _user_id()) is None
    rows = _message_rows(database)
    assert rows[:2] == before
    assert rows[2][0] == "session-old"
    assert _loaded(database, _user_id())["session-old"] == [*old_turn, *follow]
    assert _loaded(database, _user_id())["session-new"] == newer_turn


def test_new_session_starts_empty_and_keeps_the_old_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    turn_one = _exchange("one", "r1")
    turn_two = _exchange("two", "r2")
    turn_three = _exchange("three", "r3")
    calls = _run(
        monkeypatch,
        ["one", "two", "/new", "three", "/exit"],
        _ModelScript(
            replies=["r1", "r2", "r3"],
            histories=[
                turn_one,
                [*turn_one, *turn_two],
                turn_three,
            ],
            new_turns=[turn_one, turn_two, turn_three],
        ),
    )

    assert [call.prompt for call in calls] == ["one", "two", "three"]
    assert calls[1].message_history == turn_one
    assert calls[2].message_history == []
    assert len(calls.token_counts) == 3
    printed = capsys.readouterr()
    assert printed.err == ""
    lines = printed.out.splitlines()
    assert lines[:2] == ["r1", "r2"]
    new_id = lines[2]
    assert lines[3:] == ["r3"]
    loaded = _loaded(database, _user_id())
    assert loaded[new_id] == turn_three
    kept = [messages for session_id, messages in loaded.items() if session_id != new_id]
    assert kept == [[*turn_one, *turn_two]]


def test_sessions_lists_only_the_current_users_sessions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    profile = identity.resolve_user()
    store = storage.connect(database)
    other = store.ensure_session("other-user", "Other")
    store.append_turn("other-user", other, _exchange("secret", "hidden"))
    _insert_user(database, profile.user_id, profile.display_name)
    _insert_session(database, profile.user_id, "session-a", "2026-01-04T00:00:00+00:00")
    _insert_session(
        database,
        profile.user_id,
        "session-c",
        "2026-01-03T00:00:00+00:00",
        "not-json",
    )
    _insert_session(database, profile.user_id, "session-m", "2026-01-03T00:00:00+00:00")
    _insert_session(database, profile.user_id, "session-z", "2026-01-01T00:00:00+00:00")
    calls = _run(monkeypatch, ["/sessions", "/exit"], _ModelScript())

    assert calls == []
    assert calls.token_counts == []
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "session-a 2026-01-04T00:00:00+00:00\n"
        "session-c 2026-01-03T00:00:00+00:00\n"
        "session-m 2026-01-03T00:00:00+00:00\n"
        "session-z 2026-01-01T00:00:00+00:00 active\n"
    )
    assert "not-json" not in captured.out
    bodies = [body for _session, _position, body in _message_rows(database)]
    assert "not-json" in bodies


def test_resume_sends_only_the_selected_sessions_recent_turns(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_HISTORY_TURNS", "2")
    turn_1 = _exchange("one", "r1")
    turn_2 = _tool_turn("two", "r2")
    turn_3 = _exchange("three", "r3")
    turn_b = _exchange("from-b", "r4")
    turn_5 = _exchange("five", "r5")
    kept = [*turn_2, *turn_3]
    session_a = _seed_turns(database, [turn_1, turn_2, turn_3])
    calls = _run(
        monkeypatch,
        ["/new", "from-b", f"/resume {session_a}", "five", "/exit"],
        _ModelScript(
            replies=["r4", "r5"],
            histories=[turn_b, [*kept, *turn_5]],
            new_turns=[turn_b, turn_5],
        ),
    )

    assert [call.prompt for call in calls] == ["from-b", "five"]
    assert calls[0].message_history == []
    assert calls[1].message_history == kept
    assert _tool_ids(calls[1].message_history) == ["call-1", "call-1"]
    assert len(calls.token_counts) == 2
    printed = capsys.readouterr()
    assert printed.err == ""
    new_id, reply_b, resumed, reply = printed.out.splitlines()
    assert reply_b == "r4"
    assert resumed == session_a
    assert reply == "r5"
    loaded = _loaded(database, _user_id())
    assert loaded[session_a] == [*turn_1, *turn_2, *turn_3, *turn_5]
    assert loaded[new_id] == turn_b
    assert _active_session(database, _user_id()) == session_a


def test_restart_restores_a_new_empty_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    user_id = _user_id()
    follow = _exchange("hello", "reply")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(replies=["reply"], new_turns=[follow]),
    )
    _scripted_input(monkeypatch, ["  /new  ", "/exit"])
    cli.main()

    new_id = capsys.readouterr().out.strip()
    assert calls == []
    assert calls.token_counts == []
    assert _active_session(database, user_id) == new_id
    assert len(_session_ids(database, user_id)) == 2
    assert _message_rows(database) == []
    _scripted_input(monkeypatch, ["hello", "/exit"])
    cli.main()

    assert calls[0].message_history == []
    assert len(calls.token_counts) == 1
    assert [row[0] for row in _message_rows(database)] == [new_id]
    assert _active_session(database, user_id) == new_id


def test_restart_restores_a_resumed_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    user_id = _user_id()
    turn_a = _exchange("on-a", "reply-a")
    turn_b = _exchange("on-b", "reply-b")
    follow = _exchange("later", "reply-later")
    session_a = _seed_turns(database, [turn_a])
    _insert_session(
        database,
        user_id,
        "session-b",
        "2099-01-01T00:00:00+00:00",
        _body(turn_b),
    )
    calls = _replace_model(
        monkeypatch,
        _ModelScript(replies=["reply-later"], new_turns=[follow]),
    )
    _scripted_input(monkeypatch, ["/resume session-b", "/exit"])
    cli.main()

    assert calls == []
    assert capsys.readouterr().out == "session-b\n"
    assert _active_session(database, user_id) == "session-b"
    _scripted_input(monkeypatch, ["later", "/exit"])
    cli.main()

    assert calls[0].message_history == turn_b
    assert calls[0].prompt == "later"
    loaded = _loaded(database, user_id)
    assert loaded[session_a] == turn_a
    assert loaded["session-b"] == [*turn_b, *follow]
    assert _active_session(database, user_id) == "session-b"


def test_invalid_session_commands_keep_the_current_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    user_id = _user_id()
    turn_a = _exchange("hello", "saved")
    other_turn = _exchange("secret", "hidden")
    follow = _exchange("/New", "reply")
    session_a = _seed_turns(database, [turn_a])
    store = storage.connect(database)
    other = store.ensure_session("other-user", "Other")
    store.append_turn("other-user", other, other_turn)
    calls = _run(
        monkeypatch,
        [
            "/resume",
            "/resume one two",
            "/resume missing-session",
            f"/resume {other}",
            "/new now",
            "/sessions now",
            "/New",
            "/exit",
        ],
        _ModelScript(replies=["reply"], new_turns=[follow]),
    )

    assert [call.prompt for call in calls] == ["/New"]
    assert calls[0].message_history == turn_a
    assert len(calls.token_counts) == 1
    captured = capsys.readouterr()
    assert captured.out == "reply\n"
    assert captured.err == (
        f"{cli.RESUME_USAGE}\n"
        f"{cli.RESUME_USAGE}\n"
        f"{SESSION_UNAVAILABLE}\n"
        f"{SESSION_UNAVAILABLE}\n"
        f"{cli.NEW_USAGE}\n"
        f"{cli.SESSIONS_USAGE}\n"
    )
    assert _session_ids(database, user_id) == [session_a]
    assert _active_session(database, user_id) is None
    assert _loaded(database, user_id)[session_a] == [*turn_a, *follow]
    reopened = storage.connect(database)
    assert reopened.load_recent("other-user", other, 10) == other_turn


def test_failed_new_keeps_the_original_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    user_id = _user_id()
    turn_1 = _exchange("hello", "r1")
    turn_2 = _exchange("again", "r2")

    def fail_selection(
        self: storage.ConversationStore,
        user_id: str,
        session_id: str,
    ) -> None:
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(storage.ConversationStore, "_store_selection", fail_selection)
    calls = _run(
        monkeypatch,
        ["hello", "/new", "again", "/exit"],
        _ModelScript(
            replies=["r1", "r2"],
            histories=[turn_1, [*turn_1, *turn_2]],
            new_turns=[turn_1, turn_2],
        ),
    )

    assert [call.prompt for call in calls] == ["hello", "again"]
    assert calls[1].message_history == turn_1
    captured = capsys.readouterr()
    assert captured.out == "r1\nr2\n"
    assert captured.err == f"{storage.CREATE_FAILED}\n"
    assert len(_session_ids(database, user_id)) == 1
    assert _active_session(database, user_id) is None
    assert list(_loaded(database, user_id).values()) == [[*turn_1, *turn_2]]


def test_failed_resume_load_keeps_the_original_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    user_id = _user_id()
    turn_a = _exchange("on-a", "reply-a")
    follow = _exchange("next", "later")
    session_a = _seed_turns(database, [turn_a])
    _insert_session(
        database,
        user_id,
        "session-b",
        "2099-01-01T00:00:00+00:00",
        "not-json",
    )
    calls = _run(
        monkeypatch,
        ["/resume session-b", "next", "/exit"],
        _ModelScript(replies=["later"], new_turns=[follow]),
    )

    assert calls[0].message_history == turn_a
    assert len(calls.token_counts) == 1
    captured = capsys.readouterr()
    assert captured.out == "later\n"
    assert captured.err == f"{storage.READ_FAILED}\n"
    assert _active_session(database, user_id) is None
    saved = storage.connect(database)
    assert saved.load_recent(user_id, session_a, 10) == [*turn_a, *follow]
    bodies = [body for _session, _pos, body in _message_rows(database)]
    assert bodies.count("not-json") == 1


def test_failed_resume_save_keeps_the_original_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    user_id = _user_id()
    turn_a = _exchange("on-a", "reply-a")
    turn_b = _exchange("on-b", "reply-b")
    follow = _exchange("next", "later")
    session_a = _seed_turns(database, [turn_a])
    _insert_session(
        database,
        user_id,
        "session-b",
        "2099-01-01T00:00:00+00:00",
        _body(turn_b),
    )

    def fail_selection(
        self: storage.ConversationStore,
        user_id: str,
        session_id: str,
    ) -> None:
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(storage.ConversationStore, "_store_selection", fail_selection)
    calls = _run(
        monkeypatch,
        ["/resume session-b", "next", "/exit"],
        _ModelScript(replies=["later"], new_turns=[follow]),
    )

    assert calls[0].message_history == turn_a
    assert len(calls.token_counts) == 1
    captured = capsys.readouterr()
    assert captured.out == "later\n"
    assert captured.err == f"{storage.SELECTION_FAILED}\n"
    assert _active_session(database, user_id) is None
    loaded = _loaded(database, user_id)
    assert loaded[session_a] == [*turn_a, *follow]
    assert loaded["session-b"] == turn_b


def test_session_commands_do_not_call_the_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    user_id = _user_id()
    profile = identity.resolve_user()
    store = storage.connect(database)
    session_id = store.ensure_session(profile.user_id, profile.display_name)
    calls = _run(
        monkeypatch,
        [
            "  /sessions  ",
            "  /new  ",
            f"  /resume   {session_id}  ",
            "/exit",
        ],
        _ModelScript(replies=["should not be called"]),
    )

    assert calls == []
    assert calls.token_counts == []
    assert _message_rows(database) == []
    assert len(_session_ids(database, user_id)) == 2
    assert _active_session(database, user_id) == session_id
    captured = capsys.readouterr()
    assert captured.err == ""
    assert session_id in captured.out
    assert "active" in captured.out

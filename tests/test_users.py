"""Local /user commands create, list and switch profiles without the model."""

import sqlite3
from pathlib import Path
from uuid import UUID

import pytest
from pydantic_ai.messages import ModelMessage

from robot_agent import cli, identity, storage
from robot_agent.memory import MEMORY_UNAVAILABLE
from robot_agent.sessions import SESSION_UNAVAILABLE
from robot_agent.storage import ConversationStore, StorageError
from tests.test_cli import (
    RecordedCalls,
    _exchange,
    _identity_rows,
    _message_rows,
    _ModelScript,
    _replace_model,
    _scripted_input,
)
from tests.test_memory import _memory_rows
from tests.test_session_budget import (
    _seed_session_usage,
    _session_limit,
    _session_rows,
)
from tests.test_sessions import (
    _active_session,
    _ready,
    _seed_turns,
    _session_ids,
    _user_id,
)
from tests.test_summary import _store_summary, _summary_rows, _turns
from tests.test_usage import TODAY, _clock, _limit, _rows, _seed_usage

_LOCAL_LINE = "local-user Local user"


def _run_cli(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    _scripted_input(monkeypatch, lines)
    cli.main()


def _model(
    monkeypatch: pytest.MonkeyPatch,
    *pairs: tuple[str, str],
) -> RecordedCalls:
    """Script one (prompt, reply) model run per pair, in order."""
    turns = [_exchange(prompt, reply) for prompt, reply in pairs]
    script = _ModelScript(
        replies=[reply for _, reply in pairs],
        new_turns=turns,
        histories=[list(turn) for turn in turns],
    )
    return _replace_model(monkeypatch, script)


def _profile(database: Path, name: str = "Ana") -> str:
    return identity.create_profile(storage.connect(database), name)


def _users(database: Path) -> list[tuple[str, str]]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT user_id, display_name FROM users ORDER BY rowid"
        ).fetchall()
    finally:
        connection.close()
    return [(str(user_id), str(name)) for user_id, name in rows]


def _turn_count(database: Path, user_id: str, session_id: str) -> int:
    return storage.connect(database).turn_count(user_id, session_id)


def _saved(database: Path, user_id: str, session_id: str) -> list[ModelMessage]:
    return storage.connect(database).load_recent(user_id, session_id, 50)


def _fail_with(message: str):
    def fail(*_args: object) -> None:
        raise StorageError(message)

    return fail


def test_create_prints_a_new_id_and_list_shows_both_in_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ready(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript())

    _run_cli(monkeypatch, ["/user create Ana", "/user list", "/exit"])

    lines = capsys.readouterr().out.splitlines()
    new_id = lines[0]
    assert str(UUID(new_id)) == new_id
    assert new_id != "local-user"
    assert "Ana" not in new_id
    assert lines[1:] == [f"{_LOCAL_LINE} (active)", f"{new_id} Ana"]
    assert calls == []


def test_create_with_the_fixed_id_as_name_makes_a_new_uuid(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _replace_model(monkeypatch, _ModelScript())

    _run_cli(monkeypatch, ["/user create local-user", "/user list", "/exit"])

    lines = capsys.readouterr().out.splitlines()
    new_id = lines[0]
    assert str(UUID(new_id)) == new_id
    assert lines[1:] == [f"{_LOCAL_LINE} (active)", f"{new_id} local-user"]
    assert _users(database) == [
        ("local-user", "Local user"),
        (new_id, "local-user"),
    ]


def test_create_with_an_empty_or_too_long_name_stores_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["ok"]))

    _run_cli(
        monkeypatch,
        ["/user create", f"/user create {'x' * 65}", "hello", "/exit"],
    )

    captured = capsys.readouterr()
    assert captured.err == f"{cli.USER_CREATE_USAGE}\n{identity.NAME_TOO_LONG}\n"
    assert captured.out == "ok\n"
    assert _users(database) == [("local-user", "Local user")]
    assert [call.prompt for call in calls] == ["hello"]


def test_create_neither_switches_nor_creates_a_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    calls = _model(monkeypatch, ("hi", "ok"))

    _run_cli(monkeypatch, ["/user create Ana", "hi", "/exit"])

    new_id = capsys.readouterr().out.splitlines()[0]
    assert [call.user_id for call in calls] == ["local-user"]
    assert _session_ids(database, new_id) == []
    sessions = _session_ids(database, _user_id())
    assert len(sessions) == 1
    assert _turn_count(database, _user_id(), sessions[0]) == 1


def test_use_gives_a_new_profile_one_session_and_an_empty_history(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    local_session = _seed_turns(database, [])
    other = _profile(database)
    calls = _model(monkeypatch, ("hello", "ok"))

    _run_cli(monkeypatch, [f"/user use {other}", "hello", "/user list", "/exit"])

    assert capsys.readouterr().out.splitlines() == [
        other,
        "ok",
        _LOCAL_LINE,
        f"{other} Ana (active)",
    ]
    assert [call.user_id for call in calls] == [other]
    assert calls[0].message_history == []
    sessions = _session_ids(database, other)
    assert len(sessions) == 1
    assert _turn_count(database, other, sessions[0]) == 1
    assert _turn_count(database, _user_id(), local_session) == 0


def test_use_of_an_unknown_user_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    calls = _model(monkeypatch, ("hello", "ok"))

    _run_cli(monkeypatch, ["/user use nobody", "hello", "/exit"])

    captured = capsys.readouterr()
    assert captured.err == f"{cli.USER_UNAVAILABLE}\n"
    assert captured.out == "ok\n"
    assert _users(database) == [("local-user", "Local user")]
    sessions = _session_ids(database, _user_id())
    assert len(sessions) == 1
    assert [call.user_id for call in calls] == ["local-user"]
    assert _turn_count(database, _user_id(), sessions[0]) == 1


def test_use_of_the_current_user_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    saved = _exchange("q", "a")
    _seed_turns(database, [saved])
    calls = _model(monkeypatch, ("hello", "ok"))

    _run_cli(monkeypatch, ["/user use local-user", "hello", "/exit"])

    assert capsys.readouterr().out == "local-user\nok\n"
    assert calls[0].message_history == saved
    assert len(_session_ids(database, _user_id())) == 1


@pytest.mark.parametrize(
    "line",
    ["/user use", "/user use a b", "/user list x", "/user", "/user bogus"],
)
def test_invalid_user_commands_print_usage_and_change_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    line: str,
) -> None:
    database = _ready(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript())

    _run_cli(monkeypatch, [line, "/exit"])

    captured = capsys.readouterr()
    assert captured.err == f"{cli.USER_USAGE}\n"
    assert captured.out == ""
    assert calls == []
    assert _users(database) == [("local-user", "Local user")]
    assert len(_session_ids(database, _user_id())) == 1


@pytest.mark.parametrize(
    ("method", "line", "message"),
    [
        ("user_display_name", "/user use someone", storage.USER_READ_FAILED),
        ("list_users", "/user list", storage.USER_READ_FAILED),
        ("create_user", "/user create Ana", storage.USER_CREATE_FAILED),
    ],
)
def test_a_failed_profile_operation_keeps_the_chat_open(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    method: str,
    line: str,
    message: str,
) -> None:
    database = _ready(monkeypatch)
    calls = _model(monkeypatch, ("hello", "ok"))
    monkeypatch.setattr(ConversationStore, method, _fail_with(message))

    _run_cli(monkeypatch, [line, "hello", "/exit"])

    captured = capsys.readouterr()
    assert captured.err == f"{message}\n"
    assert captured.out == "ok\n"
    assert _users(database) == [("local-user", "Local user")]
    sessions = _session_ids(database, _user_id())
    assert [call.user_id for call in calls] == ["local-user"]
    assert _turn_count(database, _user_id(), sessions[0]) == 1


def test_a_failed_history_load_keeps_the_original_user_and_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    saved = _exchange("q", "a")
    local_session = _seed_turns(database, [saved])
    other = _profile(database)
    calls = _model(monkeypatch, ("hello", "ok"))
    original = ConversationStore.load_recent

    def fail_for_other(
        self: ConversationStore, user_id: str, session_id: str, limit: int
    ) -> list[ModelMessage]:
        if user_id == other:
            raise StorageError(storage.READ_FAILED)
        return original(self, user_id, session_id, limit)

    monkeypatch.setattr(ConversationStore, "load_recent", fail_for_other)

    _run_cli(monkeypatch, [f"/user use {other}", "hello", "/exit"])

    captured = capsys.readouterr()
    assert captured.err == f"{storage.READ_FAILED}\n"
    assert captured.out == "ok\n"
    assert [call.user_id for call in calls] == ["local-user"]
    assert calls[0].message_history == saved
    assert _turn_count(database, _user_id(), local_session) == 2


def test_memories_and_recall_stay_with_the_selected_user(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_RECALLED_MEMORIES", "3")
    _seed_turns(database, [_exchange("q-a", "a-a")])
    other = _profile(database)
    storage.connect(database).save_memory("local-user", "Local user", "любить чай")
    calls = _model(monkeypatch, ("чай любить", "ok"))

    _run_cli(
        monkeypatch,
        [
            f"/user use {other}",
            "/memory list",
            "/memory search чай",
            "чай любить",
            "/exit",
        ],
    )

    assert capsys.readouterr().out.splitlines() == [
        other,
        cli.NO_MEMORIES,
        cli.NO_MEMORY_MATCHES,
        "ok",
    ]
    assert calls[0].prompt == "чай любить"
    assert calls[0].message_history == []


def test_another_users_memory_cannot_be_updated_or_deleted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _seed_turns(database, [])
    other = _profile(database)
    memory_id = storage.connect(database).save_memory(
        "local-user", "Local user", "любить чай"
    )
    before = _memory_rows(database)

    _run_cli(
        monkeypatch,
        [
            f"/user use {other}",
            f"/memory update {memory_id} new",
            f"/memory delete {memory_id}",
            "/user use local-user",
            "/memory list",
            "/exit",
        ],
    )

    captured = capsys.readouterr()
    assert captured.err == f"{MEMORY_UNAVAILABLE}\n" * 2
    assert captured.out.splitlines()[-1] == f"{memory_id} любить чай"
    assert _memory_rows(database) == before


def test_sessions_and_resume_stay_with_the_selected_user(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    local_session = _seed_turns(database, [_exchange("q", "a")])
    other = _profile(database)
    _model(monkeypatch, ("hello", "ok"))

    _run_cli(
        monkeypatch,
        [f"/user use {other}", "/sessions", f"/resume {local_session}", "hello"],
    )

    captured = capsys.readouterr()
    other_session = _session_ids(database, other)[0]
    lines = captured.out.splitlines()
    assert lines[0] == other
    assert lines[1].startswith(f"{other_session} ")
    assert lines[1].endswith(" active")
    assert lines[2:] == ["ok"]
    assert local_session not in captured.out
    assert captured.err == f"{SESSION_UNAVAILABLE}\n"
    assert _turn_count(database, other, other_session) == 1
    assert _turn_count(database, _user_id(), local_session) == 1


def test_another_users_summary_is_not_sent_or_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ready(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_HISTORY_TURNS", "2")
    monkeypatch.setenv("AGENT_SUMMARY_AFTER_TURNS", "2")
    local_session = _seed_turns(database, _turns(4))
    _store_summary(database, local_session, "A private summary", 2)
    other = _profile(database)
    calls = _model(monkeypatch, ("hello", "ok"))

    _run_cli(monkeypatch, [f"/user use {other}", "hello", "/exit"])

    assert len(calls) == 1
    assert calls[0].prompt == "hello"
    assert calls[0].message_history == []
    assert _summary_rows(database) == [(local_session, "A private summary", 2)]
    assert _turn_count(database, _user_id(), local_session) == 4
    other_session = _session_ids(database, other)[0]
    assert _turn_count(database, other, other_session) == 1


def test_switching_back_restores_the_original_history_and_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ready(monkeypatch)
    saved = _exchange("q-a", "a-a")
    local_session = _seed_turns(database, [saved])
    other = _profile(database)
    sessions_before = _identity_rows(database)[1]
    calls = _model(monkeypatch, ("from b", "rb"), ("from a", "ra"))

    _run_cli(
        monkeypatch,
        [f"/user use {other}", "from b", "/user use local-user", "from a", "/exit"],
    )

    assert [call.user_id for call in calls] == [other, "local-user"]
    assert calls[0].message_history == []
    assert calls[1].message_history == saved
    local_turns = _saved(database, _user_id(), local_session)
    assert [type(m).__name__ for m in local_turns] == [
        "ModelRequest",
        "ModelResponse",
    ] * 2
    assert local_turns[:2] == saved
    other_session = _session_ids(database, other)[0]
    other_turns = _saved(database, other, other_session)
    assert len(other_turns) == 2
    sessions_after = _identity_rows(database)[1]
    assert all(row in sessions_after for row in sessions_before)


def test_switching_does_not_change_the_saved_selection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _seed_turns(database, [_exchange("q", "a")])
    other = _profile(database)
    calls = _model(monkeypatch, ("hello", "ok"))

    _run_cli(
        monkeypatch,
        ["/new", f"/user use {other}", "/user use local-user", "hello", "/exit"],
    )

    lines = capsys.readouterr().out.splitlines()
    new_session = lines[0]
    assert lines[1:] == [other, "local-user", "ok"]
    assert _active_session(database, _user_id()) == new_session
    assert len(_session_ids(database, _user_id())) == 2
    assert calls[0].message_history == []
    assert _turn_count(database, _user_id(), new_session) == 1


def test_usage_counters_follow_the_session_and_the_day_stays_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, requests="100")
    _session_limit(monkeypatch, requests="100")
    local_session = _seed_turns(database, [])
    _seed_session_usage(database, local_session, 2, 100)
    _seed_usage(database, TODAY, 3, 300)
    other = _profile(database)
    _model(monkeypatch, ("hello", "ok"))

    _run_cli(monkeypatch, [f"/user use {other}", "hello", "/exit"])

    other_session = _session_ids(database, other)[0]
    by_session = {row[0]: row[1:] for row in _session_rows(database)}
    assert by_session == {local_session: (2, 100, 0), other_session: (1, 10, 5)}
    assert _rows(database) == [(TODAY, 4, 310, 5)]


def test_a_new_process_starts_as_the_fixed_user_and_can_switch_back(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _seed_turns(database, [])
    other = _profile(database)
    _model(monkeypatch, ("hello", "ok"))
    _run_cli(monkeypatch, [f"/user use {other}", "hello", "/exit"])
    capsys.readouterr()
    first_turn = _saved(database, other, _session_ids(database, other)[0])
    calls = _model(monkeypatch, ("again", "ok2"))

    _run_cli(
        monkeypatch,
        ["/user list", f"/user use {other}", "again", "/exit"],
    )

    assert capsys.readouterr().out.splitlines() == [
        f"{_LOCAL_LINE} (active)",
        f"{other} Ana",
        other,
        "ok2",
    ]
    assert [call.user_id for call in calls] == [other]
    assert calls[0].message_history == first_turn


def test_an_existing_database_keeps_its_data_and_lists_one_user(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    session_id = _seed_turns(database, _turns(2))
    storage.connect(database).save_memory("local-user", "Local user", "любить чай")
    _store_summary(database, session_id, "Earlier summary", 1)
    messages = _message_rows(database)
    memories = _memory_rows(database)
    summaries = _summary_rows(database)

    _run_cli(monkeypatch, ["/user list", "/exit"])

    assert capsys.readouterr().out == f"{_LOCAL_LINE} (active)\n"
    assert _message_rows(database) == messages
    assert _memory_rows(database) == memories
    assert _summary_rows(database) == summaries
    assert _users(database) == [("local-user", "Local user")]


def test_user_commands_ignore_budgets_and_make_no_model_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, requests="1")
    _session_limit(monkeypatch, requests="1")
    local_session = _seed_turns(database, [])
    _seed_session_usage(database, local_session, 5, 0)
    _seed_usage(database, TODAY, 5, 0)
    other = _profile(database)
    calls = _replace_model(monkeypatch, _ModelScript())
    session_rows = _session_rows(database)
    daily_rows = _rows(database)

    _run_cli(
        monkeypatch,
        ["/user create Cy", "/user list", f"/user use {other}", "/exit"],
    )

    assert calls == []
    assert calls.token_counts == []
    assert capsys.readouterr().err == ""
    assert _rows(database) == daily_rows
    assert _session_rows(database) == session_rows
    assert _message_rows(database) == []


def test_ordinary_lines_that_mention_the_command_reach_the_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    calls = _model(
        monkeypatch,
        ("please switch to user abc", "r1"),
        ("use /user use abc", "r2"),
    )

    _run_cli(
        monkeypatch,
        ["please switch to user abc", "use /user use abc", "/exit"],
    )

    assert capsys.readouterr().out == "r1\nr2\n"
    assert [call.prompt for call in calls] == [
        "please switch to user abc",
        "use /user use abc",
    ]
    assert [call.user_id for call in calls] == ["local-user", "local-user"]
    assert _users(database) == [("local-user", "Local user")]

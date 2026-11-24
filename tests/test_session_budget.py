"""A per-session budget of requests and tokens is opt-in and persistent."""

import sqlite3
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import AgentRunError

from robot_agent import cli, storage
from robot_agent.storage import ConversationStore, StorageError
from robot_agent.tools import ClockReadError
from tests.test_cli import (
    _exchange,
    _ModelScript,
    _replace_model,
    _scripted_input,
    _usage,
)
from tests.test_sessions import _ready, _seed_turns, _session_ids, _user_id
from tests.test_summary import _install, _run, _setup, _summary_rows, _turns
from tests.test_usage import _TOKEN_REFUSAL as _DAILY_TOKEN_REFUSAL
from tests.test_usage import TODAY, _clock, _limit, _rows, _run_cli, _seed_usage

_TOKEN_REFUSAL = (
    "Session token budget reached (1000 of 1000 used in this session). "
    "Raise AGENT_SESSION_MAX_TOKENS or start a new session with /new.\n"
)
_REQUEST_REFUSAL = (
    "Session request budget reached (3 of 2 used in this session). "
    "Raise AGENT_SESSION_MAX_REQUESTS or start a new session with /new.\n"
)


def _session_rows(database: Path) -> list[tuple[str, int, int, int]]:
    connection = sqlite3.connect(database)
    try:
        found = connection.execute(
            "SELECT session_id, requests, input_tokens, output_tokens "
            "FROM session_usage ORDER BY session_id"
        ).fetchall()
    finally:
        connection.close()
    return [(str(a), int(b), int(c), int(d)) for a, b, c, d in found]


def _seed_session_usage(
    database: Path, session_id: str, requests: int, tokens_in: int
) -> None:
    store = storage.connect(database)
    store.add_session_usage(_user_id(), session_id, requests, tokens_in, 0)


def _session_limit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    requests: str | None = None,
    tokens: str | None = None,
) -> None:
    if requests is not None:
        monkeypatch.setenv("AGENT_SESSION_MAX_REQUESTS", requests)
    if tokens is not None:
        monkeypatch.setenv("AGENT_SESSION_MAX_TOKENS", tokens)


@pytest.mark.parametrize("daily", [False, True])
@pytest.mark.parametrize("raw", [None, "0"])
def test_default_budget_writes_and_reads_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raw: str | None,
    daily: bool,
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _session_limit(monkeypatch, requests=raw, tokens=raw)
    if daily:
        _limit(monkeypatch, tokens="100000")
    calls = _replace_model(monkeypatch, _ModelScript(replies=["one", "two"]))

    def fail(*_args: object) -> None:
        raise AssertionError("session usage must not be touched")

    monkeypatch.setattr(ConversationStore, "session_usage", fail)
    monkeypatch.setattr(ConversationStore, "add_session_usage", fail)

    _run_cli(monkeypatch, ["a", "b", "/exit"])

    assert [call.prompt for call in calls] == ["a", "b"]
    captured = capsys.readouterr()
    assert captured.out == "one\ntwo\n"
    assert captured.err == ""
    assert _session_rows(database) == []


def test_usage_is_summed_and_survives_restart_new_and_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ready(monkeypatch)
    _session_limit(monkeypatch, tokens="1000")
    old_session = _seed_turns(database, [])
    _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["r1", "r2", "r3"],
            usages=[_usage(1, 400, 100), _usage(1, 300, 50), _usage(1, 1, 1)],
        ),
    )

    _run_cli(monkeypatch, ["a", "b", "/exit"])
    assert _session_rows(database) == [(old_session, 2, 700, 150)]
    assert _rows(database) == []

    _run_cli(monkeypatch, ["/new", f"/resume {old_session}", "/exit"])
    assert _session_rows(database) == [(old_session, 2, 700, 150)]
    assert len(_session_ids(database, _user_id())) == 2

    _run_cli(monkeypatch, ["c", "/exit"])
    assert _session_rows(database) == [(old_session, 3, 701, 151)]


def test_a_new_session_starts_with_fresh_counters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _session_limit(monkeypatch, tokens="1000")
    first = _seed_turns(database, [])
    _seed_session_usage(database, first, 5, 1000)
    calls = _replace_model(
        monkeypatch,
        _ModelScript(replies=["fresh"], usages=[_usage(1, 30, 20)]),
    )

    _run_cli(monkeypatch, ["/new", "hello", f"/resume {first}", "again", "/exit"])

    assert [call.prompt for call in calls] == ["hello"]
    assert capsys.readouterr().err == _TOKEN_REFUSAL
    (second,) = (s for s in _session_ids(database, _user_id()) if s != first)
    counts = {row[0]: row[1:] for row in _session_rows(database)}
    assert counts == {first: (5, 1000, 0), second: (1, 30, 20)}


def test_reached_token_limit_refuses_the_line_and_keeps_the_chat_open(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _session_limit(monkeypatch, tokens="1000")
    session_id = _seed_turns(database, _turns(1))
    _seed_session_usage(database, session_id, 5, 1000)
    calls = _replace_model(monkeypatch, _ModelScript())

    _run_cli(monkeypatch, ["hello", "/memory list", "/sessions", "/time", "/exit"])

    assert calls == []
    assert calls.token_counts == []
    captured = capsys.readouterr()
    assert captured.err == _TOKEN_REFUSAL
    assert session_id in captured.out
    assert "No saved facts." in captured.out
    assert "2026-11-22T09:30:00+02:00" in captured.out
    saved = storage.connect(database).turn_count(_user_id(), session_id)
    assert saved == 1
    assert _session_rows(database) == [(session_id, 5, 1000, 0)]


def test_a_tool_turn_may_cross_the_request_limit_then_the_next_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _session_limit(monkeypatch, requests="2")
    session_id = _seed_turns(database, [])
    _seed_session_usage(database, session_id, 1, 0)
    calls = _replace_model(
        monkeypatch,
        _ModelScript(replies=["It is 09:30."], usages=[_usage(2, 20, 10)]),
    )

    _run_cli(monkeypatch, ["what time is it?", "again", "/exit"])

    assert [call.prompt for call in calls] == ["what time is it?"]
    captured = capsys.readouterr()
    assert captured.out == "It is 09:30.\n"
    assert captured.err == _REQUEST_REFUSAL
    assert _session_rows(database) == [(session_id, 3, 20, 10)]


def test_exhausted_session_refuses_the_whole_turn_without_a_summary_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _setup(monkeypatch)
    _session_limit(monkeypatch, tokens="1000")
    session_id = _seed_turns(database, _turns(4))
    _seed_session_usage(database, session_id, 1, 1000)
    backend = _install(monkeypatch, ["unused summary"])

    _run(monkeypatch, ["q5", "/exit"])

    assert backend.summarizer == []
    assert backend.main == []
    assert capsys.readouterr().err == _TOKEN_REFUSAL
    assert _summary_rows(database) == []
    assert storage.connect(database).turn_count(_user_id(), session_id) == 4


def test_summarizer_reaching_the_limit_keeps_its_summary_and_refuses_chat(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _setup(monkeypatch)
    _session_limit(monkeypatch, tokens="1000")
    session_id = _seed_turns(database, _turns(4))
    _seed_session_usage(database, session_id, 1, 400)
    backend = _install(monkeypatch, ["A short summary."], [_usage(1, 500, 100)])

    _run(monkeypatch, ["q5", "/exit"])

    assert len(backend.summarizer) == 1
    assert backend.main == []
    assert _summary_rows(database) == [(session_id, "A short summary.", 2)]
    assert _session_rows(database) == [(session_id, 2, 900, 100)]
    assert capsys.readouterr().err.endswith(_TOKEN_REFUSAL)
    assert storage.connect(database).turn_count(_user_id(), session_id) == 4


def test_non_text_chat_reply_is_rejected_but_its_usage_is_counted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _session_limit(monkeypatch, requests="10")
    session_id = _seed_turns(database, [])
    replies = cast(list[str], [{"not": "text"}])
    _replace_model(monkeypatch, _ModelScript(replies=replies, usages=[_usage(1, 8, 2)]))

    _run_cli(monkeypatch, ["hello", "/exit"])

    assert "The model returned a non-text response." in capsys.readouterr().err
    assert _session_rows(database) == [(session_id, 1, 8, 2)]
    assert storage.connect(database).turn_count(_user_id(), session_id) == 0


def test_unusable_summary_is_rejected_but_its_usage_is_counted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _setup(monkeypatch)
    _session_limit(monkeypatch, tokens="100000")
    session_id = _seed_turns(database, _turns(4))
    backend = _install(monkeypatch, [" "], [_usage(1, 50, 5), _usage(1, 7, 3)])

    _run(monkeypatch, ["q5", "/exit"])

    assert len(backend.main) == 1
    assert _summary_rows(database) == []
    assert "Warning: summary skipped. The model returned an unusable summary." in (
        capsys.readouterr().err
    )
    assert _session_rows(database) == [(session_id, 2, 57, 8)]


def test_a_failed_run_adds_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _session_limit(monkeypatch, requests="10")
    session_id = _seed_turns(database, [])
    _seed_session_usage(database, session_id, 1, 5)
    _replace_model(monkeypatch, _ModelScript(error=AgentRunError("boom")))

    _run_cli(monkeypatch, ["hello", "/exit"])

    assert "Agent request failed: boom" in capsys.readouterr().err
    assert _session_rows(database) == [(session_id, 1, 5, 0)]


def test_one_run_is_counted_in_the_daily_and_the_session_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, tokens="100000")
    _session_limit(monkeypatch, requests="10")
    session_id = _seed_turns(database, [])
    _replace_model(
        monkeypatch,
        _ModelScript(replies=["ok"], usages=[_usage(1, 30, 20)]),
    )

    _run_cli(monkeypatch, ["hello", "/exit"])

    assert _rows(database) == [(TODAY, 1, 30, 20)]
    assert _session_rows(database) == [(session_id, 1, 30, 20)]


def test_the_daily_message_is_the_only_one_when_both_limits_are_reached(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, tokens="1000")
    _session_limit(monkeypatch, tokens="1000")
    session_id = _seed_turns(database, [])
    _seed_usage(database, TODAY, 1, 1000)
    _seed_session_usage(database, session_id, 1, 1000)
    calls = _replace_model(monkeypatch, _ModelScript())

    _run_cli(monkeypatch, ["hello", "/exit"])

    assert calls == []
    assert capsys.readouterr().err == _DAILY_TOKEN_REFUSAL


def test_unreadable_session_counters_send_no_paid_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _session_limit(monkeypatch, tokens="1000")
    session_id = _seed_turns(database, [])
    calls = _replace_model(monkeypatch, _ModelScript())

    def broken(*_args: object) -> None:
        raise StorageError(storage.SESSION_USAGE_READ_FAILED)

    monkeypatch.setattr(ConversationStore, "session_usage", broken)

    _run_cli(monkeypatch, ["hello", "/sessions", "/exit"])

    assert calls == []
    captured = capsys.readouterr()
    assert captured.err == f"{storage.SESSION_USAGE_READ_FAILED}\n"
    assert session_id in captured.out
    assert _session_rows(database) == []


def test_failed_recording_warns_saves_the_turn_and_refuses_the_next_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _session_limit(monkeypatch, tokens="1000000")
    session_id = _seed_turns(database, [])
    calls = _replace_model(monkeypatch, _ModelScript(replies=["one", "two"]))

    def broken(*_args: object) -> None:
        raise StorageError(storage.SESSION_USAGE_SAVE_FAILED)

    monkeypatch.setattr(ConversationStore, "add_session_usage", broken)

    _run_cli(monkeypatch, ["first", "second", "/exit"])

    assert [call.prompt for call in calls] == ["first"]
    captured = capsys.readouterr()
    assert captured.out == "one\n"
    warning = "Warning: session usage was not recorded."
    assert f"{warning} {storage.SESSION_USAGE_SAVE_FAILED}\n" in captured.err
    assert captured.err.endswith(f"{cli.UNTRUSTED_MESSAGE}\n")
    assert storage.connect(database).turn_count(_user_id(), session_id) == 1


def test_the_session_budget_does_not_need_the_clock(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _session_limit(monkeypatch, tokens="1000")
    session_id = _seed_turns(database, [])
    _replace_model(
        monkeypatch,
        _ModelScript(replies=["fine"], usages=[_usage(1, 30, 20)]),
    )

    def broken() -> None:
        raise ClockReadError("clock broke")

    monkeypatch.setattr("robot_agent.tools.read_local_clock", broken)

    _run_cli(monkeypatch, ["hello", "/exit"])

    captured = capsys.readouterr()
    assert captured.out == "fine\n"
    assert captured.err == ""
    assert _session_rows(database) == [(session_id, 1, 30, 20)]


def test_another_users_session_usage_cannot_be_read_or_changed(
    tmp_path: Path,
) -> None:
    store = storage.connect(tmp_path / "chat.sqlite3")
    owned = store.ensure_session("user-a", "User A")
    store.ensure_session("user-b", "User B")
    store.add_session_usage("user-a", owned, 1, 2, 3)

    with pytest.raises(StorageError) as read_error:
        store.session_usage("user-b", owned)
    with pytest.raises(StorageError) as write_error:
        store.add_session_usage("user-b", owned, 10, 20, 30)

    assert read_error.value.args == (storage.NOT_OWNER,)
    assert write_error.value.args == (storage.NOT_OWNER,)
    assert store.session_usage("user-a", owned) == (1, 2, 3)


def test_a_database_without_the_usage_table_is_opened_and_repaired(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chat.sqlite3"
    store = storage.connect(path)
    session_id = store.ensure_session("user-a", "User A")
    store.append_turn("user-a", session_id, _exchange("q", "a"))
    store.save_memory("user-a", "User A", "likes tea")
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TABLE session_usage")
        connection.commit()
    finally:
        connection.close()

    reopened = storage.connect(path)

    assert reopened.session_usage("user-a", session_id) == (0, 0, 0)
    assert reopened.turn_count("user-a", session_id) == 1
    assert [row[1] for row in reopened.list_memories("user-a")] == ["likes tea"]
    assert reopened.list_sessions("user-a")[0][0] == session_id


@pytest.mark.parametrize(
    "name", ["AGENT_SESSION_MAX_REQUESTS", "AGENT_SESSION_MAX_TOKENS"]
)
@pytest.mark.parametrize("raw", ["abc", "-1", "1.5"])
def test_invalid_budget_setting_exits_before_input_is_read(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    raw: str,
) -> None:
    _ready(monkeypatch)
    monkeypatch.setenv(name, raw)
    calls = _replace_model(monkeypatch, _ModelScript())
    _scripted_input(monkeypatch, ["hello"], after="fail")

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    assert calls == []
    assert capsys.readouterr().err == (
        f"Invalid configuration: {name} must be a non-negative integer.\n"
    )

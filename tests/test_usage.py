"""A device-wide daily budget of requests and tokens is opt-in and persistent."""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import AgentRunError

from robot_agent import cli, storage
from robot_agent.storage import ConversationStore, StorageError
from robot_agent.tools import ClockReadError
from tests.test_cli import (
    _ModelScript,
    _replace_model,
    _scripted_input,
    _sent,
    _usage,
)
from tests.test_sessions import _ready, _seed_turns, _session_ids, _user_id
from tests.test_summary import _install, _run, _setup, _summary_rows, _turns

TODAY = "2026-11-22"
TOMORROW = "2026-11-23"
_ZONE = timezone(timedelta(hours=2))
_TOKEN_REFUSAL = (
    "Daily token budget reached (1000 of 1000 used today). "
    "Raise AGENT_DAILY_MAX_TOKENS or try again tomorrow.\n"
)
_REQUEST_REFUSAL = (
    "Daily request budget reached (3 of 2 used today). "
    "Raise AGENT_DAILY_MAX_REQUESTS or try again tomorrow.\n"
)


def _clock(monkeypatch: pytest.MonkeyPatch, day: str = TODAY) -> list[str]:
    """Replace the local clock. Change the returned list entry to move time."""
    current = [day]

    def read() -> datetime:
        year, month, date = (int(part) for part in current[0].split("-"))
        return datetime(year, month, date, 9, 30, tzinfo=_ZONE)

    monkeypatch.setattr("robot_agent.tools.read_local_clock", read)
    return current


def _rows(database: Path) -> list[tuple[str, int, int, int]]:
    connection = sqlite3.connect(database)
    try:
        found = connection.execute(
            "SELECT day, requests, input_tokens, output_tokens "
            "FROM daily_usage ORDER BY day"
        ).fetchall()
    finally:
        connection.close()
    return [(str(a), int(b), int(c), int(d)) for a, b, c, d in found]


def _seed_usage(database: Path, day: str, requests: int, tokens_in: int) -> None:
    storage.connect(database).add_daily_usage(day, requests, tokens_in, 0)


def _limit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    requests: str | None = None,
    tokens: str | None = None,
) -> None:
    if requests is not None:
        monkeypatch.setenv("AGENT_DAILY_MAX_REQUESTS", requests)
    if tokens is not None:
        monkeypatch.setenv("AGENT_DAILY_MAX_TOKENS", tokens)


def _run_cli(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    _scripted_input(monkeypatch, lines)
    cli.main()


@pytest.mark.parametrize("raw", [None, "0"])
def test_default_budget_writes_and_reads_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raw: str | None,
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, requests=raw, tokens=raw)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["one", "two"]))

    def fail(*_args: object) -> None:
        raise AssertionError("usage must not be touched")

    monkeypatch.setattr(ConversationStore, "daily_usage", fail)
    monkeypatch.setattr(ConversationStore, "add_daily_usage", fail)

    _run_cli(monkeypatch, ["a", "b", "/exit"])

    assert [call.prompt for call in calls] == ["a", "b"]
    captured = capsys.readouterr()
    assert captured.out == "one\ntwo\n"
    assert captured.err == ""
    assert _rows(database) == []


def test_usage_is_summed_and_survives_restart_new_and_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, tokens="1000")
    old_session = _seed_turns(database, [])
    _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["r1", "r2", "r3"],
            usages=[_usage(1, 400, 100), _usage(1, 300, 50), _usage(1, 1, 1)],
        ),
    )

    _run_cli(monkeypatch, ["a", "b", "/exit"])
    assert _rows(database) == [(TODAY, 2, 700, 150)]

    _run_cli(monkeypatch, ["/new", f"/resume {old_session}", "/exit"])
    assert _rows(database) == [(TODAY, 2, 700, 150)]
    assert len(_session_ids(database, _user_id())) == 2

    _run_cli(monkeypatch, ["c", "/exit"])
    assert _rows(database) == [(TODAY, 3, 701, 151)]


def test_reached_token_limit_refuses_the_line_and_keeps_the_chat_open(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, tokens="1000")
    session_id = _seed_turns(database, _turns(1))
    _seed_usage(database, TODAY, 5, 1000)
    calls = _replace_model(monkeypatch, _ModelScript())

    _run_cli(monkeypatch, ["hello", "/memory list", "/sessions", "/exit"])

    assert calls == []
    assert calls.token_counts == []
    captured = capsys.readouterr()
    assert captured.err == _TOKEN_REFUSAL
    assert session_id in captured.out
    assert "No saved facts." in captured.out
    saved = storage.connect(database).turn_count(_user_id(), session_id)
    assert saved == 1
    assert _rows(database) == [(TODAY, 5, 1000, 0)]


def test_a_tool_turn_may_cross_the_request_limit_then_the_next_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, requests="2")
    _seed_usage(database, TODAY, 1, 0)
    calls = _replace_model(
        monkeypatch,
        _ModelScript(replies=["It is 09:30."], usages=[_usage(2, 20, 10)]),
    )

    _run_cli(monkeypatch, ["what time is it?", "again", "/exit"])

    assert [call.prompt for call in calls] == ["what time is it?"]
    captured = capsys.readouterr()
    assert captured.out == "It is 09:30.\n"
    assert captured.err == _REQUEST_REFUSAL
    assert _rows(database) == [(TODAY, 3, 20, 10)]


def test_exhausted_budget_refuses_the_whole_turn_without_a_summary_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _setup(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, tokens="1000")
    session_id = _seed_turns(database, _turns(4))
    _seed_usage(database, TODAY, 1, 1000)
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
    _clock(monkeypatch)
    _limit(monkeypatch, tokens="1000")
    session_id = _seed_turns(database, _turns(4))
    _seed_usage(database, TODAY, 1, 400)
    backend = _install(monkeypatch, ["A short summary."], [_usage(1, 500, 100)])

    _run(monkeypatch, ["q5", "/exit"])

    assert len(backend.summarizer) == 1
    assert backend.main == []
    assert _summary_rows(database) == [(session_id, "A short summary.", 2)]
    assert _rows(database) == [(TODAY, 2, 900, 100)]
    err = capsys.readouterr().err
    assert err.endswith(_TOKEN_REFUSAL)
    assert storage.connect(database).turn_count(_user_id(), session_id) == 4


def test_a_new_local_date_starts_new_counters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    day = _clock(monkeypatch)
    _limit(monkeypatch, tokens="1000")
    _seed_usage(database, TODAY, 1, 1000)
    day[0] = TOMORROW
    calls = _replace_model(
        monkeypatch,
        _ModelScript(replies=["fine"], usages=[_usage(1, 30, 20)]),
    )

    _run_cli(monkeypatch, ["hello", "/exit"])

    assert _sent(calls) == [("test-model", "hello", "local-user")]
    assert capsys.readouterr().out == "fine\n"
    assert _rows(database) == [(TODAY, 1, 1000, 0), (TOMORROW, 1, 30, 20)]


def test_unusable_summary_is_rejected_but_its_usage_is_counted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _setup(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, tokens="100000")
    _seed_turns(database, _turns(4))
    backend = _install(monkeypatch, [" "], [_usage(1, 50, 5), _usage(1, 7, 3)])

    _run(monkeypatch, ["q5", "/exit"])

    assert len(backend.main) == 1
    assert _summary_rows(database) == []
    assert "Warning: summary skipped. The model returned an unusable summary." in (
        capsys.readouterr().err
    )
    assert _rows(database) == [(TODAY, 2, 57, 8)]


def test_non_text_chat_reply_is_rejected_but_its_usage_is_counted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, requests="10")
    replies = cast(list[str], [{"not": "text"}])
    _replace_model(monkeypatch, _ModelScript(replies=replies, usages=[_usage(1, 8, 2)]))
    _scripted_input(monkeypatch, ["hello", "/exit"])

    with pytest.raises(SystemExit):
        cli.main()

    assert "The model returned a non-text response." in capsys.readouterr().err
    assert _rows(database) == [(TODAY, 1, 8, 2)]


def test_a_failed_run_adds_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, requests="10")
    _seed_usage(database, TODAY, 1, 5)
    _replace_model(monkeypatch, _ModelScript(error=AgentRunError("boom")))
    _scripted_input(monkeypatch, ["hello", "/exit"])

    with pytest.raises(SystemExit):
        cli.main()

    assert "Agent request failed: boom" in capsys.readouterr().err
    assert _rows(database) == [(TODAY, 1, 5, 0)]


@pytest.mark.parametrize(
    "failure",
    [StorageError(storage.USAGE_READ_FAILED), ClockReadError("clock broke")],
)
def test_unreadable_counters_send_no_paid_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, tokens="1000")
    session_id = _seed_turns(database, [])
    calls = _replace_model(monkeypatch, _ModelScript())

    def broken(*_args: object) -> None:
        raise failure

    if isinstance(failure, StorageError):
        monkeypatch.setattr(ConversationStore, "daily_usage", broken)
    else:
        monkeypatch.setattr("robot_agent.tools.read_local_clock", broken)

    _run_cli(monkeypatch, ["hello", "/sessions", "/exit"])

    assert calls == []
    captured = capsys.readouterr()
    assert captured.err == f"{failure}\n"
    assert session_id in captured.out
    assert _rows(database) == []


def test_failed_recording_warns_saves_the_turn_and_refuses_the_next_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, tokens="1000000")
    session_id = _seed_turns(database, [])
    calls = _replace_model(monkeypatch, _ModelScript(replies=["one", "two"]))

    def broken(*_args: object) -> None:
        raise StorageError(storage.USAGE_SAVE_FAILED)

    monkeypatch.setattr(ConversationStore, "add_daily_usage", broken)

    _run_cli(monkeypatch, ["first", "second", "/exit"])

    assert [call.prompt for call in calls] == ["first"]
    captured = capsys.readouterr()
    assert captured.out == "one\n"
    assert (
        f"Warning: daily usage was not recorded. {storage.USAGE_SAVE_FAILED}\n"
        in captured.err
    )
    assert captured.err.endswith(f"{cli.UNTRUSTED_MESSAGE}\n")
    assert storage.connect(database).turn_count(_user_id(), session_id) == 1


@pytest.mark.parametrize("name", ["AGENT_DAILY_MAX_REQUESTS", "AGENT_DAILY_MAX_TOKENS"])
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

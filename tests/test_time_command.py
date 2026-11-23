"""The /time command prints the local time and never reaches the model."""

from datetime import datetime
from pathlib import Path

import pytest

from robot_agent import cli, storage
from robot_agent.tools import CLOCK_READ_FAILURE
from tests.test_cli import _exchange, _ModelScript, _replace_model, _scripted_input
from tests.test_sessions import _ready, _seed_turns, _session_ids, _user_id
from tests.test_summary import _install, _run, _setup, _store_summary, _summary_rows
from tests.test_summary import _turns as _summary_turns
from tests.test_tools import FIXED_TIMESTAMP, _fix_clock
from tests.test_usage import TODAY, _clock, _limit, _rows, _seed_usage

_BUDGET_TIMESTAMP = "2026-11-22T09:30:00+02:00"


def _run_cli(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    _scripted_input(monkeypatch, lines)
    cli.main()


def _turn_count(database: Path) -> int:
    store = storage.connect(database)
    total = 0
    for session_id in _session_ids(database, _user_id()):
        total += store.turn_count(_user_id(), session_id)
    return total


def test_time_prints_the_timestamp_without_a_model_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _fix_clock(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript())

    _run_cli(monkeypatch, ["  /time  ", "/exit"])

    assert calls == []
    assert calls.token_counts == []
    captured = capsys.readouterr()
    assert captured.out == f"{FIXED_TIMESTAMP}\n"
    assert captured.err == ""
    assert _turn_count(database) == 0


def test_time_leaves_the_conversation_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _fix_clock(monkeypatch)
    first = _exchange("first", "one")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["one", "two"],
            histories=[first, [*first, *_exchange("second", "two")]],
        ),
    )

    _run_cli(monkeypatch, ["first", "/time", "second", "/exit"])

    assert [call.prompt for call in calls] == ["first", "second"]
    assert calls[1].message_history == first
    assert capsys.readouterr().out == f"one\n{FIXED_TIMESTAMP}\ntwo\n"
    assert _turn_count(database) == 2


def test_time_with_an_argument_prints_usage_and_keeps_the_chat_open(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    clock_calls = _fix_clock(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["fine"]))

    _run_cli(monkeypatch, ["/time now", "hello", "/exit"])

    assert [call.prompt for call in calls] == ["hello"]
    assert clock_calls == [0]
    captured = capsys.readouterr()
    assert captured.err == "Usage: /time\n"
    assert captured.out == "fine\n"
    assert _turn_count(database) == 1


@pytest.mark.parametrize("line", ["/Time", "/TIME", "/timezone", "time"])
def test_other_lines_reach_the_model_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    line: str,
) -> None:
    _ready(monkeypatch)
    clock_calls = _fix_clock(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))

    _run_cli(monkeypatch, [line, "/exit"])

    assert [call.prompt for call in calls] == [line]
    assert clock_calls == [0]
    captured = capsys.readouterr()
    assert captured.out == "model reply\n"
    assert FIXED_TIMESTAMP not in captured.out


def test_time_ignores_a_reached_daily_budget(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, tokens="1000")
    _seed_usage(database, TODAY, 5, 1000)
    calls = _replace_model(monkeypatch, _ModelScript())

    _run_cli(monkeypatch, ["/time", "/exit"])

    assert calls == []
    captured = capsys.readouterr()
    assert captured.out == f"{_BUDGET_TIMESTAMP}\n"
    assert captured.err == ""
    assert _rows(database) == [(TODAY, 5, 1000, 0)]


def test_time_does_not_trigger_a_due_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _setup(monkeypatch)
    _fix_clock(monkeypatch)
    session_id = _seed_turns(database, _summary_turns(5))
    _store_summary(database, session_id, "Stored summary.", 1)
    backend = _install(monkeypatch)

    _run(monkeypatch, ["/time", "/exit"])

    assert backend.summarizer == []
    assert backend.main == []
    assert _summary_rows(database) == [(session_id, "Stored summary.", 1)]
    captured = capsys.readouterr()
    assert captured.out == f"{FIXED_TIMESTAMP}\n"
    assert captured.err == ""


def test_unreadable_clock_prints_the_error_once_and_keeps_the_chat_open(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ready(monkeypatch)
    clock_calls = [0]

    def broken_now() -> datetime:
        clock_calls[0] += 1
        raise OSError("clock down")

    monkeypatch.setattr("robot_agent.tools._system_now", broken_now)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["fine"]))

    _run_cli(monkeypatch, ["/time", "hello", "/exit"])

    assert clock_calls == [1]
    assert [call.prompt for call in calls] == ["hello"]
    captured = capsys.readouterr()
    assert captured.err == f"{CLOCK_READ_FAILURE}\n"
    assert captured.out == "fine\n"

"""A failed or interrupted model request returns to the prompt without saving."""

from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import AgentRunError, UsageLimitExceeded
from pydantic_ai.messages import ModelMessage

from robot_agent import cli, identity
from robot_agent.cli import INTERRUPTED_MESSAGE
from tests.test_cli import (
    RecordedCalls,
    _exchange,
    _message_rows,
    _ModelScript,
    _replace_model,
    _saved_messages,
    _scripted_input,
    _usage,
)
from tests.test_sessions import _ready, _seed_turns
from tests.test_summary import _install, _run, _saved, _setup, _summary_rows, _turns
from tests.test_tools import FIXED_TIMESTAMP, _fix_clock
from tests.test_usage import TODAY, _clock, _limit, _rows

_REQUEST_LIMIT_ERROR = "The next request would exceed the request_limit of 2"
_SECOND = _exchange("second", "ok")
_INTERRUPTED = f"{INTERRUPTED_MESSAGE}\n"
_NON_TEXT = "Agent request failed: The model returned a non-text response.\n"


def _first_fails(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException | None = None,
    token_counts: list[int] | None = None,
) -> RecordedCalls:
    """Fail the line "first" (when error is set) and answer "second" with "ok"."""
    script = _ModelScript(
        replies=["ok"],
        histories=[_SECOND],
        new_turns=[_SECOND],
        error=error,
        error_on_prompt="first",
        token_counts=token_counts or [],
    )
    return _replace_model(monkeypatch, script)


def _saved_history(database: Path) -> list[ModelMessage]:
    profile = identity.resolve_user()
    return _saved_messages(database, profile.user_id, profile.display_name)


def _non_text_script(usages: int) -> _ModelScript:
    """Scripted runs: a non-text reply first, then (optionally) a good "ok"."""
    turns = [_exchange("first", "x"), _SECOND][: usages + 1]
    return _ModelScript(
        replies=cast(list[str], [{"not": "text"}, "ok"][: usages + 1]),
        histories=turns,
        new_turns=turns,
        usages=[_usage(1, 8, 2), _usage(1, 5, 1)][: usages + 1],
    )


@pytest.mark.parametrize(
    ("error", "message"),
    [
        pytest.param(
            AgentRunError("connection failed"),
            "Agent request failed: connection failed\n",
            id="generic",
        ),
        pytest.param(
            UsageLimitExceeded(_REQUEST_LIMIT_ERROR),
            "Agent request failed: The turn exceeded the limit of 2 model requests. "
            "Raise AGENT_MAX_MODEL_REQUESTS_PER_TURN to allow more.\n",
            id="request-limit",
        ),
    ],
)
def test_a_failed_line_does_not_end_the_chat(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
    message: str,
) -> None:
    database = _ready(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_MODEL_REQUESTS_PER_TURN", "2")
    calls = _first_fails(monkeypatch, error)

    _run(monkeypatch, ["first", "second"])

    captured = capsys.readouterr()
    assert captured.err == message
    assert captured.out == "ok\n"
    assert [call.prompt for call in calls] == ["first", "second"]
    assert calls[1].message_history == []
    assert _saved_history(database) == _SECOND
    assert len(_message_rows(database)) == 1


def test_an_input_token_failure_keeps_its_message_and_the_chat_continues(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_REQUEST_INPUT_TOKENS", "100")
    calls = _first_fails(monkeypatch, token_counts=[101])

    _run(monkeypatch, ["first", "second"])

    captured = capsys.readouterr()
    assert captured.err == (
        "Agent request failed: The request exceeds the input token limit of 100.\n"
    )
    assert captured.out == "ok\n"
    assert [call.prompt for call in calls] == ["second"]
    assert _saved_history(database) == _SECOND


def test_a_non_text_failure_counts_its_usage_once_and_the_chat_continues(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, requests="10")
    _replace_model(monkeypatch, _non_text_script(usages=1))

    _run(monkeypatch, ["first", "second"])

    captured = capsys.readouterr()
    assert captured.err == _NON_TEXT
    assert captured.out == "ok\n"
    assert _rows(database) == [(TODAY, 2, 13, 3)]
    assert _saved_history(database) == _SECOND


def test_the_budget_refuses_the_next_line_after_a_failed_run_reaches_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _clock(monkeypatch)
    _limit(monkeypatch, requests="1")
    calls = _replace_model(monkeypatch, _non_text_script(usages=0))

    _run(monkeypatch, ["first", "second"])

    captured = capsys.readouterr()
    assert captured.err == (
        _NON_TEXT + "Daily request budget reached (1 of 1 used today). "
        "Raise AGENT_DAILY_MAX_REQUESTS or try again tomorrow.\n"
    )
    assert [call.prompt for call in calls] == ["first"]
    assert _rows(database) == [(TODAY, 1, 8, 2)]
    assert _message_rows(database) == []


def test_local_commands_work_after_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ready(monkeypatch)
    _fix_clock(monkeypatch)
    calls = _first_fails(monkeypatch, AgentRunError("connection failed"))

    _run(monkeypatch, ["first", "/sessions", "/memory list", "/time"])

    captured = capsys.readouterr()
    assert [call.prompt for call in calls] == ["first"]
    assert "active" in captured.out
    assert "No saved facts." in captured.out
    assert FIXED_TIMESTAMP in captured.out


def test_an_interrupted_chat_request_returns_to_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    calls = _first_fails(monkeypatch, KeyboardInterrupt())

    _run(monkeypatch, ["first", "second"])

    captured = capsys.readouterr()
    assert captured.err == _INTERRUPTED
    assert captured.out == "ok\n"
    assert [call.prompt for call in calls] == ["first", "second"]
    assert calls[1].message_history == []
    assert _saved_history(database) == _SECOND


def test_an_interrupted_summary_request_skips_the_whole_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _setup(monkeypatch)
    session_id = _seed_turns(database, _turns(4))
    backend = _install(monkeypatch, [KeyboardInterrupt()])

    _run(monkeypatch, ["q5", "/exit"])

    assert capsys.readouterr().err == _INTERRUPTED
    assert len(backend.summarizer) == 1
    assert backend.main == []
    assert _summary_rows(database) == []
    assert len(_saved(database, session_id)) == 8


def test_an_interrupt_at_the_prompt_ends_the_chat_quietly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ready(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript())

    def interrupted(_prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupted)

    cli.main()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert calls == []


def test_a_restart_restores_only_completed_turns(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    good = _exchange("good", "ok")
    _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["ok"],
            histories=[good],
            new_turns=[good],
            error=AgentRunError("connection failed"),
            error_on_prompt="bad",
        ),
    )
    _run(monkeypatch, ["bad", "good"])
    capsys.readouterr()
    restarted = _replace_model(monkeypatch, _ModelScript(error=KeyboardInterrupt()))

    _scripted_input(monkeypatch, ["stop"])
    cli.main()

    assert capsys.readouterr().err == _INTERRUPTED
    assert restarted[0].message_history == good
    assert _saved_history(database) == good
    assert len(_message_rows(database)) == 1

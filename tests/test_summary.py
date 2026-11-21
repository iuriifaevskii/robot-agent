"""Older turns can be folded into an opt-in rolling session summary."""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic_ai import AgentRunError, UsageLimitExceeded, UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
)

from robot_agent import agent, cli, storage
from robot_agent.context import SUMMARY_LABEL
from robot_agent.sessions import SESSION_UNAVAILABLE
from robot_agent.storage import StorageError
from tests.test_cli import _exchange, _forbid_input, _scripted_input, _tool_turn
from tests.test_context import _prompts
from tests.test_memory import _insert_memory, _memory_rows
from tests.test_sessions import _insert_user, _ready, _seed_turns, _user_id

_SUMMARY = "Earlier the user planned a robot head."
_LIMIT_ERROR = (
    "Exceeded the per_request_input_tokens_limit of 4096 (request_input_tokens=5000)"
)


class _Result:
    def __init__(
        self,
        output: str,
        messages: list[ModelMessage],
        produced: list[ModelMessage],
    ) -> None:
        self.output = output
        self._messages = messages
        self._produced = produced

    def all_messages(self) -> list[ModelMessage]:
        return self._messages

    def new_messages(self) -> list[ModelMessage]:
        return self._produced


@dataclass
class _Sent:
    prompt: str
    history: list[ModelMessage]
    init: dict[str, object]
    settings: dict[str, object] | None
    usage_limits: UsageLimits | None


@dataclass
class _Backend:
    """Records main and summarizer requests and scripts summarizer outcomes."""

    outcomes: list[str | Exception] = field(default_factory=list)
    main: list[_Sent] = field(default_factory=list)
    summarizer: list[_Sent] = field(default_factory=list)

    def respond(self, sent: _Sent) -> _Result:
        if "instructions" in sent.init:
            return self._summarize(sent)
        self.main.append(sent)
        reply = f"reply {len(self.main)}"
        turn = _exchange(sent.prompt, reply)
        return _Result(reply, [*sent.history, *turn], turn)

    def _summarize(self, sent: _Sent) -> _Result:
        self.summarizer.append(sent)
        if not self.outcomes:
            raise AssertionError("unexpected summarizer request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Result(outcome, [], [])


def _install(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[str | Exception] | None = None,
) -> _Backend:
    backend = _Backend(list(outcomes or []))

    class _Agent:
        def __init__(self, model: str, **kwargs: object) -> None:
            assert model == "openai:test-model"
            self.init = kwargs

        def run_sync(
            self,
            prompt: str,
            *,
            message_history: list[ModelMessage] | None = None,
            model_settings: dict[str, object] | None = None,
            usage_limits: UsageLimits | None = None,
            **_kwargs: object,
        ) -> _Result:
            history = list(message_history or [])
            return backend.respond(
                _Sent(prompt, history, self.init, model_settings, usage_limits)
            )

    monkeypatch.setattr(agent, "Agent", _Agent)
    return backend


def _run(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    _scripted_input(monkeypatch, lines)
    cli.main()


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    *,
    window: str = "2",
    after: str | None = "2",
) -> Path:
    database = _ready(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_HISTORY_TURNS", window)
    if after is not None:
        monkeypatch.setenv("AGENT_SUMMARY_AFTER_TURNS", after)
    return database


def _turns(count: int, start: int = 1) -> list[list[ModelMessage]]:
    return [_exchange(f"q{n}", f"a{n}") for n in range(start, start + count)]


def _store_summary(
    database: Path,
    session_id: str,
    text: str,
    covered: int,
    user_id: str | None = None,
) -> None:
    storage.connect(database).save_summary(
        user_id or _user_id(), session_id, text, covered
    )


def _summary_rows(database: Path) -> list[tuple[str, str, int]]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT session_id, text, covered_through FROM summaries ORDER BY 1"
        ).fetchall()
    finally:
        connection.close()
    return [(str(a), str(b), int(c)) for a, b, c in rows]


def _saved(database: Path, session_id: str) -> list[ModelMessage]:
    return storage.connect(database).load_recent(_user_id(), session_id, 50)


@pytest.mark.parametrize("raw", [None, "0"])
def test_disabled_setting_ignores_a_stored_summary(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
) -> None:
    database = _setup(monkeypatch, after=raw)
    session_id = _seed_turns(database, _turns(4))
    _store_summary(database, session_id, _SUMMARY, 2)
    backend = _install(monkeypatch)

    _run(monkeypatch, ["q5", "/exit"])

    assert backend.summarizer == []
    assert backend.main[0].prompt == "q5"
    assert _prompts(backend.main[0].history) == ["q3", "q4"]
    assert _summary_rows(database)[0][1:] == (_SUMMARY, 2)


def test_below_the_threshold_nothing_is_summarized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _setup(monkeypatch)
    _seed_turns(database, _turns(3))
    backend = _install(monkeypatch)

    _run(monkeypatch, ["q4", "/exit"])

    assert backend.summarizer == []
    assert _summary_rows(database) == []
    assert backend.main[0].prompt == "q4"
    assert _prompts(backend.main[0].history) == ["q2", "q3"]


def test_older_turns_are_summarized_and_attached(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _setup(monkeypatch)
    session_id = _seed_turns(database, _turns(4))
    backend = _install(monkeypatch, [f"  {_SUMMARY}  "])

    _run(monkeypatch, ["q5", "q6", "/exit"])

    assert len(backend.summarizer) == 1
    request = backend.summarizer[0]
    assert "tools" not in request.init
    assert request.init["instructions"] == agent.SUMMARY_INSTRUCTION
    assert request.history == []
    assert request.settings == {"max_tokens": 256}
    assert request.usage_limits is not None
    assert request.usage_limits.count_tokens_before_request
    assert request.usage_limits.per_request_input_tokens_limit == 4096
    for kept in ("q1", "a1", "q2", "a2"):
        assert kept in request.prompt
    for left_out in ("q3", "a3", "q4", "a4", "q5"):
        assert left_out not in request.prompt
    assert _summary_rows(database) == [(session_id, _SUMMARY, 2)]
    assert "Summarized 2 earlier turns." in capsys.readouterr().err
    first = backend.main[0]
    assert first.prompt.count(SUMMARY_LABEL) == 1
    assert first.prompt.count(_SUMMARY) == 1
    assert first.prompt.count("q5") == 1
    assert first.prompt.index(SUMMARY_LABEL) < first.prompt.index("q5")
    assert _prompts(first.history) == ["q3", "q4"]
    saved = _saved(database, session_id)
    assert _prompts(saved)[-2:] == ["q5", "q6"]
    assert SUMMARY_LABEL not in "".join(_prompts(saved))
    second = backend.main[1]
    assert _prompts(second.history) == ["q4", "q5"]
    assert second.prompt.count(_SUMMARY) == 1


def test_summary_rolls_forward_over_only_uncovered_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _setup(monkeypatch)
    session_id = _seed_turns(database, _turns(5))
    _store_summary(database, session_id, "S1", 2)
    backend = _install(monkeypatch, ["S2"])

    _run(monkeypatch, ["q6", "/exit"])

    assert backend.summarizer == []
    assert _prompts(backend.main[0].history) == ["q4", "q5"]
    store = storage.connect(database)
    store.append_turn(_user_id(), session_id, _exchange("q7", "a7"))

    _run(monkeypatch, ["q8", "/exit"])

    assert len(backend.summarizer) == 1
    prompt = backend.summarizer[0].prompt
    assert "S1" in prompt
    for kept in ("q3", "q4", "q5"):
        assert kept in prompt
    for left_out in ("q1", "q2", "q6", "q7", "q8"):
        assert left_out not in prompt
    assert _summary_rows(database) == [(session_id, "S2", 5)]
    assert _prompts(backend.main[1].history) == ["q6", "q7"]


def test_summary_follows_its_session_across_restart_new_and_resume(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _setup(monkeypatch)
    old_id = _seed_turns(database, _turns(4))
    other_store = storage.connect(database)
    _insert_user(database, "other-user", "Other")
    other_id = other_store.ensure_session("other-user", "Other")
    _store_summary(database, other_id, "OTHER SUMMARY", 1, "other-user")
    _store_summary(database, old_id, _SUMMARY, 2)
    backend = _install(monkeypatch)

    _run(monkeypatch, ["q5", "/new", "fresh", f"/resume {old_id}", "q6", "/exit"])

    assert backend.summarizer == []
    assert backend.main[0].prompt.count(_SUMMARY) == 1
    assert backend.main[1].prompt == "fresh"
    assert backend.main[2].prompt.count(_SUMMARY) == 1
    third = _install(monkeypatch)
    _run(monkeypatch, [f"/resume {other_id}", "/exit"])
    assert SESSION_UNAVAILABLE in capsys.readouterr().err
    assert third.main == []
    everything = "".join(call.prompt for call in backend.main)
    assert "OTHER SUMMARY" not in everything
    assert (other_id, "OTHER SUMMARY", 1) in _summary_rows(database)


def test_larger_window_after_restart_never_repeats_a_summarized_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _setup(monkeypatch, window="4")
    session_id = _seed_turns(database, _turns(4))
    _store_summary(database, session_id, _SUMMARY, 2)
    backend = _install(monkeypatch)

    _run(monkeypatch, ["q5", "/exit"])

    assert backend.summarizer == []
    assert backend.main[0].prompt.count(_SUMMARY) == 1
    assert _prompts(backend.main[0].history) == ["q3", "q4"]


def test_tool_messages_stay_out_of_the_transcript_but_in_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _setup(monkeypatch)
    turns = [_tool_turn("q1", "a1"), *_turns(3, start=2)]
    session_id = _seed_turns(database, turns)
    backend = _install(monkeypatch, [_SUMMARY])

    _run(monkeypatch, ["q5", "/exit"])

    prompt = backend.summarizer[0].prompt
    assert "q1" in prompt
    assert "a1" in prompt
    assert "get_weather" not in prompt
    assert "sunny" not in prompt
    assert "tools" not in backend.summarizer[0].init
    stored = storage.connect(database).load_turns(_user_id(), session_id, 1, 1)
    calls = [
        part
        for message in stored
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    assert [call.tool_name for call in calls] == ["get_weather"]


def test_zero_window_summarizes_and_sends_no_recent_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _setup(monkeypatch, window="0")
    _seed_turns(database, _turns(2))
    backend = _install(monkeypatch, [_SUMMARY])

    _run(monkeypatch, ["q3", "/exit"])

    prompt = backend.summarizer[0].prompt
    assert "q1" in prompt
    assert "q2" in prompt
    assert backend.main[0].history == []
    assert backend.main[0].prompt.count(_SUMMARY) == 1
    assert backend.main[0].prompt.count("q3") == 1


def test_summary_does_not_touch_saved_memories(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _setup(monkeypatch)
    session_id = _seed_turns(database, _turns(4))
    _insert_memory(database, _user_id(), "mem-1", "Likes tea", "2026-05-01T00:00:00")
    before = _memory_rows(database)
    _install(monkeypatch, [_SUMMARY])

    _run(monkeypatch, ["q5", "/memory list", "/exit"])

    assert _summary_rows(database) == [(session_id, _SUMMARY, 2)]
    assert _memory_rows(database) == before
    assert "mem-1 Likes tea" in capsys.readouterr().out
    assert all(_SUMMARY not in row.text for row in _memory_rows(database))


_BAD_OUTCOMES = [
    pytest.param(AgentRunError("provider down"), id="provider-error"),
    pytest.param(UsageLimitExceeded(_LIMIT_ERROR), id="input-limit"),
    pytest.param("   ", id="empty"),
    pytest.param("x" * 1025, id="too-long"),
]


@pytest.mark.parametrize("outcome", _BAD_OUTCOMES)
def test_failed_summary_continues_once_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str | Exception,
) -> None:
    database = _setup(monkeypatch)
    session_id = _seed_turns(database, _turns(4))
    backend = _install(monkeypatch, [outcome])

    _run(monkeypatch, ["q5", "q6", "/exit"])

    assert len(backend.summarizer) == 1
    assert "Warning" in capsys.readouterr().err
    assert _summary_rows(database) == []
    assert len(_saved(database, session_id)) == 12
    assert [call.prompt for call in backend.main] == ["q5", "q6"]
    assert _prompts(backend.main[0].history) == ["q3", "q4"]
    restarted = _install(monkeypatch, [_SUMMARY])
    _run(monkeypatch, ["q7", "/exit"])
    assert len(restarted.summarizer) == 1
    assert _summary_rows(database) == [(session_id, _SUMMARY, 4)]


def test_failed_summary_keeps_the_previous_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _setup(monkeypatch)
    session_id = _seed_turns(database, _turns(6))
    _store_summary(database, session_id, "S1", 2)
    backend = _install(monkeypatch, [AgentRunError("provider down")])

    _run(monkeypatch, ["q7", "/exit"])

    assert "S1" in backend.summarizer[0].prompt
    assert _summary_rows(database) == [(session_id, "S1", 2)]
    assert backend.main[0].prompt.count("S1") == 1
    assert _prompts(backend.main[0].history) == ["q5", "q6"]


def test_unsaved_summary_is_not_used(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _setup(monkeypatch)
    _seed_turns(database, _turns(4))
    backend = _install(monkeypatch, [_SUMMARY, _SUMMARY])

    def fail(*_args: object) -> None:
        raise StorageError(storage.SUMMARY_SAVE_FAILED)

    monkeypatch.setattr(storage.ConversationStore, "save_summary", fail)

    _run(monkeypatch, ["q5", "q6", "/exit"])

    assert len(backend.summarizer) == 1
    assert storage.SUMMARY_SAVE_FAILED in capsys.readouterr().err
    assert _summary_rows(database) == []
    assert [call.prompt for call in backend.main] == ["q5", "q6"]


def test_local_input_makes_no_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _setup(monkeypatch)
    _seed_turns(database, _turns(4))
    backend = _install(monkeypatch)

    _run(monkeypatch, ["", "/sessions", "/memory list", "/new", "   "])
    _run(monkeypatch, ["/exit"])

    assert backend.summarizer == []
    assert backend.main == []
    assert _summary_rows(database) == []


@pytest.mark.parametrize("raw", ["-1", "abc", "1.5"])
def test_invalid_setting_exits_before_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raw: str,
) -> None:
    _setup(monkeypatch, after=raw)
    backend = _install(monkeypatch)
    _forbid_input(monkeypatch)

    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 1
    assert capsys.readouterr().err == (
        "Invalid configuration: AGENT_SUMMARY_AFTER_TURNS "
        "must be a non-negative integer.\n"
    )
    assert backend.main == []
    assert backend.summarizer == []

"""Ordinary chat turns can recall a few matching saved facts."""

from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from robot_agent import cli, identity, storage
from robot_agent.context import REFERENCE_LABEL, reference_prompt
from robot_agent.storage import StorageError
from tests.test_cli import (
    _exchange,
    _forbid_input,
    _message_rows,
    _ModelScript,
    _replace_model,
    _scripted_input,
)
from tests.test_memory import _insert_memory
from tests.test_sessions import _insert_user, _ready, _run, _user_id
from tests.test_tools import (
    FIXED_TIMESTAMP,
    _fix_clock,
    _has_tool_return,
    _latest_prompt,
    _LocalFunctionModel,
    _parts,
    _prompt_text,
    _use_model,
    _user_turns,
)

_FACT = "My robot uses a Raspberry Pi 5"
_EDITED = "My robot uses a Raspberry Pi 4 board"
_OTHER = "Chosen board is a Raspberry model"
_UNRELATED = "The cat likes tuna sandwiches"
_QUESTION = "Which Raspberry Pi did I choose?"
_TIME_QUESTION = "Which Raspberry Pi did I choose, and what time is it?"
_BEST = "alpha beta gamma signal"
_TIE_EARLY = "alpha beta marker"
_TIE_SECOND = "gamma delta marker"
_OVERFLOW = "alpha gamma marker"
_LONG = "alpha beta gamma delta oversized " + ("x" * 600)
_UKRAINIAN_FACT = "Улюблена малина росте в саду."
_UKRAINIAN_MATCH = "Де росте Малина?"
_UKRAINIAN_MISS = "Котра година?"
_RANK_QUESTION = "Compare the alpha beta gamma delta options"


def _enable(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AGENT_MAX_RECALLED_MEMORIES", value)


def _prepare(database: Path) -> None:
    profile = identity.resolve_user()
    storage.connect(database).ensure_session(profile.user_id, profile.display_name)


def _store_fact(database: Path, memory_id: str, text: str, created_at: str) -> None:
    _insert_memory(database, _user_id(), memory_id, text, created_at)


def _prompts(messages: list[ModelMessage]) -> list[str]:
    found: list[str] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                found.append(part.content)
    return found


@pytest.mark.parametrize("raw", [None, "0"])
def test_disabled_recall_skips_memory_reads(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
) -> None:
    database = _ready(monkeypatch)
    if raw is not None:
        _enable(monkeypatch, raw)
    _prepare(database)
    _store_fact(database, "mem-secret", _FACT, "2026-05-01T00:00:00+00:00")

    def fail(*_args: object, **_kwargs: object) -> list[object]:
        raise AssertionError("memories should not be read")

    monkeypatch.setattr(cli, "list_memories", fail)
    calls = _run(
        monkeypatch,
        [_QUESTION, "/exit"],
        _ModelScript(replies=["ok"], histories=[_exchange(_QUESTION, "ok")]),
    )

    assert [call.prompt for call in calls] == [_QUESTION]
    assert _FACT not in calls[0].prompt
    assert REFERENCE_LABEL not in calls[0].prompt
    assert calls.token_counts[0].prompt == _QUESTION


def test_matching_fact_is_recalled_after_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ready(monkeypatch)
    _enable(monkeypatch, "1")
    saved = _run(
        monkeypatch,
        [f"/memory save {_FACT}", f"/memory save {_UNRELATED}", "/exit"],
        _ModelScript(),
    )
    assert saved == []
    assert saved.token_counts == []
    _insert_user(database, "other-user", "Other")
    _insert_memory(
        database,
        "other-user",
        "mem-other",
        _OTHER,
        "2026-05-02T00:00:00+00:00",
    )
    calls = _run(
        monkeypatch,
        ["/new", _QUESTION, "/exit"],
        _ModelScript(
            replies=["a Pi 5"],
            histories=[_exchange(_QUESTION, "a Pi 5")],
        ),
    )

    prompt = calls[0].prompt
    assert calls[0].message_history == []
    assert prompt.count(_FACT) == 1
    assert prompt.count(_QUESTION) == 1
    assert _UNRELATED not in prompt
    assert _OTHER not in prompt
    assert calls.token_counts[0].prompt == prompt


def test_ukrainian_match_ignores_case_and_skips_a_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready(monkeypatch)
    _enable(monkeypatch, "1")
    sent = reference_prompt(_UKRAINIAN_MATCH, [_UKRAINIAN_FACT])
    matched = _exchange(sent, "знайшов")
    missed = _exchange(_UKRAINIAN_MISS, "не знаю")
    calls = _run(
        monkeypatch,
        [
            f"/memory save {_UKRAINIAN_FACT}",
            _UKRAINIAN_MATCH,
            _UKRAINIAN_MISS,
            "/exit",
        ],
        _ModelScript(
            replies=["знайшов", "не знаю"],
            histories=[matched, [_exchange(_UKRAINIAN_MATCH, "знайшов"), missed]],
        ),
    )

    assert calls[0].prompt.count(_UKRAINIAN_FACT) == 1
    assert calls[0].prompt.count(_UKRAINIAN_MATCH) == 1
    assert calls[1].prompt == _UKRAINIAN_MISS
    assert REFERENCE_LABEL not in calls[1].prompt
    assert _prompts(calls[1].message_history) == [_UKRAINIAN_MATCH]


def test_recall_respects_rank_ties_and_size_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ready(monkeypatch)
    _enable(monkeypatch, "3")
    _prepare(database)
    rows = [
        ("tie-early", _TIE_EARLY, "2026-05-05T00:00:00+00:00"),
        ("long", _LONG, "2026-05-04T00:00:00+00:00"),
        ("best", _BEST, "2026-05-03T00:00:00+00:00"),
        ("tie-second", _TIE_SECOND, "2026-05-02T00:00:00+00:00"),
        ("overflow", _OVERFLOW, "2026-05-01T00:00:00+00:00"),
        ("unrelated", _UNRELATED, "2026-04-01T00:00:00+00:00"),
    ]
    for memory_id, text, created_at in rows:
        _store_fact(database, memory_id, text, created_at)
    calls = _run(
        monkeypatch,
        [_RANK_QUESTION, "/exit"],
        _ModelScript(
            replies=["picked"],
            histories=[_exchange(_RANK_QUESTION, "picked")],
        ),
    )

    prompt = calls[0].prompt
    assert prompt.index(_BEST) < prompt.index(_TIE_EARLY) < prompt.index(_TIE_SECOND)
    assert prompt.count(_RANK_QUESTION) == 1
    assert prompt.count(_BEST) == 1
    assert _LONG not in prompt
    assert _OVERFLOW not in prompt
    assert _UNRELATED not in prompt


def test_edited_and_deleted_facts_change_the_next_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ready(monkeypatch)
    _enable(monkeypatch, "1")
    _prepare(database)
    _store_fact(database, "mem-pi", _FACT, "2026-05-01T00:00:00+00:00")
    sent_fact = reference_prompt(_QUESTION, [_FACT])
    sent_edit = reference_prompt(_QUESTION, [_EDITED])
    echo_fact = _exchange(sent_fact, "the 5")
    echo_edit = _exchange(sent_edit, "the 4")
    plain_fact = _exchange(_QUESTION, "the 5")
    plain_edit = _exchange(_QUESTION, "the 4")
    plain_gone = _exchange(_QUESTION, "nothing saved")
    calls = _run(
        monkeypatch,
        [
            _QUESTION,
            f"/memory update mem-pi {_EDITED}",
            _QUESTION,
            "/memory delete mem-pi",
            _QUESTION,
            "/exit",
        ],
        _ModelScript(
            replies=["the 5", "the 4", "nothing saved"],
            histories=[
                echo_fact,
                [*plain_fact, *echo_edit],
                [*plain_fact, *plain_edit, *plain_gone],
            ],
        ),
    )

    assert calls[0].prompt == sent_fact
    assert calls[0].prompt.count(_QUESTION) == 1
    assert calls[1].prompt == sent_edit
    assert _FACT not in calls[1].prompt
    assert _prompts(calls[1].message_history) == [_QUESTION]
    assert calls[2].prompt == _QUESTION
    assert REFERENCE_LABEL not in calls[2].prompt
    follow = _exchange(_QUESTION, "still nothing")
    reopened = _run(
        monkeypatch,
        [_QUESTION, "/exit"],
        _ModelScript(replies=["still nothing"], new_turns=[follow]),
    )

    prompts = _prompts(reopened[0].message_history)
    assert prompts == [_QUESTION, _QUESTION, _QUESTION]
    assert REFERENCE_LABEL not in "".join(prompts)
    assert _FACT not in "".join(prompts)
    assert _EDITED not in "".join(prompts)
    assert reopened[0].prompt == _QUESTION


def test_recalled_tool_turn_stores_the_original_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ready(monkeypatch)
    _enable(monkeypatch, "1")
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    clock_calls = _fix_clock(monkeypatch)
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], _info: object) -> ModelResponse:
        seen.append(list(messages))
        prompt = _latest_prompt(messages)
        if _recalled_time_prompt(prompt) and not _has_tool_return(messages):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="current_local_time",
                        args={},
                        tool_call_id="time-1",
                    )
                ]
            )
        if _TIME_QUESTION in prompt:
            return ModelResponse(parts=[TextPart(content="You chose a Pi 5.")])
        return ModelResponse(parts=[TextPart(content="later")])

    model = _LocalFunctionModel(respond)
    _use_model(monkeypatch, model)
    _scripted_input(monkeypatch, [f"/memory save {_FACT}", _TIME_QUESTION])
    cli.main()

    assert clock_calls[0] == 1
    assert len(seen) == 2
    assert model.token_counts == 2
    sent = _latest_prompt(seen[0])
    assert sent.count(_FACT) == 1
    assert sent.count(_TIME_QUESTION) == 1
    assert REFERENCE_LABEL in sent
    assert "local-user" not in sent
    assert _instructions(seen)
    assert all(item is None for item in _instructions(seen))
    _scripted_input(monkeypatch, ["and now?"])
    cli.main()

    assert clock_calls[0] == 1
    assert model.token_counts == 3
    follow = _user_turns(seen)[-1]
    assert _prompt_text(follow[-1].parts[0]) == "and now?"
    prior = follow[:-1]
    assert _prompts(prior) == [_TIME_QUESTION]
    assert REFERENCE_LABEL not in "".join(_prompts(prior))
    tool_calls = [part for part in _parts(prior) if isinstance(part, ToolCallPart)]
    tool_returns = [part for part in _parts(prior) if isinstance(part, ToolReturnPart)]
    assert tool_calls[0].tool_call_id == "time-1"
    assert tool_returns[0].tool_call_id == "time-1"
    assert tool_returns[0].content == FIXED_TIMESTAMP
    texts = [part.content for part in _parts(prior) if isinstance(part, TextPart)]
    assert texts == ["You chose a Pi 5."]
    profile = identity.resolve_user()
    store = storage.connect(database)
    session_id = store.ensure_session(profile.user_id, profile.display_name)
    restored = store.load_recent(profile.user_id, session_id, 10)
    assert _prompts(restored) == [_TIME_QUESTION, "and now?"]
    assert REFERENCE_LABEL not in "".join(_prompts(restored))


def _recalled_time_prompt(prompt: str) -> bool:
    return _FACT in prompt and _TIME_QUESTION in prompt


def _instructions(seen: list[list[ModelMessage]]) -> list[str | None]:
    return [
        message.instructions
        for messages in seen
        for message in messages
        if isinstance(message, ModelRequest)
    ]


def test_memory_read_failure_sends_one_ordinary_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _enable(monkeypatch, "1")
    _prepare(database)
    _store_fact(database, "mem-pi", _FACT, "2026-05-01T00:00:00+00:00")
    original = cli.list_memories
    reads = {"count": 0}

    def flaky(store: object, user_id: str) -> list[object]:
        reads["count"] += 1
        if reads["count"] > 1:
            raise StorageError(storage.MEMORY_READ_FAILED)
        return original(store, user_id)

    monkeypatch.setattr(cli, "list_memories", flaky)
    sent = reference_prompt(_QUESTION, [_FACT])
    echo = _exchange(sent, "the 5")
    ordinary = _exchange(_QUESTION, "ordinary")
    calls = _run(
        monkeypatch,
        [_QUESTION, _QUESTION, "/exit"],
        _ModelScript(
            replies=["the 5", "ordinary"],
            histories=[echo, [_exchange(_QUESTION, "the 5"), ordinary]],
        ),
    )

    assert calls[0].prompt == sent
    assert calls[1].prompt == _QUESTION
    assert _FACT not in calls[1].prompt
    assert _prompts(calls[1].message_history) == [_QUESTION]
    assert len(calls.token_counts) == 2
    assert calls.token_counts[1].prompt == _QUESTION
    printed = capsys.readouterr()
    assert printed.err == f"{storage.MEMORY_READ_FAILED}\n"
    assert printed.out == "the 5\nordinary\n"


def test_oversized_recalled_request_is_not_sent(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    _enable(monkeypatch, "1")
    monkeypatch.setenv("AGENT_MAX_REQUEST_INPUT_TOKENS", "10")
    _prepare(database)
    _store_fact(database, "mem-pi", _FACT, "2026-05-01T00:00:00+00:00")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(replies=["nope"], token_counts=[11]),
    )
    _scripted_input(monkeypatch, [_QUESTION])

    cli.main()
    counted = calls.token_counts[0].prompt
    assert calls == []
    assert len(calls.token_counts) == 1
    assert counted.count(_FACT) == 1
    assert counted.count(_QUESTION) == 1
    assert REFERENCE_LABEL in counted
    assert _message_rows(database) == []
    assert capsys.readouterr().err == (
        "Agent request failed: The request exceeds the input token limit of 10.\n"
    )


@pytest.mark.parametrize("value", ["4", "-1", "two", "1.5"])
def test_invalid_recall_setting_exits_before_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    _ready(monkeypatch)
    _enable(monkeypatch, value)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _forbid_input(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == (
        "Invalid configuration: AGENT_MAX_RECALLED_MEMORIES must be "
        "an integer from 0 through 3.\n"
    )
    assert calls == []
    assert calls.token_counts == []


def test_local_commands_do_not_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready(monkeypatch)
    _enable(monkeypatch, "3")

    def fail(*_args: object, **_kwargs: object) -> list[object]:
        raise AssertionError("memories should not be read")

    monkeypatch.setattr(cli, "list_memories", fail)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _scripted_input(monkeypatch, [])
    cli.main()
    _scripted_input(
        monkeypatch,
        [
            "/memory save a note about raspberries here",
            "/memory update",
            "/new",
            "/sessions",
            "/resume",
            "/resume missing-session",
            "",
            "   ",
            "/exit",
        ],
    )
    cli.main()

    assert calls == []
    assert calls.token_counts == []

import runpy
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pytest
from pydantic_ai import AgentRunError, UsageLimitExceeded, UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage

from robot_agent import agent, cli, identity, storage
from robot_agent.storage import StorageError
from robot_agent.tools import ClockReadError

MISSING_BOTH = (
    "Missing configuration: OPENAI_API_KEY, OPENAI_MODEL. "
    "Copy .env.example to .env and set the values.\n"
)
MISSING_KEY = (
    "Missing configuration: OPENAI_API_KEY. "
    "Copy .env.example to .env and set the values.\n"
)
MISSING_MODEL = (
    "Missing configuration: OPENAI_MODEL. "
    "Copy .env.example to .env and set the values.\n"
)


def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "AGENT_MAX_HISTORY_TURNS",
        "AGENT_MAX_REQUEST_INPUT_TOKENS",
        "AGENT_MAX_OUTPUT_TOKENS",
        "AGENT_MAX_MODEL_REQUESTS_PER_TURN",
        "AGENT_MAX_RECALLED_MEMORIES",
        "AGENT_SUMMARY_AFTER_TURNS",
        "AGENT_DAILY_MAX_REQUESTS",
        "AGENT_DAILY_MAX_TOKENS",
        "AGENT_SESSION_MAX_REQUESTS",
        "AGENT_SESSION_MAX_TOKENS",
        "AGENT_DB_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    database = Path(tempfile.mkdtemp(prefix="robot-agent-")) / "chat.sqlite3"
    monkeypatch.setenv("AGENT_DB_PATH", str(database))
    return database


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")


class ModelCall(NamedTuple):
    model_name: str
    prompt: str
    user_id: str
    message_history: list[ModelMessage]
    model_settings: dict[str, object] | None
    usage_limits: UsageLimits | None


class TokenCount(NamedTuple):
    prompt: str
    message_history: list[ModelMessage]
    input_tokens: int
    limit: int | None


class RecordedCalls(list[ModelCall]):
    def __init__(self) -> None:
        super().__init__()
        self.token_counts: list[TokenCount] = []


def _sent(calls: list[ModelCall]) -> list[tuple[str, str, str]]:
    return [(call.model_name, call.prompt, call.user_id) for call in calls]


def _exchange(user_text: str, reply_text: str) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        ModelResponse(parts=[TextPart(content=reply_text)]),
    ]


def _tool_turn(user_text: str, reply_text: str) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="get_weather",
                    args={"city": "Kyiv"},
                    tool_call_id="call-1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="get_weather",
                    content="sunny",
                    tool_call_id="call-1",
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content=reply_text)]),
    ]


def _next_reply(pending: list[str], prompt: str) -> str:
    if not pending:
        raise AssertionError(f"unexpected model call for {prompt!r}")
    return pending.pop(0)


def _next_history(
    passed: list[ModelMessage],
    pending: list[list[ModelMessage]] | None,
) -> list[ModelMessage]:
    if pending is None:
        return passed
    if not pending:
        raise AssertionError("missing scripted history")
    return pending.pop(0)


@dataclass
class _ModelScript:
    replies: list[str] = field(default_factory=list)
    error: BaseException | None = None
    error_on_prompt: str | None = None
    histories: list[list[ModelMessage]] | None = None
    token_counts: list[int] = field(default_factory=list)
    new_turns: list[list[ModelMessage]] | None = None
    usages: list[RunUsage] = field(default_factory=list)


def _usage(requests: int, input_tokens: int, output_tokens: int) -> RunUsage:
    return RunUsage(
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class _Result:
    def __init__(
        self,
        output: str,
        messages: list[ModelMessage],
        produced: list[ModelMessage],
        usage: RunUsage | None = None,
    ) -> None:
        self.output = output
        self._messages = messages
        self._produced = produced
        self.usage = usage or _usage(1, 10, 5)

    def all_messages(self) -> list[ModelMessage]:
        return self._messages

    def new_messages(self) -> list[ModelMessage]:
        return self._produced


class _Playback:
    def __init__(self, script: _ModelScript, calls: RecordedCalls) -> None:
        self.replies = list(script.replies)
        self.error = script.error
        self.error_on_prompt = script.error_on_prompt
        self.histories = None if script.histories is None else list(script.histories)
        self.new_turns = None if script.new_turns is None else list(script.new_turns)
        self.counts = list(script.token_counts)
        self.usages = list(script.usages)
        self.calls = calls
        self.user_id = ""
        self.model_name = ""

    def bind_model(self, model: str) -> None:
        if not model.startswith("openai:"):
            raise AssertionError(f"expected an openai model, got {model}")
        self.model_name = model.removeprefix("openai:")

    def run_sync(
        self,
        prompt: str,
        message_history: list[ModelMessage] | None,
        model_settings: dict[str, object] | None,
        usage_limits: UsageLimits | None,
    ) -> _Result:
        history = list(message_history or [])
        self._count(prompt, history, usage_limits)
        self.calls.append(
            ModelCall(
                self.model_name,
                prompt,
                self.user_id,
                history,
                model_settings,
                usage_limits,
            )
        )
        if self._fails(prompt):
            assert self.error is not None
            raise self.error
        reply = _next_reply(self.replies, prompt)
        returned = _next_history(history, self.histories)
        usage = self.usages.pop(0) if self.usages else None
        return _Result(reply, returned, self._produced(history, returned), usage)

    def _produced(
        self,
        history: list[ModelMessage],
        returned: list[ModelMessage],
    ) -> list[ModelMessage]:
        if self.new_turns is None:
            return list(returned[len(history) :])
        if not self.new_turns:
            raise AssertionError("missing scripted new turn")
        return self.new_turns.pop(0)

    def _count(
        self,
        prompt: str,
        history: list[ModelMessage],
        usage_limits: UsageLimits | None,
    ) -> None:
        if usage_limits is None or not usage_limits.count_tokens_before_request:
            return
        counted = self.counts.pop(0) if self.counts else 1
        limit = usage_limits.per_request_input_tokens_limit
        self.calls.token_counts.append(TokenCount(prompt, history, counted, limit))
        if limit is not None and counted > limit:
            raise UsageLimitExceeded(
                "Exceeded the per_request_input_tokens_limit of "
                f"{limit} (request_input_tokens={counted})"
            )

    def _fails(self, prompt: str) -> bool:
        if self.error is None:
            return False
        return self.error_on_prompt is None or prompt == self.error_on_prompt


def _agent_type(playback: _Playback) -> type:
    class _Agent:
        def __init__(self, model: str, **_kwargs: object) -> None:
            playback.bind_model(model)

        def run_sync(
            self,
            prompt: str,
            *,
            message_history: list[ModelMessage] | None = None,
            model_settings: dict[str, object] | None = None,
            usage_limits: UsageLimits | None = None,
            **_kwargs: object,
        ) -> _Result:
            return playback.run_sync(
                prompt,
                message_history,
                model_settings,
                usage_limits,
            )

    return _Agent


def _replace_model(
    monkeypatch: pytest.MonkeyPatch,
    script: _ModelScript,
) -> RecordedCalls:
    calls = RecordedCalls()
    playback = _Playback(script, calls)
    original = agent.run_model

    def wrapped(
        model_name: str,
        prompt: str,
        user_id: str,
        message_history: list[ModelMessage],
        limits: agent.ModelLimits,
    ) -> tuple[str, list[ModelMessage], list[ModelMessage]]:
        playback.user_id = user_id
        return original(model_name, prompt, user_id, message_history, limits)

    monkeypatch.setattr(agent, "Agent", _agent_type(playback))
    monkeypatch.setattr(agent, "run_model", wrapped)
    return calls


def _scripted_input(
    monkeypatch: pytest.MonkeyPatch,
    lines: list[str],
    *,
    after: str = "eof",
) -> None:
    remaining = iter(lines)

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(remaining)
        except StopIteration as exc:
            if after == "eof":
                raise EOFError from exc
            raise AssertionError("input should not be read again") from exc

    monkeypatch.setattr("builtins.input", fake_input)


def _chat(path: Path) -> cli.ActiveChat:
    profile = identity.resolve_user()
    store = storage.connect(path)
    session_id = store.ensure_session(profile.user_id, profile.display_name)
    return cli.ActiveChat(
        profile.user_id,
        profile.display_name,
        session_id,
        [],
        store,
    )


def _forbid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_input(_prompt: str = "") -> str:
        raise AssertionError("input should not be read")

    monkeypatch.setattr("builtins.input", fail_input)


def test_one_line_then_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _scripted_input(monkeypatch, ["hello", "/exit"])

    cli.main()

    assert _sent(calls) == [("test-model", "hello", "local-user")]
    assert calls[0].message_history == []
    captured = capsys.readouterr()
    assert captured.out == "model reply\n"
    assert captured.err == ""


def test_second_message_receives_first_exchange_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    exchange = _exchange("first", "reply-1")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["reply-1", "reply-2"],
            histories=[exchange, [*exchange, *_exchange("second", "reply-2")]],
        ),
    )
    _scripted_input(monkeypatch, ["first", "second", "/exit"])

    cli.main()

    assert _sent(calls) == [
        ("test-model", "first", "local-user"),
        ("test-model", "second", "local-user"),
    ]
    assert calls[0].message_history == []
    assert calls[1].message_history == exchange
    assert capsys.readouterr().out == "reply-1\nreply-2\n"


def test_empty_and_whitespace_lines_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _scripted_input(monkeypatch, ["  hello  ", "", "   ", "/exit"])

    cli.main()

    assert _sent(calls) == [("test-model", "hello", "local-user")]
    assert calls[0].message_history == []
    assert len(calls.token_counts) == 1
    assert capsys.readouterr().out == "model reply\n"


def test_exit_on_first_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _scripted_input(monkeypatch, ["/exit"])

    cli.main()

    assert calls == []
    assert calls.token_counts == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_exit_with_surrounding_whitespace(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _scripted_input(monkeypatch, ["  /exit  "])

    cli.main()

    assert calls == []
    assert calls.token_counts == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_exit_word_variants_are_messages(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["a", "b", "c"]))
    _scripted_input(monkeypatch, ["exit", "/EXIT", "/exit please", "/exit"])

    cli.main()

    assert _sent(calls) == [
        ("test-model", "exit", "local-user"),
        ("test-model", "/EXIT", "local-user"),
        ("test-model", "/exit please", "local-user"),
    ]
    assert capsys.readouterr().out == "a\nb\nc\n"


def test_eof_before_any_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))

    def fail_input(_prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", fail_input)

    cli.main()

    assert calls == []
    assert calls.token_counts == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_eof_after_a_reply(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _scripted_input(monkeypatch, ["hello"])

    cli.main()

    assert _sent(calls) == [("test-model", "hello", "local-user")]
    assert calls[0].message_history == []
    assert len(calls.token_counts) == 1
    captured = capsys.readouterr()
    assert captured.out == "model reply\n"
    assert captured.err == ""


def test_tool_messages_are_passed_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    user = ModelRequest(parts=[UserPromptPart(content="weather?")])
    tool_call = ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="get_weather",
                args={"city": "Kyiv"},
                tool_call_id="call-1",
            )
        ]
    )
    tool_result = ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name="get_weather",
                content="sunny",
                tool_call_id="call-1",
            )
        ]
    )
    final = ModelResponse(parts=[TextPart(content="It is sunny.")])
    completed = [user, tool_call, tool_result, final]
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["It is sunny.", "next reply"],
            histories=[completed, completed],
        ),
    )
    _scripted_input(monkeypatch, ["weather?", "and tomorrow?", "/exit"])

    cli.main()

    assert calls[0].message_history == []
    assert calls[1].prompt == "and tomorrow?"
    recorded = calls[1].message_history
    assert len(recorded) == len(completed)
    assert all(left is right for left, right in zip(recorded, completed, strict=True))


def test_new_run_chat_starts_with_empty_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_env(monkeypatch)
    exchange = _exchange("hello", "one")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["one", "two", "later"],
            histories=[exchange, [*exchange, *_exchange("again", "two")], []],
        ),
    )
    limits = agent.ModelLimits(
        max_history_turns=6,
        max_request_input_tokens=4096,
        max_output_tokens=256,
    )
    _scripted_input(monkeypatch, ["hello", "again", "/exit"])
    cli.run_chat("test-model", limits, _chat(tmp_path / "first.sqlite3"))

    _scripted_input(monkeypatch, ["later", "/exit"])
    cli.run_chat("test-model", limits, _chat(tmp_path / "second.sqlite3"))

    assert _sent(calls) == [
        ("test-model", "hello", "local-user"),
        ("test-model", "again", "local-user"),
        ("test-model", "later", "local-user"),
    ]
    assert calls[0].message_history == []
    assert calls[1].message_history == exchange
    assert calls[2].message_history == []


def test_blank_lines_do_not_change_history(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    exchange = _exchange("hello", "model reply")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["model reply", "next reply"],
            histories=[exchange, [*exchange, *_exchange("next", "next reply")]],
        ),
    )
    _scripted_input(monkeypatch, ["hello", "", "   ", "next", "/exit"])

    cli.main()

    assert _sent(calls) == [
        ("test-model", "hello", "local-user"),
        ("test-model", "next", "local-user"),
    ]
    assert calls[1].message_history == exchange
    assert len(calls.token_counts) == 2
    captured = capsys.readouterr()
    assert captured.out == "model reply\nnext reply\n"
    assert captured.err == ""


def test_first_message_receives_empty_history_and_configured_limits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_HISTORY_TURNS", "2")
    monkeypatch.setenv("AGENT_MAX_REQUEST_INPUT_TOKENS", "1234")
    monkeypatch.setenv("AGENT_MAX_OUTPUT_TOKENS", "77")
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _scripted_input(monkeypatch, ["hello", "/exit"])

    cli.main()

    assert _sent(calls) == [("test-model", "hello", "local-user")]
    assert calls[0].message_history == []
    assert calls[0].model_settings == {"max_tokens": 77}
    usage = calls[0].usage_limits
    assert usage is not None
    assert usage.per_request_input_tokens_limit == 1234
    assert usage.count_tokens_before_request is True
    assert capsys.readouterr().out == "model reply\n"


def test_fourth_request_keeps_two_complete_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_HISTORY_TURNS", "2")
    turn_1 = _exchange("one", "r1")
    turn_2 = _tool_turn("two", "r2")
    turn_3 = _exchange("three", "r3")
    kept = [*turn_2, *turn_3]
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["r1", "r2", "r3", "r4"],
            histories=[
                turn_1,
                [*turn_1, *turn_2],
                [*turn_1, *turn_2, *turn_3],
                [*kept, *_exchange("four", "r4")],
            ],
        ),
    )
    _scripted_input(monkeypatch, ["one", "two", "three", "four", "/exit"])

    cli.main()

    assert _sent(calls) == [
        ("test-model", "one", "local-user"),
        ("test-model", "two", "local-user"),
        ("test-model", "three", "local-user"),
        ("test-model", "four", "local-user"),
    ]
    recorded = calls[3].message_history
    assert len(recorded) == len(kept)
    assert all(left is right for left, right in zip(recorded, kept, strict=True))


def test_history_limit_zero_sends_only_the_current_prompt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_HISTORY_TURNS", "0")
    first = _exchange("one", "a")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["a", "b"],
            histories=[first, [*first, *_exchange("two", "b")]],
        ),
    )
    _scripted_input(monkeypatch, ["one", "two", "/exit"])

    cli.main()

    assert _sent(calls) == [
        ("test-model", "one", "local-user"),
        ("test-model", "two", "local-user"),
    ]
    assert calls[0].message_history == []
    assert calls[1].message_history == []
    assert capsys.readouterr().out == "a\nb\n"


def test_oversized_request_is_rejected_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_REQUEST_INPUT_TOKENS", "100")
    exchange = _exchange("hello", "model reply")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["model reply"],
            histories=[exchange],
            token_counts=[1, 101],
        ),
    )
    _scripted_input(monkeypatch, ["hello", "again"])

    cli.main()
    captured = capsys.readouterr()
    assert captured.out == "model reply\n"
    assert captured.err == (
        "Agent request failed: The request exceeds the input token limit of 100.\n"
    )
    assert _sent(calls) == [("test-model", "hello", "local-user")]
    assert [count.prompt for count in calls.token_counts] == ["hello", "again"]
    assert calls.token_counts[1].message_history == exchange
    assert calls.token_counts[1].input_tokens == 101
    assert calls.token_counts[1].limit == 100


@pytest.mark.parametrize(
    ("name", "value", "requirement"),
    [
        ("AGENT_MAX_HISTORY_TURNS", "-1", "a non-negative integer"),
        ("AGENT_MAX_HISTORY_TURNS", "two", "a non-negative integer"),
        ("AGENT_MAX_REQUEST_INPUT_TOKENS", "0", "a positive integer"),
        ("AGENT_MAX_REQUEST_INPUT_TOKENS", "many", "a positive integer"),
        ("AGENT_MAX_OUTPUT_TOKENS", "-5", "a positive integer"),
        ("AGENT_MAX_OUTPUT_TOKENS", "1.5", "a positive integer"),
    ],
)
def test_invalid_limit_exits_before_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    value: str,
    requirement: str,
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    monkeypatch.setenv(name, value)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _forbid_input(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == (
        f"Invalid configuration: {name} must be {requirement}.\n"
    )
    assert calls == []
    assert calls.token_counts == []


def test_missing_only_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _forbid_input(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == MISSING_KEY
    assert calls == []


def test_missing_only_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _forbid_input(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == MISSING_MODEL
    assert calls == []


def test_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _forbid_input(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == MISSING_BOTH
    assert calls == []


def test_package_entry_point_reports_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("robot_agent", run_name="__main__")

    assert exc.value.code == 1
    assert capsys.readouterr().err == MISSING_BOTH
    assert calls == []


def test_model_call_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(
        monkeypatch,
        _ModelScript(error=AgentRunError("connection failed")),
    )
    _scripted_input(monkeypatch, ["hello"])

    cli.main()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Agent request failed: connection failed\n"
    assert _sent(calls) == [("test-model", "hello", "local-user")]
    assert calls[0].message_history == []


def test_model_failure_on_second_turn(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    exchange = _exchange("hello", "model reply")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["model reply"],
            histories=[exchange],
            error=AgentRunError("connection failed"),
            error_on_prompt="again",
        ),
    )
    _scripted_input(monkeypatch, ["hello", "again"])

    cli.main()
    captured = capsys.readouterr()
    assert captured.out == "model reply\n"
    assert captured.err == "Agent request failed: connection failed\n"
    assert _sent(calls) == [
        ("test-model", "hello", "local-user"),
        ("test-model", "again", "local-user"),
    ]
    assert calls[0].message_history == []
    assert calls[1].message_history == exchange


def _saved_messages(path: Path, user_id: str, display_name: str) -> list[ModelMessage]:
    store = storage.connect(path)
    session_id = store.ensure_session(user_id, display_name)
    return store.load_recent(user_id, session_id, 100)


def _message_rows(path: Path) -> list[tuple[str, int, str]]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            """
            SELECT session_id, position, body
            FROM messages
            ORDER BY session_id, position
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        (str(session_id), int(position), str(body))
        for session_id, position, body in rows
    ]


def _identity_rows(
    path: Path,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    connection = sqlite3.connect(path)
    try:
        users = connection.execute(
            "SELECT user_id, display_name FROM users ORDER BY user_id"
        ).fetchall()
        sessions = connection.execute(
            """
            SELECT session_id, user_id
            FROM sessions
            ORDER BY user_id, session_id
            """
        ).fetchall()
    finally:
        connection.close()
    user_rows = [(str(user_id), str(name)) for user_id, name in users]
    session_rows = [(str(session_id), str(user_id)) for session_id, user_id in sessions]
    return user_rows, session_rows


def test_fresh_database_commits_the_turn_before_the_reply(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _isolate_env(monkeypatch)
    _configure(monkeypatch)
    profile = identity.resolve_user()
    turn = _exchange("hello", "model reply")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(replies=["model reply"], histories=[turn], new_turns=[turn]),
    )
    _scripted_input(monkeypatch, ["hello", "/exit"])
    printed: list[str] = []
    write = print

    def spy(*args: object, **kwargs: object) -> None:
        users, sessions = _identity_rows(database)
        assert users == [(profile.user_id, profile.display_name)]
        assert len(sessions) == 1
        assert sessions[0][1] == profile.user_id
        saved = _saved_messages(database, profile.user_id, profile.display_name)
        assert saved == turn
        printed.append(str(args[0]))
        write(*args, **kwargs)

    monkeypatch.setattr("builtins.print", spy)
    cli.main()

    assert printed == ["model reply"]
    assert calls[0].message_history == []
    assert len(calls.token_counts) == 1
    assert capsys.readouterr().out == "model reply\n"


def test_restart_sends_only_the_latest_saved_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _isolate_env(monkeypatch)
    _configure(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_HISTORY_TURNS", "2")
    profile = identity.resolve_user()
    turn_1 = _exchange("one", "r1")
    turn_2 = _tool_turn("two", "r2")
    turn_3 = _exchange("three", "r3")
    turn_4 = _exchange("four", "r4")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["r1", "r2", "r3", "r4"],
            histories=[
                turn_1,
                [*turn_1, *turn_2],
                [*turn_1, *turn_2, *turn_3],
                [*turn_2, *turn_3, *turn_4],
            ],
            new_turns=[turn_1, turn_2, turn_3, turn_4],
        ),
    )
    _scripted_input(monkeypatch, ["one", "two", "three"])
    cli.main()

    saved = [*turn_1, *turn_2, *turn_3]
    assert _saved_messages(database, profile.user_id, profile.display_name) == saved
    rows_before = _message_rows(database)
    assert len(rows_before) == 3
    _scripted_input(monkeypatch, ["four"])
    cli.main()

    assert calls[3].message_history == [*turn_2, *turn_3]
    assert calls[3].prompt == "four"
    assert len(calls.token_counts) == 4
    rows_after = _message_rows(database)
    assert rows_after[:3] == rows_before
    assert len(rows_after) == 4
    _users, sessions = _identity_rows(database)
    assert len(sessions) == 1


def test_history_limit_zero_saves_turns_without_sending_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _isolate_env(monkeypatch)
    _configure(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_HISTORY_TURNS", "0")
    profile = identity.resolve_user()
    turn_1 = _exchange("one", "a")
    turn_2 = _exchange("two", "b")
    turn_3 = _exchange("three", "c")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["a", "b", "c"],
            histories=[turn_1, turn_2, turn_3],
            new_turns=[turn_1, turn_2, turn_3],
        ),
    )
    _scripted_input(monkeypatch, ["one", "two"])
    cli.main()

    assert calls[0].message_history == []
    assert calls[1].message_history == []
    saved = _saved_messages(database, profile.user_id, profile.display_name)
    assert saved == [*turn_1, *turn_2]
    _scripted_input(monkeypatch, ["three"])
    cli.main()

    assert calls[2].message_history == []
    assert len(calls.token_counts) == 3
    assert len(_message_rows(database)) == 3


def test_other_users_session_is_not_restored_or_appended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _isolate_env(monkeypatch)
    _configure(monkeypatch)
    profile = identity.resolve_user()
    other_turn = _exchange("secret", "hidden")
    prepared = storage.connect(database)
    other_session = prepared.ensure_session("other-user", "Other")
    prepared.append_turn("other-user", other_session, other_turn)
    local_turn = _exchange("hello", "model reply")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["model reply"],
            histories=[local_turn],
            new_turns=[local_turn],
        ),
    )
    _scripted_input(monkeypatch, ["hello", "/exit"])
    cli.main()

    assert calls[0].message_history == []
    local = _saved_messages(database, profile.user_id, profile.display_name)
    assert local == local_turn
    reopened = storage.connect(database)
    assert reopened.load_recent("other-user", other_session, 10) == other_turn
    with pytest.raises(StorageError, match=storage.NOT_OWNER):
        reopened.append_turn(profile.user_id, other_session, _exchange("nope", "no"))
    assert reopened.load_recent("other-user", other_session, 10) == other_turn
    users, sessions = _identity_rows(database)
    assert (profile.user_id, profile.display_name) in users
    assert ("other-user", "Other") in users
    assert len(sessions) == 2


def test_unopenable_database_exits_before_a_model_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _isolate_env(monkeypatch)
    _configure(monkeypatch)
    database.mkdir()
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _forbid_input(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == (
        f"Could not open the conversation database: {database}.\n"
    )
    assert calls == []
    assert calls.token_counts == []
    assert database.is_dir()


def test_non_database_file_is_left_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _isolate_env(monkeypatch)
    _configure(monkeypatch)
    database.write_bytes(b"not a database")
    snapshot = database.read_bytes()
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _forbid_input(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    message = capsys.readouterr().err
    assert message == f"Could not open the conversation database: {database}.\n"
    assert calls == []
    assert calls.token_counts == []
    assert database.read_bytes() == snapshot


def test_undecodable_turn_exits_before_a_model_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _isolate_env(monkeypatch)
    _configure(monkeypatch)
    profile = identity.resolve_user()
    store = storage.connect(database)
    session_id = store.ensure_session(profile.user_id, profile.display_name)
    store.append_turn(profile.user_id, session_id, _exchange("hello", "reply"))
    connection = sqlite3.connect(database)
    connection.execute("UPDATE messages SET body = ?", ("not-json",))
    connection.commit()
    connection.close()
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _forbid_input(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == "Could not read the saved conversation.\n"
    assert calls == []
    assert calls.token_counts == []
    assert _message_rows(database) == [(session_id, 1, "not-json")]


@pytest.mark.parametrize(
    ("error", "stderr"),
    [
        (
            AgentRunError("connection failed"),
            "Agent request failed: connection failed\n",
        ),
        (
            ClockReadError("The local clock could not be read."),
            "Agent request failed: The local clock could not be read.\n",
        ),
    ],
)
def test_failed_turn_is_not_saved(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    stderr: str,
) -> None:
    database = _isolate_env(monkeypatch)
    _configure(monkeypatch)
    profile = identity.resolve_user()
    first = _exchange("hello", "model reply")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["model reply"],
            histories=[first],
            new_turns=[first],
            error=error,
            error_on_prompt="again",
        ),
    )
    _scripted_input(monkeypatch, ["hello", "again"])

    cli.main()
    captured = capsys.readouterr()
    assert captured.out == "model reply\n"
    assert captured.err == stderr
    assert [call.prompt for call in calls] == ["hello", "again"]
    assert len(calls.token_counts) == 2
    assert _saved_messages(database, profile.user_id, profile.display_name) == first
    assert len(_message_rows(database)) == 1


def test_save_failure_hides_the_reply_and_keeps_prior_turns(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _isolate_env(monkeypatch)
    _configure(monkeypatch)
    profile = identity.resolve_user()
    first = _exchange("hello", "model reply")
    second = _exchange("again", "next reply")
    calls = _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["model reply", "next reply"],
            histories=[first, [*first, *second]],
            new_turns=[first, second],
        ),
    )
    original = storage.ConversationStore.append_turn
    attempts = {"count": 0}

    def append(
        self: storage.ConversationStore,
        user_id: str,
        session_id: str,
        messages: list[ModelMessage],
    ) -> None:
        attempts["count"] += 1
        if attempts["count"] > 1:
            raise StorageError(storage.SAVE_FAILED)
        original(self, user_id, session_id, messages)

    monkeypatch.setattr(storage.ConversationStore, "append_turn", append)
    _scripted_input(monkeypatch, ["hello", "again"], after="error")

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == "model reply\n"
    assert captured.err == f"{storage.SAVE_FAILED}\n"
    assert [call.prompt for call in calls] == ["hello", "again"]
    saved = _saved_messages(database, profile.user_id, profile.display_name)
    assert saved == first
    assert len(_message_rows(database)) == 1


@pytest.mark.parametrize("lines", [["   "], ["/exit"], []])
def test_blank_exit_and_eof_add_no_turn(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    lines: list[str],
) -> None:
    database = _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _scripted_input(monkeypatch, lines)
    cli.main()

    assert calls == []
    assert calls.token_counts == []
    assert _message_rows(database) == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

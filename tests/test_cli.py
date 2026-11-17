import runpy
from dataclasses import dataclass, field
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

from robot_agent import agent, cli

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


def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "AGENT_MAX_HISTORY_TURNS",
        "AGENT_MAX_REQUEST_INPUT_TOKENS",
        "AGENT_MAX_OUTPUT_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


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
    error: Exception | None = None
    error_on_prompt: str | None = None
    histories: list[list[ModelMessage]] | None = None
    token_counts: list[int] = field(default_factory=list)


class _Result:
    def __init__(self, output: str, messages: list[ModelMessage]) -> None:
        self.output = output
        self._messages = messages

    def all_messages(self) -> list[ModelMessage]:
        return self._messages


class _Playback:
    def __init__(self, script: _ModelScript, calls: RecordedCalls) -> None:
        self.replies = list(script.replies)
        self.error = script.error
        self.error_on_prompt = script.error_on_prompt
        self.histories = None if script.histories is None else list(script.histories)
        self.counts = list(script.token_counts)
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
        return _Result(reply, returned)

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
        def __init__(self, model: str) -> None:
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
    ) -> tuple[str, list[ModelMessage]]:
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
    cli.run_chat("test-model", "local-user", limits)

    _scripted_input(monkeypatch, ["later", "/exit"])
    cli.run_chat("test-model", "local-user", limits)

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
    _scripted_input(monkeypatch, ["hello", "again"], after="error")

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
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
    _scripted_input(monkeypatch, ["hello"], after="error")

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
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
    _scripted_input(monkeypatch, ["hello", "again"], after="error")

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == "model reply\n"
    assert captured.err == "Agent request failed: connection failed\n"
    assert _sent(calls) == [
        ("test-model", "hello", "local-user"),
        ("test-model", "again", "local-user"),
    ]
    assert calls[0].message_history == []
    assert calls[1].message_history == exchange

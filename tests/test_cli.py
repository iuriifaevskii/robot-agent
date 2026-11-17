import runpy
from typing import NamedTuple

import pytest
from pydantic_ai import AgentRunError
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
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")


class ModelCall(NamedTuple):
    model_name: str
    prompt: str
    user_id: str
    message_history: list[ModelMessage]


def _sent(calls: list[ModelCall]) -> list[tuple[str, str, str]]:
    return [(call.model_name, call.prompt, call.user_id) for call in calls]


def _exchange(user_text: str, reply_text: str) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
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


def _replace_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    replies: list[str] | None = None,
    error: Exception | None = None,
    error_on_prompt: str | None = None,
    histories: list[list[ModelMessage]] | None = None,
) -> list[ModelCall]:
    calls: list[ModelCall] = []
    pending_replies = list(replies or [])
    pending_histories = None if histories is None else list(histories)
    active_user = [""]

    class _Result:
        def __init__(self, output: str, messages: list[ModelMessage]) -> None:
            self.output = output
            self._messages = messages

        def all_messages(self) -> list[ModelMessage]:
            return self._messages

    class _Agent:
        def __init__(self, model: str) -> None:
            if not model.startswith("openai:"):
                raise AssertionError(f"expected an openai model, got {model}")
            self._model_name = model.removeprefix("openai:")

        def run_sync(
            self,
            prompt: str,
            *,
            message_history: list[ModelMessage] | None = None,
            **_kwargs: object,
        ) -> _Result:
            history = list(message_history or [])
            calls.append(ModelCall(self._model_name, prompt, active_user[0], history))
            if error is not None and (
                error_on_prompt is None or prompt == error_on_prompt
            ):
                raise error
            reply = _next_reply(pending_replies, prompt)
            returned = _next_history(history, pending_histories)
            return _Result(reply, returned)

    original = agent.run_model

    def wrapped(
        model_name: str,
        prompt: str,
        user_id: str,
        message_history: list[ModelMessage],
    ) -> tuple[str, list[ModelMessage]]:
        active_user[0] = user_id
        return original(model_name, prompt, user_id, message_history)

    monkeypatch.setattr(agent, "Agent", _Agent)
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
    calls = _replace_model(monkeypatch, replies=["model reply"])
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
        replies=["reply-1", "reply-2"],
        histories=[exchange, [*exchange, *_exchange("second", "reply-2")]],
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
    calls = _replace_model(monkeypatch, replies=["model reply"])
    _scripted_input(monkeypatch, ["  hello  ", "", "   ", "/exit"])

    cli.main()

    assert _sent(calls) == [("test-model", "hello", "local-user")]
    assert calls[0].message_history == []
    assert capsys.readouterr().out == "model reply\n"


def test_exit_on_first_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, replies=["model reply"])
    _scripted_input(monkeypatch, ["/exit"])

    cli.main()

    assert calls == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_exit_with_surrounding_whitespace(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, replies=["model reply"])
    _scripted_input(monkeypatch, ["  /exit  "])

    cli.main()

    assert calls == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_exit_word_variants_are_messages(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, replies=["a", "b", "c"])
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
    calls = _replace_model(monkeypatch, replies=["model reply"])

    def fail_input(_prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", fail_input)

    cli.main()

    assert calls == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_eof_after_a_reply(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, replies=["model reply"])
    _scripted_input(monkeypatch, ["hello"])

    cli.main()

    assert _sent(calls) == [("test-model", "hello", "local-user")]
    assert calls[0].message_history == []
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
        replies=["It is sunny.", "next reply"],
        histories=[completed, completed],
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
        replies=["one", "two", "later"],
        histories=[exchange, [*exchange, *_exchange("again", "two")], []],
    )
    _scripted_input(monkeypatch, ["hello", "again", "/exit"])
    cli.run_chat("test-model", "local-user")

    _scripted_input(monkeypatch, ["later", "/exit"])
    cli.run_chat("test-model", "local-user")

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
        replies=["model reply", "next reply"],
        histories=[exchange, [*exchange, *_exchange("next", "next reply")]],
    )
    _scripted_input(monkeypatch, ["hello", "", "   ", "next", "/exit"])

    cli.main()

    assert _sent(calls) == [
        ("test-model", "hello", "local-user"),
        ("test-model", "next", "local-user"),
    ]
    assert calls[1].message_history == exchange
    captured = capsys.readouterr()
    assert captured.out == "model reply\nnext reply\n"
    assert captured.err == ""


def test_missing_only_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    calls = _replace_model(monkeypatch, replies=["model reply"])
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
    calls = _replace_model(monkeypatch, replies=["model reply"])
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
    calls = _replace_model(monkeypatch, replies=["model reply"])
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
    calls = _replace_model(monkeypatch, replies=["model reply"])

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
    calls = _replace_model(monkeypatch, error=AgentRunError("connection failed"))
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
        replies=["model reply"],
        histories=[exchange],
        error=AgentRunError("connection failed"),
        error_on_prompt="again",
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

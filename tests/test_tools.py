import tempfile
from datetime import datetime, timedelta, timezone
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
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from robot_agent import agent, cli, identity, storage
from robot_agent.tools import format_local_timestamp

FIXED_MOMENT = datetime(2026, 11, 18, 9, 23, 15, tzinfo=timezone(timedelta(hours=2)))
FIXED_TIMESTAMP = "2026-11-18T09:23:15+02:00"
TIME_PROMPT = "what time is it?"


def _limits(**overrides: int) -> agent.ModelLimits:
    values = {
        "max_history_turns": 6,
        "max_request_input_tokens": 4096,
        "max_output_tokens": 256,
    }
    values.update(overrides)
    return agent.ModelLimits(
        max_history_turns=values["max_history_turns"],
        max_request_input_tokens=values["max_request_input_tokens"],
        max_output_tokens=values["max_output_tokens"],
    )


def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "AGENT_MAX_HISTORY_TURNS",
        "AGENT_MAX_REQUEST_INPUT_TOKENS",
        "AGENT_MAX_OUTPUT_TOKENS",
        "AGENT_MAX_RECALLED_MEMORIES",
        "AGENT_DB_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    database = Path(tempfile.mkdtemp(prefix="robot-agent-")) / "chat.sqlite3"
    monkeypatch.setenv("AGENT_DB_PATH", str(database))
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    return database


class _LocalTestModel(TestModel):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.seen: list[list[ModelMessage]] = []

    async def count_tokens(self, messages, model_settings, model_request_parameters):
        return RequestUsage(input_tokens=1)

    async def request(self, messages, model_settings, model_request_parameters):
        self.seen.append(list(messages))
        return await super().request(messages, model_settings, model_request_parameters)


class _LocalFunctionModel(FunctionModel):
    token_counts = 0

    async def count_tokens(self, messages, model_settings, model_request_parameters):
        self.token_counts += 1
        return RequestUsage(input_tokens=1)


def _use_model(
    monkeypatch: pytest.MonkeyPatch,
    model: TestModel | FunctionModel,
) -> None:
    real_agent = agent.Agent

    class _Agent(real_agent):
        def run_sync(self, *args: object, **kwargs: object):
            with self.override(model=model):
                return super().run_sync(*args, **kwargs)

    monkeypatch.setattr(agent, "Agent", _Agent)


def _fix_clock(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]

    def clock() -> datetime:
        calls[0] += 1
        return FIXED_MOMENT

    monkeypatch.setattr("robot_agent.tools.read_local_clock", clock)
    return calls


def _forbid_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    def clock() -> datetime:
        raise AssertionError("the clock tool should not run")

    monkeypatch.setattr("robot_agent.tools.read_local_clock", clock)


def _scripted_input(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    remaining = iter(lines)

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(remaining)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", fake_input)


def _user_turns(seen: list[list[ModelMessage]]) -> list[list[ModelMessage]]:
    turns: list[list[ModelMessage]] = []
    for messages in seen:
        last = messages[-1]
        if not isinstance(last, ModelRequest):
            continue
        if any(isinstance(part, UserPromptPart) for part in last.parts):
            turns.append(messages)
    return turns


def _prompt_text(part: UserPromptPart) -> str:
    content = part.content
    if isinstance(content, str):
        return content
    raise AssertionError("expected a text prompt")


def test_format_local_timestamp_uses_the_given_moment() -> None:
    assert format_local_timestamp(FIXED_MOMENT) == FIXED_TIMESTAMP


def test_time_question_runs_the_tool_and_keeps_the_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_env(monkeypatch)
    calls = _fix_clock(monkeypatch)
    model = _LocalTestModel(
        call_tools=["current_local_time"],
        custom_output_text="It is 09:23.",
    )
    _use_model(monkeypatch, model)

    reply, history, _produced, _usage = agent.request_reply(
        "test-model",
        "local-user",
        TIME_PROMPT,
        [],
        _limits(),
    )

    assert calls[0] == 1
    assert reply == "It is 09:23."
    tools = model.last_model_request_parameters
    assert tools is not None
    assert [tool.name for tool in tools.function_tools] == ["current_local_time"]
    description = tools.function_tools[0].description or ""
    assert "current date or time" in description
    assert "Do not estimate the current time" in description
    assert [type(message) for message in history] == [
        ModelRequest,
        ModelResponse,
        ModelRequest,
        ModelResponse,
    ]
    user, tool_call, tool_return, final = history
    assert _prompt_text(user.parts[0]) == TIME_PROMPT
    call = tool_call.parts[0]
    returned = tool_return.parts[0]
    assert isinstance(call, ToolCallPart)
    assert isinstance(returned, ToolReturnPart)
    assert call.tool_name == "current_local_time"
    assert returned.tool_name == "current_local_time"
    assert returned.content == FIXED_TIMESTAMP
    assert call.tool_call_id == returned.tool_call_id
    assert isinstance(final.parts[0], TextPart)
    assert final.parts[0].content == "It is 09:23."


def test_unrelated_prompt_does_not_run_the_tool(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _forbid_clock(monkeypatch)
    model = _LocalTestModel(call_tools=[], custom_output_text="hello there")
    _use_model(monkeypatch, model)
    _scripted_input(monkeypatch, ["say hello", "/exit"])

    cli.main()

    assert model.seen
    assert all(
        not isinstance(part, (ToolCallPart, ToolReturnPart))
        for messages in model.seen
        for message in messages
        for part in message.parts
    )
    captured = capsys.readouterr()
    assert captured.out == "hello there\n"
    assert captured.err == ""


def test_later_turn_receives_the_tool_exchange_within_the_history_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_env(monkeypatch)
    calls = _fix_clock(monkeypatch)
    model = _LocalTestModel(
        call_tools=["current_local_time"],
        custom_output_text="done",
    )
    _use_model(monkeypatch, model)
    limits = _limits(max_history_turns=1)
    history: list[ModelMessage] = []
    for prompt in (TIME_PROMPT, "next", "after"):
        _reply, history, _produced, _usage = agent.request_reply(
            "test-model",
            "local-user",
            prompt,
            history,
            limits,
        )

    assert calls[0] == 1
    turns = _user_turns(model.seen)
    assert len(turns) == 3
    follow_up = turns[1]
    assert _prompt_text(follow_up[-1].parts[0]) == "next"
    prior = follow_up[:-1]
    tool_returns = [
        part
        for message in prior
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(tool_returns) == 1
    assert tool_returns[0].content == FIXED_TIMESTAMP
    assert [
        type(part)
        for message in prior
        for part in message.parts
        if isinstance(part, (UserPromptPart, ToolCallPart, ToolReturnPart, TextPart))
    ] == [UserPromptPart, ToolCallPart, ToolReturnPart, TextPart]
    newest = turns[2][:-1]
    prompts = [
        _prompt_text(part)
        for message in newest
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert prompts == ["next"]
    assert not any(
        isinstance(part, (ToolCallPart, ToolReturnPart))
        for message in newest
        for part in message.parts
    )


@pytest.mark.parametrize("lines", [["   "], ["/exit"], []])
def test_blank_exit_and_eof_do_not_call_the_model_or_tool(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    lines: list[str],
) -> None:
    _isolate_env(monkeypatch)
    _forbid_clock(monkeypatch)
    model = _LocalTestModel(call_tools="all", custom_output_text="nope")
    _use_model(monkeypatch, model)
    _scripted_input(monkeypatch, lines)

    cli.main()

    assert model.seen == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_clock_failure_after_a_reply_keeps_the_chat_open_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    seen: list[list[ModelMessage]] = []
    clock_calls = [0]

    def broken_now(*_args: object, **_kwargs: object) -> datetime:
        clock_calls[0] += 1
        raise OSError("clock down")

    def respond(messages: list[ModelMessage], _info: object) -> ModelResponse:
        seen.append(list(messages))
        prompt = _latest_prompt(messages)
        if prompt == TIME_PROMPT and not _has_tool_return(messages):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="current_local_time",
                        args={},
                        tool_call_id="time-1",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="earlier reply")])

    monkeypatch.setattr("robot_agent.tools._system_now", broken_now)
    _use_model(monkeypatch, _LocalFunctionModel(respond))
    _scripted_input(monkeypatch, ["hello", TIME_PROMPT])

    cli.main()
    assert clock_calls[0] == 1
    assert len(seen) == 2
    assert not _has_tool_return(seen[1])
    captured = capsys.readouterr()
    assert captured.out == "earlier reply\n"
    assert captured.err == "Agent request failed: The local clock could not be read.\n"


def test_restart_restores_the_time_tool_turn_without_running_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _isolate_env(monkeypatch)
    profile = identity.resolve_user()
    clock_calls = _fix_clock(monkeypatch)
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], _info: object) -> ModelResponse:
        seen.append(list(messages))
        prompt = _latest_prompt(messages)
        if prompt == TIME_PROMPT and not _has_tool_return(messages):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="current_local_time",
                        args={},
                        tool_call_id="time-1",
                    )
                ]
            )
        text = "It is 09:23." if prompt == TIME_PROMPT else "later"
        return ModelResponse(parts=[TextPart(content=text)])

    model = _LocalFunctionModel(respond)
    _use_model(monkeypatch, model)
    _scripted_input(monkeypatch, [TIME_PROMPT])
    cli.main()

    assert clock_calls[0] == 1
    assert len(seen) == 2
    assert model.token_counts == 2
    assert capsys.readouterr().out == "It is 09:23.\n"
    _scripted_input(monkeypatch, ["and now?"])
    cli.main()

    assert clock_calls[0] == 1
    assert len(seen) == 3
    assert model.token_counts == 3
    captured = capsys.readouterr()
    assert captured.out == "later\n"
    assert captured.err == ""
    follow_up = _user_turns(seen)[-1]
    assert _prompt_text(follow_up[-1].parts[0]) == "and now?"
    prior = follow_up[:-1]
    assert _part_types(prior) == [
        UserPromptPart,
        ToolCallPart,
        ToolReturnPart,
        TextPart,
    ]
    prompts = [
        _prompt_text(part)
        for message in prior
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert prompts == [TIME_PROMPT]
    tool_calls = [part for part in _parts(prior) if isinstance(part, ToolCallPart)]
    tool_returns = [part for part in _parts(prior) if isinstance(part, ToolReturnPart)]
    assert len(tool_calls) == 1
    assert len(tool_returns) == 1
    assert tool_calls[0].tool_call_id == "time-1"
    assert tool_returns[0].tool_call_id == "time-1"
    assert tool_returns[0].content == FIXED_TIMESTAMP
    saved = storage.connect(database)
    session_id = saved.ensure_session(profile.user_id, profile.display_name)
    restored = saved.load_recent(profile.user_id, session_id, 6)
    restored_returns = [
        part for part in _parts(restored) if isinstance(part, ToolReturnPart)
    ]
    assert len(restored_returns) == 1


def _parts(messages: list[ModelMessage]) -> list[object]:
    return [part for message in messages for part in message.parts]


def _part_types(messages: list[ModelMessage]) -> list[type[object]]:
    return [type(part) for part in _parts(messages)]


def _latest_prompt(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                return _prompt_text(part)
    return ""


def _has_tool_return(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(part, ToolReturnPart)
        for message in messages
        for part in message.parts
    )

"""One chat turn is capped at a configurable number of model requests."""

import pytest
from pydantic_ai import UsageLimitExceeded, UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from robot_agent import agent, cli, identity
from tests.test_cli import (
    _configure,
    _isolate_env,
    _message_rows,
    _ModelScript,
    _replace_model,
    _saved_messages,
    _scripted_input,
    _sent,
    _usage,
)
from tests.test_sessions import _ready, _seed_turns
from tests.test_summary import _install, _setup, _turns
from tests.test_tools import (
    FIXED_TIMESTAMP,
    TIME_PROMPT,
    _fix_clock,
    _LocalTestModel,
    _use_model,
)
from tests.test_tools import _isolate_env as _isolate_tools_env
from tests.test_usage import _clock, _limit, _rows

SETTING = "AGENT_MAX_MODEL_REQUESTS_PER_TURN"
_REQUEST_LIMIT_ERROR = "The next request would exceed the request_limit of 2"


def _time_turn() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=TIME_PROMPT)]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="current_local_time",
                    args={},
                    tool_call_id="call-1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="current_local_time",
                    content=FIXED_TIMESTAMP,
                    tool_call_id="call-1",
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content="It is 09:23.")]),
    ]


def _failed_run(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    capsys: pytest.CaptureFixture[str],
) -> tuple[str, list[str]]:
    """Run one failing line. Return stderr and the prompts the model received."""
    calls = _replace_model(monkeypatch, _ModelScript(error=error))
    _scripted_input(monkeypatch, ["hello", "again"], after="error")

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    return captured.err, [call.prompt for call in calls]


def test_default_request_limit_and_existing_limits_are_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ready(monkeypatch)
    monkeypatch.setenv("AGENT_MAX_REQUEST_INPUT_TOKENS", "1234")
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _scripted_input(monkeypatch, ["hello", "/exit"])

    cli.main()

    usage = calls[0].usage_limits
    assert usage is not None
    assert usage.request_limit == 4
    assert usage.per_request_input_tokens_limit == 1234
    assert usage.count_tokens_before_request is True


def test_configured_request_limit_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    _ready(monkeypatch)
    monkeypatch.setenv(SETTING, "2")
    calls = _replace_model(monkeypatch, _ModelScript(replies=["model reply"]))
    _scripted_input(monkeypatch, ["hello", "/exit"])

    cli.main()

    usage = calls[0].usage_limits
    assert usage is not None
    assert usage.request_limit == 2


def test_tool_turn_within_the_limit_is_saved_with_its_tool_exchange(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    monkeypatch.setenv(SETTING, "2")
    turn = _time_turn()
    _replace_model(
        monkeypatch,
        _ModelScript(
            replies=["It is 09:23."],
            histories=[turn],
            new_turns=[turn],
            usages=[_usage(2, 20, 10)],
        ),
    )
    _scripted_input(monkeypatch, [TIME_PROMPT, "/exit"])

    cli.main()

    captured = capsys.readouterr()
    assert captured.out == "It is 09:23.\n"
    assert captured.err == ""
    profile = identity.resolve_user()
    saved = _saved_messages(database, profile.user_id, profile.display_name)
    assert saved == turn


def test_the_request_limit_stops_the_turn_without_retry_or_saving(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = _ready(monkeypatch)
    monkeypatch.setenv(SETTING, "2")
    _clock(monkeypatch)
    _limit(monkeypatch, requests="10")

    err, prompts = _failed_run(
        monkeypatch, UsageLimitExceeded(_REQUEST_LIMIT_ERROR), capsys
    )

    assert err == (
        "Agent request failed: The turn exceeded the limit of 2 model requests. "
        "Raise AGENT_MAX_MODEL_REQUESTS_PER_TURN to allow more.\n"
    )
    assert prompts == ["hello"]
    assert _message_rows(database) == []
    assert _rows(database) == []


def test_the_input_token_limit_keeps_its_own_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ready(monkeypatch)
    monkeypatch.setenv(SETTING, "2")
    monkeypatch.setenv("AGENT_MAX_REQUEST_INPUT_TOKENS", "100")
    calls = _replace_model(
        monkeypatch, _ModelScript(replies=["unused"], token_counts=[101])
    )
    _scripted_input(monkeypatch, ["hello"], after="error")

    with pytest.raises(SystemExit):
        cli.main()

    assert capsys.readouterr().err == (
        "Agent request failed: The request exceeds the input token limit of 100.\n"
    )
    assert calls == []


def test_another_usage_limit_keeps_the_generic_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _ready(monkeypatch)
    monkeypatch.setenv(SETTING, "2")
    error = UsageLimitExceeded(
        "The next tool call(s) would exceed the tool_calls_limit"
    )

    err, _prompts = _failed_run(monkeypatch, error, capsys)

    assert err == f"Agent request failed: {error}\n"
    assert SETTING not in err


@pytest.mark.parametrize("value", ["0", "-1", "abc", "", "1.5"])
def test_invalid_setting_exits_before_input_or_a_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    monkeypatch.setenv(SETTING, value)
    calls = _replace_model(monkeypatch, _ModelScript(replies=["unused"]))
    _scripted_input(monkeypatch, ["hello"], after="error")

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == (
        f"Invalid configuration: {SETTING} must be a positive integer.\n"
    )
    assert calls == []
    assert calls.token_counts == []


def test_the_summarizer_keeps_its_own_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    database = _setup(monkeypatch)
    monkeypatch.setenv(SETTING, "2")
    _seed_turns(database, _turns(4))
    backend = _install(monkeypatch, ["A short summary."])
    _scripted_input(monkeypatch, ["q5", "/exit"])

    cli.main()

    assert len(backend.summarizer) == 1
    summary_limits = backend.summarizer[0].usage_limits
    assert summary_limits is not None
    assert summary_limits.request_limit == UsageLimits().request_limit
    chat_limits = backend.main[0].usage_limits
    assert chat_limits is not None
    assert chat_limits.request_limit == 2


@pytest.mark.parametrize("ending", [["/exit"], []])
def test_local_input_makes_no_model_request(
    monkeypatch: pytest.MonkeyPatch,
    ending: list[str],
) -> None:
    _ready(monkeypatch)
    monkeypatch.setenv(SETTING, "1")
    calls = _replace_model(monkeypatch, _ModelScript())
    lines = ["", "   ", "/new", "/sessions", "/memory list", "/time", *ending]
    _scripted_input(monkeypatch, lines)

    cli.main()

    assert _sent(calls) == []
    assert calls.token_counts == []


def test_the_real_limit_message_is_recognized_and_a_tool_turn_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs the real PydanticAI limit with a local test model, no API call."""
    _isolate_tools_env(monkeypatch)
    _fix_clock(monkeypatch)
    model = _LocalTestModel(
        call_tools=["current_local_time"],
        custom_output_text="It is 09:23.",
    )
    _use_model(monkeypatch, model)

    def run(requests: int) -> agent.ModelReply:
        limits = agent.ModelLimits(
            max_history_turns=6,
            max_request_input_tokens=4096,
            max_output_tokens=256,
            max_model_requests=requests,
        )
        return agent.request_reply("test-model", "local-user", TIME_PROMPT, [], limits)

    assert run(2).text == "It is 09:23."
    with pytest.raises(agent.AgentRequestError) as exc:
        run(1)
    assert str(exc.value) == (
        "The turn exceeded the limit of 1 model requests. "
        "Raise AGENT_MAX_MODEL_REQUESTS_PER_TURN to allow more."
    )

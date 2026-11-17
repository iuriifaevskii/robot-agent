import runpy

import pytest
from pydantic_ai import AgentRunError

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


def _replace_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    replies: list[str] | None = None,
    error: Exception | None = None,
    error_on_prompt: str | None = None,
) -> list[tuple[str, str, str]]:
    calls: list[tuple[str, str, str]] = []
    pending = list(replies or [])

    def fake_run_model(model_name: str, prompt: str, user_id: str) -> str:
        calls.append((model_name, prompt, user_id))
        if error is not None and (error_on_prompt is None or prompt == error_on_prompt):
            raise error
        if not pending:
            raise AssertionError(f"unexpected model call for {prompt!r}")
        return pending.pop(0)

    monkeypatch.setattr(agent, "run_model", fake_run_model)
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

    assert calls == [("test-model", "hello", "local-user")]
    captured = capsys.readouterr()
    assert captured.out == "model reply\n"
    assert captured.err == ""


def test_two_lines_are_independent(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(monkeypatch, replies=["reply-1", "reply-2"])
    _scripted_input(monkeypatch, ["first", "second", "/exit"])

    cli.main()

    assert calls == [
        ("test-model", "first", "local-user"),
        ("test-model", "second", "local-user"),
    ]
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

    assert calls == [("test-model", "hello", "local-user")]
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

    assert calls == [
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

    assert calls == [("test-model", "hello", "local-user")]
    captured = capsys.readouterr()
    assert captured.out == "model reply\n"
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
    assert calls == [("test-model", "hello", "local-user")]


def test_model_failure_on_second_turn(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    _configure(monkeypatch)
    calls = _replace_model(
        monkeypatch,
        replies=["model reply"],
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
    assert calls == [
        ("test-model", "hello", "local-user"),
        ("test-model", "again", "local-user"),
    ]

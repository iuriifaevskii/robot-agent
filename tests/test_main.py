import pytest
from pydantic_ai import AgentRunError

import agent
import main

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
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)


def _replace_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reply: str = "",
    error: Exception | None = None,
) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_run_model(model_name: str, prompt: str) -> str:
        calls.append((model_name, prompt))
        if error is not None:
            raise error
        return reply

    monkeypatch.setattr(agent, "run_model", fake_run_model)
    return calls


def test_prints_model_reply(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    calls = _replace_model(monkeypatch, reply="model reply")

    main.main()

    assert calls == [("test-model", agent.PROMPT)]
    assert capsys.readouterr().out == "model reply\n"


def test_missing_only_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    calls = _replace_model(monkeypatch, reply="model reply")

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == MISSING_KEY
    assert calls == []


def test_missing_only_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = _replace_model(monkeypatch, reply="model reply")

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == MISSING_MODEL
    assert calls == []


def test_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    calls = _replace_model(monkeypatch, reply="model reply")

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == MISSING_BOTH
    assert calls == []


def test_model_call_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    calls = _replace_model(monkeypatch, error=AgentRunError("connection failed"))

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Agent request failed: connection failed\n"
    assert calls == [("test-model", agent.PROMPT)]

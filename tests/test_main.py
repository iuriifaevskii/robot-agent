import pytest

import main

MISSING_BOTH = (
    "Missing configuration: OPENAI_API_KEY, OPENAI_MODEL. "
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


def test_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == MISSING_BOTH


def test_missing_only_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 1
    assert capsys.readouterr().err == MISSING_MODEL


def test_configuration_loaded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    main.main()

    assert capsys.readouterr().out == "Configuration loaded.\n"

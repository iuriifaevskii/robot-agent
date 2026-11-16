import os
import sys

from dotenv import load_dotenv

REQUIRED_ENV = ("OPENAI_API_KEY", "OPENAI_MODEL")


def main() -> None:
    load_dotenv()
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        print(
            f"Missing configuration: {joined}. Copy .env.example to .env and set the values.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("Configuration loaded.")


if __name__ == "__main__":
    main()

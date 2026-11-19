"""SQLite storage for one user's session and conversation turns.

SQL stays in this module. It does not read the environment, print, or call
the model.
"""

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

OPEN_FAILED = "Could not open the conversation database"
READ_FAILED = "Could not read the saved conversation."
SAVE_FAILED = "Could not save the new conversation turn."
NOT_OWNER = "The session does not belong to this user."

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users (user_id),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS sessions_by_user ON sessions (user_id);

CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL REFERENCES sessions (session_id),
    position INTEGER NOT NULL,
    body TEXT NOT NULL,
    PRIMARY KEY (session_id, position)
);
"""


class StorageError(Exception):
    """A conversation database operation failed. The message is safe to print."""


class ConversationStore:
    """One SQLite file holding users, sessions, and complete turns."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def ensure_session(self, user_id: str, display_name: str) -> str:
        """Return this user's existing session, creating the user and session once."""
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO users (user_id, display_name)
                    VALUES (?, ?)
                    """,
                    (user_id, display_name),
                )
                row = self._connection.execute(
                    """
                    SELECT session_id FROM sessions
                    WHERE user_id = ?
                    ORDER BY created_at, session_id
                    LIMIT 1
                    """,
                    (user_id,),
                ).fetchone()
                if row is not None:
                    return str(row["session_id"])
                session_id = str(uuid4())
                self._connection.execute(
                    """
                    INSERT INTO sessions (session_id, user_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, user_id, _timestamp()),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"{OPEN_FAILED}.") from exc
        return session_id

    def load_recent(
        self,
        user_id: str,
        session_id: str,
        limit: int,
    ) -> list[ModelMessage]:
        """Return at most `limit` newest complete turns, oldest first.

        A limit below 1 returns no messages and does not decode stored turns.
        Messages are not executed.
        """
        try:
            self._require_owner(user_id, session_id)
            if limit < 1:
                return []
            rows = self._connection.execute(
                """
                SELECT body FROM messages
                WHERE session_id = ?
                ORDER BY position DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(READ_FAILED) from exc
        loaded: list[ModelMessage] = []
        for row in reversed(rows):
            loaded.extend(_deserialize(str(row["body"])))
        return loaded

    def append_turn(
        self,
        user_id: str,
        session_id: str,
        messages: Sequence[ModelMessage],
    ) -> None:
        """Commit one complete turn for a session owned by user_id."""
        body = _serialize(messages)
        try:
            with self._connection:
                self._require_owner(user_id, session_id)
                position = self._next_position(session_id)
                self._connection.execute(
                    """
                    INSERT INTO messages (session_id, position, body)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, position, body),
                )
        except sqlite3.Error as exc:
            raise StorageError(SAVE_FAILED) from exc

    def _require_owner(self, user_id: str, session_id: str) -> None:
        if self._owns(user_id, session_id):
            return
        raise StorageError(NOT_OWNER)

    def _owns(self, user_id: str, session_id: str) -> bool:
        row = self._connection.execute(
            """
            SELECT 1 FROM sessions
            WHERE session_id = ? AND user_id = ?
            """,
            (session_id, user_id),
        ).fetchone()
        return row is not None

    def _next_position(self, session_id: str) -> int:
        row = self._connection.execute(
            """
            SELECT COALESCE(MAX(position), 0) + 1
            FROM messages
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return int(row[0])


def connect(path: Path) -> ConversationStore:
    """Open the database and create tables when they are missing."""
    try:
        connection = sqlite3.connect(path)
    except sqlite3.Error as exc:
        raise StorageError(f"{OPEN_FAILED}: {path}.") from exc
    try:
        _prepare(connection)
    except sqlite3.Error as exc:
        connection.close()
        raise StorageError(f"{OPEN_FAILED}: {path}.") from exc
    return ConversationStore(connection)


def _prepare(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_SCHEMA)


def _serialize(messages: Sequence[ModelMessage]) -> str:
    payload = ModelMessagesTypeAdapter.dump_json(list(messages))
    return payload.decode()


def _deserialize(payload: str) -> list[ModelMessage]:
    try:
        return list(ModelMessagesTypeAdapter.validate_json(payload))
    except ValidationError as exc:
        raise StorageError(READ_FAILED) from exc


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()

"""SQLite storage for users, sessions, turns, summaries, memories, and usage.

SQL stays in this module. It does not read the environment, print, or call
the model.
"""

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

OPEN_FAILED = "Could not open the conversation database"
READ_FAILED = "Could not read the saved conversation."
SAVE_FAILED = "Could not save the new conversation turn."
CREATE_FAILED = "Could not create a new session."
SELECTION_FAILED = "Could not save the active session."
NOT_OWNER = "The session does not belong to this user."
SAVE_MEMORY_FAILED = "Could not save the memory."
UPDATE_MEMORY_FAILED = "Could not update the memory."
DELETE_MEMORY_FAILED = "Could not delete the memory."
MEMORY_READ_FAILED = "Could not read saved memories."
MEMORY_NOT_OWNER = "The memory does not belong to this user."
SUMMARY_READ_FAILED = "Could not read the session summary."
SUMMARY_SAVE_FAILED = "Could not save the session summary."
USAGE_READ_FAILED = "Could not read the daily usage."
USAGE_SAVE_FAILED = "Could not save the daily usage."
SESSION_USAGE_READ_FAILED = "Could not read the session usage."
SESSION_USAGE_SAVE_FAILED = "Could not save the session usage."
USER_CREATE_FAILED = "Could not create the profile."
USER_READ_FAILED = "Could not read the profiles."
CLI_MEMORY_SOURCE = "cli"

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

CREATE TABLE IF NOT EXISTS summaries (
    session_id TEXT PRIMARY KEY REFERENCES sessions (session_id),
    text TEXT NOT NULL,
    covered_through INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS active_sessions (
    user_id TEXT PRIMARY KEY REFERENCES users (user_id),
    session_id TEXT NOT NULL REFERENCES sessions (session_id)
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users (user_id),
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS memories_by_user ON memories (user_id);

CREATE TABLE IF NOT EXISTS daily_usage (
    day TEXT PRIMARY KEY,
    requests INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS session_usage (
    session_id TEXT PRIMARY KEY REFERENCES sessions (session_id),
    requests INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL
);
"""


class _MemoryInsert(NamedTuple):
    memory_id: str
    user_id: str
    text: str
    created_at: str
    source: str


class StorageError(Exception):
    """A conversation database operation failed. The message is safe to print."""


class ConversationStore:
    """One SQLite file holding users, sessions, turns, and saved facts."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def ensure_session(self, user_id: str, display_name: str) -> str:
        """Return this user's oldest session, creating the user and one session.

        Oldest is the first creation time, then session id. This does not
        change a saved active selection.
        """
        try:
            return self._oldest_session(user_id, display_name)
        except sqlite3.Error as exc:
            raise StorageError(f"{OPEN_FAILED}.") from exc

    def create_user(self, display_name: str) -> str:
        """Insert one user with a new generated id and return that id.

        No session is created and no selection is saved.
        """
        user_id = str(uuid4())
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO users (user_id, display_name) VALUES (?, ?)",
                    (user_id, display_name),
                )
        except sqlite3.Error as exc:
            raise StorageError(USER_CREATE_FAILED) from exc
        return user_id

    def list_users(self) -> list[tuple[str, str]]:
        """Return (user_id, display_name) for every user, oldest first."""
        try:
            rows = self._connection.execute(
                "SELECT user_id, display_name FROM users ORDER BY rowid"
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(USER_READ_FAILED) from exc
        return [(str(row["user_id"]), str(row["display_name"])) for row in rows]

    def user_display_name(self, user_id: str) -> str | None:
        """Return the stored display name, or None when the user is unknown."""
        try:
            row = self._connection.execute(
                "SELECT display_name FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(USER_READ_FAILED) from exc
        if row is None:
            return None
        return str(row["display_name"])

    def active_session_id(self, user_id: str) -> str | None:
        """Return the saved active session, or None when no selection exists."""
        try:
            row = self._connection.execute(
                """
                SELECT session_id FROM active_sessions
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(READ_FAILED) from exc
        if row is None:
            return None
        return str(row["session_id"])

    def create_session(self, user_id: str, display_name: str) -> str:
        """Insert an empty session and select it in one transaction."""
        session_id = str(uuid4())
        try:
            with self._connection:
                self._ensure_user(user_id, display_name)
                self._insert_session(session_id, user_id, _timestamp())
                self._store_selection(user_id, session_id)
        except sqlite3.Error as exc:
            raise StorageError(CREATE_FAILED) from exc
        return session_id

    def list_sessions(self, user_id: str) -> list[tuple[str, str]]:
        """Return (session_id, created_at), newest creation time first.

        Equal creation times are ordered by session id ascending.
        Message bodies are not read.
        """
        try:
            rows = self._connection.execute(
                """
                SELECT session_id, created_at
                FROM sessions
                WHERE user_id = ?
                ORDER BY created_at DESC, session_id ASC
                """,
                (user_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(READ_FAILED) from exc
        return [(str(row["session_id"]), str(row["created_at"])) for row in rows]

    def select_session(self, user_id: str, session_id: str) -> None:
        """Save session_id as this user's active session.

        The session must already belong to user_id.
        """
        try:
            with self._connection:
                self._require_owner(user_id, session_id)
                self._store_selection(user_id, session_id)
        except sqlite3.Error as exc:
            raise StorageError(SELECTION_FAILED) from exc

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

    def turn_count(self, user_id: str, session_id: str) -> int:
        """Return how many turns are saved in a session owned by user_id."""
        try:
            self._require_owner(user_id, session_id)
            row = self._connection.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(READ_FAILED) from exc
        return int(row[0])

    def load_turns(
        self,
        user_id: str,
        session_id: str,
        first: int,
        last: int,
    ) -> list[ModelMessage]:
        """Return turns at positions first through last, oldest first.

        Positions start at 1. The session must belong to user_id.
        """
        try:
            self._require_owner(user_id, session_id)
            rows = self._connection.execute(
                """
                SELECT body FROM messages
                WHERE session_id = ? AND position BETWEEN ? AND ?
                ORDER BY position
                """,
                (session_id, first, last),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(READ_FAILED) from exc
        loaded: list[ModelMessage] = []
        for row in rows:
            loaded.extend(_deserialize(str(row["body"])))
        return loaded

    def load_summary(self, user_id: str, session_id: str) -> tuple[str, int] | None:
        """Return (text, covered_through) for the session, or None if absent.

        The session must belong to user_id.
        """
        try:
            self._require_owner(user_id, session_id)
            row = self._connection.execute(
                """
                SELECT text, covered_through FROM summaries
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(SUMMARY_READ_FAILED) from exc
        if row is None:
            return None
        return str(row["text"]), int(row["covered_through"])

    def save_summary(
        self,
        user_id: str,
        session_id: str,
        text: str,
        covered_through: int,
    ) -> None:
        """Replace the session's summary in one transaction.

        The session must belong to user_id. Saved turns are not changed.
        """
        try:
            with self._connection:
                self._require_owner(user_id, session_id)
                self._connection.execute(
                    """
                    INSERT INTO summaries (
                        session_id, text, covered_through, updated_at
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (session_id) DO UPDATE SET
                        text = excluded.text,
                        covered_through = excluded.covered_through,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, text, covered_through, _timestamp()),
                )
        except sqlite3.Error as exc:
            raise StorageError(SUMMARY_SAVE_FAILED) from exc

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

    def daily_usage(self, day: str) -> tuple[int, int, int]:
        """Return (requests, input_tokens, output_tokens) for a local date.

        A day with no row reports zeros. Usage is device-wide, not per user.
        """
        try:
            row = self._connection.execute(
                """
                SELECT requests, input_tokens, output_tokens
                FROM daily_usage
                WHERE day = ?
                """,
                (day,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(USAGE_READ_FAILED) from exc
        if row is None:
            return 0, 0, 0
        return int(row[0]), int(row[1]), int(row[2])

    def add_daily_usage(
        self,
        day: str,
        requests: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Add to the counters for a local date in one transaction.

        The increment happens in SQL, so no read-modify-write is needed.
        """
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO daily_usage (
                        day, requests, input_tokens, output_tokens
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (day) DO UPDATE SET
                        requests = requests + excluded.requests,
                        input_tokens = input_tokens + excluded.input_tokens,
                        output_tokens = output_tokens + excluded.output_tokens
                    """,
                    (day, requests, input_tokens, output_tokens),
                )
        except sqlite3.Error as exc:
            raise StorageError(USAGE_SAVE_FAILED) from exc

    def session_usage(self, user_id: str, session_id: str) -> tuple[int, int, int]:
        """Return (requests, input_tokens, output_tokens) for one session.

        The session must belong to user_id. A session with no row reports zeros.
        """
        try:
            self._require_owner(user_id, session_id)
            row = self._connection.execute(
                """
                SELECT requests, input_tokens, output_tokens
                FROM session_usage
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(SESSION_USAGE_READ_FAILED) from exc
        if row is None:
            return 0, 0, 0
        return int(row[0]), int(row[1]), int(row[2])

    def add_session_usage(
        self,
        user_id: str,
        session_id: str,
        requests: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Add to a session's counters in one transaction.

        The session must belong to user_id. The increment happens in SQL, so
        no read-modify-write is needed.
        """
        try:
            with self._connection:
                self._require_owner(user_id, session_id)
                self._connection.execute(
                    """
                    INSERT INTO session_usage (
                        session_id, requests, input_tokens, output_tokens
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (session_id) DO UPDATE SET
                        requests = requests + excluded.requests,
                        input_tokens = input_tokens + excluded.input_tokens,
                        output_tokens = output_tokens + excluded.output_tokens
                    """,
                    (session_id, requests, input_tokens, output_tokens),
                )
        except sqlite3.Error as exc:
            raise StorageError(SESSION_USAGE_SAVE_FAILED) from exc

    def save_memory(self, user_id: str, display_name: str, text: str) -> str:
        """Insert one fact for user_id and return its new id.

        Each call creates a separate record. created_at and updated_at start
        equal. The source marks an explicit CLI save.
        """
        memory_id = str(uuid4())
        created_at = _timestamp()
        row = _MemoryInsert(memory_id, user_id, text, created_at, CLI_MEMORY_SOURCE)
        try:
            with self._connection:
                self._ensure_user(user_id, display_name)
                self._insert_memory(row)
        except sqlite3.Error as exc:
            raise StorageError(SAVE_MEMORY_FAILED) from exc
        return memory_id

    def list_memories(self, user_id: str) -> list[tuple[str, str, str, str]]:
        """Return (memory_id, text, created_at, updated_at) for user_id.

        Newest creation time comes first. Equal times use memory id ascending.
        """
        try:
            rows = self._connection.execute(
                """
                SELECT memory_id, text, created_at, updated_at
                FROM memories
                WHERE user_id = ?
                ORDER BY created_at DESC, memory_id ASC
                """,
                (user_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(MEMORY_READ_FAILED) from exc
        return [
            (
                str(row["memory_id"]),
                str(row["text"]),
                str(row["created_at"]),
                str(row["updated_at"]),
            )
            for row in rows
        ]

    def update_memory(self, user_id: str, memory_id: str, text: str) -> None:
        """Replace the text of a fact owned by user_id.

        The id and created_at stay the same. updated_at changes.
        """
        try:
            with self._connection:
                self._require_memory_owner(user_id, memory_id)
                self._update_memory_text(user_id, memory_id, text)
        except sqlite3.Error as exc:
            raise StorageError(UPDATE_MEMORY_FAILED) from exc

    def delete_memory(self, user_id: str, memory_id: str) -> None:
        """Delete one fact owned by user_id. Other rows stay in place."""
        try:
            with self._connection:
                self._require_memory_owner(user_id, memory_id)
                self._delete_memory_row(user_id, memory_id)
        except sqlite3.Error as exc:
            raise StorageError(DELETE_MEMORY_FAILED) from exc

    def _oldest_session(self, user_id: str, display_name: str) -> str:
        with self._connection:
            self._ensure_user(user_id, display_name)
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
            self._insert_session(session_id, user_id, _timestamp())
            return session_id

    def _ensure_user(self, user_id: str, display_name: str) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO users (user_id, display_name)
            VALUES (?, ?)
            """,
            (user_id, display_name),
        )

    def _insert_session(self, session_id: str, user_id: str, created_at: str) -> None:
        self._connection.execute(
            """
            INSERT INTO sessions (session_id, user_id, created_at)
            VALUES (?, ?, ?)
            """,
            (session_id, user_id, created_at),
        )

    def _store_selection(self, user_id: str, session_id: str) -> None:
        self._connection.execute(
            """
            INSERT INTO active_sessions (user_id, session_id)
            VALUES (?, ?)
            ON CONFLICT (user_id) DO UPDATE SET session_id = excluded.session_id
            """,
            (user_id, session_id),
        )

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

    def _insert_memory(self, row: _MemoryInsert) -> None:
        self._connection.execute(
            """
            INSERT INTO memories (
                memory_id, user_id, text, created_at, updated_at, source
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row.memory_id,
                row.user_id,
                row.text,
                row.created_at,
                row.created_at,
                row.source,
            ),
        )

    def _update_memory_text(self, user_id: str, memory_id: str, text: str) -> None:
        self._connection.execute(
            """
            UPDATE memories
            SET text = ?, updated_at = ?
            WHERE memory_id = ? AND user_id = ?
            """,
            (text, _timestamp(), memory_id, user_id),
        )

    def _delete_memory_row(self, user_id: str, memory_id: str) -> None:
        self._connection.execute(
            """
            DELETE FROM memories
            WHERE memory_id = ? AND user_id = ?
            """,
            (memory_id, user_id),
        )

    def _require_memory_owner(self, user_id: str, memory_id: str) -> None:
        if self._owns_memory(user_id, memory_id):
            return
        raise StorageError(MEMORY_NOT_OWNER)

    def _owns_memory(self, user_id: str, memory_id: str) -> bool:
        row = self._connection.execute(
            """
            SELECT 1 FROM memories
            WHERE memory_id = ? AND user_id = ?
            """,
            (memory_id, user_id),
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

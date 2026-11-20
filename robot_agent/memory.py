"""Save, list, search, update, and delete facts for one user.

SQL stays in storage.py. Terminal output stays in cli.py.
Only an explicit save creates a fact. Search compares text in Python so
the query is a literal substring and Ukrainian case is ignored.
"""

from dataclasses import dataclass

from robot_agent.storage import MEMORY_NOT_OWNER, ConversationStore, StorageError

MEMORY_UNAVAILABLE = "That memory is not available."


@dataclass(frozen=True)
class MemoryRecord:
    """One saved fact belonging to a user."""

    memory_id: str
    text: str
    created_at: str
    updated_at: str


def save_memory(
    store: ConversationStore,
    user_id: str,
    display_name: str,
    text: str,
) -> str:
    """Save a new fact for user_id and return its id."""
    return store.save_memory(user_id, display_name, text)


def list_memories(store: ConversationStore, user_id: str) -> list[MemoryRecord]:
    """Return user_id's facts, newest first."""
    return [_record(row) for row in store.list_memories(user_id)]


def search_memories(
    store: ConversationStore,
    user_id: str,
    query: str,
) -> list[MemoryRecord]:
    """Return user_id's facts whose text contains query, ignoring case.

    `%` and `_` are literal characters. Results keep list order.
    """
    folded = query.casefold()
    return [
        record
        for record in list_memories(store, user_id)
        if folded in record.text.casefold()
    ]


def update_memory(
    store: ConversationStore,
    user_id: str,
    memory_id: str,
    text: str,
) -> None:
    """Replace an owned fact. A missing or foreign id is unavailable."""
    try:
        store.update_memory(user_id, memory_id, text)
    except StorageError as exc:
        _reraise_unavailable(exc)


def delete_memory(store: ConversationStore, user_id: str, memory_id: str) -> None:
    """Delete an owned fact. A missing or foreign id is unavailable."""
    try:
        store.delete_memory(user_id, memory_id)
    except StorageError as exc:
        _reraise_unavailable(exc)


def _record(row: tuple[str, str, str, str]) -> MemoryRecord:
    memory_id, text, created_at, updated_at = row
    return MemoryRecord(memory_id, text, created_at, updated_at)


def _reraise_unavailable(exc: StorageError) -> None:
    if exc.args == (MEMORY_NOT_OWNER,):
        raise StorageError(MEMORY_UNAVAILABLE) from exc
    raise

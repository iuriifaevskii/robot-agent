"""Create, list, and resume sessions for one user.

SQL stays in storage.py. Terminal output stays in cli.py.
"""

from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage

from robot_agent.storage import NOT_OWNER, ConversationStore, StorageError

SESSION_UNAVAILABLE = "That session is not available."


@dataclass(frozen=True)
class SessionRecord:
    """One session's id and creation time. Message bodies are not included."""

    session_id: str
    created_at: str


def open_session(store: ConversationStore, user_id: str, display_name: str) -> str:
    """Return the saved selection, or the user's oldest session.

    A database with no saved selection keeps the previous startup rule.
    Opening a session does not record a selection.
    """
    selected = store.active_session_id(user_id)
    if selected is not None:
        return selected
    return store.ensure_session(user_id, display_name)


def create_session(store: ConversationStore, user_id: str, display_name: str) -> str:
    """Create an empty session for user_id and make it the saved selection."""
    return store.create_session(user_id, display_name)


def list_sessions(store: ConversationStore, user_id: str) -> list[SessionRecord]:
    """Return user_id's sessions, newest first, without message bodies."""
    return [
        SessionRecord(session_id, created_at)
        for session_id, created_at in store.list_sessions(user_id)
    ]


def resume_session(
    store: ConversationStore,
    user_id: str,
    session_id: str,
    history_limit: int,
) -> list[ModelMessage]:
    """Load a session owned by user_id, then save it as the selection.

    Unknown sessions and other users' sessions raise the same error.
    The selection is saved only after the requested history loads.
    """
    history = _load_available(store, user_id, session_id, history_limit)
    store.select_session(user_id, session_id)
    return history


def _load_available(
    store: ConversationStore,
    user_id: str,
    session_id: str,
    history_limit: int,
) -> list[ModelMessage]:
    try:
        return store.load_recent(user_id, session_id, history_limit)
    except StorageError as exc:
        if exc.args == (NOT_OWNER,):
            raise StorageError(SESSION_UNAVAILABLE) from exc
        raise

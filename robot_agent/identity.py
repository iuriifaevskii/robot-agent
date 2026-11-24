"""Resolve the current user at the application boundary.

The fixed profile is the startup user. Other local profiles are created and
looked up here. SQL stays in storage.py and terminal output stays in cli.py.
"""

from dataclasses import dataclass

from robot_agent.storage import ConversationStore

MAX_DISPLAY_NAME_LENGTH = 64
NAME_TOO_LONG = (
    f"The display name must be at most {MAX_DISPLAY_NAME_LENGTH} characters."
)


class InvalidDisplayName(ValueError):
    """The display name cannot be stored. The message is safe to print."""


@dataclass(frozen=True)
class UserProfile:
    """Stable local identity. The model does not choose these fields."""

    user_id: str
    display_name: str


def resolve_user() -> UserProfile:
    """Return the one fixed local profile."""
    return UserProfile(user_id="local-user", display_name="Local user")


def create_profile(store: ConversationStore, display_name: str) -> str:
    """Store a new profile under a generated id and return that id.

    The name is stripped. It is never the id and need not be unique.
    """
    name = display_name.strip()
    if not name:
        raise InvalidDisplayName("The display name must not be empty.")
    if len(name) > MAX_DISPLAY_NAME_LENGTH:
        raise InvalidDisplayName(NAME_TOO_LONG)
    return store.create_user(name)


def list_profiles(store: ConversationStore) -> list[UserProfile]:
    """Return every stored profile, oldest first."""
    return [UserProfile(user_id, name) for user_id, name in store.list_users()]


def find_profile(store: ConversationStore, user_id: str) -> UserProfile | None:
    """Return the stored profile for user_id, or None. Nothing is created."""
    name = store.user_display_name(user_id)
    if name is None:
        return None
    return UserProfile(user_id, name)

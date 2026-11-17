"""Resolve the current user at the application boundary."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UserProfile:
    """Stable local identity. The model does not choose these fields."""

    user_id: str
    display_name: str


def resolve_user() -> UserProfile:
    """Return the one fixed local profile."""
    return UserProfile(user_id="local-user", display_name="Local user")

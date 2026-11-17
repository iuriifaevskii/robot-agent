import identity


def test_resolve_user_returns_fixed_profile() -> None:
    profile = identity.resolve_user()

    assert profile.user_id == "local-user"
    assert profile.display_name == "Local user"

"""Daily usage budget decisions and message text.

This module is pure: it receives counters and limits. It does not read
storage, call a model, or print.
"""

from dataclasses import dataclass
from typing import NamedTuple

REQUESTS_LIMIT = "requests"
TOKENS_LIMIT = "tokens"
MAX_REQUESTS_SETTING = "AGENT_DAILY_MAX_REQUESTS"
MAX_TOKENS_SETTING = "AGENT_DAILY_MAX_TOKENS"
UNTRUSTED_MESSAGE = (
    "Daily usage could not be recorded earlier in this run, so no further "
    "requests are sent. Restart the program to continue."
)


class UsageCounts(NamedTuple):
    """Provider-reported usage: model requests and input and output tokens."""

    requests: int
    input_tokens: int
    output_tokens: int

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class DailyBudget:
    """Daily limits. 0 means no limit for that counter."""

    max_requests: int = 0
    max_tokens: int = 0

    @property
    def enabled(self) -> bool:
        return self.max_requests > 0 or self.max_tokens > 0


NO_BUDGET = DailyBudget()


def blocking_limit(used: UsageCounts, budget: DailyBudget) -> str | None:
    """Return the reached limit, or None when a request may be sent.

    The two limits are independent. When both are reached, requests come first.
    """
    if budget.max_requests > 0 and used.requests >= budget.max_requests:
        return REQUESTS_LIMIT
    if budget.max_tokens > 0 and used.tokens >= budget.max_tokens:
        return TOKENS_LIMIT
    return None


def refusal_message(limit: str, used: UsageCounts, budget: DailyBudget) -> str:
    """Return the message naming the reached limit and today's usage."""
    if limit == REQUESTS_LIMIT:
        return _message(
            "request", used.requests, budget.max_requests, MAX_REQUESTS_SETTING
        )
    return _message("token", used.tokens, budget.max_tokens, MAX_TOKENS_SETTING)


def _message(noun: str, used: int, limit: int, setting: str) -> str:
    return (
        f"Daily {noun} budget reached ({used} of {limit} used today). "
        f"Raise {setting} or try again tomorrow."
    )

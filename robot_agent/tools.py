"""Allowed agent tools.

These functions read the local clock only. They do not call the model,
use the terminal, or touch storage.
"""

from datetime import datetime

CLOCK_READ_FAILURE = "The local clock could not be read."


class ClockReadError(Exception):
    """The local system clock could not be read."""


def format_local_timestamp(moment: datetime) -> str:
    """Return an ISO 8601 timestamp with seconds and a numeric UTC offset."""
    if moment.utcoffset() is None:
        raise ValueError("A local timestamp needs a timezone offset.")
    return moment.isoformat(timespec="seconds")


def read_local_clock() -> datetime:
    """Return the current system-local time, including its UTC offset."""
    try:
        return _system_now().astimezone()
    except OSError as exc:
        raise ClockReadError(CLOCK_READ_FAILURE) from exc


def _system_now() -> datetime:
    return datetime.now()


def current_local_time() -> str:
    """Return the current local date and time.

    Use this tool for any question that needs the current date or time.
    Do not estimate the current time from prior knowledge.
    """
    return format_local_timestamp(read_local_clock())

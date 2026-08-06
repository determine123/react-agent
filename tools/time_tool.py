"""Date & time tool — current time, timezone conversion, date arithmetic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def get_current_time(timezone_offset: str | None = None) -> dict:
    """Return the current date and time.

    Args:
        timezone_offset: Optional UTC offset like "+8" or "-5".
                         Defaults to local system time.

    Returns:
        Dict with ``datetime``, ``date``, ``time``, ``unix_ts``, ``day_of_week``.
    """
    if timezone_offset:
        try:
            offset_hours = int(timezone_offset)
            tz = timezone(timedelta(hours=offset_hours))
            now = datetime.now(tz)
        except (ValueError, OverflowError):
            now = datetime.now()
    else:
        now = datetime.now()

    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "unix_ts": int(now.timestamp()),
        "day_of_week": now.strftime("%A"),
        "timezone": str(now.tzinfo) if now.tzinfo else "local",
    }


def days_between(date1: str, date2: str) -> dict:
    """Calculate the number of days between two dates.

    Args:
        date1: First date in YYYY-MM-DD format.
        date2: Second date in YYYY-MM-DD format.

    Returns:
        Dict with ``date1``, ``date2``, ``days``, ``description``.
    """
    try:
        d1 = datetime.strptime(date1.strip(), "%Y-%m-%d")
        d2 = datetime.strptime(date2.strip(), "%Y-%m-%d")
        diff = abs((d2 - d1).days)
        return {
            "date1": date1,
            "date2": date2,
            "days": diff,
            "description": f"There are {diff} days between {date1} and {date2}.",
        }
    except ValueError as exc:
        return {"error": f"Invalid date format: {exc}. Please use YYYY-MM-DD."}

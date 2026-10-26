"""The run's clock. Demo mode pins it, so every reviewer sees the same answers on any day."""
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from .config import Settings

LONDON = ZoneInfo("Europe/London")
# The pinned demo moment: 09:00 London time on the demo date, a working morning after the fixture snapshot.
DEMO_TIME = time(9, 0)


def now(settings: Settings) -> datetime:
    """UTC-aware current time: pinned to the demo date in demo mode, the real time in live mode."""
    if settings.live:
        return datetime.now(UTC)
    return datetime.combine(settings.demo_date, DEMO_TIME, tzinfo=LONDON).astimezone(UTC)

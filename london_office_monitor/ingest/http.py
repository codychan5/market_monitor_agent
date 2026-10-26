"""One HTTP client for every source: bounded retries with backoff for transient failures only."""
import random
import re
import time
from collections.abc import Callable
from urllib.parse import urlsplit

import httpx

from ..config import Settings

RETRY_ATTEMPTS = 3
RETRY_WAIT_MIN_SECONDS = 0.5
RETRY_WAIT_MAX_SECONDS = 8.0
# Each backoff wait is scaled by a random factor in [1 - JITTER, 1 + JITTER] so parallel clients do not retry in step.
RETRY_JITTER = 0.2
# The longest wait a server's Retry-After may impose on us; a slow source must not stall the others.
RETRY_AFTER_CAP_SECONDS = 60.0
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 15
DIGITS = re.compile(r"[0-9]+")


class FetchError(Exception):
    """A source could not be fetched. Names the host only: some feed URLs carry keys in the query string."""

    def __init__(self, host: str, reason: str):
        super().__init__(f"{host}: {reason}")
        self.host = host
        self.reason = reason


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After", "").strip()
    return min(float(value), RETRY_AFTER_CAP_SECONDS) if DIGITS.fullmatch(value) else None


class HttpClient:
    def __init__(self, settings: Settings, *, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(READ_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS), follow_redirects=True,
            headers={"User-Agent": settings.user_agent})
        self._sleep = sleep

    @staticmethod
    def _backoff(attempt: int) -> float:
        base = min(RETRY_WAIT_MAX_SECONDS, RETRY_WAIT_MIN_SECONDS * 2 ** (attempt - 1))
        return base * random.uniform(1 - RETRY_JITTER, 1 + RETRY_JITTER)

    def get(self, url: str) -> bytes:
        """The response body. Retries timeouts, network errors and 429/5xx; any other failure raises at once."""
        host = urlsplit(url).hostname or "unknown host"
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            last = attempt == RETRY_ATTEMPTS
            try:
                response = self._client.get(url)
            except TRANSIENT_ERRORS as e:
                if last:
                    raise FetchError(host, f"{type(e).__name__} after {RETRY_ATTEMPTS} attempts") from e
                self._sleep(self._backoff(attempt))
                continue
            except httpx.HTTPError as e:
                raise FetchError(host, type(e).__name__) from e
            if response.status_code in RETRYABLE_STATUSES:
                if last:
                    raise FetchError(host, f"HTTP {response.status_code} after {RETRY_ATTEMPTS} attempts")
                retry_after = _retry_after_seconds(response)
                self._sleep(retry_after if retry_after is not None else self._backoff(attempt))
                continue
            if response.status_code >= 400:
                raise FetchError(host, f"HTTP {response.status_code}")
            return response.content
        raise FetchError(host, "no attempt made")  # unreachable: the loop always returns or raises

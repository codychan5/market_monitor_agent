"""Remove secrets from any text before it is logged, stored or shown. Error text is untrusted and can carry a key."""
from pydantic import SecretStr

from .config import MIN_PASSWORD_LENGTH, Settings

REDACTED = "[redacted]"


def redact(text: str, settings: Settings) -> str:
    """Replace every secret value from settings with [redacted]. Secret fields are found by type, so a new secret
    setting is covered without editing this function."""
    for name in type(settings).model_fields:
        value = getattr(settings, name)
        if isinstance(value, SecretStr):
            secret = value.get_secret_value()
            if len(secret) >= MIN_PASSWORD_LENGTH:
                text = text.replace(secret, REDACTED)
    return text

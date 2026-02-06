from datetime import datetime, timezone
from typing import Any


def log_event(event_type: str, message: str, **fields: Any) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    extras = ""
    if fields:
        extras = " " + " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[{ts}] [{event_type}] {message}{extras}")


def send_notification(message: str) -> None:
    # Placeholder for email/Slack/webhook integration.
    print(f"[notification] {message}")

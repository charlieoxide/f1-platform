"""Logging setup. Redacts anything that looks like a token/key before it
hits disk — confidentiality requirement, not optional."""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler

_REDACT_PATTERNS = [
    re.compile(r"(sk-[a-zA-Z0-9]{16,})"),  # OpenAI/OpenRouter-style keys
    re.compile(r"(Bearer\s+[a-zA-Z0-9._-]{16,})"),  # bearer tokens
    re.compile(r"(\d{9,12}:[a-zA-Z0-9_-]{30,})"),  # telegram bot tokens
]


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for pattern in _REDACT_PATTERNS:
            msg = pattern.sub("[REDACTED]", msg)
        record.msg = msg
        record.args = ()
        return True


def setup_logging(log_path: str = "data/app.log") -> None:
    handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3)
    handler.addFilter(RedactingFilter())
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(logging.StreamHandler())

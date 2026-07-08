"""Shared logging setup for host-assistant entry points."""

from __future__ import annotations

import logging
import sys


def configure_logging(*, level: int = logging.INFO) -> None:
    """Configure a simple stderr logger once per process."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

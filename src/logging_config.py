"""Central logging configuration for the Completion API.

Historically this service wrote operational output via ``print`` straight to
stdout. The application now logs through the stdlib ``logging`` module so that
levels can be filtered and output carries a timestamp/level prefix. The default
level is INFO so the previously-printed status lines remain visible on stdout
(Cloud Run captures stdout); set ``LOG_LEVEL=DEBUG`` to surface the verbose
payload dumps, or ``LOG_LEVEL=WARNING`` to quiet routine output.
"""

import logging
import os
import sys

_CONFIGURED = False


def configure_logging() -> None:
    """Install a stdout handler on the root logger. Idempotent."""
    global _CONFIGURED

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()

    if _CONFIGURED:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True

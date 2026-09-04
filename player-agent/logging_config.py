"""Shared logging setup for all player-agent entrypoints.

Call setup_logging() once, at the very start of each subagent's main.py
(inside its `if __name__ == "__main__":` block, before anything else runs) -
every other module just does `logger = logging.getLogger(__name__)` and logs
normally, and the format/level configured here applies process-wide from
then on.

"""
from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

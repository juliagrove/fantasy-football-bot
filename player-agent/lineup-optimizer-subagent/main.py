"""Entrypoint: reviews the bot's current lineup for the active scoring period
and, via the LangGraph pipeline in lineup_agent.py, decides whether to change
it - and if so, which swaps to make.

Future addition to this subagent (not built yet): waiver_pickup, once
tools/waiver.py has a real implementation to call.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from lineup_agent import build_lineup_graph, review_lineup
from logging_config import setup_logging
from tools.team import get_lineup

logger = logging.getLogger(__name__)


def load_config() -> None:
    load_dotenv()
    required = ["LEAGUE_ID", "BOT_TEAM_ID", "ESPN_S2", "SWID", "ANTHROPIC_API_KEY"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        logger.error("Missing required .env values: %s", ", ".join(missing))
        sys.exit(1)


async def main() -> None:
    load_config()

    lineup_response = get_lineup.invoke({})
    if "error" in lineup_response:
        logger.error("Failed to fetch lineup: %s", lineup_response)
        sys.exit(1)

    scoring_period_id = lineup_response["scoring_period_id"]
    graph = build_lineup_graph(scoring_period_id=scoring_period_id)

    logger.info("Reviewing lineup for scoring period %s...", scoring_period_id)
    try:
        outcome = await review_lineup(graph, lineup_response["lineup"])
    except Exception:
        logger.exception("Failed to review lineup")
        sys.exit(1)

    logger.info("%s - %s", outcome["decision"], outcome["reasoning"])
    for change in outcome["changes"]:
        logger.info("  - %s: %s -> %s", change["player_name"], change["from_slot"], change["to_slot"])


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())

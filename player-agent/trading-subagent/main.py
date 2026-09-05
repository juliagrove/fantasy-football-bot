"""Entrypoint: reviews every pending ESPN trade and accepts/rejects each one
via the LangGraph pipeline in trading_agent.py.

Trades and roster are re-fetched before each individual decision, rather than
snapshotted once up front - executing one trade can change the roster or
invalidate another pending trade (e.g. the same player offered twice), so
each decision should see current ESPN state, not a stale list.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from logging_config import setup_logging
from tools.team import get_current_roster
from tools.trades import get_trades
from trading_agent import build_trade_graph, review_trade

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

    # Trades that raised during review are marked "attempted" so a
    # persistently broken trade (e.g. review keeps failing) doesn't spin this
    # loop forever - successfully executed trades disappear from get_trades
    # on their own once ESPN marks them non-pending.
    attempted_trade_ids: set[str] = set()

    # Tracked so a swallowed per-trade failure (see the except block below)
    # still makes the process exit non-zero at the end - otherwise a broken
    # trade would get logged and skipped, but the run would still report
    # success to GitHub Actions.
    had_failures = False

    logger.info("Checking for pending trades...")
    while True:
        trades_response = get_trades.invoke({})
        if "error" in trades_response:
            logger.error("Failed to fetch trades: %s", trades_response)
            sys.exit(1)

        pending = [t for t in trades_response["trades"] if t["trade_id"] not in attempted_trade_ids]
        if not pending:
            break

        trade = pending[0]
        roster_response = get_current_roster.invoke({})
        if "error" in roster_response:
            logger.error("Failed to fetch roster: %s", roster_response)
            sys.exit(1)

        attempted_trade_ids.add(trade["trade_id"])

        # Rebuilt per trade - trade_id/scoring_period_id are baked into the
        # bound tools as closures (see trading_agent.py), so each distinct
        # trade needs its own graph.
        graph = build_trade_graph(trade_id=trade["trade_id"], scoring_period_id=trade["scoring_period_id"])
        try:
            outcome = await review_trade(graph, trade, roster_response["roster"])
        except Exception:
            logger.exception("Failed to review trade")
            had_failures = True
            continue
        logger.info("Trade %s: %s - %s", trade["trade_id"], outcome["decision"], outcome["reasoning"])

    if had_failures:
        logger.error("Finished with one or more trade review failures.")
        sys.exit(1)

    logger.info("No pending trades.")


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())

"""Entrypoint: watches the live ESPN draft room and drives the LangGraph draft agent.

Turn detection reads the browser's own DOM ("You are on the clock!") rather
than polling ESPN's REST league API — the live draft engine runs on a
separate real-time service (fantasydraft.espn.com) that the REST API doesn't
reliably reflect, but the browser is already a live client of that state.
The bot's own roster is tracked in memory as picks succeed, for the same
reason — no dependency on REST data that may not be in sync.
"""
from __future__ import annotations

import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

from browser_agent import DraftBrowser
from draft_agent import build_draft_graph, run_draft_turn

POLL_INTERVAL_SECONDS = 1.75
MAX_ATTEMPTS_PER_TURN = 2


def load_config() -> dict:
    load_dotenv()
    required = ["LEAGUE_ID", "BOT_TEAM_ID", "ESPN_S2", "SWID", "ANTHROPIC_API_KEY"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        sys.exit(f"Missing required .env values: {', '.join(missing)}")

    return {
        "league_id": os.environ["LEAGUE_ID"],
        "bot_team_id": int(os.environ["BOT_TEAM_ID"]),
        "espn_s2": os.environ["ESPN_S2"],
        "swid": os.environ["SWID"],
        "season": os.getenv("SEASON", "2026"),
    }


async def main() -> None:
    config = load_config()

    browser = DraftBrowser(
        league_id=config["league_id"],
        season=config["season"],
        bot_team_id=config["bot_team_id"],
        espn_s2=config["espn_s2"],
        swid=config["swid"],
        headless=False,
    )
    await browser.start()

    # Built once for the whole draft — only `bot_roster` varies turn to turn,
    # and it's passed in as state on each invoke rather than rebuilt here.
    graph = build_draft_graph(browser=browser)

    print(f"Draft agent watching league {config['league_id']} as team {config['bot_team_id']}...")

    bot_roster: list = []  # [(name, position, bye), ...], appended on each confirmed pick

    # Caps how many times we'll call the (paid) agent for the *same* turn
    # before giving up and just waiting quietly — otherwise a persistently
    # broken selector burns an API call every poll cycle, forever. Resets
    # whenever the browser confirms it's no longer our turn.
    attempt_count = 0
    gave_up_this_turn = False

    try:
        while True:
            try:
                my_turn = await browser.is_my_turn()
            except Exception as exc:
                print(f"[warn] Failed to read draft room state: {exc}")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue

            print(f"[poll] is_my_turn={my_turn} attempt_count={attempt_count} roster_size={len(bot_roster)}")

            if not my_turn:
                attempt_count = 0
                gave_up_this_turn = False
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue

            if gave_up_this_turn:
                pass  # already warned below; stay quiet so we don't spam every poll
            elif attempt_count >= MAX_ATTEMPTS_PER_TURN:
                gave_up_this_turn = True
                print(
                    f"[error] Hit {MAX_ATTEMPTS_PER_TURN} failed attempts this turn. "
                    "Pausing agent calls until it's no longer our turn."
                )
            else:
                attempt_count += 1
                print(f"[info] On the clock, attempt {attempt_count}/{MAX_ATTEMPTS_PER_TURN}. Consulting agent...")
                try:
                    drafted = await run_draft_turn(graph, bot_roster)
                except Exception:
                    print("[error] Agent turn raised an exception:")
                    traceback.print_exc()
                    drafted = None

                if drafted:
                    bot_roster.append((drafted["name"], drafted["position"], drafted["bye"]))
                    attempt_count = 0
                    print(f"[info] Pick submitted: {drafted['name']} ({drafted['position']}, bye {drafted['bye']}).")
                else:
                    print("[warn] Agent turn ended without a confirmed pick; will retry if still our turn.")

            await asyncio.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n[info] Shutting down.")
    finally:
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())

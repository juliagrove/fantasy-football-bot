# Get the status of your team (players / playing status / projections / bench)

import logging
import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)

LEAGUE_ID = os.environ["LEAGUE_ID"]
SEASON = os.environ.get("SEASON", "2026")
TEAM_ID = int(os.environ["BOT_TEAM_ID"])
ESPN_S2 = os.environ["ESPN_S2"]
SWID = os.environ["SWID"]

BASE_URL = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}"

# ESPN's `defaultPositionId` on the player object - not to be confused with
# `lineupSlotId`, which encodes the specific starter/bench/flex slot instead.
POSITION_NAMES = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}

# ESPN's `lineupSlotId` - shows up both on a roster entry (its current slot)
# and in a player's `eligibleSlots` (which slots it could legally move into).
SLOT_NAMES = {
    0: "QB",
    2: "RB",
    3: "RB/WR",
    4: "WR",
    5: "WR/TE",
    6: "TE",
    7: "OP",
    16: "D/ST",
    17: "K",
    20: "BENCH",
    21: "IR",
    23: "FLEX",
}

BROWSER_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"
    ),
    "sec-ch-ua": '"Chromium";v="152", "Not?A_Brand";v="24", "Google Chrome";v="152"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
}


def _cookies() -> dict:
    return {"espn_s2": ESPN_S2, "SWID": SWID}


def _player_value_summary(player: dict) -> dict:
    """Pull fairness-relevant rank/ownership/stat fields off a raw player object."""
    ownership = player.get("ownership") or {}
    preseason_rank = (player.get("draftRanksByRankType") or {}).get("STANDARD", {}).get("rank")

    last_season_actual = None
    projected_season = None
    for stat_line in player.get("stats", []):
        if stat_line.get("statSplitTypeId") != 0:
            continue  # only care about season totals, not single-week splits
        if stat_line.get("statSourceId") == 0 and stat_line.get("seasonId") == int(SEASON) - 1:
            last_season_actual = stat_line
        elif stat_line.get("statSourceId") == 1 and stat_line.get("seasonId") == int(SEASON):
            projected_season = stat_line

    return {
        "player_id": player["id"],
        "player_name": player["fullName"],
        "position": POSITION_NAMES.get(player.get("defaultPositionId"), "UNKNOWN"),
        "preseason_rank": preseason_rank,
        "percent_owned": round(ownership["percentOwned"], 2) if "percentOwned" in ownership else None,
        "avg_draft_position": ownership.get("averageDraftPosition"),
        "last_season_points": round(last_season_actual["appliedTotal"], 1) if last_season_actual else None,
        "last_season_ppg": round(last_season_actual["appliedAverage"], 1) if last_season_actual else None,
        "projected_season_points": round(projected_season["appliedTotal"], 1) if projected_season else None,
        "projected_ppg": round(projected_season["appliedAverage"], 1) if projected_season else None,
    }


def _weekly_projection(player: dict, scoring_period_id: int) -> float | None:
    """Pull this week's projected points, as opposed to the season-level totals
    _player_value_summary already covers. Distinguished from the season-total
    stat line by statSplitTypeId == 1 (single period) and a matching
    scoringPeriodId, with statSourceId == 1 (projected, not actual)."""
    for stat_line in player.get("stats", []):
        if (
            stat_line.get("statSourceId") == 1
            and stat_line.get("statSplitTypeId") == 1
            and stat_line.get("scoringPeriodId") == scoring_period_id
        ):
            return round(stat_line["appliedTotal"], 1)
    return None


@tool
def get_lineup() -> dict:
    """Fetch the bot's current starting lineup and bench for the active scoring period.

    Returns a dict: {"team_name": str, "scoring_period_id": int, "lineup": [...]}.
    Each entry has the same rank/ownership/season-points context as
    get_current_roster, plus `slot` (current slot, e.g. "QB", "FLEX", "BENCH"),
    `eligible_slots` (which slots that player could legally move into), and
    `projected_this_week` (projected points for the active scoring period, not
    a season total). Use this to judge whether swapping a bench player into a
    starting slot would raise the team's projected total for the week.
    """
    response = requests.get(
        BASE_URL,
        params={"view": ["mRoster", "mTeam"]},
        headers=BROWSER_HEADERS,
        cookies=_cookies(),
    )
    if not response.ok:
        logger.error("get_lineup failed: %s %s", response.status_code, response.text)
        return {"error": response.status_code, "details": response.json()}

    data = response.json()
    scoring_period_id = data["scoringPeriodId"]
    team = next((t for t in data["teams"] if t["id"] == TEAM_ID), None)
    if team is None:
        logger.error("get_lineup: no team with id %s", TEAM_ID)
        return {"error": "not_found", "details": f"No team with id {TEAM_ID}"}

    lineup = []
    for entry in team["roster"]["entries"]:
        player = entry["playerPoolEntry"]["player"]
        summary = _player_value_summary(player)
        summary["slot"] = SLOT_NAMES.get(entry["lineupSlotId"], entry["lineupSlotId"])
        summary["eligible_slots"] = [SLOT_NAMES.get(s, s) for s in player.get("eligibleSlots", [])]
        summary["projected_this_week"] = _weekly_projection(player, scoring_period_id)
        lineup.append(summary)

    logger.info("get_lineup: %d player(s), scoring_period_id=%s", len(lineup), scoring_period_id)
    return {"team_name": team["name"], "scoring_period_id": scoring_period_id, "lineup": lineup}


@tool
def get_current_roster() -> dict:
    """Fetch the bot's current fantasy football roster from ESPN.

    Returns a dict: {"team_name": str, "roster": [...]}. Each roster entry has
    the player's position plus the same rank/ownership/season-points fairness
    context as get_trades. Use this alongside get_trades to judge whether a
    proposed trade actually improves the roster.
    """
    response = requests.get(
        BASE_URL,
        params={"view": ["mRoster", "mTeam"]},
        headers=BROWSER_HEADERS,
        cookies=_cookies(),
    )
    if not response.ok:
        logger.error("get_current_roster failed: %s %s", response.status_code, response.text)
        return {"error": response.status_code, "details": response.json()}

    data = response.json()
    team = next((t for t in data["teams"] if t["id"] == TEAM_ID), None)
    if team is None:
        logger.error("get_current_roster: no team with id %s", TEAM_ID)
        return {"error": "not_found", "details": f"No team with id {TEAM_ID}"}

    roster = [
        _player_value_summary(entry["playerPoolEntry"]["player"])
        for entry in team["roster"]["entries"]
    ]

    logger.info("get_current_roster: %d player(s) on roster", len(roster))
    return {"team_name": team["name"], "roster": roster}

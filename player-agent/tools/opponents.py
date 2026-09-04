# Get an individual Opponents team/info

import logging
import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)

LEAGUE_ID = os.environ["LEAGUE_ID"]
SEASON = os.environ.get("SEASON", "2026")
OPPONENT_TEAM_ID = os.environ.get("OPPONENT_TEAM_ID")
ESPN_S2 = os.environ["ESPN_S2"]
SWID = os.environ["SWID"]

BASE_URL = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}"

# ESPN's `defaultPositionId` on the player object - not to be confused with
# `lineupSlotId`, which encodes the specific starter/bench/flex slot instead.
POSITION_NAMES = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}

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


@tool
def get_opponents_roster() -> dict:
    """Fetch this week's opponent's current fantasy football roster from ESPN.

    Reads which team is the opponent from the OPPONENT_TEAM_ID env var (the
    numeric team id, found e.g. on the team's ESPN page URL or via get_trades'
    team names) - update it each week as the matchup changes.

    Returns a dict: {"team_name": str, "roster": [...]}. Each roster entry has
    the same rank/ownership/season-points fairness context as
    get_current_roster. Use this alongside get_current_roster to scout the
    opponent before setting a lineup or proposing a trade.
    """
    if not OPPONENT_TEAM_ID:
        return {"error": "OPPONENT_TEAM_ID is not set"}

    response = requests.get(
        BASE_URL,
        params={"view": ["mRoster", "mTeam"]},
        headers=BROWSER_HEADERS,
        cookies=_cookies(),
    )
    if not response.ok:
        logger.error("get_opponents_roster failed: %s %s", response.status_code, response.text)
        return {"error": response.status_code, "details": response.json()}

    data = response.json()
    opponent_team_id = int(OPPONENT_TEAM_ID)
    team = next((t for t in data["teams"] if t["id"] == opponent_team_id), None)
    if team is None:
        logger.error("get_opponents_roster: no team with id %s", opponent_team_id)
        return {"error": "not_found", "details": f"No team with id {opponent_team_id}"}

    roster = [
        _player_value_summary(entry["playerPoolEntry"]["player"])
        for entry in team["roster"]["entries"]
    ]

    logger.info("get_opponents_roster: %d player(s) on %s's roster", len(roster), team["name"])
    return {"team_name": team["name"], "roster": roster}

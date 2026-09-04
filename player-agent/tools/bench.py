# Modify your bench - move players
"""
MOVE PLAYER endpoint - same lm-api-writes /transactions/ endpoint as
accept_trade/reject_trade in trades.py, but `type: "ROSTER"` with an `items`
array instead of a `relatedTransactionId`. Each item is one player's move:
{"playerId": ..., "type": "LINEUP", "fromLineupSlotId": ..., "toLineupSlotId": ...}.
Multiple items in one call apply as a single atomic transaction (useful for a
true swap - move both players in one request).

curl --url 'https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl/seasons/2026/segments/0/leagues/2057324920/transactions/?platformVersion=96e7cdc122a61e6c778b4087703c10d123d0565d' \
  -H 'accept: application/json' \
  -H 'content-type: application/json' \
  -H 'origin: https://fantasy.espn.com' \
  -H 'referer: https://fantasy.espn.com/' \
  -H 'x-fantasy-platform: espn-fantasy-web' \
  -H 'x-fantasy-source: kona' \
  --cookie "espn_s2=${ESPN_S2}; SWID=${SWID}" \
  --data-raw '{"isLeagueManager":false,"teamId":1,"type":"ROSTER","memberId":"'"${SWID}"'","scoringPeriodId":1,"executionType":"EXECUTE","items":[{"playerId":3916387,"type":"LINEUP","fromLineupSlotId":0,"toLineupSlotId":20}]}'

Response (201, status EXECUTED on success):
{
    "bidAmount": 0,
    "executionType": "EXECUTE",
    "id": "2bf9d12d-6daf-4876-9166-06b566a7c168",
    "isActingAsTeamOwner": false,
    "isLeagueManager": false,
    "isPending": false,
    "items": [
        {
            "fromLineupSlotId": 20,
            "fromTeamId": 0,
            "isKeeper": false,
            "overallPickNumber": 0,
            "playerId": 3916387,
            "toLineupSlotId": 0,
            "toTeamId": 0,
            "type": "LINEUP"
        }
    ],
    "memberId": "<SWID>",
    "proposedDate": 1788399344656,
    "rating": 0,
    "scoringPeriodId": 1,
    "skipTransactionCounters": false,
    "status": "EXECUTED",
    "subOrder": 0,
    "teamId": 1,
    "type": "ROSTER"
}
"""

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

# Build hash captured from ESPN's frontend network requests - may need to
# be re-captured from a fresh network tab request if writes start 400ing.
PLATFORM_VERSION = os.environ["PLATFORM_VERSION"]

WRITE_BASE_URL = (
    f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl"
    f"/seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}/transactions/"
)

# Mirrors tools/team.py's SLOT_NAMES, reversed - change_lineup receives slot
# names (as produced by get_lineup) and has to send ESPN back the numeric
# lineupSlotId each one came from.
SLOT_IDS = {
    "QB": 0,
    "RB": 2,
    "RB/WR": 3,
    "WR": 4,
    "WR/TE": 5,
    "TE": 6,
    "OP": 7,
    "D/ST": 16,
    "K": 17,
    "BENCH": 20,
    "IR": 21,
    "FLEX": 23,
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


@tool
def change_lineup(changes: list[dict], scoring_period_id: int) -> dict:
    """Move one or more players into different lineup slots on ESPN.

    Args:
        changes: List of {"player_id": int, "player_name": str, "from_slot": str,
            "to_slot": str} dicts describing the swaps to make. Slot names must
            match get_lineup's output (e.g. "BENCH", "TE", "FLEX").
        scoring_period_id: The active scoring_period_id, from get_lineup().

    All changes are submitted as one atomic transaction. Returns the ESPN
    transaction response on success, or an error dict with the status code
    and details on failure. Always call get_lineup first to confirm current
    slots/eligibility and get the active scoring_period_id.
    """
    items = [
        {
            "playerId": change["player_id"],
            "type": "LINEUP",
            "fromLineupSlotId": SLOT_IDS[change["from_slot"]],
            "toLineupSlotId": SLOT_IDS[change["to_slot"]],
        }
        for change in changes
    ]

    response = requests.post(
        WRITE_BASE_URL,
        params={"platformVersion": PLATFORM_VERSION},
        headers={
            **BROWSER_HEADERS,
            "Content-Type": "application/json",
            "Origin": "https://fantasy.espn.com",
            "Referer": "https://fantasy.espn.com/",
            "x-fantasy-platform": "espn-fantasy-web",
            "x-fantasy-source": "kona",
        },
        cookies=_cookies(),
        json={
            "isLeagueManager": False,
            "teamId": TEAM_ID,
            "type": "ROSTER",
            "memberId": SWID,
            "scoringPeriodId": scoring_period_id,
            "executionType": "EXECUTE",
            "items": items,
        },
    )
    if not response.ok:
        logger.error("change_lineup failed: %s %s", response.status_code, response.text)
        return {"error": response.status_code, "details": response.json()}

    data = response.json()
    logger.info("change_lineup: %d swap(s) submitted, status=%s", len(changes), data.get("status"))
    return data
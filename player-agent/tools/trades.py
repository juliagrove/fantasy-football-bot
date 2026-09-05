import logging
import os
from datetime import datetime, timezone

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


def _to_iso(epoch_ms: int | None) -> str | None:
    if epoch_ms is None:
        return None
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        "preseason_rank": preseason_rank,
        "percent_owned": round(ownership["percentOwned"], 2) if "percentOwned" in ownership else None,
        "avg_draft_position": ownership.get("averageDraftPosition"),
        "last_season_points": round(last_season_actual["appliedTotal"], 1) if last_season_actual else None,
        "last_season_ppg": round(last_season_actual["appliedAverage"], 1) if last_season_actual else None,
        "projected_season_points": round(projected_season["appliedTotal"], 1) if projected_season else None,
        "projected_ppg": round(projected_season["appliedAverage"], 1) if projected_season else None,
    }


@tool
def get_trades() -> dict:
    """Fetch all pending trade proposals involving the bot's team on ESPN Fantasy Football.

    Returns a dict: {"count": int, "teams": [...], "trades": [...]}. `teams`
    lists every team in the league as {"team_id": int, "team_name": str} -
    use it to look up a team's numeric id (e.g. for OPPONENT_TEAM_ID), even
    if that team isn't part of a pending trade. Each trade in `trades` shows
    the proposing team/member (with ids), expiration, which players the bot
    would receive (`you_receive`) and give up (`you_give_up`) - each with
    rank/ownership/season-points context for judging fairness and the other
    team's id - and how other teams in the trade have already responded
    (`team_responses`). Use this before calling accept_trade or reject_trade.
    """
    response = requests.get(
        BASE_URL,
        params={"view": ["mPendingTransactions", "mRoster", "mTeam"]},
        headers=BROWSER_HEADERS,
        cookies=_cookies(),
    )
    if not response.ok:
        logger.error("get_trades failed: %s %s", response.status_code, response.text)
        return {"error": response.status_code, "details": response.json()}

    data = response.json()
    team_names = {t["id"]: t["name"] for t in data["teams"]}
    member_names = {m["id"]: m["displayName"] for m in data.get("members", [])}

    players = {}
    for t in data["teams"]:
        for entry in t["roster"]["entries"]:
            player = entry["playerPoolEntry"]["player"]
            players[player["id"]] = player

    trades = []
    for txn in data.get("pendingTransactions", []):
        if txn.get("type") != "TRADE_PROPOSAL" or txn.get("status") != "PENDING":
            continue

        items = txn.get("items", [])
        if not any(item["fromTeamId"] == TEAM_ID or item["toTeamId"] == TEAM_ID for item in items):
            continue

        you_receive = []
        you_give_up = []
        for item in items:
            player = players.get(item["playerId"])
            if player is None:
                continue
            summary = _player_value_summary(player)
            if item["toTeamId"] == TEAM_ID:
                summary["from_team"] = team_names.get(item["fromTeamId"])
                summary["from_team_id"] = item["fromTeamId"]
                you_receive.append(summary)
            elif item["fromTeamId"] == TEAM_ID:
                summary["to_team"] = team_names.get(item["toTeamId"])
                summary["to_team_id"] = item["toTeamId"]
                you_give_up.append(summary)

        trades.append(
            {
                "trade_id": txn["id"],
                "status": txn["status"],
                "scoring_period_id": txn.get("scoringPeriodId"),
                "proposed_by_team": team_names.get(txn.get("teamId")),
                "proposed_by_team_id": txn.get("teamId"),
                "proposed_by_member": member_names.get(txn.get("memberId")),
                "proposed_date": _to_iso(txn.get("proposedDate")),
                "expires_date": _to_iso(txn.get("expirationDate")),
                "you_receive": you_receive,
                "you_give_up": you_give_up,
                "team_responses": {
                    team_names.get(int(team_id), team_id): action
                    for team_id, action in txn.get("teamActions", {}).items()
                },
            }
        )

    teams = [{"team_id": team_id, "team_name": team_name} for team_id, team_name in team_names.items()]

    logger.info("get_trades: %d pending trade(s) found", len(trades))
    return {"count": len(trades), "teams": teams, "trades": trades}


PLATFORM_VERSION = os.environ["PLATFORM_VERSION"]

WRITE_BASE_URL = (
    f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl"
    f"/seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}/transactions/"
)


def _submit_trade_response(trade_id: str, txn_type: str, scoring_period_id: int, comment: str | None = None) -> dict:
    payload = {
        "isLeagueManager": False,
        "teamId": TEAM_ID,
        "type": txn_type,
        "memberId": SWID,
        "scoringPeriodId": scoring_period_id,
        "executionType": "EXECUTE",
        "relatedTransactionId": trade_id,
    }
    if comment is not None:
        payload["comment"] = comment

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
        json=payload,
    )
    if not response.ok:
        logger.error(
            "Trade %s (%s) failed: %s %s", trade_id, txn_type, response.status_code, response.text
        )
        return {"error": response.status_code, "details": response.json()}

    data = response.json()
    logger.info("Trade (%s) status=%s", txn_type, data.get("status"))
    return data


@tool
def accept_trade(trade_id: str, scoring_period_id: int) -> dict:
    """Accept a pending trade proposal on ESPN Fantasy Football.

    Args:
        trade_id: The `trade_id` of the proposal to accept, from get_trades().
        scoring_period_id: The `scoring_period_id` of that same trade, from get_trades().

    Returns the ESPN transaction response on success, or an error dict with
    the status code and details on failure. Always call get_trades first to
    confirm the trade still exists and get its current trade_id/scoring_period_id.
    """
    return _submit_trade_response(trade_id, "TRADE_ACCEPT", scoring_period_id)


@tool
def reject_trade(trade_id: str, scoring_period_id: int, comment: str = "") -> dict:
    """Reject a pending trade proposal on ESPN Fantasy Football.

    Args:
        trade_id: The `trade_id` of the proposal to reject, from get_trades().
        scoring_period_id: The `scoring_period_id` of that same trade, from get_trades().
        comment: Optional message shown to the other team explaining the rejection.

    Returns the ESPN transaction response on success, or an error dict with
    the status code and details on failure. Always call get_trades first to
    confirm the trade still exists and get its current trade_id/scoring_period_id.
    """
    return _submit_trade_response(trade_id, "TRADE_DECLINE", scoring_period_id, comment=comment)
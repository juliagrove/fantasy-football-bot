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
ESPN_S2 = os.environ["ESPN_S2"]
SWID = os.environ["SWID"]
TOPIC_ID = os.environ["TOPIC_ID"]
OPPONENT_CHAT_ID = os.environ.get("OPPONENT_CHAT_ID")

COMMUNICATION_URL = (
    f"https://lm-api-communication.fantasy.espn.com/apis/v3/games/ffl"
    f"/seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}/communication/topics"
)


def _topic_url(topic_id: str) -> str:
    return f"{COMMUNICATION_URL}/{topic_id}"

# The chat endpoint only ever gives an `author` as a member SWID - resolving
# that to a team name needs a separate call to the league-reads endpoint
# (same one trades.py/team.py use), whose `teams[].owners` lists the SWID(s)
# that own each team.
READ_BASE_URL = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}"

# Emulate a real Chrome/macOS browser session - ESPN's edge/WAF 
# rejects requests that don't look like they came from a browser 
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


def _to_iso(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _team_names_by_swid() -> dict:
    """Map each team's owner SWID(s) to that team's name."""
    response = requests.get(
        READ_BASE_URL,
        params={"view": "mTeam"},
        headers=BROWSER_HEADERS,
        cookies=_cookies(),
    )
    if not response.ok:
        logger.warning(
            "Failed to fetch team names for chat author resolution: %s %s",
            response.status_code,
            response.text,
        )
        return {}
    teams = response.json().get("teams", [])

    # A SWID can be listed as a secondary co-owner on a team it doesn't
    # actually belong to. Map primaryOwner first so it always wins.
    mapping: dict[str, str] = {}
    for team in teams:
        primary = team.get("primaryOwner")
        if primary:
            mapping[primary] = team["name"]
    for team in teams:
        for owner in team.get("owners", []):
            mapping.setdefault(owner, team["name"])
    return mapping


def _get_chat(topic_id: str, label: str) -> list[dict]:
    response = requests.get(
        _topic_url(topic_id),
        params={"view": "chat_conversation", "platform": "chat"},
        headers=BROWSER_HEADERS,
        cookies=_cookies(),
    )
    if not response.ok:
        logger.error("%s failed: %s %s", label, response.status_code, response.text)
        return {"error": response.status_code, "details": response.json()}

    messages = response.json().get("messages", [])
    real_messages = [message for message in messages if message.get("author") != "LM"]
    real_messages.sort(key=lambda message: message["date"])
    real_messages = real_messages[-10:]

    team_names = _team_names_by_swid()

    result = [
        {
            "author": team_names.get(message["author"], message["author"]),
            "content": message.get("content"),
            "date": _to_iso(message["date"]),
        }
        for message in real_messages
    ]
    logger.info("%s: %d message(s) fetched", label, len(result))
    return result


def _send_chat(topic_id: str, message: str, label: str) -> dict:
    response = requests.post(
        f"{_topic_url(topic_id)}/messages/",
        params={"source": SWID, "platform": "chat"},
        headers={
            **BROWSER_HEADERS,
            "Content-Type": "application/json",
            "Origin": "https://chat.espn.com",
            "Referer": "https://chat.espn.com/",
        },
        cookies=_cookies(),
        json={"author": SWID, "content": message, "messageTypeId": ""},
    )
    if not response.ok:
        logger.error("%s failed: %s %s", label, response.status_code, response.text)
        return {"error": response.status_code, "details": response.json()}

    data = response.json()
    logger.info("%s: posted message id=%s", label, data[0].get("id") if data else None)
    return data


@tool
def get_league_chat() -> list[dict]:
    """Fetch the league's all-members group chat messages from ESPN Fantasy Football.

    Returns at most the 10 most recent messages, sorted chronologically
    (oldest first - ESPN's own order is neither newest-first nor
    insertion-order, so this can't be trusted as-is), each reduced to just
    `author` (the sender's team name, resolved from their SWID - falls back
    to the raw SWID if it can't be resolved to a team), `content`, and an
    ISO 8601 UTC `date`. System-generated entries (author "LM" - e.g.
    automated trade notifications, which have no `content` field, just
    player/team id `attributes`) are filtered out. Use this to read what's
    been said in the league chat, e.g. before deciding whether to reply.
    """
    return _get_chat(TOPIC_ID, "get_league_chat")


@tool
def send_league_message(message: str) -> dict:
    """Post a message to the league's all-members group chat on ESPN Fantasy Football.

    Args:
        message: The text content to send to the league chat.

    Returns the created message object from ESPN (author, content, id, timestamp)
    on success, or an error dict with the status code and details on failure.
    """
    return _send_chat(TOPIC_ID, message, "send_league_message")


@tool
def get_opponent_chat() -> list[dict] | dict:
    """Fetch the 1-on-1 chat messages with this week's opponent on ESPN Fantasy Football.

    Reads the conversation topic from the OPPONENT_CHAT_ID env var (find it
    via devtools in chat.espn.com's network tab - the topic id in the URL of
    the 1-on-1 thread). Returns the same shape as get_league_chat, or an
    error dict if OPPONENT_CHAT_ID isn't configured.
    """
    if not OPPONENT_CHAT_ID:
        return {"error": "OPPONENT_CHAT_ID is not set"}
    return _get_chat(OPPONENT_CHAT_ID, "get_opponent_chat")


@tool
def send_opponent_message(message: str) -> dict:
    """Post a message to the 1-on-1 chat with this week's opponent on ESPN Fantasy Football.

    Args:
        message: The text content to send to the opponent chat.

    Reads the conversation topic from the OPPONENT_CHAT_ID env var. Returns
    the created message object from ESPN on success, or an error dict
    (including if OPPONENT_CHAT_ID isn't configured) on failure.
    """
    if not OPPONENT_CHAT_ID:
        return {"error": "OPPONENT_CHAT_ID is not set"}
    return _send_chat(OPPONENT_CHAT_ID, message, "send_opponent_message")

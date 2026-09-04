import logging
import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)

LEAGUE_ID = os.environ["LEAGUE_ID"]
SEASON = os.environ.get("SEASON", "2026")
TEAM_ID = os.environ["BOT_TEAM_ID"]
ESPN_S2 = os.environ["ESPN_S2"]
SWID = os.environ["SWID"]
PLATFORM_VERSION = os.environ["PLATFORM_VERSION"]

BASE_URL = (
    f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl"
    f"/seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}/teams/{TEAM_ID}"
)

TEAM_ABBREV = "RRT"

# Emulate a real Chrome/macOS browser session
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
def update_team_name(name: str) -> dict:
    """Update the bot's fantasy team name on ESPN.

    Args:
        name: The new team name to set.

    The team abbreviation is hardcoded to "RRT". Returns the updated team
    object from ESPN on success, or an error dict with the status code and
    details on failure.
    """
    response = requests.post(
        BASE_URL,
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
        json={"abbrev": TEAM_ABBREV, "name": name},
    )
    if not response.ok:
        logger.error("update_team_name failed: %s %s", response.status_code, response.text)
        return {"error": response.status_code, "details": response.json()}

    data = response.json()
    logger.info("update_team_name: renamed to '%s'", name)
    return data

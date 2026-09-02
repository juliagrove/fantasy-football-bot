"""One-time interactive login: capture a full ESPN browser session so the draft
agent doesn't have to rely on raw cookie injection, which ESPN's SPA rejects.

Run this once (`python login.py`), log in normally in the window it opens,
then press Enter in the terminal. The saved session is reused by every
`python main.py` run afterward via browser_agent.AUTH_STATE_PATH.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from browser_agent import AUTH_STATE_PATH, build_draft_url


def main() -> None:
    load_dotenv()
    draft_url = build_draft_url(
        league_id=os.environ["LEAGUE_ID"],
        season=os.getenv("SEASON", "2026"),
        bot_team_id=os.environ["BOT_TEAM_ID"],
        swid=os.environ["SWID"],
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        # Log in on the exact same URL main.py will use — localStorage is
        # scoped per-origin/route, so logging in somewhere else may not
        # populate what the actual draft room route needs.
        page.goto(draft_url)

        input(
            "A MyDisney login prompt should appear on the ESPN Fantasy page. Log "
            "in fully there (email/password/2FA as needed) until the prompt is "
            "gone and you see the actual fantasy.espn.com page. Then press Enter "
            "here...\n"
        )

        context.storage_state(path=AUTH_STATE_PATH)
        print(f"Saved session to {AUTH_STATE_PATH}. You can now run: python main.py")
        browser.close()


if __name__ == "__main__":
    main()

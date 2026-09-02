"""Playwright execution engine that drives the live ESPN draft room UI.

Uses Playwright's *async* API deliberately: this browser is driven from
inside a LangGraph tool call, and LangGraph executes tools on an asyncio
event loop (or a worker thread, for sync tools). Playwright's sync API is
hard-bound to the OS thread that created it, so calling it from LangGraph's
tool-execution thread raises `greenlet.error: Cannot switch to a different
thread`. Running everything (agent + browser) on one asyncio event loop
avoids that entirely.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from playwright.async_api import Page, async_playwright

DRAFT_URL_TEMPLATE = (
    "https://fantasy.espn.com/football/draft"
    "?leagueId={league_id}&seasonId={season}&teamId={team_id}&memberId={swid}"
)

# Path to a full session (cookies + localStorage) saved by `login.py`. ESPN's
# draft-room SPA does a client-side auth check beyond just the espn_s2/SWID
# cookies (it's backed by Disney's unified login), so a real, manually
# authenticated session is more reliable than injecting the two cookies alone.
AUTH_STATE_PATH = "espn_auth_state.json"

DRAFT_BUTTON_RE = re.compile(r"^draft$", re.IGNORECASE)
# Any position's dropdown option is a reliable, stable way to identify the
# position filter <select> among the page's other filter dropdowns (sort,
# team) without depending on their exact class names or DOM order.
POSITION_OPTION_RE = re.compile(r"^QB$", re.IGNORECASE)
# ESPN renders this banner the instant it's actually our turn. The live
# draft engine runs on a separate real-time service (fantasydraft.espn.com,
# confirmed via the browser's own network traffic) that the REST league API
# doesn't reliably reflect — but the browser itself, already a live client
# of that real-time state, always shows this correctly.
ON_THE_CLOCK_RE = re.compile(r"you are on the clock", re.IGNORECASE)
# Bye weeks are single/double-digit (1-18). Best-effort extraction from a
# row's full text after stripping the player's name — the row also contains
# rank, team, and decimal projection numbers, so this isn't bulletproof.
# Flagged as the most likely spot to need adjustment once checked live.
BYE_WEEK_RE = re.compile(r"\b(1[0-8]|[1-9])\b")


def _extract_bye_week(row_text: str, player_name: str) -> Optional[int]:
    remainder = row_text.replace(player_name, "")
    match = BYE_WEEK_RE.search(remainder)
    return int(match.group(1)) if match else None


def build_draft_url(league_id: str, season: str, bot_team_id, swid: str) -> str:
    return DRAFT_URL_TEMPLATE.format(
        league_id=league_id, season=season, team_id=bot_team_id, swid=swid
    )


class DraftBrowser:
    """Owns a visible Chromium session logged into ESPN via a saved auth session."""

    def __init__(
        self,
        league_id: str,
        season: str,
        bot_team_id: int,
        espn_s2: str,
        swid: str,
        headless: bool = False,
    ):
        self.league_id = league_id
        self.season = season
        self.bot_team_id = bot_team_id
        self.espn_s2 = espn_s2
        self.swid = swid
        self.headless = headless

        self._playwright = None
        self._browser = None
        self._context = None
        self.page: Page | None = None

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)

        if os.path.exists(AUTH_STATE_PATH):
            self._context = await self._browser.new_context(storage_state=AUTH_STATE_PATH)
        else:
            print(
                f"[warn] No saved session at {AUTH_STATE_PATH} — falling back to raw "
                "cookie injection, which may get redirected to a login screen. "
                "Run `python login.py` once to fix this."
            )
            self._context = await self._browser.new_context()
            await self._context.add_cookies(
                [
                    {
                        "name": "espn_s2",
                        "value": self.espn_s2,
                        "domain": ".espn.com",
                        "path": "/",
                        "secure": True,
                        "sameSite": "None",
                    },
                    {
                        "name": "SWID",
                        "value": self.swid,
                        "domain": ".espn.com",
                        "path": "/",
                        "secure": True,
                        "sameSite": "None",
                    },
                ]
            )
        self.page = await self._context.new_page()
        await self.page.goto(build_draft_url(self.league_id, self.season, self.bot_team_id, self.swid))
        await self.page.wait_for_load_state("networkidle")

    async def close(self) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def is_my_turn(self) -> bool:
        """Check the draft room itself for the "You are on the clock!" banner.

        This is the real-time signal, sourced directly from the same live
        state the browser is already rendering — not from a REST poll.
        """
        if self.page is None:
            raise RuntimeError("DraftBrowser.start() must be called before is_my_turn()")
        return await self.page.get_by_text(ON_THE_CLOCK_RE).first.is_visible()

    async def _apply_position_filter(self, position: str) -> None:
        page = self.page
        # Identify the position filter <select> by the presence of a "QB"
        # option — the sort/team filter dropdowns share the same CSS class
        # but don't have that option, so this disambiguates without relying
        # on DOM order or exact class names.
        position_select = page.locator("select").filter(
            has=page.locator("option", has_text=POSITION_OPTION_RE)
        ).first
        await position_select.select_option(label=position)
        await page.wait_for_timeout(750)  # let the filtered list re-render

    async def get_top_candidates_at_position(self, position: str, n: int) -> list:
        """Filter the player list to `position` and read the top `n` ranked
        players (name + bye week) without drafting anyone.

        Uses ESPN's own live-ranked, position-filtered list as the ranking
        source rather than a static local ranking.
        """
        if self.page is None:
            raise RuntimeError(
                "DraftBrowser.start() must be called before get_top_candidates_at_position()"
            )

        page = self.page
        await self._apply_position_filter(position)

        draft_buttons = page.get_by_role("button", name=DRAFT_BUTTON_RE)
        names_locator = page.locator("div.player-details")
        available = min(n, await draft_buttons.count(), await names_locator.count())
        # div.player-details stacks multiple lines (name, team, position,
        # injury tag) -- only the first line is the actual player name.
        # Playwright's has_text does substring matching against normalized
        # (whitespace-collapsed) text, so a raw multi-line string here would
        # never match anything downstream when drafting.
        names = []
        for i in range(available):
            raw = (await names_locator.nth(i).inner_text()).strip()
            names.append(raw.splitlines()[0].strip())

        # Read the whole list's text once and slice out each candidate's
        # segment by name position, rather than re-running an expensive
        # page-wide "find the div containing this text + a button" query
        # once per candidate (that was slow enough across 5 candidates to
        # look like a hang).
        bye_by_name = {}
        try:
            full_text = await page.inner_text("body")
            for idx, name in enumerate(names):
                start = full_text.find(name)
                if start == -1:
                    continue
                end = full_text.find(names[idx + 1]) if idx + 1 < len(names) else start + 200
                bye_by_name[name] = _extract_bye_week(full_text[start:end], name)
        except Exception:
            pass

        return [{"name": name, "bye": bye_by_name.get(name)} for name in names]

    async def draft_chosen_player_at_position(self, position: str, player_name: str) -> None:
        """Filter to `position` (if not already) and draft the named player."""
        if self.page is None:
            raise RuntimeError(
                "DraftBrowser.start() must be called before draft_chosen_player_at_position()"
            )

        page = self.page
        await self._apply_position_filter(position)

        draft_buttons = page.get_by_role("button", name=DRAFT_BUTTON_RE)
        row = page.locator("div").filter(has_text=player_name).filter(has=draft_buttons).last
        await row.wait_for(state="visible", timeout=5000)

        draft_button = row.get_by_role("button", name=DRAFT_BUTTON_RE)
        await draft_button.click()

        # A confirmation dialog is common in ESPN's draft flow; click through
        # it if one appears, but don't fail the pick if it doesn't.
        try:
            confirm_button = page.get_by_role("button", name=DRAFT_BUTTON_RE)
            await confirm_button.click(timeout=3000)
        except Exception:
            pass

        await page.wait_for_timeout(2000)  # give ESPN's draft state time to update

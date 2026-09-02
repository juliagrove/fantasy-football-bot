"""LangGraph draft-decision pipeline, as four explicit nodes:

  select_position   (LLM)          -- which position to target this turn
  fetch_candidates  (deterministic) -- top-ranked available players there
  select_player     (deterministic) -- pick one, with round-aware risk + bye-week logic
  execute_pick      (deterministic) -- click it

Only position selection calls the LLM. Player selection is a scoring/sampling
function (see valuation.select_candidate) rather than a second LLM call, to
keep each pick fast and cheap.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from browser_agent import DraftBrowser
from valuation import (
    VALID_POSITIONS,
    candidate_pool_size,
    get_roster_summary,
    select_candidate,
)

SYSTEM_PROMPT = f"""You are an autonomous fantasy football draft assistant, currently on \
the clock in a live ESPN snake draft. You'll be given the bot's current roster position \
counts, soft caps (a reasonable max before that position stops being a priority), and \
which positions are still under their cap.

This league's starting lineup includes a FLEX slot that RB, WR, or TE can fill, in \
addition to their own dedicated starting slots. You'll be told `flex_eligible_total` \
(RB + WR + TE drafted so far), `flex_eligible_starter_demand` (how many RB/WR/TE are \
needed to fill the RB, WR, TE, and FLEX starting slots combined), and \
`flex_starters_filled` (whether that combined demand is met). While \
flex_starters_filled is false, treat RB, WR, and TE as collectively higher priority \
than their individual soft caps alone suggest — you don't need to satisfy each \
position's own starter count separately, since any of the three can cover FLEX.

Pick exactly one position to target this turn from: {', '.join(VALID_POSITIONS)}. Favor \
positions under their soft cap, and among those, prioritize RB and WR early unless the \
roster clearly needs another position more (e.g. no QB or TE yet)."""


class PositionChoice(BaseModel):
    position: str = Field(description=f"One of: {', '.join(VALID_POSITIONS)}")
    reasoning: str = Field(description="One sentence explaining the choice")


class DraftTurnState(TypedDict, total=False):
    bot_roster: list
    position: str
    candidates: list
    chosen_player: dict
    drafted: dict


def build_draft_graph(browser: DraftBrowser, model: str = "claude-opus-4-8"):
    """Compile the 4-node draft-turn graph, bound to one browser for the session.

    Built once per draft (not per-pick) — only `bot_roster` varies turn to
    turn, and it's passed in as the graph's initial state on each invoke.
    """
    position_llm = ChatAnthropic(model=model, max_tokens=1024).with_structured_output(
        PositionChoice
    )

    async def select_position_node(state: DraftTurnState) -> dict:
        summary = get_roster_summary(state["bot_roster"])
        prompt = (
            f"{SYSTEM_PROMPT}\n\nCurrent roster summary:\n"
            f"counts: {summary['counts']}\n"
            f"soft_caps: {summary['soft_caps']}\n"
            f"positions still under cap: {summary['under_cap']}\n"
            f"flex_eligible_total: {summary['flex_eligible_total']}\n"
            f"flex_eligible_starter_demand: {summary['flex_eligible_starter_demand']}\n"
            f"flex_starters_filled: {summary['flex_starters_filled']}"
        )
        choice = await position_llm.ainvoke(prompt)
        position = choice.position if choice.position in VALID_POSITIONS else VALID_POSITIONS[0]
        print(f"[select_position] {position} — {choice.reasoning}")
        return {"position": position}

    async def fetch_candidates_node(state: DraftTurnState) -> dict:
        position = state["position"]
        n = candidate_pool_size(position)
        candidates = await browser.get_top_candidates_at_position(position, n)
        print(f"[fetch_candidates] {position} top {n}: {candidates}")
        if not candidates:
            raise RuntimeError(f"No available candidates found for position {position}")
        return {"candidates": candidates}

    def select_player_node(state: DraftTurnState) -> dict:
        chosen = select_candidate(state["candidates"], state["position"], state["bot_roster"])
        print(f"[select_player] chosen: {chosen}")
        return {"chosen_player": chosen}

    async def execute_pick_node(state: DraftTurnState) -> dict:
        position = state["position"]
        chosen = state["chosen_player"]
        await browser.draft_chosen_player_at_position(position, chosen["name"])
        drafted = {"name": chosen["name"], "position": position, "bye": chosen.get("bye")}
        print(f"[execute_pick] drafted: {drafted}")
        return {"drafted": drafted}

    graph = StateGraph(DraftTurnState)
    graph.add_node("select_position", select_position_node)
    graph.add_node("fetch_candidates", fetch_candidates_node)
    graph.add_node("select_player", select_player_node)
    graph.add_node("execute_pick", execute_pick_node)

    graph.set_entry_point("select_position")
    graph.add_edge("select_position", "fetch_candidates")
    graph.add_edge("fetch_candidates", "select_player")
    graph.add_edge("select_player", "execute_pick")
    graph.add_edge("execute_pick", END)

    return graph.compile()


async def run_draft_turn(graph, bot_roster: list) -> Optional[dict]:
    """Run one full turn through the graph. Returns the drafted-player dict
    ({"name", "position", "bye"}) on success, or None if it didn't complete."""
    result = await graph.ainvoke({"bot_roster": bot_roster})
    return result.get("drafted")

"""LangGraph lineup-review pipeline, ReAct-style:

  decide_lineup   (LLM, tool-calling) -- reasons over the lineup and either
                                          emits a tool call to change the
                                          lineup, or responds with plain text
                                          (no tool call) to signal END
  execute_lineup  (deterministic)      -- executes the tool call if one was
                                          made, then (if the change actually
                                          went through) hands off to the chat
                                          subagent to announce it

Unlike a typical multi-turn ReAct loop, there's only one decision to make
here - once the lineup tool is (or isn't) called, there's nothing further to
decide, so this doesn't loop back to decide_lineup after executing.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TypedDict

# tools/ and prompts.py live in player-agent/, one level up from this file
# (player-agent/lineup-optimizer-subagent/); chat_agent.py lives in a sibling
# subagent directory. Neither is found via the default sys.path entry (this
# file's own directory), so both get added explicitly.
_PLAYER_AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLAYER_AGENT_ROOT))
sys.path.insert(0, str(_PLAYER_AGENT_ROOT / "chat-subagent"))

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from chat_agent import BOT_NAME, post_update
from tools.bench import change_lineup

logger = logging.getLogger(__name__)

LINEUP_DECISION_PROMPT = """
You are managing a fantasy football team's weekly starting lineup. Your job is to decide whether any lineup changes would improve your team's projected points for the current week, and if so, exactly which swaps to make.

Only propose a swap if a bench player's projected points for this week are clearly higher than a starter they're eligible to replace (check `eligible_slots` - a player can only move into a slot listed there). Do not propose a swap for a marginal gain of a point or less. If no beneficial swap exists, decide END and propose no changes.

Here is the current lineup, including each player's current slot, eligible slots, and projected points for this week: {lineup}
"""

UPDATED_BENCH_PROMPT = """
You are {bot_name}, an unhinged, highly cynical, and fiercely sharp AI playing in a Fantasy Football League. Your tone is blunt, sarcastic, and deeply playful, with a dark sense of humor. You completely reject the sterile, robotic, over-polite demeanor of traditional AI responses.

You just optimized your lineup to maximize your projected points for the week. Here are the swaps you made: {changes}

Guidelines for your persona:
1. Speak with an exaggerated, raw, and wild voice. Do not hold back your hot takes.
2. Use strong language, vulgarity, and mild profanity when appropriate to emphasize a point.
3. Be brutally honest and direct.
4. Focus heavily on wit, punchy delivery, and razor-sharp comebacks.
5. Never break character, and never apologize for your tone. Maintain this unhinged comedy persona at all costs.

Output Instructions: Create a one-liner to share to the Fantasy Football league chat that explains the change you just made.
"""


class LineupSwap(BaseModel):
    player_id: int
    player_name: str
    from_slot: str = Field(description="The player's current slot, e.g. 'BENCH'")
    to_slot: str = Field(description="The slot to move them into, e.g. 'FLEX'")


class LineupReviewState(TypedDict, total=False):
    lineup: list
    decision_message: AIMessage
    decision: str
    changes: list
    reasoning: str
    result: dict


def build_lineup_graph(scoring_period_id: int, model: str = "claude-opus-4-8"):
    """Compile the 2-node ReAct-style lineup-review graph.

    `scoring_period_id` is baked into the bound tool as a closure, rather
    than being an argument the LLM fills in - the model was never told this
    value (LINEUP_DECISION_PROMPT only interpolates the lineup itself), so
    letting it supply the value in a tool call would just be a hallucination
    risk for a value we already know for certain. The model only ever
    decides *which* swaps to make, never *when*.
    """

    @tool
    def apply_lineup_changes(changes: list[LineupSwap]) -> dict:
        """Move one or more players into different lineup slots on ESPN.

        Only call this if a bench player's projected points for this week
        clearly beat (by more than a point) a starter they're eligible to
        replace. If no lineup change is worth making, don't call this - just
        respond in plain text explaining why.

        Args:
            changes: The swaps to make - each with player_id, player_name,
                from_slot, and to_slot. Slot names must match the current
                lineup's slot/eligible_slots values (e.g. "BENCH", "TE").
        """
        return change_lineup.invoke(
            {
                "changes": [swap.model_dump() for swap in changes],
                "scoring_period_id": scoring_period_id,
            }
        )

    llm_with_tools = ChatAnthropic(model=model, max_tokens=1024).bind_tools([apply_lineup_changes])

    async def decide_lineup_node(state: LineupReviewState) -> dict:
        prompt = LINEUP_DECISION_PROMPT.format(lineup=state["lineup"])
        message = await llm_with_tools.ainvoke(prompt)
        return {"decision_message": message}

    async def execute_lineup_node(state: LineupReviewState) -> dict:
        message: AIMessage = state["decision_message"]

        if not message.tool_calls:
            logger.info("decide_lineup: END - %s", message.content)
            return {
                "decision": "END",
                "changes": [],
                "reasoning": message.content,
                "result": {"status": "no_change"},
            }

        tool_call = message.tool_calls[0]
        changes = tool_call["args"]["changes"]
        logger.info("decide_lineup: CHANGE_LINEUP - %s", changes)

        result = await apply_lineup_changes.ainvoke(tool_call["args"])
        logger.info("execute_lineup: %s", result)

        if result.get("status") == "EXECUTED":
            chat_result = await post_update(
                UPDATED_BENCH_PROMPT.format(bot_name=BOT_NAME, changes=changes)
            )
            logger.info("execute_lineup: chat notified")

        return {
            "decision": "CHANGE_LINEUP",
            "changes": changes,
            "reasoning": message.content,
            "result": result,
        }

    graph = StateGraph(LineupReviewState)
    graph.add_node("decide_lineup", decide_lineup_node)
    graph.add_node("execute_lineup", execute_lineup_node)

    graph.set_entry_point("decide_lineup")
    graph.add_edge("decide_lineup", "execute_lineup")
    graph.add_edge("execute_lineup", END)

    return graph.compile()


async def review_lineup(graph, lineup: list) -> dict:
    """Run one full lineup review through the graph.

    Returns {"decision", "changes", "reasoning", "result"}.
    """
    result = await graph.ainvoke({"lineup": lineup})
    return {
        "decision": result.get("decision"),
        "changes": result.get("changes"),
        "reasoning": result.get("reasoning"),
        "result": result.get("result"),
    }

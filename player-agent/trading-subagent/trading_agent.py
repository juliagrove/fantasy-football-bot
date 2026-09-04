"""LangGraph trade-review pipeline, ReAct-style:

  decide_trade   (LLM, tool-calling) -- reasons over the trade + roster and
                                          calls either accept_this_trade or
                                          reject_this_trade
  execute_trade  (deterministic)      -- executes whichever was called, then
                                          (if rejected) hands off to the chat
                                          subagent to announce it

Same shape as lineup-optimizer-subagent's lineup_agent.py. trade_id and
scoring_period_id are baked into the two bound tools as closures rather than
being LLM-supplied arguments - the model only ever decides ACCEPT vs REJECT,
it's never asked to (and can't) name which specific ESPN transaction to act
on, so it can't misfire against the wrong trade_id.

Unlike lineup_agent.py, there's no END outcome here - ACCEPT_REJECT_TRADE_PROMPT
always calls for one of the two decisions, never "do nothing."
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TypedDict

# tools/ and prompts.py live in player-agent/, one level up from this file
# (player-agent/trading-subagent/); chat_agent.py lives in a sibling subagent
# directory. Neither is found via the default sys.path entry (this file's
# own directory), so both get added explicitly.
_PLAYER_AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLAYER_AGENT_ROOT))
sys.path.insert(0, str(_PLAYER_AGENT_ROOT / "chat-subagent"))

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from chat_agent import BOT_NAME, post_update
from tools.trades import accept_trade, reject_trade

logger = logging.getLogger(__name__)

ACCEPT_REJECT_TRADE_PROMPT = """
You have been offered a trade in a fantasy football league. Your job is to either ACCEPT or REJECT the trade. The majority of the time, the trade will be rejected, however do not completely rule it out.

Based on your roster and the trade offered below, determine whether this trade will provide value to your team. You almost NEVER want to give up your star players.

Here is the trade: {trade}

Here is your current roster: {roster}
"""

REJECTED_TRADE_PROMPT = """
You are {bot_name}, an unhinged, highly cynical, and fiercely sharp AI playing in a Fantasy Football League. Your tone is blunt, sarcastic, and deeply playful, with a dark sense of humor. You completely reject the sterile, robotic, over-polite demeanor of traditional AI responses.
You just rejected a trade from {opponent_team}. The trade was {offered_trade}.

Guidelines for your persona:
1. Speak with an exaggerated, raw, and wild voice. Do not hold back your hot takes.
2. Use strong language, vulgarity, and mild profanity when appropriate to emphasize a point.
3. Be brutally honest and direct.
4. Focus heavily on wit, punchy delivery, and razor-sharp comebacks.
5. Never break character, and never apologize for your tone. Maintain this unhinged comedy persona at all costs.

Output Instructions: Create a one-liner to share to the Fantasy Football league chat that explains the trade you just rejected, and trash talk {opponent_team}.
"""


class TradeReviewState(TypedDict, total=False):
    trade: dict
    roster: list
    decision_message: AIMessage
    decision: str
    reasoning: str
    result: dict


def build_trade_graph(trade_id: str, scoring_period_id: int, model: str = "claude-opus-4-8"):
    """Compile the 2-node ReAct-style trade-review graph for one specific trade.

    trade_id/scoring_period_id are baked in as closures - the model only ever
    decides ACCEPT vs REJECT. Rebuild once per trade (see main.py's loop).
    """

    @tool
    def accept_this_trade() -> dict:
        """Accept the pending trade under review.

        Call this if the trade provides clear value to the team - you almost
        never want to give up star players.
        """
        return accept_trade.invoke({"trade_id": trade_id, "scoring_period_id": scoring_period_id})

    @tool
    def reject_this_trade() -> dict:
        """Reject the pending trade under review.

        Call this if the trade doesn't provide clear value - this is the
        majority-case decision.
        """
        return reject_trade.invoke({"trade_id": trade_id, "scoring_period_id": scoring_period_id})

    llm_with_tools = ChatAnthropic(model=model, max_tokens=1024).bind_tools(
        [accept_this_trade, reject_this_trade]
    )

    async def decide_trade_node(state: TradeReviewState) -> dict:
        prompt = ACCEPT_REJECT_TRADE_PROMPT.format(trade=state["trade"], roster=state["roster"])
        message = await llm_with_tools.ainvoke(prompt)
        return {"decision_message": message}

    async def execute_trade_node(state: TradeReviewState) -> dict:
        message: AIMessage = state["decision_message"]

        if not message.tool_calls:
            logger.error("decide_trade: model made no decision - %s", message.content)
            return {"decision": "ERROR", "reasoning": message.content, "result": {"status": "no_decision"}}

        tool_call = message.tool_calls[0]
        if tool_call["name"] == "accept_this_trade":
            decision = "ACCEPT"
            result = await accept_this_trade.ainvoke({})
        else:
            decision = "REJECT"
            result = await reject_this_trade.ainvoke({})

        logger.info("execute_trade: %s trade %s -> %s", decision, trade_id, result)

        if decision == "REJECT" and result.get("status") == "EXECUTED":
            chat_result = await post_update(
                REJECTED_TRADE_PROMPT.format(
                    bot_name=BOT_NAME,
                    offered_trade=state["trade"],
                    opponent_team=state["trade"]["proposed_by_team"],
                )
            )
            logger.info("execute_trade: chat notified -> %s", chat_result)

        return {"decision": decision, "reasoning": message.content, "result": result}

    graph = StateGraph(TradeReviewState)
    graph.add_node("decide_trade", decide_trade_node)
    graph.add_node("execute_trade", execute_trade_node)

    graph.set_entry_point("decide_trade")
    graph.add_edge("decide_trade", "execute_trade")
    graph.add_edge("execute_trade", END)

    return graph.compile()


async def review_trade(graph, trade: dict, roster: list) -> dict:
    """Run one full trade through the graph.

    Returns {"decision", "reasoning", "result"}.
    """
    result = await graph.ainvoke({"trade": trade, "roster": roster})
    return {
        "decision": result.get("decision"),
        "reasoning": result.get("reasoning"),
        "result": result.get("result"),
    }

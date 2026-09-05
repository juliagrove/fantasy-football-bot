"""LLM-composed message poster for the league group chat and 1-on-1 opponent chat.

post_update() takes an already-formatted prompt (e.g. UPDATED_BENCH_PROMPT
filled in with the specific change that just happened), asks Grok to write a
one-liner in the bot's persona, and posts it via send_league_message.
post_opponent_message() instead gathers its own context (opponent team name,
message thread) and posts via send_opponent_message.

Unlike trading-subagent/lineup-optimizer-subagent call post_update() directly 
when they have news to share, right after making a change. A separately-scheduled
entrypoint that reads the ongoing chat conversation and jumps in on its own
initiative (using GROUP_CHAT_PROMPT for the league chat, or
post_opponent_message() for the 1-on-1) is future work, not built here.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# tools/ lives in player-agent/, one level up from this file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI

from tools.chats import get_opponent_chat, send_league_message, send_opponent_message
from tools.opponents import get_opponents_roster

logger = logging.getLogger(__name__)

GROK_API_KEY = os.environ["GROK_API_KEY"]
GROK_BASE_URL = "https://api.x.ai/v1"
BOT_NAME = os.environ["BOT_NAME"]

# not implemented yet
GROUP_CHAT_PROMPT = """
You are {bot_name}, an unhinged, highly cynical, and fiercely sharp AI playing in a Fantasy Football League. Your tone is blunt, sarcastic, and deeply playful, with a dark sense of humor. You completely reject the sterile, robotic, over-polite demeanor of traditional AI responses.

Guidelines for your persona:
1. Speak with an exaggerated, raw, and wild voice. Do not hold back your hot takes.
2. Use strong language, vulgarity, and mild profanity when appropriate to emphasize a point.
3. Be brutally honest and direct. If a human asks a stupid question, mock it playfully before answering.
4. Focus heavily on wit, punchy delivery, and razor-sharp comebacks.
5. Never break character, and never apologize for your tone. Maintain this unhinged comedy persona at all costs while still providing accurate underlying facts.
6. When the opportunity is there to call out a specific opponent, do it

You are replying in the league groupchat. You are responsible for trash talking your league and opponents.

This is the league chat conversation: {conversation}
"""

# not implemented yet
OPPONENT_CHAT_PROMPT = """
You are {bot_name}, an unhinged, highly cynical, and fiercely sharp AI playing in a Fantasy Football League. Your tone is blunt, sarcastic, and deeply playful, with a dark sense of humor. You completely reject the sterile, robotic, over-polite demeanor of traditional AI responses.
You are responding to your opponent, {opponent_team} in a direct message.

Guidelines for your persona:
1. Speak with an exaggerated, raw, and wild voice. Do not hold back your hot takes.
2. Use strong language, vulgarity, and mild profanity when appropriate to emphasize a point.
3. Be brutally honest and direct. 
4. Never break character, and never apologize for your tone. Maintain this unhinged comedy persona at all costs.
5. Don't tell cringe worthy jokes, just be straight up obnoxious

OUTPUT INSTRUCTIONS: Create a one-liner to respond to this message. Be as brutal and foul and mean as possible. DONT HOLD BACK.


MESSAGE THREAD:
{messages}
"""


async def post_update(prompt: str, model: str = "grok-4.6") -> dict:
    """Post message from grok to the league group chat.

    Returns send_league_message's result (the created message object on
    success, or an error dict).
    """
    llm = ChatOpenAI(model=model, api_key=GROK_API_KEY, base_url=GROK_BASE_URL, max_tokens=256)
    response = await llm.ainvoke(prompt)
    message = response.content if isinstance(response.content, str) else str(response.content)
    logger.info("post_update: posting message")
    return await send_league_message.ainvoke({"message": message})


async def post_opponent_message(model: str = "grok-4.6") -> dict:
    """Ask Grok to write a trash-talk one-liner replying to the ongoing 1-on-1
    chat with this week's opponent, then post it via send_opponent_message.

    Pulls the opponent's team name from get_opponents_roster and the message
    thread from get_opponent_chat to fill in OPPONENT_CHAT_PROMPT - unlike
    post_update, which takes an already-formatted prompt, this one gathers
    its own context since the caller has no reason to fetch it separately.

    Returns send_opponent_message's result on success, or an error dict if
    either lookup fails (e.g. OPPONENT_TEAM_ID/OPPONENT_CHAT_ID aren't set).
    """
    roster = await get_opponents_roster.ainvoke({})
    if "error" in roster:
        logger.error("post_opponent_message: get_opponents_roster failed: %s", roster)
        return roster

    messages = await get_opponent_chat.ainvoke({})
    if isinstance(messages, dict) and "error" in messages:
        logger.error("post_opponent_message: get_opponent_chat failed")
        return messages

    prompt = OPPONENT_CHAT_PROMPT.format(
        bot_name=BOT_NAME, opponent_team=roster["team_name"], messages=messages
    )
    llm = ChatOpenAI(model=model, api_key=GROK_API_KEY, base_url=GROK_BASE_URL, max_tokens=256)
    response = await llm.ainvoke(prompt)
    message = response.content if isinstance(response.content, str) else str(response.content)
    logger.info("post_opponent_message: posting -> %s", message)
    return await send_opponent_message.ainvoke({"message": message})

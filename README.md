# fantasy-football-bot

My family and I needed one more person to fill our Fantasy Football league. Instead of finding someone outside the family, I built this "bot" to fill the 10th seat — a fully autonomous AI agent that plays a full Fantasy Football season on its own.

## What it does

Key agentic features include:
- Drafting the team live
- Accepting / rejecting trades
- Lineup optimization
- Waiver pickup (in progress)
- Posting to the chat (in progress)

My personal favorite: the chat agent has "trash talk" capabilities, mimicking the persona of Tesla's "Grok Unhinged" mode.

## How it works

Running on a cron schedule, the bot is triggered to check its roster, chats, lineup, etc. to determine whether it needs to make any changes or not.

1. **Draft agent** — run once, by hand, right before the live draft. It drives a real browser session against ESPN's draft room and picks players in real time.
2. **Player agent** — runs on a GitHub Actions cron schedule for the rest of the season, made up of three subagents:
   - `trading-subagent` — reviews incoming trade offers and accepts/rejects them
   - `lineup-optimizer-subagent` — sets the weekly starting lineup
   - `chat-subagent` — posts trash talk to the league chat

Each subagent runs as a LangGraph state machine (e.g. pick a position → fetch candidates → select a player → execute the pick), reading and writing league state through ESPN's private Fantasy API.

## Tech Stack

- **Python**
- **LangGraph** — powers each agent's decision-making as a graph of nodes
- **LangChain** — Claude drives drafting, trading, and lineup decisions; Grok drives the trash-talk chat agent
- **LangSmith** — tracing and eval for agent runs
- **Playwright** — browser automation for the draft agent
- **ESPN Fantasy API** — undocumented private-league API (cookie-based auth) for rosters, trades, matchups, and league chat
- **GitHub Actions** — cron-scheduled workflows trigger the trading and lineup agents throughout the season

## Architecture
_(in progress)_

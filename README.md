# fantasy-football-bot

My family and I needed one more person to fill our Fantasy Football league. Instead of finding someone outside the family, I built this "bot" to fill the 10th seat. 

### This project is a fully autonomous AI agent that runs a Fantasy Football season.

Key agentic features include:
- Drafting the team live
- Accepting / rejecting trades
- Lineup optimization
- Waiver pickup (in progress)
- Chatting (in progress)

My personal favorite:
- The chatting agent has "trash talk" capabilities, mimicking the persona of Tesla's "Grok Unhinged" mode

### How does it work?
Running on a cron schedule, the bot is triggered to check its roster, chats, lineup, etc. to determine whether it needs to make any changes or not.

## Tech Stack
- **Python**
- **LangGraph** — powers each agent's decision-making as a graph of nodes (e.g. pick a position → fetch candidates → select a player → execute the pick)
- **LangChain** — Claude drives drafting, trading, and lineup decisions; Grok drives the trash-talk chat agent
- **LangSmith** — tracing and eval for agent runs
- **Playwright** — browser automation for the draft agent
- **ESPN Fantasy API** — undocumented private-league API (cookie-based auth) for rosters, trades, matchups, and league chat
- **GitHub Actions** — cron-scheduled workflows trigger the trading and lineup agents throughout the season

## Architecture
_(in progress)_

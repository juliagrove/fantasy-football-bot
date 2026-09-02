"""Roster-construction and pick-selection logic.

Player valuation itself is not done locally — the draft agent reads ESPN's
own live, position-filtered player list (already ranked by ESPN's default
projections) directly from the browser instead of a static CSV. This module
covers what happens with that ranked list: which position to target, and
which of the top few ranked players to actually take.
"""
from __future__ import annotations

import random

# This league's starting lineup (from ESPN League Settings -> Roster).
# FLEX can be filled by an RB, WR, or TE.
STARTING_LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "D/ST": 1}
FLEX_ELIGIBLE_POSITIONS = {"RB", "WR", "TE"}
# Total RB/WR/TE players needed to fill the RB, WR, TE, and FLEX starting
# slots combined (2 + 2 + 1 + 1 = 6) -- not any single position's own cap.
FLEX_ELIGIBLE_STARTER_DEMAND = sum(
    STARTING_LINEUP[p] for p in FLEX_ELIGIBLE_POSITIONS
) + STARTING_LINEUP["FLEX"]

# Bench depth carried beyond the starting lineup, per position. RB/WR get
# more since they're the most FLEX-eligible and injury-prone; K/D-ST get
# none since a backup kicker/defense isn't worth a roster spot.
BENCH_DEPTH = {"QB": 1, "RB": 3, "WR": 3, "TE": 1, "K": 0, "D/ST": 0}

# Soft roster caps used to avoid over-drafting a position early (e.g. a 3rd QB
# before starters elsewhere are filled) -- derived from the lineup + bench
# depth above, not arbitrary. `VALID_POSITIONS` doubles as the set of labels
# ESPN's position filter dropdown accepts.
POSITION_SOFT_CAPS = {
    position: STARTING_LINEUP[position] + BENCH_DEPTH[position]
    for position in ("QB", "RB", "WR", "TE", "K", "D/ST")
}
VALID_POSITIONS = list(POSITION_SOFT_CAPS)

# How many ranked candidates to consider for a pick. QB gets a narrower pool
# since backup-tier QBs drop off in value faster than RB/WR/TE; K/D-ST never
# use their pool for randomness (see select_candidate) so 1 is enough.
POSITION_CANDIDATE_POOL = {"QB": 3, "RB": 4, "WR": 4, "TE": 4, "K": 1, "D/ST": 1}

# Roster size (picks already made) at which point selection stops always
# taking the top-ranked candidate and starts allowing some risk/variance —
# i.e. after the first 4 rounds.
RANDOMNESS_START_ROUND = 4

# Positions that always take the top-ranked candidate regardless of round —
# there's no meaningful upside/downside tradeoff worth "risking" here.
NO_RANDOMNESS_POSITIONS = {"K", "D/ST"}

# Weight subtracted per roster player already on a candidate's bye week, to
# discourage (but not outright forbid) stacking byes at randomness-eligible
# picks. Weights are floored above zero so a clearly-best candidate is never
# fully excluded just for a bye collision.
BYE_COLLISION_PENALTY = 2
MIN_CANDIDATE_WEIGHT = 0.5


def get_roster_summary(bot_roster: list) -> dict:
    """Return the bot's current position counts, soft caps, and which positions
    still have room under their cap (i.e. are reasonable to target next)."""
    counts = {position: 0 for position in VALID_POSITIONS}
    for _, position, _ in bot_roster:
        counts[position] = counts.get(position, 0) + 1

    under_cap = [p for p in VALID_POSITIONS if counts[p] < POSITION_SOFT_CAPS[p]]

    # FLEX can be filled by RB, WR, or TE, so their combined count -- not any
    # one position's own soft cap -- determines whether that shared demand
    # (RB + WR + TE + FLEX starting slots) is satisfied.
    flex_eligible_total = sum(counts[p] for p in FLEX_ELIGIBLE_POSITIONS)

    return {
        "counts": counts,
        "soft_caps": POSITION_SOFT_CAPS,
        # Positions still worth targeting. If every position is at/over its
        # soft cap, fall back to all positions rather than an empty list.
        "under_cap": under_cap or VALID_POSITIONS,
        "flex_eligible_total": flex_eligible_total,
        "flex_eligible_starter_demand": FLEX_ELIGIBLE_STARTER_DEMAND,
        "flex_starters_filled": flex_eligible_total >= FLEX_ELIGIBLE_STARTER_DEMAND,
    }


def candidate_pool_size(position: str) -> int:
    return POSITION_CANDIDATE_POOL.get(position, 5)


def select_candidate(candidates: list, position: str, bot_roster: list) -> dict:
    """Pick one candidate ({"name", "bye"}) from the ranked list ESPN returned.

    Rounds 1-4, and K/D-ST at any round, always take the top-ranked
    candidate. From round 5 on (for other positions), do a weighted-random
    pick across the candidate pool, favoring higher rank and penalizing
    candidates whose bye week collides with players already on the roster.
    """
    if not candidates:
        raise ValueError("No candidates to select from")

    picks_made = len(bot_roster)
    if position in NO_RANDOMNESS_POSITIONS or picks_made < RANDOMNESS_START_ROUND:
        return candidates[0]

    existing_byes = [bye for _, _, bye in bot_roster if bye is not None]

    weights = []
    for rank_index, candidate in enumerate(candidates):
        weight = len(candidates) - rank_index  # rank 1 (index 0) gets the highest weight
        bye = candidate.get("bye")
        if bye is not None:
            weight -= BYE_COLLISION_PENALTY * existing_byes.count(bye)
        weights.append(max(weight, MIN_CANDIDATE_WEIGHT))

    return random.choices(candidates, weights=weights, k=1)[0]

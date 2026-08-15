"""Kaggriculture submission entry point.

Kaggle loads this file and calls `agent(obs)` (or `agent(obs, config)`) once
per turn for whichever player index we've been assigned. Keep this file
self-contained (or bundle extra modules into a tar.gz with this at the root
per AGENTS.md) since it's shipped standalone to the competition runner.

Strategy (v0 — "wheat loop"): buy wheat seeds, water, harvest, sell, repeat
on the farmer's current tile. This is intentionally the simplest possible
profitable loop; see data/README.md and data/AGENTS.md for the full rules
and ideas for a stronger policy (expanding land, hiring hands, animals,
higher-value crops, market timing).
"""

from __future__ import annotations

WHEAT_SEED_COST = 10
WHEAT_FIRST_YIELD_DAY = 2


def agent(obs, config=None):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]

    market: list[list] = []

    seeds = private["seeds"].get("WHEAT", 0)
    if seeds == 0 and me["money"] >= WHEAT_SEED_COST:
        market.append(["BUY_SEED", "WHEAT", 1])

    wheat_in_shed = private["shed"].get("WHEAT", 0)
    if wheat_in_shed > 0:
        market.append(["SELL", "WHEAT", wheat_in_shed])

    farmer_op = ["PASS"]
    if tile is None and seeds > 0:
        farmer_op = ["PLANT", "WHEAT"]
    elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop_age = obs["day"] - tile["planted_day"]
        if crop_age >= WHEAT_FIRST_YIELD_DAY:
            farmer_op = ["HARVEST"]
        elif not tile["watered_today"]:
            farmer_op = ["WATER"]

    return {"farmer": farmer_op, "hands": [], "market": market}

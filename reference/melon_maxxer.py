"""Melon Maxxer -- reference scripted agent from Kaggle's official
"kaggriculture-getting-started" notebook (kaggle.com/code/bovard/
kaggriculture-getting-started, owner: bovard), downloaded via
`kaggle kernels pull bovard/kaggriculture-getting-started` -- source and
metadata for the full notebook are in reference/kaggriculture-getting-started/.

Grows and sells only MELON on a single tile: buy a seed whenever out of stock
and affordable, walk to/plant/water/harvest the nearest melon tile, hold sales
until the market price clears SELL_THRESHOLD (dumps the entire shed in one
order once it does). Confined to the starting NW quadrant -- never hires
hands or buys land.

Documented weaknesses (per the notebook itself, unmodified here): never
expands land or hires hands, only ever grows one crop so it has nothing to
sell when other prices are better, never fertilizes (leaves a yield bonus on
the table), and dumps its entire shed in a single SELL order which can crash
the price mid-sale (data/README.md: price moves unit-by-unit within an
order). Useful as a second, more competent (than `random`) but still clearly
beatable scripted opponent alongside `starter`, for win-rate evaluation
across a field instead of a single baseline -- see rl/README.md's Guiding
Principles ("rank by win rate against a field, never margin against one
baseline").

Usage: same `.py`-agent-file convention as main.py / checkpoints/v0/main.py --
pass this path directly as --opponent to rl/train.py, rl/play.py,
rl/analyze_actions.py, or rl/noop_audit.py, or as --p1/--p2 to
scripts/run_local.py.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

MELON_SEED_COST = CROPS["MELON"]["seed"]
MELON_MAX_YIELD_DAY = CROPS["MELON"]["max_yield_day"]
SELL_THRESHOLD = 200


def _step_toward(fx, fy, tx, ty):
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"
    return None


def _find_target_tile(farm, board_size, have_seed):
    fx, fy = farm["farmer"]
    candidates = []
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile["crop"] == "MELON":
                purpose = None
                # Harvest if ripe.
                age_ok = tile["yield_units"] > 0
                if age_ok and tile.get("planted_day") is not None:
                    purpose = "harvest"
                if not tile["watered_today"]:
                    purpose = "water" if purpose is None else purpose
                if purpose:
                    candidates.append((x, y, purpose))
            elif tile is None and have_seed:
                candidates.append((x, y, "plant"))

    if not candidates:
        return None

    priority = {"harvest": 0, "water": 1, "plant": 2}
    candidates.sort(key=lambda c: (priority[c[2]], abs(c[0] - fx) + abs(c[1] - fy)))
    return candidates[0]


def agent(obs):
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    board_size = len(farm["tiles"])
    fx, fy = farm["farmer"]
    tile = farm["tiles"][fy][fx]
    day = obs.get("day", 0)

    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    market_prices = (obs.get("market", {}) or {}).get("prices", {})
    melon_price = market_prices.get("MELON", 0)

    market = []

    # Sell melons only when the market is paying enough.
    melons_in_shed = shed.get("MELON", 0)
    if melons_in_shed > 0 and melon_price >= SELL_THRESHOLD:
        market.append(["SELL", "MELON", melons_in_shed])

    # Top up seed inventory so the next empty tile can be planted.
    if seeds.get("MELON", 0) == 0 and farm["money"] >= MELON_SEED_COST:
        market.append(["BUY_SEED", "MELON", 1])

    # Decide farmer action.
    farmer = ["PASS"]

    if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile["crop"] == "MELON":
        age = day - tile["planted_day"]
        if age >= MELON_MAX_YIELD_DAY and tile["yield_units"] > 0:
            farmer = ["HARVEST"]
        elif not tile["watered_today"]:
            farmer = ["WATER"]
        else:
            target = _find_target_tile(farm, board_size, seeds.get("MELON", 0) > 0)
            if target:
                step = _step_toward(fx, fy, target[0], target[1])
                if step:
                    farmer = [step]
    elif tile is None and seeds.get("MELON", 0) > 0:
        farmer = ["PLANT", "MELON"]
    else:
        target = _find_target_tile(farm, board_size, seeds.get("MELON", 0) > 0)
        if target:
            step = _step_toward(fx, fy, target[0], target[1])
            if step:
                farmer = [step]

    return {"farmer": farmer, "hands": [], "market": market}

"""Observation encoding shared by the RL env, the training script, and self-play
opponents (a frozen SB3 checkpoint acting as the opponent needs the exact same
encoder the policy trained with).

This is training-time infra, not the submission artifact -- main.py stays a
plain, dependency-free `agent(obs)` function. A trained policy gets exported to
something main.py can ship separately, once there's a checkpoint worth
submitting (see rl/README.md).
"""

from __future__ import annotations

import numpy as np

BOARD_SIZE = 10

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]

# All sellable products, per the Object Types / Price Function tables in data/README.md.
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

BASE_PRICE = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
    "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
}
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
MARKET_I0 = 10_000  # starting inventory every product is calibrated against

# The four tiles PICKUP/DROP require standing on -- confirmed in kaggriculture.py:
# _shed_access_tiles() -> [(half-1,half-1), (half,half-1), (half-1,half), (half,half)]
# for half = board_size // 2, and both ops silently no-op via _is_shed_adjacent()
# before doing anything else if the farmer isn't on one of them. Nothing in the
# observation flagged this before noop_audit.py found PICKUP/DROP failing ~100% of
# the time in practice -- see rl/README.md.
SHED_ADJACENT_TILES = {
    (BOARD_SIZE // 2 - 1, BOARD_SIZE // 2 - 1), (BOARD_SIZE // 2, BOARD_SIZE // 2 - 1),
    (BOARD_SIZE // 2 - 1, BOARD_SIZE // 2), (BOARD_SIZE // 2, BOARD_SIZE // 2),
}

# Per-tile channels: locked, empty, weed, is_plant, crop one-hot (5), watered_today,
# yield_units (normalized), fertilized, farmer-is-here.
# v1 has no BUILD_COOP/BUILD_PASTURE/PLACE ops, so animal structures never appear on
# a farm we control and get no dedicated channel yet -- see rl/README.md.
GRID_CHANNELS = 4 + len(CROPS) + 4

# 6 scalars + 1 (near-shed flag) + seed counts + PRODUCTS x{shed, held-in-inventory,
# market price, market inventory} -- see _encode_vector.
VECTOR_DIM = 7 + len(CROPS) + len(PRODUCTS) * 4


def _encode_farm_grid(farm) -> np.ndarray:
    grid = np.zeros((GRID_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if tile is None:
                grid[1, y, x] = 1.0
            elif tile == "LOCKED":
                grid[0, y, x] = 1.0
            else:
                kind = tile.get("kind")
                if kind == "WEED":
                    grid[2, y, x] = 1.0
                elif kind == "PLANT":
                    grid[3, y, x] = 1.0
                    crop = tile.get("crop")
                    if crop in CROPS:
                        grid[4 + CROPS.index(crop), y, x] = 1.0
                    grid[9, y, x] = 1.0 if tile.get("watered_today") else 0.0
                    grid[10, y, x] = min(tile.get("yield_units", 0) / 6.0, 2.0)
                    grid[11, y, x] = 1.0 if tile.get("fertilized_until_day", -1) >= 0 else 0.0
                # COOP/PASTURE: no channel yet (see module docstring).
    fx, fy = farm["farmer"]
    grid[12, fy, fx] = 1.0
    return grid


def _encode_vector(obs) -> np.ndarray:
    player = obs["player"]
    opponent = 1 - player
    me = obs["farms"][player]
    opp = obs["farms"][opponent]
    private = obs["private"]
    market = obs["market"]

    parts = [
        me["money"] / 1000.0,
        opp["money"] / 1000.0,
        obs["day"] / 30.0,
        obs["hour"] / 24.0,
        len(me.get("unlocked_quadrants", [])) / 4.0,
        len(obs["town"].get("unlocked_shops", [])) / 8.0,
        1.0 if tuple(me["farmer"]) in SHED_ADJACENT_TILES else 0.0,
    ]
    seeds = private["seeds"]
    parts.extend(seeds.get(c, 0) / 10.0 for c in CROPS)
    shed = private["shed"]
    parts.extend(shed.get(p, 0) / 50.0 for p in PRODUCTS)
    # HARVEST lands produce in the acting unit's personal inventory, not the shed --
    # it only moves to the shed (and becomes sellable) at end-of-day (see rl/env.py's
    # _potential docstring). Without this, the policy can't see what it just
    # harvested until that transfer happens, up to turnsPerDay-1 turns later.
    inventories = private.get("inventories", [])
    inv_totals: dict[str, int] = {}
    for inv in inventories:
        for item, count in inv.items():
            inv_totals[item] = inv_totals.get(item, 0) + count
    parts.extend(inv_totals.get(p, 0) / 50.0 for p in PRODUCTS)
    prices = market["prices"]
    parts.extend(prices.get(p, BASE_PRICE[p]) / BASE_PRICE[p] for p in PRODUCTS)
    inventory = market["inventory"]
    parts.extend(inventory.get(p, MARKET_I0) / MARKET_I0 for p in PRODUCTS)
    return np.array(parts, dtype=np.float32)


def encode_obs(obs) -> dict[str, np.ndarray]:
    """Encode a raw Kaggriculture observation into the RL env's Dict observation space.

    Symmetric in `obs["player"]` -- works whether called for the agent under
    training (usually player 0) or for a frozen checkpoint acting as the
    opponent (player 1), since it only ever reads `obs["player"]` to figure out
    which side is "own" vs "opp".
    """
    player = obs["player"]
    opponent = 1 - player
    return {
        "own_grid": _encode_farm_grid(obs["farms"][player]),
        "opp_grid": _encode_farm_grid(obs["farms"][opponent]),
        "vector": _encode_vector(obs),
    }

"""Observation encoding shared by the RL env, the training script, and self-play"""

from __future__ import annotations

import numpy as np

BOARD_SIZE = 10

MAX_HANDS = 6

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

BASE_PRICE = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
    "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
}
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
MARKET_I0 = 10_000

SHED_ADJACENT_TILES = {
    (BOARD_SIZE // 2 - 1, BOARD_SIZE // 2 - 1), (BOARD_SIZE // 2, BOARD_SIZE // 2 - 1),
    (BOARD_SIZE // 2 - 1, BOARD_SIZE // 2), (BOARD_SIZE // 2, BOARD_SIZE // 2),
}

GRID_CHANNELS = 4 + len(CROPS) + 5

HAND_VECTOR_DIM = 5

VECTOR_DIM = 7 + len(CROPS) + len(PRODUCTS) * 4 + MAX_HANDS * HAND_VECTOR_DIM


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
    for hx, hy in farm.get("hands", []):
        grid[GRID_CHANNELS - 2, hy, hx] = 1.0
    fx, fy = farm["farmer"]
    grid[GRID_CHANNELS - 1, fy, fx] = 1.0
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


def _encode_hands_vector(obs) -> np.ndarray:
    """Per-hand-slot features for our own hands, MAX_HANDS slots fixed regardless"""
    player = obs["player"]
    farm = obs["farms"][player]
    hands = farm.get("hands", [])
    inventories = obs["private"].get("inventories", [])
    parts = []
    for i in range(MAX_HANDS):
        if i < len(hands):
            hx, hy = hands[i]
            inv = inventories[i + 1] if i + 1 < len(inventories) else {}
            inv_count = sum(inv.values()) if isinstance(inv, dict) else 0
            near_shed = 1.0 if (hx, hy) in SHED_ADJACENT_TILES else 0.0
            parts.extend([hx / BOARD_SIZE, hy / BOARD_SIZE, 1.0, inv_count / 20.0, near_shed])
        else:
            parts.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    return np.array(parts, dtype=np.float32)


def encode_obs(obs) -> dict[str, np.ndarray]:
    """Encode a raw Kaggriculture observation into the RL env's Dict observation space."""
    player = obs["player"]
    opponent = 1 - player
    return {
        "own_grid": _encode_farm_grid(obs["farms"][player]),
        "opp_grid": _encode_farm_grid(obs["farms"][opponent]),
        "vector": np.concatenate([_encode_vector(obs), _encode_hands_vector(obs)]),
    }

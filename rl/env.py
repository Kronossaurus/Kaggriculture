"""Gymnasium wrapper around Kaggriculture, for RL training via kaggle_environments'"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make

from .features import (
    BASE_PRICE,
    BOARD_SIZE,
    CROPS,
    GRID_CHANNELS,
    MAX_HANDS,
    PRODUCTS,
    SEED_COST,
    VECTOR_DIM,
    encode_obs,
)

FARMER_OPS = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST",
    "WATER", "HARVEST", "FERTILIZE", "DIG", "PICKUP", "DROP",
    *(f"PLANT_{c}" for c in CROPS),
]

BUY_SEED_OPS = ["NONE", *CROPS]
BUY_PRODUCT_OPS = ["NONE", "FERTILIZER"]
BUY_LAND_OPS = ["NONE", "BUY_LAND"]
SELL_OPS = ["NONE", *PRODUCTS]

HIRE_OPS = ["NONE", "HIRE"]

OpponentSpec = str | Callable[[dict], dict]


def _decode_unit_op(op: str) -> list:
    """Turn one FARMER_OPS name into the Kaggriculture action-list form for a"""
    if op.startswith("PLANT_"):
        return ["PLANT", op.split("_", 1)[1]]
    if op == "PICKUP":
        return ["PICKUP", "FERTILIZER", 1]
    return [op]


def decode_action(action, obs) -> dict:
    """Turn a (farmer_op_idx, buy_seed_idx, buy_product_idx, buy_land_idx,"""
    farmer_idx, buy_seed_idx, buy_product_idx, buy_land_idx, sell_idx, hire_idx, *hand_idxs = (
        int(v) for v in action
    )
    private = obs["private"]
    market: list[list] = []

    if buy_seed_idx > 0:
        market.append(["BUY_SEED", BUY_SEED_OPS[buy_seed_idx], 1])
    if buy_product_idx > 0:
        market.append(["BUY_PRODUCT", BUY_PRODUCT_OPS[buy_product_idx], 1])
    if buy_land_idx > 0:
        market.append(["BUY_LAND"])
    if sell_idx > 0:
        item = SELL_OPS[sell_idx]
        qty = private["shed"].get(item, 0)
        if qty > 0:
            market.append(["SELL", item, qty])
    if hire_idx > 0:
        market.append(["HIRE"])

    farmer_op = _decode_unit_op(FARMER_OPS[farmer_idx])
    hands = [_decode_unit_op(FARMER_OPS[i]) for i in hand_idxs]

    return {"farmer": farmer_op, "hands": hands, "market": market[:10]}


def _encode_unit_op(unit_action) -> int:
    """Turn one unit's ['OP', ...] action list (farmer or a hand slot) back"""
    if not unit_action:
        return FARMER_OPS.index("PASS")
    op = unit_action[0]
    if op == "PLANT" and len(unit_action) > 1 and f"PLANT_{unit_action[1]}" in FARMER_OPS:
        return FARMER_OPS.index(f"PLANT_{unit_action[1]}")
    if op in FARMER_OPS:
        return FARMER_OPS.index(op)
    return FARMER_OPS.index("PASS")


def encode_action(action_dict: dict, obs) -> np.ndarray:
    """Reverse of decode_action: turn a full Kaggriculture action dict (as a"""
    farmer_idx = _encode_unit_op(action_dict.get("farmer"))

    buy_seed_idx, buy_product_idx, buy_land_idx, sell_idx, hire_idx = 0, 0, 0, 0, 0
    for order in action_dict.get("market", []):
        kind = order[0] if order else None
        if kind == "BUY_SEED" and buy_seed_idx == 0 and len(order) > 1 and order[1] in BUY_SEED_OPS:
            buy_seed_idx = BUY_SEED_OPS.index(order[1])
        elif kind == "BUY_PRODUCT" and buy_product_idx == 0 and len(order) > 1 and order[1] in BUY_PRODUCT_OPS:
            buy_product_idx = BUY_PRODUCT_OPS.index(order[1])
        elif kind == "BUY_LAND" and buy_land_idx == 0:
            buy_land_idx = BUY_LAND_OPS.index("BUY_LAND")
        elif kind == "SELL" and sell_idx == 0 and len(order) > 1 and order[1] in SELL_OPS:
            sell_idx = SELL_OPS.index(order[1])
        elif kind == "HIRE" and hire_idx == 0:
            hire_idx = HIRE_OPS.index("HIRE")

    hands_actions = action_dict.get("hands") or []
    hand_idxs = [_encode_unit_op(hands_actions[i] if i < len(hands_actions) else None) for i in range(MAX_HANDS)]

    return np.array(
        [farmer_idx, buy_seed_idx, buy_product_idx, buy_land_idx, sell_idx, hire_idx, *hand_idxs], dtype=np.int64
    )


class KaggricultureEnv(gym.Env):
    """Single-agent Gymnasium view of Kaggriculture: our policy is player 0,"""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    GAMMA = 0.99

    IDLE_PENALTY = 0.02

    NEGLECT_PENALTY = 0.02

    RELATIVE_OUTCOME_SCALE = 1.0

    def __init__(self, opponent: OpponentSpec = "starter", episode_steps: int = 720, seed: int | None = None):
        super().__init__()
        self.opponent = opponent
        config: dict[str, Any] = {"episodeSteps": episode_steps}
        if seed is not None:
            config["seed"] = seed
        self._kaggle_env = make("kaggriculture", configuration=config)
        self._trainer = None
        self._last_obs = None
        self._prev_money = 0.0
        self._prev_potential = 0.0

        self.action_space = spaces.MultiDiscrete(
            [
                len(FARMER_OPS), len(BUY_SEED_OPS), len(BUY_PRODUCT_OPS), len(BUY_LAND_OPS), len(SELL_OPS),
                len(HIRE_OPS), *([len(FARMER_OPS)] * MAX_HANDS),
            ]
        )
        self.observation_space = spaces.Dict(
            {
                "own_grid": spaces.Box(0.0, np.inf, shape=(GRID_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                "opp_grid": spaces.Box(0.0, np.inf, shape=(GRID_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                "vector": spaces.Box(0.0, np.inf, shape=(VECTOR_DIM,), dtype=np.float32),
            }
        )

    @staticmethod
    def _own_money(raw_obs) -> float:
        return float(raw_obs["farms"][raw_obs["player"]]["money"])

    @staticmethod
    def _opponent_money(raw_obs) -> float:
        return float(raw_obs["farms"][1 - raw_obs["player"]]["money"])

    @staticmethod
    def _potential(raw_obs) -> float:
        """Shaping potential Phi(s): rough resale value of stuff we're holding that"""
        prices = raw_obs["market"]["prices"]
        day = raw_obs["day"]

        farm = raw_obs["farms"][raw_obs["player"]]
        total = 0.0
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    crop = tile.get("crop")
                    if crop in BASE_PRICE:
                        standing_value = tile.get("yield_units", 0) * prices.get(crop, BASE_PRICE[crop])
                        total += max(standing_value, SEED_COST.get(crop, 0))
                        if tile.get("fertilized_until_day", -1) >= day:
                            total += prices.get("FERTILIZER", BASE_PRICE.get("FERTILIZER", 0))
        for crop, count in raw_obs["private"]["seeds"].items():
            total += count * SEED_COST.get(crop, 0)
        for item, count in raw_obs["private"]["shed"].items():
            total += count * prices.get(item, BASE_PRICE.get(item, 0))
        for inv in raw_obs["private"].get("inventories", []):
            for item, count in inv.items():
                total += count * prices.get(item, BASE_PRICE.get(item, 0))
        return total / 100.0

    @staticmethod
    def _is_idle(raw_obs) -> bool:
        """True if we're holding no seed and have no standing PLANT tile -- i.e. not"""
        private = raw_obs["private"]
        if any(count > 0 for count in private["seeds"].values()):
            return False
        farm = raw_obs["farms"][raw_obs["player"]]
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    return False
        return True

    @staticmethod
    def _unwatered_crop_count(raw_obs) -> int:
        """Count of our own standing PLANT tiles not yet watered today"""
        farm = raw_obs["farms"][raw_obs["player"]]
        count = 0
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT" and not tile.get("watered_today"):
                    count += 1
        return count

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._trainer = self._kaggle_env.train([None, self.opponent])
        raw_obs = self._trainer.reset()
        self._last_obs = raw_obs
        self._prev_money = self._own_money(raw_obs)
        self._prev_potential = self._potential(raw_obs)
        return encode_obs(raw_obs), {}

    def step(self, action):
        action_dict = decode_action(action, self._last_obs)
        raw_obs, _trainer_reward, done, info = self._trainer.step(action_dict)
        self._last_obs = raw_obs
        done = bool(done)

        money = self._own_money(raw_obs)
        money_reward = (money - self._prev_money) / 100.0
        self._prev_money = money

        potential = 0.0 if done else self._potential(raw_obs)
        shaping = self.GAMMA * potential - self._prev_potential
        self._prev_potential = potential

        idle_penalty = self.IDLE_PENALTY if (not done and self._is_idle(raw_obs)) else 0.0

        neglect_penalty = 0.0 if done else self.NEGLECT_PENALTY * self._unwatered_crop_count(raw_obs)

        relative_outcome = self.RELATIVE_OUTCOME_SCALE * (money - self._opponent_money(raw_obs)) / 100.0 if done else 0.0

        reward = money_reward + shaping - idle_penalty - neglect_penalty + relative_outcome

        return encode_obs(raw_obs), reward, done, False, (dict(info) if info else {})

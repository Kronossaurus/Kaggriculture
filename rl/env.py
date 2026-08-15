"""Gymnasium wrapper around Kaggriculture, for RL training via kaggle_environments'
built-in `env.train()` trainer.

IMPORTANT: `env.train()`'s reward is documented as a per-step diff, but
Kaggriculture's own interpreter (kaggriculture.py) only ever writes
`state.reward` on the terminal step(s) -- every other turn it's left
untouched. That makes the trainer's "diff" a near-total no-op for 719 of 720
turns, then a single terminal jump equal to raw final money (not a real gain,
since the previous recorded reward was never money to begin with). Training
against that signal collapses PPO to a do-nothing policy (confirmed: v1
converged to spamming a single movement op for the entire episode). This
wrapper ignores the trainer's `reward` and computes a real per-turn money
diff itself, straight from the observation.

v1 action scope: the policy controls two independent per-turn choices:

- a **farmer op** (PASS/NORTH/SOUTH/EAST/WEST/WATER/HARVEST/FERTILIZE/DIG/
  PLANT_<crop>) for the main farmer on the starting NW quadrant, and
- a **buy-seed choice** (NONE or one of CROPS) for a `BUY_SEED` market order.

These are separate `MultiDiscrete` dimensions, not one combined op, because
they're separate in the game itself -- a turn's action dict has independent
`farmer` and `market` keys (see data/README.md), so a real player can e.g.
move AND buy a seed the same turn. An earlier version of this wrapper modeled
BUY_SEED_<crop> as one more option competing with movement/tile ops for a
single discrete choice, which meant *choosing to buy a seed cost the farmer
its whole turn* -- not how the game actually works, and an unnecessary tax on
top of the credit-assignment problem below. Selling is still auto-generated
(the shed is emptied via SELL market orders every turn, independent of both
policy choices) -- no learned sell-timing yet, see rl/README.md. If there's no
seed in inventory, PLANT_<crop> is just a silent no-op like any other invalid
action, same as BUY_SEED is if unaffordable -- the policy has to learn the
buy-then-plant sequence itself, nothing auto-buys a seed for it. Farm hands,
land expansion (BUY_LAND), hiring (HIRE), and animals are out of scope for v1.
See rl/README.md for the plan to extend the action space further.

Reward has four parts, added together each step -- see step() and _potential():
1. real per-turn money diff (the actual signal we care about);
2. potential-based shaping on standing-crop/seed/shed value, floored at cost basis
   for standing crops, so BUY_SEED/PLANT/HARVEST are shaping-neutral rather than
   penalized;
3. a flat per-turn IDLE_PENALTY while holding no seed and no standing crop, which
   is deliberately *not* potential-based (see step()) -- it's the difference
   between "buying a seed doesn't hurt" (part 2) and "sitting empty-handed
   costs you something, so go buy a seed" (part 3);
4. a per-tile NEGLECT_PENALTY for each standing crop not yet watered today, same
   rationale one link further down the loop -- "watering doesn't hurt" (part 2)
   vs. "not watering today costs you something" (part 4).
"""

from __future__ import annotations

from typing import Any, Callable, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make

from .features import BASE_PRICE, BOARD_SIZE, CROPS, GRID_CHANNELS, SEED_COST, VECTOR_DIM, encode_obs

FARMER_OPS = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST",
    "WATER", "HARVEST", "FERTILIZE", "DIG",
    *(f"PLANT_{c}" for c in CROPS),
]

# A separate, independent per-turn choice from FARMER_OPS -- see module docstring.
# "NONE" (index 0) means don't queue a BUY_SEED order this turn.
BUY_SEED_OPS = ["NONE", *CROPS]

# A built-in name ("random"/"pass"/"starter"), a path to an agent .py file, or a
# callable obs -> action dict (e.g. a wrapped SB3 checkpoint for self-play).
OpponentSpec = Union[str, Callable[[dict], dict]]


def decode_action(action, obs) -> dict:
    """Turn a (farmer_op_idx, buy_seed_idx) pair -- a MultiDiscrete action, see
    module docstring -- into a full Kaggriculture action dict. Selling is the one
    market order still auto-generated: the shed is emptied every turn regardless
    of what the policy picked (no learned sell-timing yet, see rl/README.md)."""
    farmer_idx, buy_idx = (int(v) for v in action)
    op = FARMER_OPS[farmer_idx]
    private = obs["private"]
    market: list[list] = []

    if buy_idx > 0:
        market.append(["BUY_SEED", BUY_SEED_OPS[buy_idx], 1])

    farmer_op = ["PLANT", op.split("_", 1)[1]] if op.startswith("PLANT_") else [op]

    for item, qty in private["shed"].items():
        if qty > 0:
            market.append(["SELL", item, qty])

    return {"farmer": farmer_op, "hands": [], "market": market[:10]}


class KaggricultureEnv(gym.Env):
    """Single-agent Gymnasium view of Kaggriculture: our policy is player 0,
    `opponent` (an `OpponentSpec`, see above) drives player 1.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    # Must match PPO's `gamma` in rl/train.py (SB3 default 0.99) for the potential-based
    # shaping below to keep its policy-invariance guarantee -- see step().
    GAMMA = 0.99

    # Per-turn tax applied while holding no seed and no standing crop -- see step()'s
    # docstring note on why this is deliberately *not* folded into Phi.
    IDLE_PENALTY = 0.02

    # Per-turn, per-tile tax for each of our own standing PLANT tiles not yet watered
    # today -- same rationale as IDLE_PENALTY (Phi's floor-fix makes watering
    # non-negative but creates no urgency to do it *today*), aimed at the next link in
    # the loop: buy -> plant -> water -> harvest -> sell.
    NEGLECT_PENALTY = 0.02

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

        # [farmer_op_idx, buy_seed_idx] -- two independent choices, see module docstring.
        self.action_space = spaces.MultiDiscrete([len(FARMER_OPS), len(BUY_SEED_OPS)])
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
    def _potential(raw_obs) -> float:
        """Shaping potential Phi(s): rough resale value of stuff we're holding that
        the raw money-diff reward can't see yet -- crops still standing on our own
        tiles plus unplanted seeds in inventory (at their purchase cost), scaled
        like the money reward.

        The raw money-diff reward only pays off at SELL time -- for wheat that's
        ~WHEAT_FIRST_YIELD_DAY (2) days / ~48 turns after PLANT, per main.py's
        heuristic -- which starved early PPO updates of any signal for the buy-
        seed/plant/water/harvest ops (see rl/README.md analysis). Phi gives credit
        turn by turn instead of one lump sum at harvest.

        A standing crop's value is floored at its own seed cost, not just
        `yield_units * price`: right after PLANT, yield_units is 0, so scoring the
        tile at raw yield value made planting itself look like a shaping *loss*
        (Phi drops from the seed's held value to ~0 the instant it's planted) --
        an accidental penalty on the exact op we want the policy to take, and
        exactly why `checkpoints/v2` kept learning to buy seeds and never plant/
        water them (see rl/README.md). Flooring at cost basis makes PLANT
        shaping-neutral like BUY_SEED already is; Phi only rises for real once
        yield_units grows past that floor (rewarding watering/waiting), and if
        the crop instead dies from neglect (2 unwatered days -> WEED, no longer
        `kind == "PLANT"`), Phi drops from the floor to 0 -- a real, deserved
        penalty for abandoning a started plant/water/harvest flow instead of
        finishing it.

        Harvested-but-unsold produce in the shed also counts, at full market price
        (no flooring needed -- it's already realized, nothing left to lose). Same
        rationale one more link down the chain: HARVEST moves a tile's standing
        value into the shed, and `decode_action` builds each turn's auto-SELL
        orders from the *pre-action* obs, so a turn that harvests doesn't also sell
        that same produce -- it sits in the shed until the turn after. Without
        counting the shed, Phi would drop the instant HARVEST empties/reduces a
        tile (or removes it entirely, for one-time crops) a full turn before the
        matching money-diff reward arrives, making HARVEST look like a shaping
        *loss* -- the exact same bug PLANT had before the floor-fix above, just one
        step later in the loop.
        """
        # Live per-resource sell prices (data/README.md: "Sell prices ... move
        # dynamically per resource and persist across days") -- already exposed in
        # the observation and used by features.py's encoder, just not previously
        # wired into Phi, which used the static BASE_PRICE table instead. Falls back
        # to BASE_PRICE for anything missing from the quote (matches features.py).
        # Doesn't affect the policy-invariance guarantee -- market prices are just
        # another part of the observable state Phi is a function of. Seed cost is
        # NOT looked up dynamically: the market only moves *sell* prices, buy prices
        # for seeds are fixed (data/README.md), so SEED_COST is already correct.
        prices = raw_obs["market"]["prices"]

        farm = raw_obs["farms"][raw_obs["player"]]
        total = 0.0
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    crop = tile.get("crop")
                    if crop in BASE_PRICE:
                        standing_value = tile.get("yield_units", 0) * prices.get(crop, BASE_PRICE[crop])
                        total += max(standing_value, SEED_COST.get(crop, 0))
        for crop, count in raw_obs["private"]["seeds"].items():
            total += count * SEED_COST.get(crop, 0)
        for item, count in raw_obs["private"]["shed"].items():
            total += count * prices.get(item, BASE_PRICE.get(item, 0))
        return total / 100.0

    @staticmethod
    def _is_idle(raw_obs) -> bool:
        """True if we're holding no seed and have no standing PLANT tile -- i.e. not
        anywhere in the buy->plant->water->harvest loop right now. Used by the idle
        penalty in step(), not by _potential()."""
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
        """Count of our own standing PLANT tiles not yet watered today
        (`watered_today` resets False at each day's refresh, per data/README.md) --
        used by the neglect penalty in step()."""
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

        # Computed ourselves, not from the trainer's returned reward -- see
        # module docstring for why that field can't be trusted per-turn.
        money = self._own_money(raw_obs)
        # Money deltas run tens-to-low-hundreds per meaningful event; scale down
        # so PPO sees roughly O(1) rewards. Tune if training is unstable.
        money_reward = (money - self._prev_money) / 100.0
        self._prev_money = money

        # Potential-based shaping (Ng, Harada & Russell 1999): shaped = raw +
        # gamma*Phi(s') - Phi(s). Provably leaves the optimal policy unchanged
        # (it telescopes to the same total return over an episode) while making
        # the per-turn reward dense instead of sparse. Phi(terminal) = 0 by
        # convention -- no future turns left to harvest into.
        potential = 0.0 if done else self._potential(raw_obs)
        shaping = self.GAMMA * potential - self._prev_potential
        self._prev_potential = potential

        # Idle tax: NOT potential-based, deliberately. Phi's floor-fix already makes
        # BUY_SEED/PLANT roughly shaping-neutral (the seed's cost basis carries
        # through), but "neutral" only removes the disincentive -- it doesn't create
        # any pull *toward* buying. Potential-based shaping can't add that pull either:
        # it's a function of state only, so telescoping makes it invariant to how long
        # you dwell in a state and only pays out on transitions. A flat per-turn
        # penalty while idle (no seed held, no crop standing) breaks that invariance on
        # purpose -- it's a real, accumulating cost for leaving the buy->plant->water->
        # harvest->sell loop empty, which is exactly the nudge needed to get BUY_SEED
        # picked over PASS/movement.
        idle_penalty = self.IDLE_PENALTY if (not done and self._is_idle(raw_obs)) else 0.0

        # Same rationale, one link further down the loop: a standing crop not yet
        # watered today costs something *now*, not just "eventually turns to weed" --
        # see NEGLECT_PENALTY.
        neglect_penalty = 0.0 if done else self.NEGLECT_PENALTY * self._unwatered_crop_count(raw_obs)

        reward = money_reward + shaping - idle_penalty - neglect_penalty

        return encode_obs(raw_obs), reward, done, False, (dict(info) if info else {})

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

Action scope: the policy controls five independent per-turn choices, one
`MultiDiscrete` dimension each:

- a **farmer op** (PASS/NORTH/SOUTH/EAST/WEST/WATER/HARVEST/FERTILIZE/DIG/
  PICKUP/DROP/PLANT_<crop>) for the main farmer on the starting NW quadrant;
- a **buy-seed choice** (NONE or one of CROPS) for a `BUY_SEED` order;
- a **buy-product choice** (NONE or FERTILIZER) for a `BUY_PRODUCT` order --
  WHEAT is the only other product `BUY_PRODUCT` can target, but that's for
  feeding animals (out of scope, see rl/README.md), so it's not offered here;
- a **buy-land choice** (NONE or BUY_LAND) -- `BUY_LAND` itself takes no
  target quadrant (the game unlocks NE, then SW, then SE, in a fixed order at
  increasing fixed prices), so this is a plain per-turn toggle;
- a **sell choice** (NONE or one of PRODUCTS) for a `SELL` order, which
  liquidates *all* of that product currently in the shed (no partial-quantity
  sub-choice) -- this replaces an earlier version that auto-sold the entire
  shed every turn regardless of what the policy picked. Selling is now a real
  learned decision like buying is, which reopens the same credit-assignment
  question BUY_SEED originally had (does SELL-then-money-goes-up get
  connected before exploration gives up on it?) -- untested at the scale
  BUY_SEED needed IDLE_PENALTY to solve, see rl/README.md for the current
  state.

These are separate `MultiDiscrete` dimensions, not one combined op, because
they're separate in the game itself -- a turn's action dict has independent
`farmer` and `market` keys (see data/README.md), so a real player can e.g.
move AND buy a seed AND buy land the same turn. An earlier version of this
wrapper modeled BUY_SEED_<crop> as one more option competing with
movement/tile ops for a single discrete choice, which meant *choosing to buy
a seed cost the farmer its whole turn* -- not how the game actually works,
and an unnecessary tax on top of the credit-assignment problem below.

PLANT_<crop> stays one op per crop rather than a single grouped PLANT with
the crop inferred from inventory -- a grouped version was tried and reverted
because it removes the policy's ability to choose *which* held seed to plant
when multiple types are in inventory at once (an earlier justification -- that
the same-turn BUY_SEED can never be the seed a same-turn PLANT consumes
anyway, per data/README.md's Turn Processing Order -- only covers the
single-seed-type case, not this one). If there's no seed of the named crop on
hand, PLANT_<crop> is just a silent no-op like any other invalid action, same
as BUY_SEED is if unaffordable -- the policy has to learn the buy-then-plant
sequence itself, nothing auto-buys a seed for it.

PICKUP always targets FERTILIZER -- the only item in the current scope worth
moving out of the shed. Per data/README.md, `BUY_PRODUCT FERTILIZER` lands
fertilizer in the shed, but `FERTILIZE` consumes from the *active farmer's
inventory* (confirmed in kaggriculture.py: `FERTILIZE` calls
`_inv_take(inv, "FERTILIZER", 1)` where `inv` is the farmer's own inventory,
not the shed) -- so the buy -> pickup -> fertilize chain needs all three
pieces; FERTILIZE was previously in FARMER_OPS but always a no-op because
neither BUY_PRODUCT nor PICKUP existed anywhere in the action space. DROP
dumps the farmer's entire inventory into the shed on demand. Without it, the
only way harvested produce ever reaches the shed (and becomes sellable -- see
data/README.md, `SELL` explicitly checks `private["shed"]`, never personal
inventory) is the automatic end-of-day drop, up to `turnsPerDay - 1` turns
after HARVEST. DROP lets the policy shorten that gap to whatever it wants,
including zero.

Farm hands, hiring (`HIRE`), and animals are still out of scope. See
rl/README.md for what's left and the plan to extend the action space further.

Reward has five parts, added together each step -- see step() and _potential():
1. real per-turn money diff (the actual signal we care about);
2. potential-based shaping on standing-crop/seed/shed/inventory value, floored
   at cost basis for standing crops (plus an active-fertilizer credit, see
   _potential), so BUY_SEED/PLANT/HARVEST/PICKUP/DROP/FERTILIZE are
   shaping-neutral rather than penalized;
3. a flat per-turn IDLE_PENALTY while holding no seed and no standing crop, which
   is deliberately *not* potential-based (see step()) -- it's the difference
   between "buying a seed doesn't hurt" (part 2) and "sitting empty-handed
   costs you something, so go buy a seed" (part 3);
4. a per-tile NEGLECT_PENALTY for each standing crop not yet watered today, same
   rationale one link further down the loop -- "watering doesn't hurt" (part 2)
   vs. "not watering today costs you something" (part 4);
5. a terminal-only RELATIVE_OUTCOME_SCALE * (own money - opponent's money) bonus
   on the last step -- the actual win condition (data/README.md: most money at
   the end wins) is relative, but parts 1-4 are all purely about our own money.
   See step()'s docstring note for why nothing else in this reward, including
   Kaggriculture's own discarded `state.reward`, ever compares to the opponent.

No new penalty term was added for BUY_LAND or SELL -- unlike seeds/watering,
neither is required for the core loop to produce money at all, so nudging the
policy toward them the same way would be premature (no evidence yet that they
need it). SELL in particular already has a natural end-of-episode pressure:
Phi(terminal) = 0 by convention (see step()), so unsold shed/inventory value
the policy is still sitting on when the episode ends is a real, uncompensated
loss in the shaped reward -- that alone should discourage holding forever
without needing an explicit tax.

BUY_LAND was deliberately left *without* a Phi credit for the land itself,
unlike everything above -- and this is NOT the same "unfairly penalized" bug
class. Seeds/crops/produce/fertilizer all have concrete, already-known
present value (what was paid, or what they'd sell for right now) that Phi was
just failing to count; newly-unlocked land's payoff is genuinely prospective
(more tiles, spread across many future turns) with no honest way to price it
today, so inventing a credit would be guessing, not fixing a measurement
bug -- a real player has to weigh the same $1k/$2k/$4k up-front hit against
uncertain future return, that's the intended trade-off. Worth flagging
separately though: at `/100` scale that's a -10 to -40 reward swing in a
single turn, 10-40x any other single event in this reward function (SB3 PPO's
default `normalize_advantage=True` smooths per-batch outliers somewhat, but
doesn't eliminate the risk this scale mismatch destabilizes training the same
way the original 720-turn sparse-reward problem did) -- untested at this
scale, revisit if BUY_LAND turns out to never get explored.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make

from .features import BASE_PRICE, BOARD_SIZE, CROPS, GRID_CHANNELS, PRODUCTS, SEED_COST, VECTOR_DIM, encode_obs

FARMER_OPS = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST",
    "WATER", "HARVEST", "FERTILIZE", "DIG", "PICKUP", "DROP",
    *(f"PLANT_{c}" for c in CROPS),
]

# Four more independent per-turn choices -- see module docstring. "NONE" (index 0)
# always means don't queue that market order this turn.
BUY_SEED_OPS = ["NONE", *CROPS]
BUY_PRODUCT_OPS = ["NONE", "FERTILIZER"]
BUY_LAND_OPS = ["NONE", "BUY_LAND"]
SELL_OPS = ["NONE", *PRODUCTS]

# A built-in name ("random"/"pass"/"starter"), a path to an agent .py file, or a
# callable obs -> action dict (e.g. a wrapped SB3 checkpoint for self-play).
OpponentSpec = str | Callable[[dict], dict]


def decode_action(action, obs) -> dict:
    """Turn a (farmer_op_idx, buy_seed_idx, buy_product_idx, buy_land_idx, sell_idx)
    tuple -- a MultiDiscrete action, see module docstring -- into a full
    Kaggriculture action dict."""
    farmer_idx, buy_seed_idx, buy_product_idx, buy_land_idx, sell_idx = (int(v) for v in action)
    op = FARMER_OPS[farmer_idx]
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

    if op.startswith("PLANT_"):
        farmer_op = ["PLANT", op.split("_", 1)[1]]
    elif op == "PICKUP":
        farmer_op = ["PICKUP", "FERTILIZER", 1]
    else:
        farmer_op = [op]

    return {"farmer": farmer_op, "hands": [], "market": market[:10]}


class KaggricultureEnv(gym.Env):
    """Single-agent Gymnasium view of Kaggriculture: our policy is player 0,
    `opponent` (an `OpponentSpec`, see above) drives player 1.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

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

    # Multiplier on the terminal-step own-vs-opponent money margin -- see step()'s
    # docstring note on why the win/loss outcome needed its own term (nothing else
    # in this reward ever compares to the opponent). 1.0 keeps it on the same /100
    # scale as everything else; tune here if it turns out too strong/weak relative
    # to the per-turn shaping.
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

        # [farmer_op_idx, buy_seed_idx, buy_product_idx, buy_land_idx, sell_idx] --
        # five independent choices, see module docstring.
        self.action_space = spaces.MultiDiscrete(
            [len(FARMER_OPS), len(BUY_SEED_OPS), len(BUY_PRODUCT_OPS), len(BUY_LAND_OPS), len(SELL_OPS)]
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
        """Shaping potential Phi(s): rough resale value of stuff we're holding that
        the raw money-diff reward can't see yet -- crops still standing on our own
        tiles, unplanted seeds (at purchase cost), and shed/inventory contents (at
        market price) -- scaled like the money reward.

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

        Shed contents count at full market price (no flooring needed -- it's
        already realized, nothing left to lose). Personal-inventory contents
        (`private["inventories"]`, one dict per active unit) count the same way,
        and matter for the same reason: HARVEST lands produce in the acting
        unit's *inventory*, not the shed (confirmed in kaggriculture.py:
        `_inv_add(inv, tile["crop"], units)` where `inv` is the farmer's own
        inventory) -- it only reaches the shed, and becomes sellable at all (SELL
        explicitly checks `private["shed"]`, never inventory), at the automatic
        end-of-day drop, up to `turnsPerDay - 1` turns later. An earlier version
        of this function counted only the shed, on the mistaken assumption
        HARVEST deposited there directly (its regression test used a synthetic
        state that didn't match the real transition) -- that meant Phi dropped a
        tile's standing value at HARVEST with nothing counted to replace it until
        the shed-drop, up to a day later: the same shaping-penalty-on-the-op-we-
        want bug PLANT had, just with a much longer window. Counting inventory
        closes it for real. FERTILIZER held in the shed or picked up into
        inventory is valued the same generic way (BASE_PRICE has an entry for
        it), no special-casing needed.

        A tile with an active fertilizer bonus (`fertilized_until_day >= day`,
        per data/README.md: doubles the per-day yield bonus for 3 days) gets an
        extra credit equal to one FERTILIZER unit's price, added on top of (not
        merged into) the seed-cost floor above -- same bug class, one more link
        down the chain: FERTILIZE consumes a real, Phi-valued FERTILIZER unit
        from inventory (see the shed/inventory paragraph above) in exchange for
        a benefit -- the doubled yield -- that `yield_units` doesn't reflect
        until subsequent WATER calls actually bank it. Without this credit, Phi
        drops by the fertilizer's value the instant FERTILIZE is called with
        nothing to replace it, the exact shaping-penalty-on-the-op-we-want
        pattern PLANT had. The credit disappears when `fertilized_until_day`
        lapses (3 days later, per kaggriculture.py: `max(existing, day + 2)`) --
        by then the doubled watering should have grown `yield_units` past it
        for a smooth handoff; if the bonus window was wasted (not watered), Phi
        drops for real -- a deserved penalty, same as letting a crop die from
        neglect above.
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

        # Win/loss outcome: NOT potential-based, only ever nonzero on the terminal
        # step. Every other term above is about our own money -- nothing anywhere
        # in this reward compares to the opponent, even though the actual win
        # condition (data/README.md: "whoever has the most money in the bank at
        # the end... wins") is relative, not absolute. Kaggriculture's own
        # `state.reward` doesn't help either -- confirmed in kaggriculture.py, it's
        # just `float(own money)` on the terminal step, same absolute quantity we
        # already compute per-turn, no comparison to the opponent baked in. This
        # margin is the real win condition: positive if we'd win the match right
        # now, negative if we'd lose. It doesn't replace the dense per-turn shaping
        # above (still needed for credit assignment within the episode) -- it adds
        # the one signal that was completely missing: an episode that ends with
        # more money than the opponent should score better than one that doesn't,
        # regardless of the absolute amount either side has. Standard RL credit
        # assignment (the value function, via GAE) then propagates "this trajectory
        # led to a win" back across the whole episode's actions -- not the same as
        # imitating the opponent's specific choices (still not observed), but it's
        # how winning strategies actually get reinforced without behavioral cloning.
        relative_outcome = self.RELATIVE_OUTCOME_SCALE * (money - self._opponent_money(raw_obs)) / 100.0 if done else 0.0

        reward = money_reward + shaping - idle_penalty - neglect_penalty + relative_outcome

        return encode_obs(raw_obs), reward, done, False, (dict(info) if info else {})

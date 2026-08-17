"""Kaggriculture submission entry point.

Kaggle loads this file and calls `agent(obs)` (or `agent(obs, config)`) once
per turn for whichever player index we've been assigned. Keep this file
self-contained (or bundle extra modules into a tar.gz with this at the root
per AGENTS.md) since it's shipped standalone to the competition runner.

Strategy ("farm maxxer"): a scripted multi-unit farm loop built on top
of `reference/melon_maxxer.py` (Kaggle's own official example agent, see its
docstring), fixing three of its four documented weaknesses (see the
Correction note below for the fourth):

  - melon_maxxer never expands land or hires hands -> this agent buys land
    greedily and hires hands every day (hiring is *extremely* cheap: fib(n)
    dollars for the n-th hire that day, i.e. $1, $1, $2, $3, $5, ... -- see
    kaggriculture.py's `_hire_cost`), then drives every hand with the same
    per-unit task logic as the farmer.
  - melon_maxxer only ever grows MELON -> this agent grows a mix of MELON
    (highest value/tile/day by a wide margin, per data/README.md's Object
    Types table), WHEAT and CARROT (cheap, fast-cycling fillers that keep
    tiles turning over while melons mature, and diversify what's sellable
    when one crop's market price is depressed).
  - melon_maxxer never fertilizes -> this agent fertilizes WHEAT/CARROT
    (one-time crops whose *unfertilized* yield cap is strictly below their
    fertilized cap, per data/README.md's Harvest Yields section -- melon's
    cap is reachable by watering alone, fertilizer there only shortens the
    growth window, so it's skipped as lower value for the added complexity).

**Correction -- the fourth "fix" (throttled selling) was reverted, it was a
regression**: melon_maxxer dumps its entire shed in one SELL order once a
threshold is crossed, which can crash the price mid-sale (price moves
unit-by-unit *within* an order, see data/README.md's Market Mechanics). An
earlier version of this agent capped how much of each crop it sold per turn
to avoid that, on paper a strict improvement. Measured instead: a deterministic
20-seed head-to-head between that version and an ablation that dumps
everything (`reference/main_dump_sell.py`) was a 19/20-seed *loss* for the
capped version. Root cause: holding harvested value in the shed waiting for
the cap to free up slows reinvestment into land/seeds/hands more than the
occasional price dip from a big dump costs, at this agent's actual
production scale -- the theoretical concern was real but small relative to
the throttling's own opportunity cost. `_build_sell_orders` now dumps in
full again, like melon_maxxer, with the shed-overflow-guard threshold-bypass
(`SHED_OVERFLOW_GUARD`) kept since that part was never in question.

Animal support (GOOSE/COW/SHEEP) was implemented and A/B tested
(`reference/main_with_animals_v1.py`) and found to be a clear net negative --
roughly a 70% cut to final money vs. this animal-free version, deterministic
A/B vs. `starter` across 3 seeds -- even after fixing four real bugs along
the way. The daily per-animal upkeep (FEED/CARE/COLLECT_FERTILIZER) competes
for the same scarce farmer/hand turns as the much higher value/tile/day crop
loop, and a single missed feeding cycle loses an entire $300-500 purchase
outright. See that file's docstring for the full writeup before attempting
this again.

Still out of scope: any opponent modeling -- this agent only ever reacts to
its own farm/market state. See rl/README.md for the parallel (currently
stalled) attempt at a learned policy.

**Endgame liquidation**: unsold shed contents (and anything still sitting in
a farmer/hand's personal inventory) don't count towards the final score --
see data/README.md's Reward section. Everything above ignores that until
close to the last day, so this agent also tracks `days_remaining` (from
`config["episodeSteps"]`/`config["turnsPerDay"]`, which `kaggle_environments`
does pass through -- confirmed via `agent.py`'s `co_argcount` inspection) and,
as the season winds down: stops buying seeds/land that can't pay back in the
time left, has units prioritize walking harvested produce back to the shed
over watering a crop that won't finish in time, and drops the sell-price
threshold entirely so nothing is still sitting unsold when the last turn's
market closes.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_PARAMS,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

CASH_CROPS = ("MELON", "WHEAT", "CARROT")
CROP_TARGET_SHARE = {"MELON": 0.5, "WHEAT": 0.3, "CARROT": 0.2}
FERTILIZE_CROPS = {"WHEAT", "CARROT"}

SEED_RESERVE = {"MELON": 3, "WHEAT": 3, "CARROT": 3}
FERTILIZER_RESERVE = 2
FERTILIZER_MAX_PRICE_FRAC = 1.2  # don't buy fertilizer more than 20% over base

SELL_PARAMS = {
    # threshold: fraction of base price required to sell at all -- see
    # _build_sell_orders for why there's no per-turn cap.
    "MELON":  {"threshold": 0.65},
    "WHEAT":  {"threshold": 0.45},
    "CARROT": {"threshold": 0.45},
}

SHED_CAPACITY = 100  # data/README.md default; not exposed in `obs`
SHED_OVERFLOW_GUARD = 0.85  # start dumping regardless of price above this fraction full

MAX_HANDS = 6
SAFETY_RESERVE = 300  # cash kept in reserve before optional buys/land/fertilizer
MAX_MARKET_ORDERS = 10  # data/README.md default; not exposed in `obs`

DEFAULT_EPISODE_STEPS = 720  # data/README.md defaults, used if `config` is unavailable
DEFAULT_TURNS_PER_DAY = 24
SEED_BUFFER_DAYS = 1  # a crop needs first_yield_day + this many days left to be worth planting
LAND_CUTOFF_DAYS = 5  # stop buying new land this close to the end, it won't pay back in time
ENDGAME_LIQUIDATION_DAYS = 2  # ignore the sell-price threshold this close to the end


def _cfg_get(config, key, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


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


def _shed_adjacent_tiles(board_size):
    half = board_size // 2
    return {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}


def _is_fertilizable(tile, day):
    return (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile["crop"] in FERTILIZE_CROPS
        and tile["fertilized_until_day"] < day
        and day - tile["planted_day"] <= CROPS[tile["crop"]]["max_yield_day"]
    )


def _any_fertilizable(farm, day):
    return any(_is_fertilizable(tile, day) for row in farm["tiles"] for tile in row)


def _is_harvestable(tile, day):
    """yield_units > 0 alone is NOT "ready" for one-time crops: `_new_plant`
    seeds it at 1 immediately on planting (kaggriculture.py), but HARVEST is
    still a no-op until `first_yield_day` (age 2 for wheat/carrot, 10 for
    melon). Confusing the two makes a unit spam a no-op HARVEST instead of
    WATER, and the crop dies to neglect before ever becoming harvestable."""
    if not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0:
        return False
    if tile.get("kind") != "PLANT":
        return True  # animal structures have no such gate; out of scope here anyway
    return day - tile["planted_day"] >= CROPS[tile["crop"]]["first_yield_day"]


class _CropPlanner:
    """Tracks, for one turn, which crop each new PLANT should use.

    Buying happens the turn *before* planting (market orders resolve after
    farmer/hand actions each turn -- see data/README.md's Turn Processing
    Order), so this only ever spends seeds already sitting in
    `private["seeds"]`, never seeds bought this same turn.
    """

    def __init__(self, private, farm, days_remaining):
        self.seed_budget = dict(private.get("seeds", {}))
        # A crop that can't reach first_yield_day (+ a day of slack to sell it
        # after) before the season ends is a pure loss to plant -- see the
        # module docstring's Endgame liquidation note.
        self.plantable_crops = {
            c for c in CASH_CROPS if days_remaining >= CROPS[c]["first_yield_day"] + SEED_BUFFER_DAYS
        }
        self.counts = {c: 0 for c in CASH_CROPS}
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile["crop"] in CASH_CROPS:
                    self.counts[tile["crop"]] += 1
        self.total = sum(self.counts.values()) or 1

    def has_plantable(self):
        return any(self.seed_budget.get(c, 0) > 0 for c in self.plantable_crops)

    def choose(self):
        """Pick whichever affordable crop is furthest below its target mix share."""
        available = [c for c in self.plantable_crops if self.seed_budget.get(c, 0) > 0]
        if not available:
            return None
        crop = min(available, key=lambda c: self.counts[c] / self.total - CROP_TARGET_SHARE[c])
        self.seed_budget[crop] -= 1
        self.counts[crop] += 1
        self.total += 1
        return crop


def _find_target(pos, farm, day, board_size, claimed, inv, planner, shed_tiles, want_fertilizer, liquidating):
    """Scan the whole farm for the nearest unclaimed tile worth walking to.

    `liquidating` (close to the last day, see the module docstring's Endgame
    liquidation note) swaps water's and shed-return's priority: watering a
    crop that won't finish growing in time is wasted effort, while getting
    already-harvested produce into the shed guarantees it can still be sold.
    """
    fx, fy = pos
    water_priority = 5 if liquidating else 1
    shed_priority = 1 if liquidating else 5
    candidates = []
    for y in range(board_size):
        for x in range(board_size):
            if (x, y) in claimed:
                continue
            tile = farm["tiles"][y][x]
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                if _is_harvestable(tile, day):
                    candidates.append((x, y, 0))  # harvest
                elif not tile["watered_today"]:
                    candidates.append((x, y, water_priority))
                elif inv.get("FERTILIZER", 0) > 0 and _is_fertilizable(tile, day):
                    candidates.append((x, y, 3))  # deliver fertilizer
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                candidates.append((x, y, 2))  # clear a weed to reclaim the tile
            elif tile is None and planner.has_plantable():
                candidates.append((x, y, 4))  # plant

    carrying_produce = any(k != "FERTILIZER" and n > 0 for k, n in inv.items())
    needs_fertilizer = inv.get("FERTILIZER", 0) == 0 and want_fertilizer
    if carrying_produce or needs_fertilizer:
        for (sx, sy) in shed_tiles:
            if (sx, sy) not in claimed:
                candidates.append((sx, sy, shed_priority))  # return to shed

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[2], abs(c[0] - fx) + abs(c[1] - fy)))
    return candidates[0]


def _decide_unit(pos, inv, farm, private, day, board_size, claimed, planner, shed_tiles, want_fertilizer, liquidating):
    fx, fy = pos
    tile = farm["tiles"][fy][fx]

    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if _is_harvestable(tile, day):
            claimed.add((fx, fy))
            return ["HARVEST"]
        if not tile["watered_today"]:
            claimed.add((fx, fy))
            return ["WATER"]
        if inv.get("FERTILIZER", 0) > 0 and _is_fertilizable(tile, day):
            claimed.add((fx, fy))
            return ["FERTILIZE"]

    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        claimed.add((fx, fy))
        return ["DIG"]

    if (fx, fy) in shed_tiles:
        carried_produce = {k: n for k, n in inv.items() if k != "FERTILIZER" and n > 0}
        if carried_produce:
            claimed.add((fx, fy))
            return ["DROP"]
        if inv.get("FERTILIZER", 0) == 0 and want_fertilizer and private["shed"].get("FERTILIZER", 0) > 0:
            claimed.add((fx, fy))
            return ["PICKUP", "FERTILIZER", 1]

    if tile is None:
        crop = planner.choose()
        if crop:
            claimed.add((fx, fy))
            return ["PLANT", crop]

    target = _find_target(pos, farm, day, board_size, claimed, inv, planner, shed_tiles, want_fertilizer, liquidating)
    if target:
        claimed.add((target[0], target[1]))
        step = _step_toward(fx, fy, target[0], target[1])
        if step:
            return [step]
    return ["PASS"]


def _build_sell_orders(shed, prices, days_remaining):
    """Dumps the entire shed stock of a crop in one order once its price
    threshold is met -- same behavior as melon_maxxer, deliberately reverted
    from an earlier version that capped units sold per turn (see the module
    docstring's Correction note). A per-turn cap was tried expecting it to
    avoid crashing the price via a big single-order dump; instead it was a
    measured net negative (a 19/20-seed head-to-head loss to this dump
    version, `reference/main_dump_sell.py`): holding harvested value in the
    shed waiting for the cap to free up slows reinvestment into land/seeds/
    hands more than an occasional price dip costs, at this agent's actual
    production scale.
    """
    orders = []
    shed_total = sum(shed.values())
    crowded = shed_total >= SHED_CAPACITY * SHED_OVERFLOW_GUARD
    liquidating = days_remaining <= ENDGAME_LIQUIDATION_DAYS
    for crop, params in SELL_PARAMS.items():
        qty = shed.get(crop, 0)
        if qty <= 0:
            continue
        base = MARKET_PARAMS[crop]["base"]
        price = prices.get(crop, base)
        if crowded or liquidating or price >= base * params["threshold"]:
            orders.append(["SELL", crop, qty])
    return orders


def _build_land_orders(farm, days_remaining):
    if days_remaining < LAND_CUTOFF_DAYS:
        return []  # a fresh quadrant can't earn back its cost this close to the end
    n_extra = len(farm["unlocked_quadrants"]) - 1  # NW is always owned
    if n_extra >= len(LAND_ORDER):
        return []
    if farm["money"] - SAFETY_RESERVE >= LAND_PRICES[n_extra]:
        return [["BUY_LAND"]]
    return []


def _build_hire_orders(farm):
    desired = min(MAX_HANDS, 2 * len(farm["unlocked_quadrants"]))
    remaining = desired - farm["hires_today"]
    if remaining <= 0:
        return []
    return [["HIRE"] for _ in range(remaining)]


def _build_seed_orders(seeds, money, days_remaining):
    orders = []
    remaining = money - SAFETY_RESERVE
    for crop, reserve in SEED_RESERVE.items():
        if days_remaining < CROPS[crop]["first_yield_day"] + SEED_BUFFER_DAYS:
            continue  # wouldn't mature (and sell) before the season ends
        have = seeds.get(crop, 0)
        cost = CROPS[crop]["seed"]
        while have < reserve and remaining >= cost:
            orders.append(["BUY_SEED", crop, 1])
            have += 1
            remaining -= cost
    return orders


def _build_fertilizer_orders(private, prices, money):
    have = private["shed"].get("FERTILIZER", 0) + sum(
        inv.get("FERTILIZER", 0) for inv in private["inventories"]
    )
    if have >= FERTILIZER_RESERVE:
        return []
    base = MARKET_PARAMS["FERTILIZER"]["base"]
    price = prices.get("FERTILIZER", base)
    if price > base * FERTILIZER_MAX_PRICE_FRAC:
        return []
    if money - SAFETY_RESERVE < price:
        return []
    return [["BUY_PRODUCT", "FERTILIZER", 1]]


def agent(obs, config=None):
    player = obs.get("player", 0)
    farms = obs.get("farms", [])
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    private = obs.get("private", {}) or {}
    day = obs.get("day", 0)
    board_size = len(farm["tiles"])
    shed_tiles = _shed_adjacent_tiles(board_size)

    turns_per_day = int(_cfg_get(config, "turnsPerDay", DEFAULT_TURNS_PER_DAY)) or DEFAULT_TURNS_PER_DAY
    episode_steps = int(_cfg_get(config, "episodeSteps", DEFAULT_EPISODE_STEPS)) or DEFAULT_EPISODE_STEPS
    last_day = max(0, -(-episode_steps // turns_per_day) - 1)  # ceil(steps/turns_per_day) - 1
    days_remaining = last_day - day
    liquidating = days_remaining <= ENDGAME_LIQUIDATION_DAYS

    market_prices = (obs.get("market", {}) or {}).get("prices", {})
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    inventories = private.get("inventories", [{}])

    want_fertilizer = _any_fertilizable(farm, day)
    planner = _CropPlanner(private, farm, days_remaining)

    claimed = set()
    units = [farm["farmer"]] + list(farm["hands"])
    actions = [
        _decide_unit(
            pos,
            inventories[i] if i < len(inventories) else {},
            farm,
            private,
            day,
            board_size,
            claimed,
            planner,
            shed_tiles,
            want_fertilizer,
            liquidating,
        )
        for i, pos in enumerate(units)
    ]
    farmer_action, hands_actions = actions[0], actions[1:]

    market = []
    market += _build_sell_orders(shed, market_prices, days_remaining)
    market += _build_land_orders(farm, days_remaining)
    market += _build_hire_orders(farm)
    market += _build_seed_orders(seeds, farm["money"], days_remaining)
    market += _build_fertilizer_orders(private, market_prices, farm["money"])
    market = market[:MAX_MARKET_ORDERS]

    return {"farmer": farmer_action, "hands": hands_actions, "market": market}

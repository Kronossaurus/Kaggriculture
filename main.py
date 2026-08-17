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

**Tried a second time, still reverted**: `reference/x492_nah_i_d_win.py`
(see `scripts/pool_winrate.py`) runs animals profitably by never deciding
about them live at all -- their placement/feed/harvest schedule is baked
into a precomputed action tape offline, with no runtime turn-contention
risk. Adapting that idea to this agent's reactive architecture, 2 of
`MAX_HANDS`'s 6 hands were carved out as permanently dedicated animal
hands, never touching a crop tile, specifically to remove the turn-
contention failure mode above. It didn't help. A replay first caught a
second, unrelated starvation cause even with dedicated hands: WHEAT is
both a sellable crop and animal feed, and `_build_sell_orders` dumps the
entire shed the moment its price clears threshold, so the shed sat at 0
WHEAT for turns at a stretch -- fixed by buying feed WHEAT directly from
the market, decoupled from the crop side. With that fixed too (animals no
longer mass-starved), a 10-seed pool comparison was still a severe net
loss across the board -- e.g. -66% mean money vs `random`, -62% vs
`melon_maxxer` (which flipped from a 10/10-seed win to 6/10), -60% vs
`starter` (9/10). The root cause was never actually "contention" or
"starvation logistics" -- it's that 2 of 6 hands (a third of the crew) is
too much crop-tending capacity to give up for what animals produce in a
30-day season, at this agent's crop economics. Don't retry without first
finding a much cheaper way to fund animal upkeep than a fixed hand tax.

**Price-aware planting**: MELON's price curve is punished quadratically for
oversupply above equilibrium (`data/README.md`'s Price Function, the `sq`
above-target shape -- the steepest in the whole table) and only recovers
~1 unit/day: no town shop demands MELON, only the flat town-center trickle
(`data/README.md`'s Town Buildings table). A replay showed `_CropPlanner`
kept spending its 50% target share on MELON turn after turn right after a
big sell dump had just crashed its price (250 -> 86 base over one season,
seed 1 vs `starter`) -- replanting straight back into a market it had just
flooded. `_CropPlanner.choose` now skips a crop from consideration while
its current price is below the same threshold `_build_sell_orders` already
uses to decide whether to sell it, falling back to ignoring price only if
every plantable crop is currently depressed (so land never sits idle over
a temporary dip).

**Land utilization gate**: `_build_land_orders` used to buy the next
quadrant purely on cash available, with no regard for whether the land
already owned was even being worked. A replay showed this agent buying its
4th quadrant (all 100 tiles) by day 11 while plateauing at ~40-48% of
tiles actually planted for the rest of the season.
`reference/x492_nah_i_d_win.py` (the strongest agent in the pool, see
`scripts/pool_winrate.py`) never buys that 4th quadrant at all -- it stops
at 3 (75 tiles) and holds ~70-73% utilization there instead, ending the
same paired-seed season with far more money. `_build_land_orders` now
gates each new `BUY_LAND` on the current quadrant(s) already being at
least `LAND_UTILIZATION_GATE` planted -- buying land faster than it can be
staffed just leaves it idle. **A companion fix was tried and reverted**:
`MAX_HANDS = 6` (a self-imposed number, not a game rule -- hiring is
nearly free, fib-cost/day per `data/README.md`'s Hiring section) looked
like the reason utilization plateaued, and x492 itself runs up to 12
hands, so `MAX_HANDS` was raised to 12 with hires scaled to match.
Measured instead, 10-seed pool comparison: the land gate alone was a clean
win against every pool opponent (e.g. +21% mean money vs `random`, +41% vs
x492), while adding the raised hand cap on top made it *worse* than the
gate alone almost everywhere (still better than doing nothing, just a
smaller gain) -- more hands means more simultaneous harvests dumped into
the same finite-demand market, which crashes prices faster than the extra
fib-scaling hire cost pays back. `MAX_HANDS` and the hire formula are back
to their original values; only the land gate stuck.

**TOMATO/STRAWBERRY**: this agent only ever grew the three one-time crops
(MELON/WHEAT/CARROT). `data/README.md`'s Harvest Yields section documents
two more, TOMATO and STRAWBERRY, as *ongoing* crops -- once planted they
keep producing on a fixed schedule (tomato at ages 8/9/10/11, strawberry at
10/12/14/16) without needing to be replanted between harvests, unlike a
one-time crop's single payout. `_is_harvestable`/`_decide_unit`'s generic
PLANT-tile field checks (`yield_units`, `watered_today`) already handle
this correctly with no changes -- the engine keeps the tile alive after
HARVEST for an ongoing crop instead of clearing it, and just keeps refilling
`yield_units` on schedule, so a unit that waters an ongoing-crop tile daily
and harvests whenever `yield_units > 0` was already the right behavior.
Both are now in `CASH_CROPS`/`SELL_PARAMS`/`SEED_RESERVE`. **Measured, not
assumed, how much of MELON's share to give them**: the obvious-looking move
was cutting MELON's dominant 0.5 share hard to make room (it's the crop
whose price crashes quadratically, per the Price-aware planting note
above) -- 10-seed pool comparison showed that alone (MELON 0.35, no
tomato/strawberry, redistributed to WHEAT/CARROT) was a *clean loss*
against every pool opponent, including x492. MELON's price gate already
protects it from replanting into a crashed market, so de-emphasizing it
further just traded away its far-higher base price ($250 vs WHEAT's $25)
for nothing. What actually pays off is adding TOMATO/STRAWBERRY as a small
addition without touching MELON much: `CROP_TARGET_SHARE` settled at
MELON 0.45 / WHEAT 0.25 / CARROT 0.15 / TOMATO 0.10 / STRAWBERRY 0.05 --
roughly flat (+/-2%) against every weaker pool opponent and +33% mean
money against x492 (their gentler above-equilibrium price curves, `sqrt`/
`linear` vs MELON's `sq`, pay off specifically when the market is
contested). Fertilizing them was initially left out: `_is_fertilizable`'s
eligibility window used `CROPS[crop]["max_yield_day"]` as an upper bound,
which for these two crops equals `first_yield_day` (a placeholder value,
not "last production day"), so reusing it as-is would only ever allow
fertilizing the *first* scheduled production and silently miss the rest.
Fixed with `_last_yield_day`, an ongoing-crop-aware bound
(`first_yield_day + (max_yield - 1) * interval`, matching the real
schedule -- tomato through age 11, strawberry through age 16); both are
now in `FERTILIZE_CROPS` too. 10-seed pool comparison: a clean win
everywhere, +2 to +7% mean money with no regressions (largest gain against
x492, +7%, where every bit of yield matters most).

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

CASH_CROPS = ("MELON", "WHEAT", "CARROT", "TOMATO", "STRAWBERRY")
CROP_TARGET_SHARE = {"MELON": 0.45, "WHEAT": 0.25, "CARROT": 0.15, "TOMATO": 0.10, "STRAWBERRY": 0.05}
FERTILIZE_CROPS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY"}

SEED_RESERVE = {"MELON": 3, "WHEAT": 3, "CARROT": 3, "TOMATO": 2, "STRAWBERRY": 2}
FERTILIZER_RESERVE = 2
FERTILIZER_MAX_PRICE_FRAC = 1.2  # don't buy fertilizer more than 20% over base

SELL_PARAMS = {
    # threshold: fraction of base price required to sell at all -- see
    # _build_sell_orders for why there's no per-turn cap.
    "MELON":      {"threshold": 0.65},
    "WHEAT":      {"threshold": 0.45},
    "CARROT":     {"threshold": 0.45},
    "TOMATO":     {"threshold": 0.50},
    "STRAWBERRY": {"threshold": 0.55},
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
LAND_UTILIZATION_GATE = 0.6  # don't buy the next quadrant until this much of what we
                              # already own is planted -- see the module docstring's
                              # Land utilization gate note
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


def _is_plant(tile):
    return isinstance(tile, dict) and tile.get("kind") == "PLANT"


def _is_weed(tile):
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def _last_yield_day(crop):
    """The last day a crop can still produce, i.e. the true upper bound for
    _is_fertilizable -- CROPS[crop]["max_yield_day"] is only that for
    one-time crops. For an *ongoing* crop (TOMATO/STRAWBERRY) it's just a
    placeholder equal to first_yield_day (data/README.md's Harvest Yields
    section): the real schedule runs first_yield_day, +interval, +interval,
    ... for max_yield productions total (tomato ages 8/9/10/11, strawberry
    10/12/14/16), so the placeholder alone would only ever allow
    fertilizing the first of the four."""
    data = CROPS[crop]
    if data["ongoing"]:
        return data["first_yield_day"] + (data["max_yield"] - 1) * data["interval"]
    return data["max_yield_day"]


def _is_fertilizable(tile, day):
    return (
        _is_plant(tile)
        and tile["crop"] in FERTILIZE_CROPS
        and tile["fertilized_until_day"] < day
        and day - tile["planted_day"] <= _last_yield_day(tile["crop"])
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

    def __init__(self, private, farm, days_remaining, prices=None):
        self.seed_budget = dict(private.get("seeds", {}))
        self.prices = prices or {}
        # A crop that can't reach first_yield_day (+ a day of slack to sell it
        # after) before the season ends is a pure loss to plant -- see the
        # module docstring's Endgame liquidation note.
        self.plantable_crops = {
            c for c in CASH_CROPS if days_remaining >= CROPS[c]["first_yield_day"] + SEED_BUFFER_DAYS
        }
        self.counts = {c: 0 for c in CASH_CROPS}
        for row in farm["tiles"]:
            for tile in row:
                if _is_plant(tile) and tile["crop"] in CASH_CROPS:
                    self.counts[tile["crop"]] += 1
        self.total = sum(self.counts.values()) or 1

    def has_plantable(self):
        return any(self.seed_budget.get(c, 0) > 0 for c in self.plantable_crops)

    def _price_ok(self, crop):
        """Would we currently be willing to *sell* this crop? If not (and we
        have price data at all), planting more of it just grows tomorrow's
        shed pile in a market that's already below what we're willing to
        accept -- see the module docstring's Price-aware planting note."""
        price = self.prices.get(crop)
        if price is None:
            return True
        base = MARKET_PARAMS[crop]["base"]
        return price >= base * SELL_PARAMS[crop]["threshold"]

    def choose(self):
        """Pick whichever affordable, currently-worth-selling crop is
        furthest below its target mix share. Falls back to ignoring price
        if every plantable crop is currently depressed, so land never sits
        idle over a temporary dip."""
        available = [c for c in self.plantable_crops if self.seed_budget.get(c, 0) > 0]
        if not available:
            return None
        pool = [c for c in available if self._price_ok(c)] or available
        crop = min(pool, key=lambda c: self.counts[c] / self.total - CROP_TARGET_SHARE[c])
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
            if _is_plant(tile):
                if _is_harvestable(tile, day):
                    candidates.append((x, y, 0))  # harvest
                elif not tile["watered_today"]:
                    candidates.append((x, y, water_priority))
                elif inv.get("FERTILIZER", 0) > 0 and _is_fertilizable(tile, day):
                    candidates.append((x, y, 3))  # deliver fertilizer
            elif _is_weed(tile):
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

    if _is_plant(tile):
        if _is_harvestable(tile, day):
            claimed.add((fx, fy))
            return ["HARVEST"]
        if not tile["watered_today"]:
            claimed.add((fx, fy))
            return ["WATER"]
        if inv.get("FERTILIZER", 0) > 0 and _is_fertilizable(tile, day):
            claimed.add((fx, fy))
            return ["FERTILIZE"]

    if _is_weed(tile):
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
    unlocked = [tile for row in farm["tiles"] for tile in row if tile != "LOCKED"]
    planted = sum(1 for tile in unlocked if _is_plant(tile))
    if unlocked and planted / len(unlocked) < LAND_UTILIZATION_GATE:
        return []  # can't staff what we already own -- more land would just sit idle
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
    planner = _CropPlanner(private, farm, days_remaining, market_prices)

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

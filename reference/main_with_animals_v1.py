"""Snapshot of `main.py` WITH animal support (GOOSE/COW/SHEEP), kept for the
record after A/B testing showed it's a clear net negative vs. the animal-free
baseline (`reference/main_pre_animals.py`) -- NOT the active submission.

Deterministic A/B vs. `starter`, 3 seeds: this version scored ~$7-8k;
`main_pre_animals.py` scored ~$22-24k under otherwise identical conditions.
That's roughly a 70% cut to final money, and it held up after fixing four
real bugs found along the way (dead capital from a missing PICKUP-animal
step, a turn-0 purchase spree that drained starting cash, structures built
too far from the shed for daily feeding to be practical, and FEED losing
distance tiebreaks against far more numerous crop-watering candidates until
it was given strictly higher priority). The residual problem looks
structural, not another bug: animals need constant per-day attention (FEED,
CARE, COLLECT_FERTILIZER) that competes for the same scarce farmer/hand
turns as the already-tuned, much higher value/tile/day crop loop (melon
especially), and a single missed feeding cycle can lose an entire $300-500
purchase outright. Revisit if a future iteration of this agent has
meaningfully more spare unit-turns (more hands, cheaper travel) to spend on
upkeep without cannibalizing the crop loop.

Kaggle loads this file and calls `agent(obs)` (or `agent(obs, config)`) once
per turn for whichever player index we've been assigned. Keep this file
self-contained (or bundle extra modules into a tar.gz with this at the root
per AGENTS.md) since it's shipped standalone to the competition runner.

Strategy ("farm maxxer"): a scripted multi-unit farm loop built on top
of `reference/melon_maxxer.py` (Kaggle's own official example agent, see its
docstring), fixing its four documented weaknesses:

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
  - melon_maxxer dumps its entire shed in a single SELL order once a
    threshold is crossed, which can crash the price mid-sale (price moves
    unit-by-unit *within* an order, see data/README.md's Market Mechanics)
    -> this agent caps how much of each crop it sells per turn, spreading
    a big harvest over several turns instead of one dump, and only lifts
    the cap when the shed is close to overflowing (losing produce to a full
    shed is strictly worse than a bad sale price).

**Animals (v2)**: melon_maxxer never touches coops/pastures either, but this
agent does -- GOOSE/COW/SHEEP, mixed by the same target-share logic as the
crops (`_AnimalPlanner`, mirroring `_CropPlanner`). Unlike crops, a built
COOP/PASTURE occupies its tile for the rest of the game instead of cycling
back to empty (animals are ongoing/indefinite -- data/README.md's Object
Types table), so this is a small, capped, diversifying land allocation
(`MAX_ANIMAL_STRUCTURES`), not the primary use of land the way crops
(especially melon) are. Two synergies worth calling out: `COLLECT_FERTILIZER`
feeds the same FERTILIZE pipeline built for WHEAT/CARROT above at zero
marginal cost once an animal is producing, and EGG/MILK/WOOL are markets we
don't otherwise touch, so animal income doesn't compete with crop sales for
the same price curve. `FEED` needs WHEAT carried in a unit's inventory (not
the shed directly, same PICKUP dance as fertilizer), so a fraction of shed
wheat is held back from sale as a standing feed reserve
(`WHEAT_FEED_RESERVE_PER_ANIMAL`) rather than always being offered for SELL.

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
    ANIMALS,
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
    # threshold: fraction of base price required to sell at all.
    # cap: max units of this crop sold per turn (spreads a big harvest out
    # instead of dumping it in one order -- see module docstring).
    "MELON":  {"threshold": 0.65, "cap": 6},
    "WHEAT":  {"threshold": 0.45, "cap": 20},
    "CARROT": {"threshold": 0.45, "cap": 15},
    "EGG":    {"threshold": 0.50, "cap": 10},
    "MILK":   {"threshold": 0.50, "cap": 8},
    "WOOL":   {"threshold": 0.50, "cap": 8},
}

ANIMAL_TYPES = ("GOOSE", "COW", "SHEEP")
ANIMAL_TARGET_SHARE = {"GOOSE": 0.4, "COW": 0.35, "SHEEP": 0.25}
# Unlike crops, a built COOP/PASTURE occupies its tile for the rest of the
# game (animals are ongoing/indefinite -- data/README.md's Object Types
# table) instead of cycling back to empty, so this is a small, one-time,
# diversifying land allocation, not the primary use of land the way crops
# (especially melon) are -- see the module docstring.
MAX_ANIMAL_STRUCTURES = 6
ANIMAL_MAX_SHED_DISTANCE = 4  # keep structures close enough that a daily FEED round trip is cheap -- see _find_target
ANIMAL_BUFFER_DAYS = 1  # like SEED_BUFFER_DAYS: needs first_yield_day + this much time left to be worth it
WHEAT_FEED_RESERVE_PER_ANIMAL = 2  # days of feed kept back from sale per live animal
WHEAT_FEED_MAX_PRICE_FRAC = 1.5  # willing to pay more for feed-wheat than for fertilizer -- see _build_wheat_feed_orders
ANIMAL_SAFETY_RESERVE = 800  # higher than SAFETY_RESERVE: animals are a chunky, slow-payoff spend that
# shouldn't crowd out land/hiring/seed money early -- see _build_animal_orders

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


def _dist_to_shed(x, y, shed_tiles):
    return min(abs(x - sx) + abs(y - sy) for sx, sy in shed_tiles)


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


def _is_animal_tile(tile):
    return isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE")


def _count_live_animals(farm):
    return sum(
        1 for row in farm["tiles"] for tile in row if _is_animal_tile(tile) and tile.get("animal")
    )


def _has_unoccupied_structure(farm, structure_kind):
    return any(
        isinstance(tile, dict) and tile.get("kind") == structure_kind and not tile.get("animal")
        for row in farm["tiles"]
        for tile in row
    )


def _count_hungry_animals(farm):
    """FEED has no age/window gate like FERTILIZE -- an animal needs it every
    single day it's alive, or it escapes after 2 consecutive misses."""
    return sum(
        1
        for row in farm["tiles"]
        for tile in row
        if _is_animal_tile(tile) and tile.get("animal") and not tile["fed_today"]
    )


def _wants_animal_pickup(farm, shed):
    """A bought animal sitting in the shed is dead capital until it's PICKUP'd
    and PLACEd -- this is what makes that happen. Without it, BUY_ANIMAL
    orders would spend real money on an animal that never gets used."""
    return any(shed.get(a, 0) > 0 and _has_unoccupied_structure(farm, ANIMALS[a]["structure"]) for a in ANIMAL_TYPES)


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


class _AnimalPlanner:
    """Mirrors `_CropPlanner` for animal structures/species, with one key
    difference: a built COOP/PASTURE never goes back to empty (animals are
    ongoing/indefinite), so this only ever grows towards
    `MAX_ANIMAL_STRUCTURES` -- it never "replants" like the crop planner does.
    """

    def __init__(self, farm, days_remaining):
        self.viable = {
            a for a in ANIMAL_TYPES if days_remaining >= ANIMALS[a]["first_yield_day"] + ANIMAL_BUFFER_DAYS
        }
        self.counts = {a: 0 for a in ANIMAL_TYPES}
        self.total_structures = 0
        for row in farm["tiles"]:
            for tile in row:
                if not _is_animal_tile(tile):
                    continue
                self.total_structures += 1
                animal = tile.get("animal")
                if animal:
                    self.counts[animal] += 1

    def wants_more_structures(self):
        return bool(self.viable) and self.total_structures < MAX_ANIMAL_STRUCTURES

    def choose_species(self):
        """Pick whichever viable species is furthest below its target mix share."""
        if not self.wants_more_structures():
            return None
        share_total = self.total_structures or 1
        species = min(self.viable, key=lambda a: self.counts[a] / share_total - ANIMAL_TARGET_SHARE[a])
        self.counts[species] += 1
        self.total_structures += 1
        return species

    @staticmethod
    def species_to_place(inv, structure_kind):
        """Which held (already-purchased) animal fits the structure a unit is standing on."""
        for a in ANIMAL_TYPES:
            if inv.get(a, 0) > 0 and ANIMALS[a]["structure"] == structure_kind:
                return a
        return None


def _find_target(
    pos, farm, day, board_size, claimed, inv, planner, animal_planner, shed_tiles, want_fertilizer, want_feed,
    want_animal_pickup, liquidating,
):
    """Scan the whole farm for the nearest unclaimed tile worth walking to.

    Priority tiers (lower = more urgent): 0 harvest, 1 feed, 2 water, 3 dig
    weed, 4 collect animal fertilizer, 5 deliver crop fertilizer, 6 build a
    structure, 7 plant a crop / place a carried animal, 8 care (animal,
    opportunistic filler), 9 return to shed. Feed sits *above* water, not
    tied with it: an earlier version tied them, and with dozens of crop
    tiles typically outnumbering a handful of animals, ties were broken by
    distance and animals almost always lost that tiebreak to nearer crops --
    confirmed the root cause of a starvation problem via instrumentation
    (most purchased animals never getting fed in time despite guaranteed
    feed supply). Missing one day of water costs a bonus; missing one day of
    feed (twice in a row) is an animal's entire purchase price gone for good.

    `liquidating` (close to the last day, see the module docstring's Endgame
    liquidation note) swaps feed/water's and shed-return's priority: tending
    something that won't finish in time is wasted effort, while getting
    already-harvested produce into the shed guarantees it can still be sold.
    """
    fx, fy = pos
    feed_priority = 9 if liquidating else 1
    water_priority = 9 if liquidating else 2
    shed_priority = 1 if liquidating else 9
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
                    candidates.append((x, y, 5))  # deliver fertilizer
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                candidates.append((x, y, 3))  # clear a weed to reclaim the tile
            elif _is_animal_tile(tile):
                if tile.get("animal"):
                    if tile["yield_units"] > 0:
                        candidates.append((x, y, 0))  # harvest eggs/milk/wool
                    elif not tile["fed_today"]:
                        candidates.append((x, y, feed_priority))
                    elif tile.get("fertilizer_available"):
                        candidates.append((x, y, 4))  # collect fertilizer byproduct
                    elif not tile["cared_today"]:
                        candidates.append((x, y, 8))  # care: cheap, no urgency, opportunistic
                elif _AnimalPlanner.species_to_place(inv, tile["kind"]):
                    candidates.append((x, y, 7))  # place a carried animal here
            elif tile is None:
                if animal_planner.wants_more_structures() and _dist_to_shed(x, y, shed_tiles) <= ANIMAL_MAX_SHED_DISTANCE:
                    candidates.append((x, y, 6))  # build a coop/pasture, close enough to feed daily
                elif planner.has_plantable():
                    candidates.append((x, y, 7))  # plant

    carrying_produce = any(k != "FERTILIZER" and k not in ANIMAL_TYPES and n > 0 for k, n in inv.items())
    carrying_animal = any(inv.get(a, 0) > 0 for a in ANIMAL_TYPES)
    needs_fertilizer = inv.get("FERTILIZER", 0) == 0 and want_fertilizer
    needs_wheat = inv.get("WHEAT", 0) == 0 and want_feed and not carrying_animal
    needs_animal = not carrying_animal and want_animal_pickup
    # Fetching feed-wheat is step one of feeding, so it needs feed's own
    # urgency, not the general low-priority shed-return tier -- tying it to
    # `shed_priority` (as an earlier version did) meant FEED being the map's
    # top priority accomplished nothing, since almost no unit ever carried
    # wheat to act on it in the first place. Confirmed via instrumentation:
    # escapes went *up*, not down, when only the on-tile FEED priority was
    # raised without this.
    shed_reasons = []
    if needs_wheat:
        shed_reasons.append(feed_priority)
    if carrying_produce or (needs_fertilizer and not carrying_animal) or needs_animal:
        shed_reasons.append(shed_priority)
    if shed_reasons:
        best = min(shed_reasons)
        for (sx, sy) in shed_tiles:
            if (sx, sy) not in claimed:
                candidates.append((sx, sy, best))  # return to shed

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[2], abs(c[0] - fx) + abs(c[1] - fy)))
    return candidates[0]


def _decide_unit(
    pos, inv, farm, private, day, board_size, claimed, planner, animal_planner, shed_tiles, want_fertilizer,
    want_feed, hungry_count, want_animal_pickup, liquidating,
):
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

    if _is_animal_tile(tile):
        if tile.get("animal"):
            if tile["yield_units"] > 0:
                claimed.add((fx, fy))
                return ["HARVEST"]
            if not tile["fed_today"] and inv.get("WHEAT", 0) > 0:
                claimed.add((fx, fy))
                return ["FEED"]
            if not tile["cared_today"]:
                claimed.add((fx, fy))
                return ["CARE"]
            if tile.get("fertilizer_available"):
                claimed.add((fx, fy))
                return ["COLLECT_FERTILIZER"]
        else:
            species = _AnimalPlanner.species_to_place(inv, tile["kind"])
            if species:
                claimed.add((fx, fy))
                return ["PLACE", species]

    if (fx, fy) in shed_tiles:
        carried_produce = {k: n for k, n in inv.items() if k != "FERTILIZER" and k not in ANIMAL_TYPES and n > 0}
        if carried_produce:
            claimed.add((fx, fy))
            return ["DROP"]
        shed_wheat = private["shed"].get("WHEAT", 0)
        if inv.get("WHEAT", 0) == 0 and want_feed and shed_wheat > 0:
            # Checked before fertilizer/animal pickup -- feeding is the most
            # time-critical use of a shed visit (see _find_target's priority
            # note). Grab enough for every currently-hungry animal in one
            # trip, not just one -- FEED (unlike WATER) needs the unit to be
            # carrying the item, so a 1-at-a-time pickup meant a full shed
            # round trip per animal per day. That logistics tax was the root
            # cause of a confirmed starvation problem (13 of 19 purchased
            # animals escaped from neglect in one full-season run before
            # this fix).
            claimed.add((fx, fy))
            return ["PICKUP", "WHEAT", min(shed_wheat, max(1, hungry_count))]
        if inv.get("FERTILIZER", 0) == 0 and want_fertilizer and private["shed"].get("FERTILIZER", 0) > 0:
            claimed.add((fx, fy))
            return ["PICKUP", "FERTILIZER", 1]
        if want_animal_pickup and not any(inv.get(a, 0) > 0 for a in ANIMAL_TYPES):
            for a in ANIMAL_TYPES:
                if private["shed"].get(a, 0) > 0 and _has_unoccupied_structure(farm, ANIMALS[a]["structure"]):
                    claimed.add((fx, fy))
                    return ["PICKUP", a, 1]

    if tile is None:
        if animal_planner.wants_more_structures() and _dist_to_shed(fx, fy, shed_tiles) <= ANIMAL_MAX_SHED_DISTANCE:
            species = animal_planner.choose_species()
            if species:
                claimed.add((fx, fy))
                return [f"BUILD_{ANIMALS[species]['structure']}"]
        crop = planner.choose()
        if crop:
            claimed.add((fx, fy))
            return ["PLANT", crop]

    target = _find_target(
        pos, farm, day, board_size, claimed, inv, planner, animal_planner, shed_tiles, want_fertilizer, want_feed,
        want_animal_pickup, liquidating,
    )
    if target:
        claimed.add((target[0], target[1]))
        step = _step_toward(fx, fy, target[0], target[1])
        if step:
            return [step]
    return ["PASS"]


def _build_sell_orders(shed, prices, days_remaining, wheat_feed_reserve):
    orders = []
    shed_total = sum(shed.values())
    crowded = shed_total >= SHED_CAPACITY * SHED_OVERFLOW_GUARD
    liquidating = days_remaining <= ENDGAME_LIQUIDATION_DAYS
    dump_uncapped = crowded or days_remaining <= 0  # no more turns left to protect the price
    for crop, params in SELL_PARAMS.items():
        qty = shed.get(crop, 0)
        # Keep enough wheat back to feed live animals -- unless the game is
        # basically over, in which case there's no more feeding left to do.
        if crop == "WHEAT" and days_remaining > 0:
            qty = max(0, qty - wheat_feed_reserve)
        if qty <= 0:
            continue
        base = MARKET_PARAMS[crop]["base"]
        price = prices.get(crop, base)
        if crowded or liquidating or price >= base * params["threshold"]:
            cap = qty if dump_uncapped else params["cap"]
            orders.append(["SELL", crop, min(qty, cap)])
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


def _build_wheat_feed_orders(shed, prices, money, live_animal_count):
    """Top up the feed reserve directly via BUY_PRODUCT rather than only
    hoping our own wheat crop harvests happen to cover it -- WHEAT_FEED
    reserve (see _build_sell_orders) only stops the reserve from being SOLD
    away, it doesn't refill it. A willingness to pay somewhat above base
    price is deliberate: an unfed animal escaping wastes its entire $300-500
    purchase, a loss many times a wheat order's cost.
    """
    if live_animal_count <= 0:
        return []
    target = live_animal_count * WHEAT_FEED_RESERVE_PER_ANIMAL
    have = shed.get("WHEAT", 0)
    if have >= target:
        return []
    base = MARKET_PARAMS["WHEAT"]["base"]
    price = prices.get("WHEAT", base)
    if price > base * WHEAT_FEED_MAX_PRICE_FRAC:
        return []
    affordable = int((money - SAFETY_RESERVE) // price) if price > 0 else 0
    need = min(target - have, max(0, affordable))
    if need <= 0:
        return []
    return [["BUY_PRODUCT", "WHEAT", need]]


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


def _build_animal_orders(farm, private, money, days_remaining):
    """Buy toward MAX_ANIMAL_STRUCTURES's target mix, one animal per turn.

    Independently recomputes farm state rather than sharing
    `_AnimalPlanner`'s mutated counts (same style as `_build_seed_orders` vs.
    `_CropPlanner`) since this runs in a separate phase (market-order
    building, after unit actions are already decided) against the same
    pre-turn snapshot. Capped to one purchase/turn and gated behind a reserve
    well above `SAFETY_RESERVE` -- an early run showed all 5-6 structures'
    worth of animals ($300-500 each) getting bought in a single turn 0,
    which is a real capital cost (unlike the free BUILD_COOP/PASTURE) and,
    queued that aggressively, starved land/hiring/seed money for the first
    several days. Land expansion and the crop cash-flow loop fund everything
    else, so they get first claim on early money; animals are the slower,
    longer-horizon payoff and can afford to be paced out.
    """
    viable = [
        a for a in ANIMAL_TYPES if days_remaining >= ANIMALS[a]["first_yield_day"] + ANIMAL_BUFFER_DAYS
    ]
    if not viable:
        return []

    counts = {a: 0 for a in ANIMAL_TYPES}
    total_structures = 0
    for row in farm["tiles"]:
        for tile in row:
            if not _is_animal_tile(tile):
                continue
            total_structures += 1
            animal = tile.get("animal")
            if animal:
                counts[animal] += 1

    # What gates a new purchase is *unoccupied* structures, not the total
    # built -- BUILD_COOP/PASTURE is free and gets built up to
    # MAX_ANIMAL_STRUCTURES quickly, well before animals (paced to one
    # purchase/turn, see below) catch up to fill them. Gating on the total
    # instead (an earlier version of this function did) meant buying stopped
    # dead the moment structures hit the cap, even with most of them still
    # empty -- confirmed via instrumentation: only 2 of 6 built structures
    # ever got an animal in a full-season run.
    unoccupied = sum(
        1 for row in farm["tiles"] for tile in row if _is_animal_tile(tile) and not tile.get("animal")
    )
    shed = private.get("shed", {})
    in_stock = sum(shed.get(a, 0) for a in ANIMAL_TYPES)  # bought, awaiting PICKUP + PLACE
    if unoccupied - in_stock <= 0:
        return []

    remaining_cash = money - ANIMAL_SAFETY_RESERVE
    candidates = [a for a in viable if ANIMALS[a]["cost"] <= remaining_cash]
    if not candidates:
        return []
    share_total = (total_structures + in_stock) or 1
    species = min(candidates, key=lambda a: counts[a] / share_total - ANIMAL_TARGET_SHARE[a])
    return [["BUY_ANIMAL", species, 1]]


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
    hungry_count = _count_hungry_animals(farm)
    want_feed = hungry_count > 0
    want_animal_pickup = _wants_animal_pickup(farm, shed)
    planner = _CropPlanner(private, farm, days_remaining)
    animal_planner = _AnimalPlanner(farm, days_remaining)

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
            animal_planner,
            shed_tiles,
            want_fertilizer,
            want_feed,
            hungry_count,
            want_animal_pickup,
            liquidating,
        )
        for i, pos in enumerate(units)
    ]
    farmer_action, hands_actions = actions[0], actions[1:]

    live_animal_count = _count_live_animals(farm)
    wheat_feed_reserve = live_animal_count * WHEAT_FEED_RESERVE_PER_ANIMAL
    market = []
    market += _build_sell_orders(shed, market_prices, days_remaining, wheat_feed_reserve)
    market += _build_land_orders(farm, days_remaining)
    market += _build_hire_orders(farm)
    market += _build_wheat_feed_orders(shed, market_prices, farm["money"], live_animal_count)
    market += _build_seed_orders(seeds, farm["money"], days_remaining)
    market += _build_fertilizer_orders(private, market_prices, farm["money"])
    market += _build_animal_orders(farm, private, farm["money"], days_remaining)
    market = market[:MAX_MARKET_ORDERS]

    return {"farmer": farmer_action, "hands": hands_actions, "market": market}

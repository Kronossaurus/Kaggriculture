"""Frozen snapshot of `main.py` as submitted to Kaggle on 2026-08-17 (see
`scripts/pool_winrate.py`'s DEFAULT_POOL and the module docstring's snapshot
convention in `reference/main_dump_sell.py`). Captures the state after: the
land-utilization-gate recalibration + hire-capacity cap (_land_worth_buying),
animal husbandry attempt 3 (land-share-sized PASTURE/COOP, shared crop/animal
crew), the animal-zone + quadrant-utilization spatial clustering fix, and
fertilizer selling. See git log around this commit and main.py's own
docstring for the full investigation history. Not maintained going forward --
a frozen comparison point, not a second strategy to keep developing.
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

CASH_CROPS = ("MELON", "WHEAT", "CARROT", "TOMATO", "STRAWBERRY")
CROP_TARGET_SHARE = {"MELON": 0.45, "WHEAT": 0.25, "CARROT": 0.15, "TOMATO": 0.10, "STRAWBERRY": 0.05}
FERTILIZE_CROPS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY"}

SEED_RESERVE = {"MELON": 3, "WHEAT": 3, "CARROT": 3, "TOMATO": 2, "STRAWBERRY": 2}
FERTILIZER_RESERVE = 2
FERTILIZER_MAX_PRICE_FRAC = 1.2  # don't buy fertilizer more than 20% over base

SELL_PARAMS = {
    # threshold: fraction of base price required to sell at all -- see
    # _build_sell_orders for why there's no per-turn cap. Shared by
    # _CropPlanner and _AnimalPlanner's own _price_ok, so a crop/animal
    # kind's plant/build eligibility always matches its sell eligibility.
    "MELON":      {"threshold": 0.65},
    "WHEAT":      {"threshold": 0.45},
    "CARROT":     {"threshold": 0.45},
    "TOMATO":     {"threshold": 0.50},
    "STRAWBERRY": {"threshold": 0.55},
    "MILK":       {"threshold": 0.50},
    "WOOL":       {"threshold": 0.60},  # WOOL's above-equilibrium price shape is nearly
                                         # as steep as MELON's, see the Price-aware
                                         # planting note
    "EGG":        {"threshold": 0.35},
    "FERTILIZER": {"threshold": 0.50},  # collected free from animal structures (COLLECT_FERTILIZER)
                                         # in excess of what crops currently need -- see the module
                                         # docstring's Selling fertilizer note
}

SHED_CAPACITY = 100  # data/README.md default; not exposed in `obs`
SHED_OVERFLOW_GUARD = 0.85  # start dumping regardless of price above this fraction full

ANIMAL_KINDS = ("COW", "SHEEP", "GOOSE")
ANIMAL_PRODUCT = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
ANIMAL_STRUCTURE = {"COW": "PASTURE", "SHEEP": "PASTURE", "GOOSE": "COOP"}
ANIMAL_PRODUCTS = ("MILK", "WOOL", "EGG")  # finished products a unit carries to the shed
ANIMAL_TARGET_SHARE = {"COW": 0.60, "SHEEP": 0.40, "GOOSE": 0.0}  # GOOSE has the
    # lowest $/day of the three ($50: EGG base 50 / 1-day interval, vs COW's $80 and
    # SHEEP's $66.7) despite being the cheapest to buy -- 5/5 downloaded top-ladder
    # replays (3 different players) never place one; see the module docstring's
    # Animal husbandry (attempt 3) note
ANIMAL_LAND_SHARE = 0.18  # target fraction of owned tiles reserved for animal
                          # structures -- see the module docstring's Animal husbandry
                          # (attempt 3) note; top-ladder replays hold this steady
                          # around 15-24% throughout the season
ANIMAL_RESERVE = 1  # spare bought-but-not-yet-placed animal tokens to keep queued, per kind
FEED_RESERVE_PER_ANIMAL = 2  # days of WHEAT buffer per live animal, bought directly from
                              # the market -- see _build_feed_orders for why
FEED_MAX_PRICE_FRAC = 5.0  # a near-nonbinding backstop, not a real cap: missing a feed
                            # loses the whole animal ($300-500+), dwarfing even a wildly
                            # inflated wheat price -- see the module docstring's Animal
                            # husbandry (attempt 2) note for why a tighter cap is a trap

MAX_HANDS = 6
SAFETY_RESERVE = 300  # cash kept in reserve before optional buys/land/fertilizer
MAX_MARKET_ORDERS = 10  # data/README.md default; not exposed in `obs`

DEFAULT_EPISODE_STEPS = 720  # data/README.md defaults, used if `config` is unavailable
DEFAULT_TURNS_PER_DAY = 24
SEED_BUFFER_DAYS = 1  # a crop needs first_yield_day + this many days left to be worth planting
LAND_CUTOFF_DAYS = 5  # stop buying new land this close to the end, it won't pay back in time
LAND_UTILIZATION_GATE = 0.45  # don't buy the next quadrant until this much of what we
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


def _dist_to_shed(x, y, board_size):
    return min(abs(x - sx) + abs(y - sy) for sx, sy in _shed_adjacent_tiles(board_size))


def _quadrant_of(x, y, board_size):
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def _quadrant_utilization(farm, board_size):
    """Fraction of each unlocked quadrant's own tiles currently in use
    (planted, or holding an animal structure) -- used to steer new
    plantings toward whichever quadrant is most neglected, see the module
    docstring's spatial clustering (crops) note. A locked quadrant's
    utilization is irrelevant (no empty-tile candidate is ever generated
    there, its tiles are the "LOCKED" string, not None) and defaults to 0.0
    for the div-by-zero case, which never actually gets used."""
    used = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}
    total = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if tile == "LOCKED":
                continue
            q = _quadrant_of(x, y, board_size)
            total[q] += 1
            if _is_plant(tile) or (isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP")):
                used[q] += 1
    return {q: (used[q] / total[q] if total[q] else 0.0) for q in used}


def _animal_zone(farm, board_size, needed):
    """The `needed` empty tiles closest to the shed -- animal structures
    claim this zone first, leaving the rest of the map free for crops to
    spread into, instead of both competing for whichever empty tile happens
    to be nearest the unit that finds it first. See the module docstring's
    spatial clustering (animals) note: real top-ladder play keeps every
    animal structure in a tight ring around the shed (FEED is daily and
    unmissable, so a short walk there matters), with crops filling the rest
    of the map edge-to-edge."""
    if needed <= 0:
        return set()
    empties = [(x, y) for y in range(board_size) for x in range(board_size) if farm["tiles"][y][x] is None]
    empties.sort(key=lambda p: _dist_to_shed(p[0], p[1], board_size))
    return set(empties[:needed])


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


def _is_animal_tile(tile):
    return isinstance(tile, dict) and bool(tile.get("animal"))


def _is_empty_structure(tile):
    return isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP") and not tile.get("animal")


def _carried_animal_for(tile, inv):
    """Which animal kind (if any) in `inv` matches this empty structure's type."""
    for kind in ANIMAL_KINDS:
        if ANIMAL_STRUCTURE[kind] == tile.get("kind") and inv.get(kind, 0) > 0:
            return kind
    return None


class _AnimalPlanner:
    """Tracks, for one turn, which animal kind to build a structure for or
    fetch from the shed next -- mirrors _CropPlanner's price-aware,
    share-balancing logic. Unlike attempt 2 (see the module docstring's
    Animal husbandry notes), capacity is sized as a target *fraction of
    owned land* (`ANIMAL_LAND_SHARE`), not a fixed hand allocation, and any
    unit -- farmer or hand -- may act on an animal task; there's no
    animal-only crew. `wants_structure`/`wants_pickup` are read-only (safe
    during candidate scanning in _find_target); `choose_structure`/
    `choose_pickup` mutate reserve counters, so only call them once a unit
    is actually standing on the tile it's about to act on."""

    def __init__(self, farm, shed, prices):
        self.counts = {kind: 0 for kind in ANIMAL_KINDS}
        self.empty_structures = {"PASTURE": 0, "COOP": 0}
        unlocked_tiles = 0
        for row in farm["tiles"]:
            for tile in row:
                if tile == "LOCKED":
                    continue
                unlocked_tiles += 1
                if not (isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP")):
                    continue
                animal = tile.get("animal")
                if animal:
                    self.counts[animal] += 1
                else:
                    self.empty_structures[tile["kind"]] += 1
        self.total = sum(self.counts.values()) or 1
        self.shed = dict(shed)
        self.prices = prices or {}
        self.target = round(unlocked_tiles * ANIMAL_LAND_SHARE)

    def _committed(self):
        return sum(self.counts.values()) + sum(self.empty_structures.values())

    def remaining_structures(self):
        """How many more structures still need to be built to reach the
        land-share target -- used to size the near-shed zone _animal_zone
        reserves, not to decide whether to build (choose_structure/
        wants_structure remain the source of truth for that)."""
        return max(0, self.target - self._committed())

    def _price_ok(self, kind):
        product = ANIMAL_PRODUCT[kind]
        price = self.prices.get(product)
        if price is None:
            return True
        base = MARKET_PARAMS[product]["base"]
        return price >= base * SELL_PARAMS[product]["threshold"]

    def wants_structure(self):
        if self._committed() >= self.target:
            return False
        return any(self.empty_structures[ANIMAL_STRUCTURE[k]] == 0 for k in ANIMAL_KINDS)

    def wants_pickup(self):
        return any(
            self.empty_structures[ANIMAL_STRUCTURE[k]] > 0 and self.shed.get(k, 0) > 0 for k in ANIMAL_KINDS
        )

    def choose_structure(self):
        """Pick a kind with no spare (built-but-empty) structure yet, preferring
        whichever affordable, currently-worth-selling kind is furthest below
        its target mix share -- same fallback as _CropPlanner.choose().
        Excludes kinds with a 0 target share (a "structure type has zero
        spares" check alone would otherwise let a 0-share kind win as soon
        as its structure type is the only one with no spare waiting --
        checked at the module docstring's Animal husbandry (attempt 3) note
        after that let GOOSE keep getting built despite a 0 target share)."""
        if self._committed() >= self.target:
            return None
        needing = [
            k for k in ANIMAL_KINDS
            if ANIMAL_TARGET_SHARE[k] > 0 and self.empty_structures[ANIMAL_STRUCTURE[k]] == 0
        ]
        if not needing:
            return None
        pool = [k for k in needing if self._price_ok(k)] or needing
        kind = min(pool, key=lambda k: self.counts[k] / self.total - ANIMAL_TARGET_SHARE[k])
        self.empty_structures[ANIMAL_STRUCTURE[kind]] += 1  # reserve so two units don't both build one
        return kind

    def choose_pickup(self):
        """Pick a kind that already has a spare structure waiting AND a
        bought token sitting in the shed."""
        ready = [
            k for k in ANIMAL_KINDS
            if ANIMAL_TARGET_SHARE[k] > 0
            and self.empty_structures[ANIMAL_STRUCTURE[k]] > 0
            and self.shed.get(k, 0) > 0
        ]
        if not ready:
            return None
        pool = [k for k in ready if self._price_ok(k)] or ready
        kind = min(pool, key=lambda k: self.counts[k] / self.total - ANIMAL_TARGET_SHARE[k])
        self.empty_structures[ANIMAL_STRUCTURE[kind]] -= 1  # reserve so two units don't grab the same slot
        self.shed[kind] = self.shed.get(kind, 0) - 1
        return kind


def _find_target(
    pos, farm, day, board_size, claimed, inv, planner, animal_planner, shed_tiles,
    want_fertilizer, shed_wheat_available, liquidating, animal_zone, quadrant_util,
):
    """Scan the whole farm for the nearest (or, for a fresh planting,
    nearest-adjusted-for-quadrant-balance -- see below) unclaimed tile worth
    walking to, across both crops and animals -- one shared crew, one shared
    priority order, so a unit works whichever is more urgent right now
    instead of being permanently assigned to one or the other (see the
    module docstring's Animal husbandry (attempt 3) note).

    `liquidating` (close to the last day, see the module docstring's Endgame
    liquidation note) swaps water's and shed-return's priority: watering a
    crop that won't finish growing in time is wasted effort, while getting
    already-harvested produce into the shed guarantees it can still be sold.

    Every candidate carries a precomputed `sort_dist` (its 4th field) rather
    than the sort key recomputing raw distance uniformly: a fresh PLANT
    candidate gets its raw distance inflated by how utilized its quadrant
    already is (`quadrant_util`), nudging new plantings toward whichever
    quadrant is most neglected instead of always the literal nearest empty
    tile -- see the module docstring's spatial clustering (crops) note. Every
    other candidate type keeps plain, unadjusted distance.
    """
    fx, fy = pos
    water_priority = 5 if liquidating else 1
    shed_priority = 1 if liquidating else 5
    spread_penalty = board_size // 2  # scale of the quadrant-balance nudge, in tiles
    candidates = []
    any_unfed = False
    for y in range(board_size):
        for x in range(board_size):
            if (x, y) in claimed:
                continue
            tile = farm["tiles"][y][x]
            dist = abs(x - fx) + abs(y - fy)
            if _is_plant(tile):
                if _is_harvestable(tile, day):
                    candidates.append((x, y, 0, dist))  # harvest
                elif not tile["watered_today"]:
                    candidates.append((x, y, water_priority, dist))
                elif inv.get("FERTILIZER", 0) > 0 and _is_fertilizable(tile, day):
                    candidates.append((x, y, 3, dist))  # deliver fertilizer
            elif _is_weed(tile):
                candidates.append((x, y, 2, dist))  # clear a weed to reclaim the tile
            elif _is_animal_tile(tile):
                if tile.get("yield_units", 0) > 0:
                    candidates.append((x, y, 0, dist))  # harvest animal product
                elif not tile.get("fed_today"):
                    any_unfed = True
                    if inv.get("WHEAT", 0) > 0:
                        candidates.append((x, y, -1, dist))  # feed -- more urgent than
                                                               # anything; 2 misses = the
                                                               # animal escapes
                elif tile.get("fertilizer_available"):
                    candidates.append((x, y, 3, dist))  # collect fertilizer
                elif not tile.get("cared_today"):
                    candidates.append((x, y, 3, dist))  # care (free next-yield bonus)
            elif _is_empty_structure(tile) and _carried_animal_for(tile, inv):
                candidates.append((x, y, 0, dist))  # place a carried animal -- don't leave it idle
            elif tile is None:
                if (x, y) in animal_zone:
                    candidates.append((x, y, 4, dist))  # reserved for an animal structure
                elif planner.has_plantable():
                    q = _quadrant_of(x, y, board_size)
                    candidates.append((x, y, 4, dist + quadrant_util[q] * spread_penalty))  # plant
                elif animal_planner.wants_structure():
                    candidates.append((x, y, 4, dist))  # build a structure (overflow beyond the zone)

    carrying_produce = any(
        n > 0 and k not in ("WHEAT", *ANIMAL_KINDS) and (k != "FERTILIZER" or not want_fertilizer)
        for k, n in inv.items()
    )
    needs_fertilizer = inv.get("FERTILIZER", 0) == 0 and want_fertilizer
    needs_wheat = inv.get("WHEAT", 0) <= 0 and any_unfed and shed_wheat_available
    needs_animal_pickup = animal_planner.wants_pickup() and not any(inv.get(k, 0) > 0 for k in ANIMAL_KINDS)
    if carrying_produce or needs_fertilizer or needs_wheat or needs_animal_pickup:
        priority = -1 if needs_wheat else shed_priority
        for (sx, sy) in shed_tiles:
            if (sx, sy) not in claimed:
                candidates.append((sx, sy, priority, abs(sx - fx) + abs(sy - fy)))  # return to shed

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[2], c[3]))
    return candidates[0]


def _build_action(kind):
    return ["BUILD_COOP" if ANIMAL_STRUCTURE[kind] == "COOP" else "BUILD_PASTURE"]


def _decide_unit(
    pos, inv, farm, private, day, board_size, claimed, planner, animal_planner, shed_tiles,
    want_fertilizer, liquidating, animal_zone, quadrant_util,
):
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

    if _is_animal_tile(tile):
        if tile.get("yield_units", 0) > 0:
            claimed.add((fx, fy))
            return ["HARVEST"]
        if not tile.get("fed_today") and inv.get("WHEAT", 0) > 0:
            claimed.add((fx, fy))
            return ["FEED"]
        if tile.get("fertilizer_available"):
            claimed.add((fx, fy))
            return ["COLLECT_FERTILIZER"]
        if not tile.get("cared_today"):
            claimed.add((fx, fy))
            return ["CARE"]

    if _is_empty_structure(tile):
        kind = _carried_animal_for(tile, inv)
        if kind:
            claimed.add((fx, fy))
            return ["PLACE", kind]

    if (fx, fy) in shed_tiles:
        carried_product = {k: n for k, n in inv.items() if k in ANIMAL_PRODUCTS and n > 0}
        if carried_product:
            claimed.add((fx, fy))
            item, n = next(iter(carried_product.items()))
            return ["PLACE", item, n]
        carried_produce = {
            k: n for k, n in inv.items()
            if n > 0 and k not in ("WHEAT", *ANIMAL_KINDS) and (k != "FERTILIZER" or not want_fertilizer)
        }
        if carried_produce:
            claimed.add((fx, fy))
            return ["DROP"]
        if inv.get("FERTILIZER", 0) == 0 and want_fertilizer and private["shed"].get("FERTILIZER", 0) > 0:
            claimed.add((fx, fy))
            return ["PICKUP", "FERTILIZER", 1]
        if inv.get("WHEAT", 0) <= 0 and private["shed"].get("WHEAT", 0) > 0:
            claimed.add((fx, fy))
            return ["PICKUP", "WHEAT", min(4, private["shed"]["WHEAT"])]
        if not any(inv.get(k, 0) > 0 for k in ANIMAL_KINDS):
            kind = animal_planner.choose_pickup()
            if kind:
                claimed.add((fx, fy))
                return ["PICKUP", kind, 1]

    if tile is None:
        if (fx, fy) in animal_zone:
            kind = animal_planner.choose_structure()
            if kind:
                claimed.add((fx, fy))
                return _build_action(kind)
        crop = planner.choose()
        if crop:
            claimed.add((fx, fy))
            return ["PLANT", crop]
        kind = animal_planner.choose_structure()
        if kind:
            claimed.add((fx, fy))
            return _build_action(kind)

    shed_wheat_available = private["shed"].get("WHEAT", 0) > 0
    target = _find_target(
        pos, farm, day, board_size, claimed, inv, planner, animal_planner, shed_tiles,
        want_fertilizer, shed_wheat_available, liquidating, animal_zone, quadrant_util,
    )
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


def _hands_target(quadrants):
    return min(MAX_HANDS, 2 * quadrants)


def _land_worth_buying(quadrants_now):
    """The next quadrant only pays for itself if it actually raises hire
    capacity -- see the module docstring's Land utilization gate note. The
    hire formula (2*quadrants, capped at MAX_HANDS) hits its ceiling at
    quadrants_now == MAX_HANDS // 2 (3, at the default MAX_HANDS=6); a
    replay showed the same MAX_HANDS=6 hands spread across a 4th quadrant's
    extra land planting no more tiles in absolute terms than 3 quadrants'
    worth -- just diluted thinner, plus the quadrant's own price wasted."""
    return _hands_target(quadrants_now + 1) > _hands_target(quadrants_now)


def _build_land_orders(farm, days_remaining):
    if days_remaining < LAND_CUTOFF_DAYS:
        return []  # a fresh quadrant can't earn back its cost this close to the end
    quadrants_now = len(farm["unlocked_quadrants"])
    n_extra = quadrants_now - 1  # NW is always owned
    if n_extra >= len(LAND_ORDER) or not _land_worth_buying(quadrants_now):
        return []
    unlocked = [tile for row in farm["tiles"] for tile in row if tile != "LOCKED"]
    used = sum(
        1 for tile in unlocked
        if _is_plant(tile) or (isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP"))
    )
    if unlocked and used / len(unlocked) < LAND_UTILIZATION_GATE:
        return []  # can't staff what we already own -- more land would just sit idle
    if farm["money"] - SAFETY_RESERVE >= LAND_PRICES[n_extra]:
        return [["BUY_LAND"]]
    return []


def _build_hire_orders(farm):
    desired = _hands_target(len(farm["unlocked_quadrants"]))
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


def _build_feed_orders(farm, private, prices, money):
    """Buy WHEAT directly from the market for animal feed, decoupled from
    the crop side's own WHEAT production/selling. Without this, animals
    starve even with plenty of attention: `_build_sell_orders` dumps every
    WHEAT harvest as soon as its price clears the sell threshold, so the
    shed can sit at 0 WHEAT for turns at a stretch -- see the module
    docstring's Animal husbandry (attempt 2) note. Placed ahead of
    `_build_seed_orders` in `agent()`'s market list: missing a seed buy
    delays a planting one turn, missing a feed loses the whole animal."""
    live = sum(
        1 for row in farm["tiles"] for tile in row
        if isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP") and tile.get("animal")
    )
    if live == 0:
        return []
    reserve = live * FEED_RESERVE_PER_ANIMAL
    have = private.get("shed", {}).get("WHEAT", 0)
    needed = reserve - have
    if needed <= 0:
        return []
    base = MARKET_PARAMS["WHEAT"]["base"]
    price = prices.get("WHEAT", base)
    if price > base * FEED_MAX_PRICE_FRAC:
        return []
    remaining = money - SAFETY_RESERVE
    orders = []
    while needed > 0 and remaining >= price:
        orders.append(["BUY_PRODUCT", "WHEAT", 1])
        remaining -= price
        needed -= 1
    return orders


def _build_animal_orders(farm, private, money):
    """Keep at most ANIMAL_RESERVE bought-but-unplaced tokens per kind in
    the shed so a unit always has something to grab -- mirrors
    _build_seed_orders. Capped against the same land-share target
    _AnimalPlanner uses, counted by *live animals* (not structures --
    BUILD_PASTURE/BUILD_COOP is a free action, so structures reach the
    target well before purchases can catch up; counting empty structures
    as already "committed" here stopped this from ever buying enough
    animals to fill them, leaving land built but empty for the rest of the
    season)."""
    orders = []
    shed = private.get("shed", {})
    unlocked_tiles = sum(1 for row in farm["tiles"] for tile in row if tile != "LOCKED")
    target = round(unlocked_tiles * ANIMAL_LAND_SHARE)
    committed = sum(shed.get(kind, 0) for kind in ANIMAL_KINDS)
    committed += sum(
        1 for row in farm["tiles"] for tile in row
        if isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP") and tile.get("animal")
    )
    remaining = money - SAFETY_RESERVE
    for kind in ANIMAL_KINDS:
        if ANIMAL_TARGET_SHARE[kind] <= 0:
            continue  # e.g. GOOSE -- see ANIMAL_TARGET_SHARE's own comment
        if committed >= target:
            break
        if shed.get(kind, 0) >= ANIMAL_RESERVE:
            continue
        cost = ANIMALS[kind]["cost"]
        if remaining < cost:
            continue
        orders.append(["BUY_ANIMAL", kind, 1])
        remaining -= cost
        committed += 1
    return orders


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
    animal_planner = _AnimalPlanner(farm, shed, market_prices)
    animal_zone = _animal_zone(farm, board_size, animal_planner.remaining_structures())
    quadrant_util = _quadrant_utilization(farm, board_size)

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
            liquidating,
            animal_zone,
            quadrant_util,
        )
        for i, pos in enumerate(units)
    ]
    farmer_action, hands_actions = actions[0], actions[1:]

    market = []
    market += _build_sell_orders(shed, market_prices, days_remaining)
    market += _build_land_orders(farm, days_remaining)
    market += _build_hire_orders(farm)
    market += _build_feed_orders(farm, private, market_prices, farm["money"])
    market += _build_seed_orders(seeds, farm["money"], days_remaining)
    market += _build_fertilizer_orders(private, market_prices, farm["money"])
    market += _build_animal_orders(farm, private, farm["money"])
    market = market[:MAX_MARKET_ORDERS]

    return {"farmer": farmer_action, "hands": hands_actions, "market": market}

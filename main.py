"""Kaggriculture submission entry point."""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_I0,
    MARKET_PARAMS,
    PRICE_FLOOR,
    market_price,
)

CASH_CROPS = ("MELON", "WHEAT", "CARROT", "TOMATO", "STRAWBERRY")
MELON_TARGET_SHARE = 0.45
WHEAT_TARGET_SHARE = 0.25
CARROT_TARGET_SHARE = 0.15
TOMATO_TARGET_SHARE = 0.10
STRAWBERRY_TARGET_SHARE = 0.05
CROP_TARGET_SHARE = {
    "MELON": MELON_TARGET_SHARE, "WHEAT": WHEAT_TARGET_SHARE, "CARROT": CARROT_TARGET_SHARE,
    "TOMATO": TOMATO_TARGET_SHARE, "STRAWBERRY": STRAWBERRY_TARGET_SHARE,
}
STRAWBERRY_LATE_SHARE = 0.478
CROP_MIX_RAMP_START_DAY = 8
CROP_MIX_RAMP_DURATION = 2
STRAWBERRY_RESERVE_LATE = 3
FERTILIZE_CROPS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY"}
PRICE_ATTRACTIVENESS_WEIGHT = 0.26

PLANT_PRIORITY = 4
PLANT_PRIORITY_UNDER_GATE = 1.5
SHED_RETURN_URGENT_PRIORITY = 2
CARRIED_PRODUCE_URGENT_THRESHOLD = 3
HARVEST_AT_RISK_DIST_BIAS = 1000
HARVEST_SAFE_PRIORITY = 1.206
WATER_YIELD_INERT_PRIORITY = 1.387
WATER_ROUTINE_PRIORITY = 2.294
WHEAT_SHED_RETURN_THRESHOLD = 5

MELON_SEED_RESERVE = 3
WHEAT_SEED_RESERVE = 3
CARROT_SEED_RESERVE = 3
TOMATO_SEED_RESERVE = 2
STRAWBERRY_SEED_RESERVE = 2
SEED_RESERVE = {
    "MELON": MELON_SEED_RESERVE, "WHEAT": WHEAT_SEED_RESERVE, "CARROT": CARROT_SEED_RESERVE,
    "TOMATO": TOMATO_SEED_RESERVE, "STRAWBERRY": STRAWBERRY_SEED_RESERVE,
}
SEED_RESERVE_PRICE_CAP = 2.0
FERTILIZER_RESERVE = 2
FERTILIZER_STOCKPILE_RESERVE = 10
FERTILIZER_CHEAP_PRICE_FRAC = 0.35
FERTILIZER_MAX_PRICE_FRAC = 1.92

MELON_SELL_THRESHOLD = 0.65
WHEAT_SELL_THRESHOLD = 0.854
CARROT_SELL_THRESHOLD = 0.45
TOMATO_SELL_THRESHOLD = 0.50
STRAWBERRY_SELL_THRESHOLD = 0.55
MILK_SELL_THRESHOLD = 0.50
WOOL_SELL_THRESHOLD = 0.60
EGG_SELL_THRESHOLD = 0.35
FERTILIZER_SELL_THRESHOLD = 0.50
SELL_PARAMS = {
    "MELON": {"threshold": MELON_SELL_THRESHOLD},
    "WHEAT": {"threshold": WHEAT_SELL_THRESHOLD},
    "CARROT": {"threshold": CARROT_SELL_THRESHOLD},
    "TOMATO": {"threshold": TOMATO_SELL_THRESHOLD},
    "STRAWBERRY": {"threshold": STRAWBERRY_SELL_THRESHOLD},
    "MILK": {"threshold": MILK_SELL_THRESHOLD},
    "WOOL": {"threshold": WOOL_SELL_THRESHOLD},
    "EGG": {"threshold": EGG_SELL_THRESHOLD},
    "FERTILIZER": {"threshold": FERTILIZER_SELL_THRESHOLD},
}

SHED_CAPACITY = 100
SHED_OVERFLOW_GUARD = 0.633

ANIMAL_KINDS = ("COW", "SHEEP", "GOOSE")
ANIMAL_PRODUCT = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
ANIMAL_STRUCTURE = {"COW": "PASTURE", "SHEEP": "PASTURE", "GOOSE": "COOP"}
ANIMAL_PRODUCTS = ("MILK", "WOOL", "EGG")
ANIMAL_TARGET_SHARE_COW = 0.629
ANIMAL_TARGET_SHARE_SHEEP = 0.456
ANIMAL_TARGET_SHARE_GOOSE = 0.0
ANIMAL_TARGET_SHARE = {
    "COW": ANIMAL_TARGET_SHARE_COW, "SHEEP": ANIMAL_TARGET_SHARE_SHEEP, "GOOSE": ANIMAL_TARGET_SHARE_GOOSE,
}
SHEEP_MAX = 6
ANIMAL_LAND_SHARE = 0.18
ANIMAL_MAX_PER_UNIT = 1.0
ANIMAL_RESERVE = 1
FEED_RESERVE_PER_ANIMAL = 2
FEED_MAX_PRICE_FRAC = 4.22
WHEAT_SELL_RESERVE_DAYS = 3

FARM_HAND_COST_MULT = 1

HANDS_TARGET_Q1 = 8
HANDS_BASE = 6
HANDS_PER_QUADRANT = 2
MAX_HANDS = 16
SAFETY_RESERVE = 561
MAX_MARKET_ORDERS = 10

DEFAULT_EPISODE_STEPS = 720
DEFAULT_TURNS_PER_DAY = 24
SEED_BUFFER_DAYS = 1
LAND_CUTOFF_DAYS = 5
LAND_UTILIZATION_GATE = 0.493
LAND_QUADRANT_CAP = 3
LAND_RESERVE_PER_ANIMAL = 45
LAND_RESERVE_FLAT = 60
SELL_PROCEEDS_DISCOUNT = 0.9
ENDGAME_LIQUIDATION_DAYS = 2
LAST_CHANCE_HOURS = 4


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
    """Fraction of each unlocked quadrant's own tiles currently in use"""
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
    """The `needed` empty tiles closest to the shed -- animal structures"""
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
    """The last day a crop can still produce, i.e. the true upper bound for"""
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


def _any_unfed_animal(farm):
    """Mirrors `_any_fertilizable` -- used to gate the WHEAT endgame-liquidation DROP escape"""
    return any(_is_animal_tile(tile) and not tile.get("fed_today") for row in farm["tiles"] for tile in row)


def _is_harvestable(tile, day, liquidating):
    """yield_units > 0 alone is NOT "ready" for one-time crops: `_new_plant`"""
    if not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0:
        return False
    if tile.get("kind") != "PLANT":
        return True
    crop_data = CROPS[tile["crop"]]
    age = day - tile["planted_day"]
    if age < crop_data["first_yield_day"]:
        return False
    if crop_data["ongoing"]:
        return True
    return (
        age >= crop_data["max_yield_day"]
        or tile["yield_units"] >= crop_data["max_yield"]
        or liquidating
    )


def _is_yield_inert(tile, day):
    """A one-time crop's WATER action doesn't start adding yield_units the"""
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        return False
    crop_data = CROPS[tile["crop"]]
    if crop_data["ongoing"]:
        return False
    window_start = (crop_data["max_yield_day"] + 1) // 2
    age = day - tile["planted_day"]
    return age < window_start


def _crop_mix_ramp_fraction(day):
    """0.0 before CROP_MIX_RAMP_START_DAY, 1.0 once CROP_MIX_RAMP_DURATION days past it,"""
    if CROP_MIX_RAMP_DURATION <= 0:
        return 1.0 if day >= CROP_MIX_RAMP_START_DAY else 0.0
    return max(0.0, min(1.0, (day - CROP_MIX_RAMP_START_DAY) / CROP_MIX_RAMP_DURATION))


def _crop_target_share(day):
    """CROP_TARGET_SHARE, but with STRAWBERRY's share ramped toward STRAWBERRY_LATE_SHARE as"""
    frac = _crop_mix_ramp_fraction(day)
    base_strawberry = CROP_TARGET_SHARE["STRAWBERRY"]
    strawberry_share = base_strawberry + frac * (STRAWBERRY_LATE_SHARE - base_strawberry)
    other_scale = (1 - strawberry_share) / (1 - base_strawberry)
    return {
        crop: strawberry_share if crop == "STRAWBERRY" else share * other_scale
        for crop, share in CROP_TARGET_SHARE.items()
    }


def _strawberry_seed_reserve(day):
    """SEED_RESERVE["STRAWBERRY"], ramped toward STRAWBERRY_RESERVE_LATE on the same schedule"""
    frac = _crop_mix_ramp_fraction(day)
    base = SEED_RESERVE["STRAWBERRY"]
    return base + frac * (STRAWBERRY_RESERVE_LATE - base)


def _above_sell_threshold(prices, resource):
    """Is `resource`'s current market price at or above its own SELL_PARAMS threshold (True,"""
    price = prices.get(resource)
    if price is None:
        return True
    base = MARKET_PARAMS[resource]["base"]
    return price >= base * SELL_PARAMS[resource]["threshold"]


def _price_gated_pool(candidates, price_ok):
    """Prefer only currently-worth-selling candidates, but never leave the pool empty (falls back"""
    return [c for c in candidates if price_ok(c)] or candidates


class _CropPlanner:
    """Tracks, for one turn, which crop each new PLANT should use."""

    def __init__(self, private, farm, days_remaining, day=0, prices=None):
        self.target_share = _crop_target_share(day)
        self.seed_budget = dict(private.get("seeds", {}))
        self.prices = prices or {}
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
        """Would we currently be willing to *sell* this crop? If not, planting more of it just"""
        return _above_sell_threshold(self.prices, crop)

    def _score(self, crop):
        """Lower (more negative) wins the min() in choose(): how far below"""
        gap = self.counts[crop] / self.total - self.target_share[crop]
        price = self.prices.get(crop)
        if price is None:
            return gap
        price_ratio = price / MARKET_PARAMS[crop]["base"]
        return gap - PRICE_ATTRACTIVENESS_WEIGHT * (price_ratio - 1)

    def choose(self):
        """Pick whichever affordable, currently-worth-selling crop scores"""
        available = [c for c in self.plantable_crops if self.seed_budget.get(c, 0) > 0]
        if not available:
            return None
        pool = _price_gated_pool(available, self._price_ok)
        crop = min(pool, key=self._score)
        self.seed_budget[crop] -= 1
        self.counts[crop] += 1
        self.total += 1
        return crop


def _is_animal_tile(tile):
    return isinstance(tile, dict) and bool(tile.get("animal"))


def _live_animal_count(farm):
    """Shared by `_build_feed_orders`, `_build_animal_orders`, and"""
    return sum(1 for row in farm["tiles"] for tile in row if _is_animal_tile(tile))


def _is_empty_structure(tile):
    return isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP") and not tile.get("animal")


def _carried_animal_for(tile, inv):
    """Which animal kind (if any) in `inv` matches this empty structure's type."""
    for kind in ANIMAL_KINDS:
        if ANIMAL_STRUCTURE[kind] == tile.get("kind") and inv.get(kind, 0) > 0:
            return kind
    return None


def _animal_target(farm):
    """Target live-animal count -- the smaller of a land-share target"""
    unlocked_tiles = sum(1 for row in farm["tiles"] for tile in row if tile != "LOCKED")
    land_target = round(unlocked_tiles * ANIMAL_LAND_SHARE)
    total_units = 1 + len(farm["hands"])
    capacity_target = round(total_units * ANIMAL_MAX_PER_UNIT)
    return min(land_target, capacity_target)


class _AnimalPlanner:
    """Tracks, for one turn, which animal kind to build a structure for or"""

    def __init__(self, farm, shed, prices):
        self.counts = {kind: 0 for kind in ANIMAL_KINDS}
        self.empty_structures = {"PASTURE": 0, "COOP": 0}
        for row in farm["tiles"]:
            for tile in row:
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
        self.target = _animal_target(farm)

    def _committed(self):
        return sum(self.counts.values()) + sum(self.empty_structures.values())

    def remaining_structures(self):
        """How many more structures still need to be built to reach the"""
        return max(0, self.target - self._committed())

    def _price_ok(self, kind):
        return _above_sell_threshold(self.prices, ANIMAL_PRODUCT[kind])

    def wants_structure(self):
        if self._committed() >= self.target:
            return False
        return any(self.empty_structures[ANIMAL_STRUCTURE[k]] == 0 for k in ANIMAL_KINDS)

    def wants_pickup(self):
        return any(
            self.empty_structures[ANIMAL_STRUCTURE[k]] > 0 and self.shed.get(k, 0) > 0 for k in ANIMAL_KINDS
        )

    def _at_hard_cap(self, kind):
        """SHEEP_MAX: a hard ceiling on top of ANIMAL_TARGET_SHARE's proportional split --"""
        return kind == "SHEEP" and self.counts["SHEEP"] >= SHEEP_MAX

    def choose_structure(self):
        """Pick a kind with no spare (built-but-empty) structure yet, preferring"""
        if self._committed() >= self.target:
            return None
        needing = [
            k for k in ANIMAL_KINDS
            if ANIMAL_TARGET_SHARE[k] > 0 and self.empty_structures[ANIMAL_STRUCTURE[k]] == 0
            and not self._at_hard_cap(k)
        ]
        if not needing:
            return None
        pool = _price_gated_pool(needing, self._price_ok)
        kind = min(pool, key=lambda k: self.counts[k] / self.total - ANIMAL_TARGET_SHARE[k])
        self.empty_structures[ANIMAL_STRUCTURE[kind]] += 1
        return kind

    def choose_pickup(self):
        """Pick a kind that already has a spare structure waiting AND a"""
        ready = [
            k for k in ANIMAL_KINDS
            if ANIMAL_TARGET_SHARE[k] > 0
            and self.empty_structures[ANIMAL_STRUCTURE[k]] > 0
            and self.shed.get(k, 0) > 0
            and not self._at_hard_cap(k)
        ]
        if not ready:
            return None
        pool = _price_gated_pool(ready, self._price_ok)
        kind = min(pool, key=lambda k: self.counts[k] / self.total - ANIMAL_TARGET_SHARE[k])
        self.empty_structures[ANIMAL_STRUCTURE[kind]] -= 1
        self.shed[kind] = self.shed.get(kind, 0) - 1
        return kind


def _find_personal_target(
    pos, farm, day, board_size, claimed, inv, animal_planner, shed_tiles,
    want_fertilizer, shed_wheat_available, liquidating, feed_futile=False,
):
    """The subset of this agent's old single-scan target search that genuinely depends on"""
    fx, fy = pos
    shed_priority = 1 if liquidating else 5
    carries_fertilizer = inv.get("FERTILIZER", 0) > 0
    carries_wheat = inv.get("WHEAT", 0) > 0
    carries_animal_token = any(inv.get(k, 0) > 0 for k in ANIMAL_KINDS)
    candidates = []
    if carries_fertilizer or carries_wheat or carries_animal_token:
        for y in range(board_size):
            for x in range(board_size):
                if (x, y) in claimed:
                    continue
                tile = farm["tiles"][y][x]
                dist = abs(x - fx) + abs(y - fy)
                if _is_plant(tile):
                    if (carries_fertilizer and tile.get("watered_today")
                            and not _is_harvestable(tile, day, liquidating) and _is_fertilizable(tile, day)):
                        candidates.append((x, y, 3, dist))
                elif _is_animal_tile(tile):
                    if carries_wheat and tile.get("yield_units", 0) <= 0 and not tile.get("fed_today"):
                        candidates.append((x, y, -1, dist))
                elif carries_animal_token and _is_empty_structure(tile) and _carried_animal_for(tile, inv):
                    candidates.append((x, y, 0, dist))

    carried_produce_total = sum(
        n for k, n in inv.items()
        if n > 0 and k not in ("WHEAT", *ANIMAL_KINDS) and (k != "FERTILIZER" or not want_fertilizer or liquidating)
    )
    carrying_produce = carried_produce_total > 0
    needs_fertilizer = inv.get("FERTILIZER", 0) == 0 and want_fertilizer and not liquidating
    needs_wheat = (
        inv.get("WHEAT", 0) <= 0 and not feed_futile and _any_unfed_animal(farm) and shed_wheat_available
    )
    needs_animal_pickup = animal_planner.wants_pickup() and not carries_animal_token
    returning_wheat = inv.get("WHEAT", 0) >= WHEAT_SHED_RETURN_THRESHOLD or (
        feed_futile and inv.get("WHEAT", 0) > 0
    )
    if carrying_produce or needs_fertilizer or needs_wheat or needs_animal_pickup or returning_wheat:
        if needs_wheat:
            priority = -1
        elif carried_produce_total >= CARRIED_PRODUCE_URGENT_THRESHOLD:
            priority = SHED_RETURN_URGENT_PRIORITY
        else:
            priority = shed_priority
        for (sx, sy) in shed_tiles:
            if (sx, sy) not in claimed:
                candidates.append((sx, sy, priority, abs(sx - fx) + abs(sy - fy)))

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[2], c[3]))
    return candidates[0]


def _build_tile_jobs(
    farm, day, board_size, claimed, animal_zone, quadrant_util, plant_priority, liquidating,
    planner, animal_planner, last_chance_window=False,
):
    """The unit-independent subset of this agent's old single-scan target search --"""
    water_priority = 5 if liquidating else WATER_ROUTINE_PRIORITY
    animal_upkeep_priority = min(3, plant_priority)
    spread_penalty = board_size // 2
    jobs = []
    for y in range(board_size):
        for x in range(board_size):
            if (x, y) in claimed:
                continue
            tile = farm["tiles"][y][x]
            if _is_plant(tile):
                if _is_harvestable(tile, day, liquidating):
                    at_risk = not liquidating and tile.get("consecutive_unwatered", 0) >= 1
                    priority = 0 if at_risk else HARVEST_SAFE_PRIORITY
                    jobs.append((x, y, priority, -HARVEST_AT_RISK_DIST_BIAS if at_risk else 0))
                elif not tile["watered_today"]:
                    at_risk = not liquidating and tile.get("consecutive_unwatered", 0) >= 1
                    urgent = at_risk and last_chance_window
                    if urgent:
                        priority = 0
                    elif at_risk:
                        priority = 0.5
                    elif not liquidating and _is_yield_inert(tile, day):
                        priority = WATER_YIELD_INERT_PRIORITY
                    else:
                        priority = water_priority
                    jobs.append((x, y, priority, 0))
            elif _is_weed(tile):
                jobs.append((x, y, 2, 0))
            elif _is_animal_tile(tile):
                if tile.get("yield_units", 0) > 0:
                    jobs.append((x, y, 0, 0))
                elif tile.get("fed_today") and (
                    tile.get("fertilizer_available") or not tile.get("cared_today")
                ):
                    jobs.append((x, y, animal_upkeep_priority, 0))
            elif tile is None:
                if (x, y) in animal_zone:
                    jobs.append((x, y, 4, 0))
                elif planner.has_plantable():
                    q = _quadrant_of(x, y, board_size)
                    jobs.append((x, y, plant_priority, quadrant_util[q] * spread_penalty))
                elif animal_planner.wants_structure():
                    jobs.append((x, y, 4, 0))
    return jobs


def _assign_tile_jobs(units, jobs, personal):
    """Global (unit x job) match for `_build_tile_jobs`'s shared pool, plus each unit's own"""
    pairs = []
    for i, pos in units:
        fx, fy = pos
        for j, (x, y, prio, bias) in enumerate(jobs):
            dist = abs(x - fx) + abs(y - fy) + bias
            pairs.append((prio, dist, i, j))
        pc = personal.get(i)
        if pc is not None:
            pairs.append((pc[2], pc[3], i, None))
    pairs.sort(key=lambda pair: (pair[0], pair[1]))
    assigned_units = set()
    assigned_jobs = set()
    result = {}
    n_units = len(units)
    for prio, dist, i, j in pairs:
        if i in assigned_units:
            continue
        if j is None:
            x, y = personal[i][0], personal[i][1]
        else:
            if j in assigned_jobs:
                continue
            x, y = jobs[j][0], jobs[j][1]
            assigned_jobs.add(j)
        result[i] = (x, y)
        assigned_units.add(i)
        if len(assigned_units) == n_units:
            break
    return result


def _build_action(kind):
    return ["BUILD_COOP" if ANIMAL_STRUCTURE[kind] == "COOP" else "BUILD_PASTURE"]


def _decide_unit(
    pos, inv, farm, private, day, claimed, planner, animal_planner, shed_tiles,
    want_fertilizer, liquidating, animal_zone, feed_futile=False,
):
    """Resolves a unit's action if it's standing on an actionable tile -- unconditionally"""
    fx, fy = pos
    tile = farm["tiles"][fy][fx]

    if _is_plant(tile):
        if _is_harvestable(tile, day, liquidating):
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
        wheat_safe_to_liquidate = liquidating and (feed_futile or not _any_unfed_animal(farm))
        carried_produce = {
            k: n for k, n in inv.items()
            if n > 0 and k not in ANIMAL_KINDS
            and (k != "FERTILIZER" or not want_fertilizer or liquidating)
            and (k != "WHEAT" or n >= WHEAT_SHED_RETURN_THRESHOLD or wheat_safe_to_liquidate)
        }
        if carried_produce:
            claimed.add((fx, fy))
            return ["DROP"]
        if (
            inv.get("FERTILIZER", 0) == 0 and want_fertilizer and not liquidating
            and private["shed"].get("FERTILIZER", 0) > 0
        ):
            claimed.add((fx, fy))
            return ["PICKUP", "FERTILIZER", 1]
        if inv.get("WHEAT", 0) <= 0 and private["shed"].get("WHEAT", 0) > 0 and not wheat_safe_to_liquidate:
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

    return None


def _build_sell_orders(shed, prices, days_remaining, farm):
    """Dumps the entire shed stock of a crop in one order once its price"""
    orders = []
    shed_total = sum(shed.values())
    crowded = shed_total >= SHED_CAPACITY * SHED_OVERFLOW_GUARD
    liquidating = days_remaining <= ENDGAME_LIQUIDATION_DAYS
    live_animals = None
    for crop, params in SELL_PARAMS.items():
        qty = shed.get(crop, 0)
        if qty <= 0:
            continue
        if crop == "WHEAT" and not (crowded or liquidating):
            if live_animals is None:
                live_animals = _live_animal_count(farm)
            reserve = live_animals * min(WHEAT_SELL_RESERVE_DAYS, max(0, days_remaining))
            qty -= reserve
            if qty <= 0:
                continue
        base = MARKET_PARAMS[crop]["base"]
        price = prices.get(crop, base)
        if crowded or liquidating or price >= base * params["threshold"]:
            orders.append(["SELL", crop, qty])
    return orders


def _hands_target(quadrants):
    if quadrants <= 1:
        return HANDS_TARGET_Q1
    return min(MAX_HANDS, HANDS_BASE + HANDS_PER_QUADRANT * quadrants)


def _land_worth_buying(quadrants_now):
    """The next quadrant only pays for itself if it actually raises hire"""
    return _hands_target(quadrants_now + 1) > _hands_target(quadrants_now)


def _land_utilization(farm):
    """Fraction of currently-owned (non-LOCKED) tiles that are planted or"""
    unlocked = [tile for row in farm["tiles"] for tile in row if tile != "LOCKED"]
    if not unlocked:
        return 0.0
    used = sum(
        1 for tile in unlocked
        if _is_plant(tile) or (isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP"))
    )
    return used / len(unlocked)


def _fib(n):
    """Indexed so _fib(0)=1, _fib(1)=1, _fib(2)=2, _fib(3)=3, _fib(4)=5... --"""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _sell_proceeds(item, qty, inventory):
    """X578/X594 kernel-pull reference lever 4 of 4 (see the module docstring's Harvest-timing"""
    inv = inventory
    total = 0
    for _ in range(qty):
        price = market_price(item, inv)
        total += price
        if price > PRICE_FLOOR:
            inv += 1
    return total


def _buy_cost(item, qty, inventory):
    """Mirror of _sell_proceeds for the buy side -- found independently while auditing"""
    inv = inventory
    total = 0
    for _ in range(qty):
        price = market_price(item, inv)
        total += price
        inv -= 1
    return total


def _orders_cash_cost(orders, farm, market_inventory):
    """16th market-order-slot-starvation attempt: estimate the total cash"""
    total = 0.0
    n_extra = len(farm["unlocked_quadrants"]) - 1
    hires_so_far = farm["hires_today"]
    for order in orders:
        op = order[0]
        if op == "SELL":
            item = order[1]
            qty = order[2] if len(order) > 2 else 1
            inv = market_inventory.get(item, MARKET_I0)
            total -= _sell_proceeds(item, qty, inv) * SELL_PROCEEDS_DISCOUNT
        elif op == "BUY_PRODUCT":
            item = order[1]
            qty = order[2] if len(order) > 2 else 1
            inv = market_inventory.get(item, MARKET_I0)
            total += _buy_cost(item, qty, inv)
        elif op == "BUY_SEED":
            item = order[1]
            qty = order[2] if len(order) > 2 else 1
            total += qty * CROPS[item]["seed"]
        elif op == "BUY_ANIMAL":
            item = order[1]
            qty = order[2] if len(order) > 2 else 1
            total += qty * ANIMALS[item]["cost"]
        elif op == "BUY_LAND":
            if 0 <= n_extra < len(LAND_PRICES):
                total += LAND_PRICES[n_extra]
            n_extra += 1
        elif op == "HIRE":
            total += FARM_HAND_COST_MULT * _fib(hires_so_far)
            hires_so_far += 1
    return total


def _land_cash_reserve(farm):
    """X578/X594 kernel-pull reference lever 2 of 4 (see the module docstring's Harvest-timing"""
    return _live_animal_count(farm) * LAND_RESERVE_PER_ANIMAL + LAND_RESERVE_FLAT


def _build_land_orders(farm, days_remaining, money):
    """`money` is the caller's same-turn cash budget -- see agent()'s own comment on why hire"""
    if days_remaining < LAND_CUTOFF_DAYS:
        return []
    quadrants_now = len(farm["unlocked_quadrants"])
    n_extra = quadrants_now - 1
    if quadrants_now >= LAND_QUADRANT_CAP or n_extra >= len(LAND_ORDER) or not _land_worth_buying(quadrants_now):
        return []
    if _land_utilization(farm) < LAND_UTILIZATION_GATE:
        return []
    if money - _land_cash_reserve(farm) >= LAND_PRICES[n_extra]:
        return [["BUY_LAND"]]
    return []


def _build_hire_orders(farm, money_left):
    """`money_left` is the caller's same-turn cash budget (see agent()'s own comment on why"""
    desired = _hands_target(len(farm["unlocked_quadrants"]))
    remaining = desired - farm["hires_today"]
    if remaining <= 0:
        return []
    orders = []
    budget = max(0, money_left)
    for i in range(remaining):
        cost = FARM_HAND_COST_MULT * _fib(farm["hires_today"] + i)
        if budget < cost:
            break
        orders.append(["HIRE"])
        budget -= cost
    return orders


def _build_seed_orders(seeds, money, days_remaining, prices, day=0):
    """Price-reactive seed reserve: each crop's target buffer is"""
    orders = []
    remaining = money - SAFETY_RESERVE
    strawberry_reserve = _strawberry_seed_reserve(day)
    for crop, base_reserve in SEED_RESERVE.items():
        if crop == "STRAWBERRY":
            base_reserve = strawberry_reserve
        if days_remaining < CROPS[crop]["first_yield_day"] + SEED_BUFFER_DAYS:
            continue
        base_price = MARKET_PARAMS[crop]["base"]
        price = prices.get(crop, base_price)
        price_ratio = max(1.0, price / base_price) if base_price else 1.0
        reserve = round(base_reserve * min(price_ratio, SEED_RESERVE_PRICE_CAP))
        have = seeds.get(crop, 0)
        cost = CROPS[crop]["seed"]
        while have < reserve and remaining >= cost:
            orders.append(["BUY_SEED", crop, 1])
            have += 1
            remaining -= cost
    return orders


def _build_fertilizer_orders(private, prices, money, want_fertilizer):
    """Tops up to the small standing FERTILIZER_RESERVE normally, but"""
    have = private["shed"].get("FERTILIZER", 0) + sum(
        inv.get("FERTILIZER", 0) for inv in private["inventories"]
    )
    base = MARKET_PARAMS["FERTILIZER"]["base"]
    price = prices.get("FERTILIZER", base)
    cheap = price <= base * FERTILIZER_CHEAP_PRICE_FRAC
    reserve = FERTILIZER_STOCKPILE_RESERVE if (cheap and want_fertilizer) else FERTILIZER_RESERVE
    if have >= reserve:
        return []
    if price > base * FERTILIZER_MAX_PRICE_FRAC:
        return []
    if money - SAFETY_RESERVE < price:
        return []
    return [["BUY_PRODUCT", "FERTILIZER", 1]]


def _build_feed_orders(farm, private, prices, money):
    """Buy WHEAT directly from the market for animal feed, decoupled from"""
    live = _live_animal_count(farm)
    if live == 0:
        return []
    reserve = live * FEED_RESERVE_PER_ANIMAL
    shed_wheat = private.get("shed", {}).get("WHEAT", 0)
    carried_wheat = sum(inv.get("WHEAT", 0) for inv in private.get("inventories", []))
    have = shed_wheat + carried_wheat
    needed = reserve - have
    if needed <= 0:
        return []
    base = MARKET_PARAMS["WHEAT"]["base"]
    price = prices.get("WHEAT", base)
    if price > base * FEED_MAX_PRICE_FRAC:
        return []
    remaining = money - SAFETY_RESERVE
    affordable = int(remaining // price) if price > 0 else 0
    qty = min(needed, max(0, affordable))
    if qty <= 0:
        return []
    return [["BUY_PRODUCT", "WHEAT", qty]]


def _build_animal_orders(farm, private, money):
    """Keep at most ANIMAL_RESERVE bought-but-unplaced tokens per kind in"""
    orders = []
    shed = private.get("shed", {})
    target = _animal_target(farm)
    committed = sum(shed.get(kind, 0) for kind in ANIMAL_KINDS)
    committed += _live_animal_count(farm)
    live_sheep = sum(
        1 for row in farm["tiles"] for tile in row
        if isinstance(tile, dict) and tile.get("animal") == "SHEEP"
    )
    committed_sheep = shed.get("SHEEP", 0) + live_sheep
    remaining = money - SAFETY_RESERVE
    for kind in ANIMAL_KINDS:
        if ANIMAL_TARGET_SHARE[kind] <= 0:
            continue
        if committed >= target:
            break
        if kind == "SHEEP" and committed_sheep >= SHEEP_MAX:
            continue
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
    hour = obs.get("hour", 0)
    board_size = len(farm["tiles"])
    shed_tiles = _shed_adjacent_tiles(board_size)

    turns_per_day = int(_cfg_get(config, "turnsPerDay", DEFAULT_TURNS_PER_DAY)) or DEFAULT_TURNS_PER_DAY
    episode_steps = int(_cfg_get(config, "episodeSteps", DEFAULT_EPISODE_STEPS)) or DEFAULT_EPISODE_STEPS
    last_day = max(0, -(-episode_steps // turns_per_day) - 1)
    days_remaining = last_day - day
    liquidating = days_remaining <= ENDGAME_LIQUIDATION_DAYS
    feed_futile = days_remaining <= 0
    last_chance_window = hour >= turns_per_day - LAST_CHANCE_HOURS

    market_obs = obs.get("market", {}) or {}
    market_prices = market_obs.get("prices", {})
    market_inventory = market_obs.get("inventory", {})
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    inventories = private.get("inventories", [{}])

    want_fertilizer = _any_fertilizable(farm, day)
    planner = _CropPlanner(private, farm, days_remaining, day, market_prices)
    animal_planner = _AnimalPlanner(farm, shed, market_prices)
    animal_zone = _animal_zone(farm, board_size, animal_planner.remaining_structures())
    quadrant_util = _quadrant_utilization(farm, board_size)
    plant_priority = (
        PLANT_PRIORITY_UNDER_GATE if _land_utilization(farm) < LAND_UTILIZATION_GATE else PLANT_PRIORITY
    )

    claimed = set()
    units = [farm["farmer"]] + list(farm["hands"])
    actions = [None] * len(units)
    pending = []
    for i, pos in enumerate(units):
        result = _decide_unit(
            pos,
            inventories[i] if i < len(inventories) else {},
            farm,
            private,
            day,
            claimed,
            planner,
            animal_planner,
            shed_tiles,
            want_fertilizer,
            liquidating,
            animal_zone,
            feed_futile,
        )
        if result is None:
            pending.append((i, pos))
        else:
            actions[i] = result

    if pending:
        shed_wheat_available = shed.get("WHEAT", 0) > 0
        personal = {}
        for i, pos in pending:
            p = _find_personal_target(
                pos, farm, day, board_size, claimed,
                inventories[i] if i < len(inventories) else {},
                animal_planner, shed_tiles, want_fertilizer, shed_wheat_available, liquidating,
                feed_futile,
            )
            if p is not None:
                personal[i] = p
                claimed.add((p[0], p[1]))
        jobs = _build_tile_jobs(
            farm, day, board_size, claimed, animal_zone, quadrant_util, plant_priority,
            liquidating, planner, animal_planner, last_chance_window,
        )
        assignment = _assign_tile_jobs(pending, jobs, personal) if (jobs or personal) else {}
        for i, pos in pending:
            fx, fy = pos
            tx_ty = assignment.get(i)
            if tx_ty is not None:
                claimed.add(tx_ty)
                step = _step_toward(fx, fy, tx_ty[0], tx_ty[1])
                actions[i] = [step] if step else ["PASS"]
            else:
                actions[i] = ["PASS"]

    farmer_action, hands_actions = actions[0], actions[1:]

    market = []
    sell_orders = _build_sell_orders(shed, market_prices, days_remaining, farm)
    market += sell_orders
    cash_reserved = _orders_cash_cost(sell_orders, farm, market_inventory)

    hire_orders = _build_hire_orders(farm, farm["money"] - cash_reserved)
    market += hire_orders
    cash_reserved += _orders_cash_cost(hire_orders, farm, market_inventory)

    land_orders = _build_land_orders(farm, days_remaining, farm["money"] - cash_reserved)
    market += land_orders
    cash_reserved += _orders_cash_cost(land_orders, farm, market_inventory)

    feed_orders = _build_feed_orders(farm, private, market_prices, farm["money"] - cash_reserved)
    market += feed_orders
    cash_reserved += _orders_cash_cost(feed_orders, farm, market_inventory)

    seed_orders = _build_seed_orders(seeds, farm["money"] - cash_reserved, days_remaining, market_prices, day)
    market += seed_orders
    cash_reserved += _orders_cash_cost(seed_orders, farm, market_inventory)

    fertilizer_orders = _build_fertilizer_orders(
        private, market_prices, farm["money"] - cash_reserved, want_fertilizer
    )
    market += fertilizer_orders
    cash_reserved += _orders_cash_cost(fertilizer_orders, farm, market_inventory)

    market += _build_animal_orders(farm, private, farm["money"] - cash_reserved)
    market = market[:MAX_MARKET_ORDERS]

    return {"farmer": farmer_action, "hands": hands_actions, "market": market}

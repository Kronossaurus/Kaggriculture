"""Regression tests for main.py -- pure functions over plain dicts, so no"""

from __future__ import annotations

import itertools
import sys

import main


def _tile(crop="WHEAT", planted_day=0, watered_today=False, yield_units=1, fertilized_until_day=-1):
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": planted_day,
        "watered_today": watered_today,
        "consecutive_unwatered": 0 if watered_today else 1,
        "yield_units": yield_units,
        "max_lifespan_step": -1,
        "fertilized_until_day": fertilized_until_day,
    }


def _farm(tiles):
    """A 3x3 board with `tiles` (a {(x, y): tile} sparse map) on top of empty."""
    board = [[None for _ in range(3)] for _ in range(3)]
    for (x, y), tile in tiles.items():
        board[y][x] = tile
    return {
        "money": 3000.0,
        "tiles": board,
        "farmer": [0, 0],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }


_NEUTRAL_QUADRANT_UTIL = {"NW": 0.0, "NE": 0.0, "SW": 0.0, "SE": 0.0}


def _farm_with_one_fertilizable_tile():
    """A 3x3 board, all LOCKED except the unit's own shed-adjacent tile"""
    board = [["LOCKED" for _ in range(3)] for _ in range(3)]
    board[0][0] = None
    board[2][2] = _tile(crop="WHEAT", planted_day=0, watered_today=True)
    return {
        "money": 3000.0, "tiles": board, "farmer": [0, 0], "hands": [],
        "unlocked_quadrants": ["NW"], "hires_today": 0,
    }


def test_is_harvestable_false_immediately_after_planting():
    """The core bug: WHEAT's first_yield_day is 2, but _new_plant seeds"""
    fresh = _tile(crop="WHEAT", planted_day=5, yield_units=1)
    assert not main._is_harvestable(fresh, day=5, liquidating=False)
    assert not main._is_harvestable(fresh, day=6, liquidating=False)


def test_is_harvestable_false_at_first_yield_day_still_growing():
    """Harvest-timing note: legal (age >= first_yield_day) is NOT the same as"""
    ripe = _tile(crop="WHEAT", planted_day=5, yield_units=1)
    assert not main._is_harvestable(ripe, day=7, liquidating=False)


def test_is_harvestable_true_once_max_yield_day_reached():
    mature = _tile(crop="WHEAT", planted_day=5, yield_units=3)
    assert main._is_harvestable(mature, day=9, liquidating=False)


def test_is_harvestable_true_once_yield_cap_reached_even_before_max_yield_day():
    """Nothing left to gain by waiting once yield_units hits max_yield (e.g."""
    capped = _tile(crop="WHEAT", planted_day=5, yield_units=6)
    assert main._is_harvestable(capped, day=8, liquidating=False)


def test_is_harvestable_true_early_when_liquidating():
    """Mirrors the X578/X594 kernel-pull reference's own harvest-timing"""
    early = _tile(crop="WHEAT", planted_day=5, yield_units=1)
    assert not main._is_harvestable(early, day=7, liquidating=False)
    assert main._is_harvestable(early, day=7, liquidating=True)


def test_is_harvestable_ongoing_crop_unaffected_by_max_yield_day():
    """TOMATO/STRAWBERRY are unaffected: their own scheduled-production"""
    tomato = _tile(crop="TOMATO", planted_day=0, yield_units=1)
    assert main._is_harvestable(tomato, day=8, liquidating=False)


def test_is_harvestable_false_when_yield_units_zero():
    empty = _tile(crop="MELON", planted_day=0, yield_units=0)
    assert not main._is_harvestable(empty, day=20, liquidating=False)


def test_is_yield_inert_true_before_the_watering_bonus_window():
    """WHEAT's watering bonus window_start = (max_yield_day+1)//2 = (4+1)//2"""
    fresh = _tile(crop="WHEAT", planted_day=5)
    assert main._is_yield_inert(fresh, day=5)
    assert main._is_yield_inert(fresh, day=6)


def test_is_yield_inert_false_once_the_bonus_window_starts():
    tile = _tile(crop="WHEAT", planted_day=5)
    assert not main._is_yield_inert(tile, day=7)


def test_is_yield_inert_melon_window_starts_before_first_yield_day():
    """MELON's window_start ((12+1)//2=6) is well before its first_yield_day"""
    melon = _tile(crop="MELON", planted_day=0)
    assert main._is_yield_inert(melon, day=5)
    assert not main._is_yield_inert(melon, day=6)


def test_is_yield_inert_false_for_ongoing_crops():
    tomato = _tile(crop="TOMATO", planted_day=0)
    assert not main._is_yield_inert(tomato, day=0)


def test_is_fertilizable_covers_full_ongoing_crop_schedule():
    """TOMATO's max_yield_day (8) is just a placeholder equal to"""
    tile = _tile(crop="TOMATO", planted_day=0, fertilized_until_day=-1)
    assert main._is_fertilizable(tile, day=11)
    assert not main._is_fertilizable(tile, day=12)


def test_decide_unit_digs_a_weed_instead_of_stalling():
    """Without DIG, a WEED tile is permanently lost -- PLANT/WATER/SELL all"""
    weed = {"kind": "WEED"}
    farm = _farm({(0, 0): weed})
    claimed = set()
    planner = main._CropPlanner({"seeds": {}}, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    action = main._decide_unit(
        pos=[0, 0], inv={}, farm=farm, private={"shed": {}}, day=0, claimed=claimed,
        planner=planner, animal_planner=animal_planner, shed_tiles=set(), want_fertilizer=False, liquidating=False,
        animal_zone=set(),
    )
    assert action == ["DIG"]


def test_decide_unit_drops_surplus_fertilizer_at_the_shed():
    """FERTILIZER used to be excluded from "carried produce" unconditionally,"""
    farm = _farm({})
    claimed = set()
    planner = main._CropPlanner({"seeds": {}}, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    action = main._decide_unit(
        pos=[0, 0], inv={"FERTILIZER": 3}, farm=farm, private={"shed": {}}, day=0,
        claimed=claimed, planner=planner, animal_planner=animal_planner, shed_tiles={(0, 0)},
        want_fertilizer=False, liquidating=False, animal_zone=set(),
    )
    assert action == ["DROP"]


def test_decide_unit_keeps_freshly_picked_up_fertilizer_when_still_needed():
    """A unit standing at the shed holding just-fetched FERTILIZER must NOT"""
    farm = _farm_with_one_fertilizable_tile()
    planner = main._CropPlanner({"seeds": {}}, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    action = main._decide_unit(
        pos=[0, 0], inv={"FERTILIZER": 1}, farm=farm, private={"shed": {}}, day=0,
        claimed=set(), planner=planner, animal_planner=animal_planner, shed_tiles={(0, 0)},
        want_fertilizer=True, liquidating=False, animal_zone=set(),
    )
    assert action is None
    target = main._find_personal_target(
        pos=[0, 0], farm=farm, day=0, board_size=3, claimed=set(), inv={"FERTILIZER": 1},
        animal_planner=animal_planner, shed_tiles={(0, 0)},
        want_fertilizer=True, shed_wheat_available=False, liquidating=False,
    )
    assert (target[0], target[1]) == (2, 2)


def test_decide_unit_keeps_freshly_picked_up_wheat_below_the_threshold():
    """Same livelock family as the FERTILIZER test above, for WHEAT: a unit"""
    farm = _farm({})
    planner = main._CropPlanner({"seeds": {}}, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    action = main._decide_unit(
        pos=[0, 0], inv={"WHEAT": 4}, farm=farm, private={"shed": {}}, day=0,
        claimed=set(), planner=planner, animal_planner=animal_planner, shed_tiles={(0, 0)},
        want_fertilizer=False, liquidating=False, animal_zone=set(),
    )
    assert action != ["DROP"]


def test_decide_unit_sells_wheat_below_threshold_during_liquidation():
    """See the module docstring's Endgame liquidation (WHEAT) note: without"""
    farm = _farm({})
    planner = main._CropPlanner({"seeds": {}}, farm, days_remaining=1)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    action = main._decide_unit(
        pos=[0, 0], inv={"WHEAT": 4}, farm=farm, private={"shed": {}}, day=0,
        claimed=set(), planner=planner, animal_planner=animal_planner, shed_tiles={(0, 0)},
        want_fertilizer=False, liquidating=True, animal_zone=set(),
    )
    assert action == ["DROP"]


def test_decide_unit_keeps_wheat_during_liquidation_if_an_animal_still_needs_it():
    """See the module docstring's Endgame liquidation (WHEAT) note, third"""
    farm = _farm({(2, 2): {"kind": "PASTURE", "animal": "COW", "fed_today": False, "cared_today": True}})
    planner = main._CropPlanner({"seeds": {}}, farm, days_remaining=1)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    action = main._decide_unit(
        pos=[0, 0], inv={"WHEAT": 4}, farm=farm, private={"shed": {}}, day=0,
        claimed=set(), planner=planner, animal_planner=animal_planner, shed_tiles={(0, 0)},
        want_fertilizer=False, liquidating=True, animal_zone=set(),
    )
    assert action != ["DROP"]


def test_decide_unit_sells_fertilizer_during_liquidation_even_if_still_wanted():
    """See the module docstring's Endgame liquidation (fertilizer) note: a"""
    farm = _farm_with_one_fertilizable_tile()
    planner = main._CropPlanner({"seeds": {}}, farm, days_remaining=1)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    action = main._decide_unit(
        pos=[0, 0], inv={"FERTILIZER": 1}, farm=farm, private={"shed": {}}, day=0,
        claimed=set(), planner=planner, animal_planner=animal_planner, shed_tiles={(0, 0)},
        want_fertilizer=True, liquidating=True, animal_zone=set(),
    )
    assert action == ["DROP"]


def test_step_toward_moves_one_axis_at_a_time():
    assert main._step_toward(0, 0, 5, 0) == "EAST"
    assert main._step_toward(5, 0, 0, 0) == "WEST"
    assert main._step_toward(0, 0, 0, 5) == "SOUTH"
    assert main._step_toward(0, 5, 0, 0) == "NORTH"
    assert main._step_toward(3, 3, 3, 3) is None


def test_build_sell_orders_dumps_the_full_shed_stock():
    """Locks the reverted fix (see module docstring): no per-turn cap --"""
    shed = {"WHEAT": 47}
    prices = {"WHEAT": 30}
    farm = _farm({})
    orders = main._build_sell_orders(shed, prices, days_remaining=20, farm=farm)
    assert orders == [["SELL", "WHEAT", 47]]


def test_build_sell_orders_holds_below_threshold():
    shed = {"MELON": 10}
    prices = {"MELON": 50}
    farm = _farm({})
    orders = main._build_sell_orders(shed, prices, days_remaining=20, farm=farm)
    assert orders == []


def test_build_sell_orders_holds_back_a_wheat_reserve_for_live_animals():
    """With live animals on the farm, WHEAT sells shed-qty minus"""
    farm = _farm({(0, 0): _animal_tile(), (1, 0): _animal_tile()})
    shed = {"WHEAT": 20}
    prices = {"WHEAT": 30}
    orders = main._build_sell_orders(shed, prices, days_remaining=20, farm=farm)
    assert orders == [["SELL", "WHEAT", 14]]


def test_build_sell_orders_wheat_reserve_ignored_when_liquidating():
    """The reserve never withholds the endgame liquidation dump -- missing"""
    farm = _farm({(0, 0): _animal_tile(), (1, 0): _animal_tile()})
    shed = {"WHEAT": 20}
    prices = {"WHEAT": 5}
    orders = main._build_sell_orders(shed, prices, days_remaining=1, farm=farm)
    assert orders == [["SELL", "WHEAT", 20]]


def test_hands_target_uses_the_q1_override_only_at_one_quadrant():
    """See HANDS_TARGET_Q1's own comment: a surgical override for the"""
    assert main._hands_target(1) == main.HANDS_TARGET_Q1
    assert main._hands_target(2) == min(main.MAX_HANDS, main.HANDS_BASE + main.HANDS_PER_QUADRANT * 2)
    assert main._hands_target(3) == min(main.MAX_HANDS, main.HANDS_BASE + main.HANDS_PER_QUADRANT * 3)


def test_build_hire_orders_requests_the_full_target_when_affordable():
    """The common case: plenty of cash -- every hire up to the target gets requested,"""
    farm = _farm({})
    orders = main._build_hire_orders(farm, money_left=100000)
    assert len(orders) == main.HANDS_TARGET_Q1
    assert all(o == ["HIRE"] for o in orders)


def test_build_hire_orders_stops_once_the_next_hire_is_unaffordable():
    """See _build_hire_orders's own docstring: without this, requesting a hire the engine"""
    farm = _farm({})
    farm["hires_today"] = 0
    orders = main._build_hire_orders(farm, money_left=5)
    assert len(orders) == 3


def test_build_land_orders_blocked_below_utilization_gate():
    """See module docstring's Land utilization gate note: a replay showed"""
    tiles = {(0, 0): _tile(), (1, 0): _tile()}
    farm = _farm(tiles)
    farm["money"] = 5000.0
    orders = main._build_land_orders(farm, days_remaining=20, money=farm["money"])
    assert orders == []


def test_build_land_orders_buys_once_utilization_gate_is_met():
    tiles = {(x, y): _tile() for x in range(3) for y in range(2)}
    farm = _farm(tiles)
    farm["money"] = 5000.0
    orders = main._build_land_orders(farm, days_remaining=20, money=farm["money"])
    assert orders == [["BUY_LAND"]]


def test_build_land_orders_does_not_buy_the_4th_quadrant_under_land_quadrant_cap():
    """LAND_QUADRANT_CAP blocks a 4th-quadrant purchase outright, even at"""
    tiles = {(x, y): _tile() for x in range(3) for y in range(3)}
    farm = _farm(tiles)
    farm["unlocked_quadrants"] = ["NW", "NE", "SW"]
    farm["money"] = 50000.0
    orders = main._build_land_orders(farm, days_remaining=20, money=farm["money"])
    assert orders == []


def test_orders_cash_cost_sums_across_order_types():
    """16th market-order-slot-starvation attempt: _orders_cash_cost is the"""
    farm = {"unlocked_quadrants": ["NW", "NE"], "hires_today": 1}
    orders = [
        ["BUY_PRODUCT", "WHEAT", 2],
        ["BUY_SEED", "MELON", 1],
        ["BUY_ANIMAL", "COW", 1],
        ["BUY_LAND"],
        ["HIRE"],
    ]
    cost = main._orders_cash_cost(orders, farm, market_inventory={})
    expected = (
        main._buy_cost("WHEAT", 2, main.MARKET_I0)
        + main.CROPS["MELON"]["seed"]
        + main.ANIMALS["COW"]["cost"]
        + main.LAND_PRICES[len(farm["unlocked_quadrants"]) - 1]
        + main.FARM_HAND_COST_MULT * main._fib(farm["hires_today"])
    )
    assert cost == expected


def test_orders_cash_cost_sell_orders_are_negative_revenue():
    """SELL orders earn money rather than spend it, so they contribute a"""
    farm = {"unlocked_quadrants": ["NW"], "hires_today": 0}
    orders = [["SELL", "WHEAT", 4]]
    inventory = {"WHEAT": main.MARKET_I0}
    cost = main._orders_cash_cost(orders, farm, market_inventory=inventory)
    assert cost == -(main._sell_proceeds("WHEAT", 4, main.MARKET_I0) * main.SELL_PROCEEDS_DISCOUNT)
    assert cost == -87.3


def test_orders_cash_cost_buy_product_uses_per_unit_price_walk():
    """Mirror of the SELL test above, for the buy side (see _buy_cost's own docstring): a"""
    farm = {"unlocked_quadrants": ["NW"], "hires_today": 0}
    orders = [["BUY_PRODUCT", "WHEAT", 4]]
    inventory = {"WHEAT": main.MARKET_I0}
    cost = main._orders_cash_cost(orders, farm, market_inventory=inventory)
    assert cost == main._buy_cost("WHEAT", 4, main.MARKET_I0)
    assert cost > 4 * main.market_price("WHEAT", main.MARKET_I0)


def test_sell_proceeds_price_declines_across_units_in_one_order():
    """See _sell_proceeds's own docstring: each unit sold nudges the market"""
    inv = main.MARKET_I0
    first_unit_price = main.market_price("WHEAT", inv)
    total = main._sell_proceeds("WHEAT", 10, inv)
    assert total < 10 * first_unit_price


def _animal_tile(kind="PASTURE"):
    return {"kind": kind, "animal": "COW" if kind == "PASTURE" else "GOOSE"}


def test_build_feed_orders_skipped_when_reserve_already_met():
    tiles = {(0, 0): _animal_tile()}
    farm = _farm(tiles)
    private = {"shed": {"WHEAT": 999}}
    prices = {"WHEAT": 25}
    assert main._build_feed_orders(farm, private, prices, money=10000.0) == []


def test_build_feed_orders_counts_carried_wheat_and_consolidates_into_one_order():
    """The 18th WHEAT-consolidation attempt (kept despite a measured"""
    farm = _farm({(0, 0): _animal_tile()})
    private = {"shed": {"WHEAT": 0}, "inventories": [{"WHEAT": 3}]}
    prices = {"WHEAT": 25}
    orders = main._build_feed_orders(farm, private, prices, money=10000.0)
    assert orders == []


def test_build_feed_orders_consolidates_a_real_shortfall_into_one_order():
    """Same accounting as above but with a genuine shortfall: 2 animals ->"""
    farm = _farm({(0, 0): _animal_tile(), (1, 0): _animal_tile()})
    private = {"shed": {"WHEAT": 0}, "inventories": [{"WHEAT": 1}]}
    prices = {"WHEAT": 25}
    orders = main._build_feed_orders(farm, private, prices, money=10000.0)
    assert orders == [["BUY_PRODUCT", "WHEAT", 3]]


def test_build_seed_orders_scales_reserve_up_with_an_attractive_price():
    """Locks the price-reactive seed reserve fix (see the module"""
    seeds = {"MELON": 0}
    base = main.MARKET_PARAMS["MELON"]["base"]
    prices = {"MELON": base * main.SEED_RESERVE_PRICE_CAP}
    orders = main._build_seed_orders(seeds, money=100000.0, days_remaining=20, prices=prices)
    melon_orders = [o for o in orders if o[1] == "MELON"]
    expected = round(main.SEED_RESERVE["MELON"] * main.SEED_RESERVE_PRICE_CAP)
    assert len(melon_orders) == expected


def test_build_seed_orders_does_not_shrink_reserve_below_baseline_on_a_dip():
    """A depressed price (below base) must not shrink the reserve below"""
    seeds = {"MELON": 0}
    base = main.MARKET_PARAMS["MELON"]["base"]
    prices = {"MELON": base * 0.3}
    orders = main._build_seed_orders(seeds, money=100000.0, days_remaining=20, prices=prices)
    melon_orders = [o for o in orders if o[1] == "MELON"]
    assert len(melon_orders) == main.SEED_RESERVE["MELON"]


def test_crop_target_share_unramped_before_start_day():
    """Before CROP_MIX_RAMP_START_DAY the mix must be byte-identical to the flat"""
    share = main._crop_target_share(day=0)
    assert share == main.CROP_TARGET_SHARE


def test_crop_target_share_fully_ramped_after_duration():
    """Once CROP_MIX_RAMP_DURATION days past CROP_MIX_RAMP_START_DAY, STRAWBERRY's share"""
    day = main.CROP_MIX_RAMP_START_DAY + main.CROP_MIX_RAMP_DURATION
    share = main._crop_target_share(day)
    assert abs(share["STRAWBERRY"] - main.STRAWBERRY_LATE_SHARE) < 1e-9
    assert abs(sum(share.values()) - 1.0) < 1e-9
    others = [c for c in main.CASH_CROPS if c != "STRAWBERRY"]
    for a, b in itertools.pairwise(others):
        base_ratio = main.CROP_TARGET_SHARE[a] / main.CROP_TARGET_SHARE[b]
        ramped_ratio = share[a] / share[b]
        assert abs(ramped_ratio - base_ratio) < 1e-9


def test_crop_target_share_ramps_linearly_at_the_midpoint():
    day = main.CROP_MIX_RAMP_START_DAY + main.CROP_MIX_RAMP_DURATION // 2
    share = main._crop_target_share(day)
    base = main.CROP_TARGET_SHARE["STRAWBERRY"]
    expected_midpoint = (base + main.STRAWBERRY_LATE_SHARE) / 2
    assert abs(share["STRAWBERRY"] - expected_midpoint) < 0.01


def test_strawberry_seed_reserve_ramps_on_the_same_schedule():
    assert main._strawberry_seed_reserve(day=0) == main.SEED_RESERVE["STRAWBERRY"]
    fully_ramped_day = main.CROP_MIX_RAMP_START_DAY + main.CROP_MIX_RAMP_DURATION
    assert main._strawberry_seed_reserve(fully_ramped_day) == main.STRAWBERRY_RESERVE_LATE


def test_build_fertilizer_orders_stockpiles_when_price_is_cheap_and_wanted():
    """Locks the price-reactive fertilizer buying fix (see the module"""
    private = {"shed": {"FERTILIZER": main.FERTILIZER_RESERVE}, "inventories": [{}]}
    base = main.MARKET_PARAMS["FERTILIZER"]["base"]
    prices = {"FERTILIZER": base * main.FERTILIZER_CHEAP_PRICE_FRAC}
    orders = main._build_fertilizer_orders(private, prices, money=10000.0, want_fertilizer=True)
    assert orders == [["BUY_PRODUCT", "FERTILIZER", 1]]


def test_build_fertilizer_orders_stops_at_the_normal_reserve_when_not_cheap():
    private = {"shed": {"FERTILIZER": main.FERTILIZER_RESERVE}, "inventories": [{}]}
    base = main.MARKET_PARAMS["FERTILIZER"]["base"]
    prices = {"FERTILIZER": base}
    assert main._build_fertilizer_orders(private, prices, money=10000.0, want_fertilizer=True) == []


def test_build_fertilizer_orders_stops_at_the_normal_reserve_when_nothing_needs_it():
    """Cheap price alone isn't enough -- without a fertilizable crop"""
    private = {"shed": {"FERTILIZER": main.FERTILIZER_RESERVE}, "inventories": [{}]}
    base = main.MARKET_PARAMS["FERTILIZER"]["base"]
    prices = {"FERTILIZER": base * main.FERTILIZER_CHEAP_PRICE_FRAC}
    assert main._build_fertilizer_orders(private, prices, money=10000.0, want_fertilizer=False) == []


def test_build_fertilizer_orders_cheap_threshold_stays_below_the_sell_threshold():
    """The stockpile-buy trigger and the sell trigger must never overlap,"""
    assert main.FERTILIZER_CHEAP_PRICE_FRAC < main.SELL_PARAMS["FERTILIZER"]["threshold"]


def test_crop_planner_balances_toward_target_share():
    """MELON's target share (0.45) is furthest from 0 among an empty farm"""
    farm = _farm({})
    private = {"seeds": {"MELON": 5, "WHEAT": 5, "CARROT": 5}}
    planner = main._CropPlanner(private, farm, days_remaining=20)
    assert planner.choose() == "MELON"


def test_crop_planner_avoids_depressed_price_crop():
    """MELON's price crashes hard on a big sell dump and recovers slowly."""
    farm = _farm({})
    private = {"seeds": {"MELON": 5, "WHEAT": 5, "CARROT": 5}}
    prices = {"MELON": 50, "WHEAT": 25, "CARROT": 35}
    planner = main._CropPlanner(private, farm, days_remaining=20, prices=prices)
    assert planner.choose() in ("WHEAT", "CARROT")


def test_crop_planner_price_attractiveness_can_override_target_share_gap():
    """_price_ok is a binary pass/fail -- once both crops clear it, a real"""
    farm = _farm({})
    private = {"seeds": {"MELON": 5, "STRAWBERRY": 5}}
    prices = {"MELON": 250, "STRAWBERRY": 240}
    planner = main._CropPlanner(private, farm, days_remaining=20, prices=prices)
    planner.counts["MELON"] = 40
    planner.counts["STRAWBERRY"] = 3
    planner.total = 100
    assert planner.choose() == "STRAWBERRY"


def test_crop_planner_falls_back_when_every_price_is_depressed():
    """Land should never sit idle just because the whole market dipped at"""
    farm = _farm({})
    private = {"seeds": {"MELON": 5, "WHEAT": 5, "CARROT": 5}}
    prices = {"MELON": 1, "WHEAT": 1, "CARROT": 1}
    planner = main._CropPlanner(private, farm, days_remaining=20, prices=prices)
    assert planner.choose() == "MELON"


def test_crop_planner_excludes_crops_that_cannot_mature_in_time():
    """See main.py's Endgame liquidation note: MELON needs"""
    farm = _farm({})
    private = {"seeds": {"MELON": 5, "WHEAT": 5, "CARROT": 5}}
    planner = main._CropPlanner(private, farm, days_remaining=5)
    assert "MELON" not in planner.plantable_crops
    assert planner.choose() in ("WHEAT", "CARROT")


def _resolve_target(
    pos, farm, day, board_size, inv, planner, animal_planner, shed_tiles,
    want_fertilizer, shed_wheat_available, liquidating, animal_zone,
    quadrant_util=None, plant_priority=None, last_chance_window=False,
):
    """Test helper mirroring agent()'s post-on-tile routing for a single unit -- the"""
    if quadrant_util is None:
        quadrant_util = _NEUTRAL_QUADRANT_UTIL
    if plant_priority is None:
        plant_priority = main.PLANT_PRIORITY
    claimed = set()
    personal = main._find_personal_target(
        pos, farm, day, board_size, claimed, inv, animal_planner, shed_tiles,
        want_fertilizer, shed_wheat_available, liquidating,
    )
    if personal is not None:
        claimed.add((personal[0], personal[1]))
    jobs = main._build_tile_jobs(
        farm, day, board_size, claimed, animal_zone, quadrant_util, plant_priority,
        liquidating, planner, animal_planner, last_chance_window,
    )
    fx, fy = pos
    scored = [(x, y, prio, abs(x - fx) + abs(y - fy) + bias) for (x, y, prio, bias) in jobs]
    if personal is not None:
        scored.append(personal)
    if not scored:
        return None
    scored.sort(key=lambda c: (c[2], c[3]))
    x, y, prio, _ = scored[0]
    return (x, y, prio)


def test_find_target_shed_return_stays_routine_below_the_carry_threshold():
    """Locks the targeted shed-return fix (see the module docstring's"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[0][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": True, "cared_today": False}
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    small_load = {"STRAWBERRY": main.CARRIED_PRODUCE_URGENT_THRESHOLD - 1}
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv=small_load,
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles={(4, 4)},
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert (target[0], target[1]) == (1, 0)


def test_find_target_shed_return_turns_urgent_above_the_carry_threshold():
    """Same setup, but at the threshold: now walks home instead."""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[0][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": True, "cared_today": False}
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    big_load = {"STRAWBERRY": main.CARRIED_PRODUCE_URGENT_THRESHOLD}
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv=big_load,
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles={(4, 4)},
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert (target[0], target[1]) == (4, 4)


def test_find_target_wheat_below_threshold_stays_held_outside_liquidation():
    """Companion to the two tests above, but for WHEAT specifically -- see"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[0][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": True, "cared_today": False}
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    small_wheat = {"WHEAT": main.WHEAT_SHED_RETURN_THRESHOLD - 1}
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv=small_wheat,
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles={(4, 4)},
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert (target[0], target[1]) == (1, 0)


def test_find_target_wheat_below_threshold_still_stays_held_during_liquidation():
    """Same setup, but liquidating: deliberately UNCHANGED (an earlier"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[0][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": True, "cared_today": False}
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    small_wheat = {"WHEAT": main.WHEAT_SHED_RETURN_THRESHOLD - 1}
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv=small_wheat,
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=1),
        animal_planner=animal_planner, shed_tiles={(4, 4)},
        want_fertilizer=False, shed_wheat_available=False, liquidating=True,
        animal_zone=set(),
    )
    assert (target[0], target[1]) == (1, 0)


def test_find_target_prioritizes_a_tile_about_to_become_a_weed():
    """See the module docstring's Neglected-tile urgency note: a tile with"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    routine = _tile(watered_today=False)
    routine["consecutive_unwatered"] = 0
    at_risk = _tile(watered_today=False)
    tiles[0][1] = routine
    tiles[4][4] = at_risk
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv={},
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert (target[0], target[1]) == (4, 4)


def test_find_target_at_risk_water_stays_priority_half_outside_last_chance_window():
    """Third variant in [[harvest-early-at-risk-salvage-regression]]: outside"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    at_risk = _tile(watered_today=False)
    tiles[0][1] = at_risk
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv={},
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(), last_chance_window=False,
    )
    assert target[2] == 0.5


def test_find_target_at_risk_water_escalates_to_harvest_tier_within_last_chance_window():
    """Same tile, but within LAST_CHANCE_HOURS of day's end: now ties the"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    at_risk = _tile(watered_today=False)
    tiles[0][1] = at_risk
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv={},
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(), last_chance_window=True,
    )
    assert target[2] == 0


def test_find_target_water_deprioritized_when_yield_inert_and_not_at_risk():
    """See _is_yield_inert's own docstring: WHEAT age 0 is well before its"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    fresh = _tile(crop="WHEAT", planted_day=0, watered_today=False)
    fresh["consecutive_unwatered"] = 0
    tiles[0][1] = fresh
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv={},
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert target[2] == main.WATER_YIELD_INERT_PRIORITY


def test_find_target_water_normal_priority_once_past_yield_inert_window():
    """Same tile, one day later: age 2 == WHEAT's window_start, so today's"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tile = _tile(crop="WHEAT", planted_day=0, watered_today=False)
    tile["consecutive_unwatered"] = 0
    tiles[0][1] = tile
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=2, board_size=board_size, inv={},
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert target[2] == main.WATER_ROUTINE_PRIORITY


def test_find_target_safe_harvestable_tile_uses_harvest_safe_priority():
    """A ripe tile with consecutive_unwatered==0 has a full day of slack"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    safe = _tile(crop="WHEAT", planted_day=0, watered_today=True, yield_units=4)
    tiles[0][1] = safe
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=4, board_size=board_size, inv={},
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert target[2] == main.HARVEST_SAFE_PRIORITY


def test_find_target_prioritizes_an_at_risk_harvestable_tile_over_a_closer_one():
    """See the module docstring's Harvest-ready tiles rotting to weeds note:"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    safe = _tile(crop="WHEAT", planted_day=0, watered_today=True, yield_units=4)
    at_risk = _tile(crop="WHEAT", planted_day=0, watered_today=False, yield_units=4)
    tiles[0][1] = safe
    tiles[4][4] = at_risk
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=4, board_size=board_size, inv={},
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert (target[0], target[1]) == (4, 4)
    assert target[2] == 0


def test_find_target_plant_priority_loses_to_weeding_at_baseline():
    """See the module docstring's Planting priority under the gate note:"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[0][1] = {"kind": "WEED"}
    farm = {"tiles": tiles, "hands": []}
    private = {"seeds": {"WHEAT": 5}}
    planner = main._CropPlanner(private, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv={},
        planner=planner, animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert (target[0], target[1]) == (1, 0)


def test_find_target_plant_priority_beats_weeding_under_the_gate():
    """Same setup as above, but with PLANT_PRIORITY_UNDER_GATE (1.5, below"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[4][4] = {"kind": "WEED"}
    farm = {"tiles": tiles, "hands": []}
    private = {"seeds": {"WHEAT": 5}}
    planner = main._CropPlanner(private, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv={},
        planner=planner, animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(), plant_priority=main.PLANT_PRIORITY_UNDER_GATE,
    )
    assert target[2] == main.PLANT_PRIORITY_UNDER_GATE


def test_find_target_animal_upkeep_ties_plant_priority_under_the_gate():
    """Animal upkeep (collect-fertilizer/care) floors at"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[0][0] = "LOCKED"
    tiles[0][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": True,
                    "fertilizer_available": False, "cared_today": False}
    tiles[4][4] = None
    farm = {"tiles": tiles, "hands": []}
    private = {"seeds": {"WHEAT": 5}}
    planner = main._CropPlanner(private, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv={},
        planner=planner, animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(), plant_priority=main.PLANT_PRIORITY_UNDER_GATE,
    )
    assert (target[0], target[1]) == (1, 0)


def test_find_target_animal_upkeep_priority_unchanged_at_baseline():
    """Outside the gate (plant_priority=PLANT_PRIORITY=4), animal upkeep"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[0][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": True,
                    "fertilizer_available": False, "cared_today": False}
    farm = {"tiles": tiles, "hands": []}
    private = {"seeds": {}}
    planner = main._CropPlanner(private, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv={},
        planner=planner, animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert target[2] == 3


def test_land_utilization_selects_boosted_plant_priority_below_the_gate():
    """Locks agent()'s own selection logic (see the module docstring's"""
    tiles = {(0, 0): _tile()}
    farm = _farm(tiles)
    assert main._land_utilization(farm) < main.LAND_UTILIZATION_GATE


def test_find_target_prefers_the_less_planted_quadrant_on_a_distance_tie():
    """See the module docstring's spatial clustering (crops) note: a replay"""
    board_size = 4
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    for (x, y) in [(0, 0), (1, 0), (0, 1)]:
        tiles[y][x] = _tile(watered_today=True)
    farm = {"tiles": tiles, "hands": []}
    private = {"seeds": {"WHEAT": 5}}
    planner = main._CropPlanner(private, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    quadrant_util = main._quadrant_utilization(farm, board_size)
    target = _resolve_target(
        pos=[1, 0], farm=farm, day=0, board_size=board_size, inv={},
        planner=planner, animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(), quadrant_util=quadrant_util,
    )
    assert (target[0], target[1]) == (2, 0)


def test_assign_tile_jobs_matches_globally_not_order_dependent():
    """The actual behavior difference this rework is for: two units and two tile-only jobs"""
    jobs = [(1, 0, 2, 0), (9, 0, 2, 0)]
    units = [(0, (0, 0)), (1, (9, 1))]
    assignment = main._assign_tile_jobs(units, jobs, {})
    assert assignment[0] == (1, 0)
    assert assignment[1] == (9, 0)


def test_assign_tile_jobs_personal_candidate_competes_fairly_with_the_pool():
    """See _decide_unit's own docstring: a personal candidate must NOT win outright over a"""
    jobs = [(5, 5, 0, 0)]
    personal = {0: (0, 1, 5, 1)}
    units = [(0, (0, 0))]
    assignment = main._assign_tile_jobs(units, jobs, personal)
    assert assignment[0] == (5, 5)


def test_animal_target_caps_at_crew_capacity_not_just_land_share():
    """See the module docstring's Animal capacity vs crop capacity note: a"""
    big_land_tiny_crew = {"tiles": [[None] * 10 for _ in range(10)], "hands": []}
    assert main._animal_target(big_land_tiny_crew) == 1

    small_land_big_crew = {"tiles": [[None] * 3 for _ in range(3)], "hands": [[0, 0]] * 6}
    assert main._animal_target(small_land_big_crew) == 2


def test_animal_zone_prefers_tiles_closest_to_shed():
    """See the module docstring's spatial clustering (animals) note: animal"""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    for (sx, sy) in main._shed_adjacent_tiles(board_size):
        tiles[sy][sx] = {"kind": "PLANT"}
    farm = {"tiles": tiles}
    zone = main._animal_zone(farm, board_size, needed=1)
    ((zx, zy),) = zone
    assert main._dist_to_shed(zx, zy, board_size) == 1
    assert main._dist_to_shed(0, 0, board_size) > 1


def _farm_with_n_live_sheep(n, board_size=10):
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    for i in range(n):
        tiles[0][i] = {"kind": "PASTURE", "animal": "SHEEP"}
    return {"tiles": tiles, "hands": list(range(15))}


def test_animal_planner_choose_structure_respects_sheep_hard_cap():
    """SHEEP_MAX: ANIMAL_TARGET_SHARE alone is a proportional split with no ceiling, so it grows"""
    farm = _farm_with_n_live_sheep(main.SHEEP_MAX)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    animal_planner.target = 20
    assert animal_planner.choose_structure() == "COW"


def test_animal_planner_choose_pickup_respects_sheep_hard_cap():
    farm = _farm_with_n_live_sheep(main.SHEEP_MAX)
    farm["tiles"][1][0] = {"kind": "PASTURE"}
    animal_planner = main._AnimalPlanner(farm, {"SHEEP": 1}, {})
    animal_planner.target = 20
    assert animal_planner.choose_pickup() is None


def test_build_animal_orders_stops_buying_sheep_at_the_hard_cap():
    farm = _farm_with_n_live_sheep(main.SHEEP_MAX)
    farm["unlocked_quadrants"] = ["NW", "NE", "SW"]
    private = {"shed": {}}
    orders = main._build_animal_orders(farm, private, money=100000.0)
    assert not any(o[0] == "BUY_ANIMAL" and o[1] == "SHEEP" for o in orders)
    assert any(o[0] == "BUY_ANIMAL" and o[1] == "COW" for o in orders)


def _all_tests():
    return [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]


if __name__ == "__main__":
    failures = []
    for name, fn in _all_tests():
        try:
            fn()
            print(f"  ok  {name}")
        except AssertionError as e:
            failures.append(name)
            print(f"FAIL  {name}: {e}")
    total = len(_all_tests())
    print(f"\n{total - len(failures)}/{total} passed")
    if failures:
        sys.exit(1)

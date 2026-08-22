"""Regression tests for main.py -- pure functions over plain dicts, so no
live kaggle_environments instance is needed. No pytest dependency: functions
are named test_* (pytest would pick them up unmodified if the project ever
adds it) but this file also runs standalone via `uv run test_main.py`. Same
style and philosophy as rl/test_env.py.

Every function tested here caused a real bug this session hit once already
(see main.py's docstring and this session's history):
  - `_is_harvestable`: yield_units > 0 alone isn't "ready" for a freshly
    planted one-time crop -- the original bug that made the farmer spam a
    no-op HARVEST instead of WATER, letting the crop die to neglect.
  - `_is_weed` / `_decide_unit`'s DIG branch: the other half of that same
    incident -- without DIG, a tile that turned into a weed was permanently
    lost (SELL/WATER/PLANT all no-op on a WEED tile).
  - `_build_sell_orders`: an earlier version capped units sold per turn,
    which measured as a 19/20-seed regression against dumping in full (see
    reference/main_dump_sell.py and main.py's docstring's Correction note).
    This locks the dump-in-full behavior so it can't quietly regress back.
  - `_CropPlanner.choose`: the crop-mix balancing logic multiple other
    functions (planting, seed-buying) depend on being correct.
  - `_build_land_orders`: a replay showed this agent buying all 4
    quadrants by day 11 while plateauing at ~40-48% of tiles actually
    planted for the rest of the season -- land purchases now gate on
    utilization of what's already owned (see main.py's docstring's Land
    utilization gate note).
  - `_is_fertilizable` / `_last_yield_day`: TOMATO/STRAWBERRY's
    `max_yield_day` is just a placeholder equal to `first_yield_day`, not
    "last production day" -- using it directly would only ever fertilize
    the first of four scheduled productions (see main.py's docstring's
    TOMATO/STRAWBERRY note).
"""

from __future__ import annotations

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


_NEUTRAL_QUADRANT_UTIL = {"NW": 0.0, "NE": 0.0, "SW": 0.0, "SE": 0.0}  # no quadrant-balance
                                                                        # bias -- for tests that
                                                                        # don't exercise it


def _farm_with_one_fertilizable_tile():
    """A 3x3 board, all LOCKED except the unit's own shed-adjacent tile
    (0, 0) and one fertilizable WHEAT tile at (2, 2) -- shared setup for
    the fertilizer-delivery-vs-liquidation tests below."""
    board = [["LOCKED" for _ in range(3)] for _ in range(3)]
    board[0][0] = None
    board[2][2] = _tile(crop="WHEAT", planted_day=0, watered_today=True)
    return {
        "money": 3000.0, "tiles": board, "farmer": [0, 0], "hands": [],
        "unlocked_quadrants": ["NW"], "hires_today": 0,
    }


def test_is_harvestable_false_immediately_after_planting():
    """The core bug: WHEAT's first_yield_day is 2, but _new_plant seeds
    yield_units at 1 the instant it's planted (kaggriculture.py) -- so
    yield_units > 0 alone must NOT mean ready."""
    fresh = _tile(crop="WHEAT", planted_day=5, yield_units=1)
    assert not main._is_harvestable(fresh, day=5, liquidating=False)
    assert not main._is_harvestable(fresh, day=6, liquidating=False)  # age 1, still < first_yield_day of 2


def test_is_harvestable_false_at_first_yield_day_still_growing():
    """Harvest-timing note: legal (age >= first_yield_day) is NOT the same as
    fully grown -- WHEAT's yield_units keeps growing (watered) through
    max_yield_day (age 4). Harvesting the instant it's legal locks in
    yield_units=1 instead of up to 4 (watered) or 6 (fertilized)."""
    ripe = _tile(crop="WHEAT", planted_day=5, yield_units=1)
    assert not main._is_harvestable(ripe, day=7, liquidating=False)  # age 2 == first_yield_day, not max_yield_day (4)


def test_is_harvestable_true_once_max_yield_day_reached():
    mature = _tile(crop="WHEAT", planted_day=5, yield_units=3)
    assert main._is_harvestable(mature, day=9, liquidating=False)  # age 4 == WHEAT's max_yield_day


def test_is_harvestable_true_once_yield_cap_reached_even_before_max_yield_day():
    """Nothing left to gain by waiting once yield_units hits max_yield (e.g.
    fertilized growth reaching the cap early) -- no reason to keep the tile
    occupied past that point."""
    capped = _tile(crop="WHEAT", planted_day=5, yield_units=6)  # WHEAT's max_yield
    assert main._is_harvestable(capped, day=8, liquidating=False)  # age 3, still < max_yield_day (4)


def test_is_harvestable_true_early_when_liquidating():
    """Mirrors the X578/X594 kernel-pull reference's own harvest-timing
    trade-off: with no season left to wait for full growth, grab whatever
    yield exists now instead of risking it never getting harvested/sold."""
    early = _tile(crop="WHEAT", planted_day=5, yield_units=1)
    assert not main._is_harvestable(early, day=7, liquidating=False)  # age 2, would normally wait
    assert main._is_harvestable(early, day=7, liquidating=True)  # same tile, liquidating -- harvest now


def test_is_harvestable_ongoing_crop_unaffected_by_max_yield_day():
    """TOMATO/STRAWBERRY are unaffected: their own scheduled-production
    cadence (see _is_fertilizable's ongoing-crop note) already governs when
    yield_units appears, so the one-time-crop maturity wait doesn't apply."""
    tomato = _tile(crop="TOMATO", planted_day=0, yield_units=1)
    assert main._is_harvestable(tomato, day=8, liquidating=False)  # age 8 == first_yield_day == "max_yield_day" placeholder


def test_is_harvestable_false_when_yield_units_zero():
    empty = _tile(crop="MELON", planted_day=0, yield_units=0)
    assert not main._is_harvestable(empty, day=20, liquidating=False)  # long past first_yield_day, nothing to harvest though


def test_is_fertilizable_covers_full_ongoing_crop_schedule():
    """TOMATO's max_yield_day (8) is just a placeholder equal to
    first_yield_day (kaggriculture.py's CROPS table) -- not "last
    production day". Its real schedule runs ages 8/9/10/11 (interval 1,
    max_yield 4), so fertilizing must stay eligible through age 11, not
    cut off after the first scheduled production."""
    tile = _tile(crop="TOMATO", planted_day=0, fertilized_until_day=-1)
    assert main._is_fertilizable(tile, day=11)  # last scheduled production
    assert not main._is_fertilizable(tile, day=12)  # past the last production


def test_decide_unit_digs_a_weed_instead_of_stalling():
    """Without DIG, a WEED tile is permanently lost -- PLANT/WATER/SELL all
    no-op on it (kaggriculture.py), so it must be the top priority action
    when a unit is standing on one."""
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
    """See the module docstring's Selling fertilizer note: FERTILIZER
    collected from animal structures used to be excluded from "carried
    produce" unconditionally, so once nothing was left to fertilize a unit
    holding surplus collected FERTILIZER had no candidate to chase at all
    and just sat on it for the rest of the game instead of banking it in
    the shed to sell. A unit standing at the shed with no fertilizable crop
    anywhere should drop it, not PASS."""
    farm = _farm({})  # no plant tiles anywhere -> nothing is fertilizable
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
    """The regression the first version of the fix above caused (see the
    module docstring's Selling fertilizer note): a unit standing at the
    shed holding FERTILIZER it just fetched for delivery must NOT
    immediately DROP it back the moment want_fertilizer is still true, or
    every fertilizer pickup livelocks PICKUP/DROP forever -- a 10-seed pool
    comparison with that version collapsed mean money from ~$36-38k to
    ~$2,290 against `random` alone.

    `_decide_unit` itself no longer walks (that's agent()'s job now, via
    `_find_personal_target` -- see the module docstring's Routing efficiency
    note): confirm it doesn't DROP (returns None, not an action, since (0,0)
    is a shed tile with nothing to do standing still), and separately that
    the unit's personal target really is the fertilizable tile, not the shed
    itself -- i.e. it would walk to deliver, not livelock on DROP."""
    farm = _farm_with_one_fertilizable_tile()
    planner = main._CropPlanner({"seeds": {}}, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    action = main._decide_unit(
        pos=[0, 0], inv={"FERTILIZER": 1}, farm=farm, private={"shed": {}}, day=0,
        claimed=set(), planner=planner, animal_planner=animal_planner, shed_tiles={(0, 0)},
        want_fertilizer=True, liquidating=False, animal_zone=set(),
    )
    assert action is None  # not DROP -- doesn't resolve here, agent() routes it onward
    target = main._find_personal_target(
        pos=[0, 0], farm=farm, day=0, board_size=3, claimed=set(), inv={"FERTILIZER": 1},
        animal_planner=animal_planner, shed_tiles={(0, 0)},
        want_fertilizer=True, shed_wheat_available=False, liquidating=False,
    )
    assert (target[0], target[1]) == (2, 2)  # the fertilizable tile, not another DROP at the shed


def test_decide_unit_keeps_freshly_picked_up_wheat_below_the_threshold():
    """Same livelock family as the FERTILIZER test above, for WHEAT (12th
    market-order-slot-starvation attempt, see the module docstring): a
    unit standing at the shed holding a freshly PICKUP'd WHEAT batch for
    animal feed (up to 4 units -- see _decide_unit's own PICKUP WHEAT
    call) must NOT immediately DROP it back, or every feed pickup
    livelocks PICKUP/DROP forever. This was the actual bug behind the
    11th attempt's weed-count explosion (8->34 of 75 tiles) -- diffing
    that replay against a corrected one showed only 1 of 720 turns
    differed, proving it wasn't a trip-generation issue at all.
    WHEAT_SHED_RETURN_THRESHOLD=5 must exceed PICKUP's 4-unit cap for
    this guard to actually hold."""
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
    """See the module docstring's Endgame liquidation (WHEAT) note: without
    `or liquidating`, a carried WHEAT batch below WHEAT_SHED_RETURN_THRESHOLD
    (e.g. left over from a feed PICKUP) has no path to the shed and sits
    stranded in unit inventory through game end -- unsold inventory doesn't
    count toward final score. Mirrors the FERTILIZER liquidation test above."""
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
    """See the module docstring's Endgame liquidation (WHEAT) note, third
    attempt: a first version dropped sub-threshold WHEAT during liquidation
    unconditionally and measurably lost real money, because it let a unit
    DROP feed WHEAT at the shed instead of walking it to a still-hungry
    animal -- missing a feed this late loses the whole animal, a bigger
    loss than a few unsold units. With an unfed animal on the farm, the
    guard (`_any_unfed_animal`) must keep the WHEAT held even while
    liquidating."""
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
    """See the module docstring's Endgame liquidation (fertilizer) note: a
    replay showed real, ongoing dollar value sitting in unit inventories
    that never made it back to the shed before the season ended. Once
    liquidating, a further fertilizer-delivery trip isn't worth the travel
    time even if some crop is nominally still fertilizable -- carry it to
    the shed and sell it instead of holding out for a delivery."""
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
    assert main._step_toward(3, 3, 3, 3) is None  # already there


def test_build_sell_orders_dumps_the_full_shed_stock():
    """Locks the reverted fix (see module docstring): no per-turn cap --
    a big harvest sells in one order, not spread across several turns.
    No animals on this farm, so the 18th attempt's WHEAT reserve is 0 and
    doesn't interfere."""
    shed = {"WHEAT": 47}
    prices = {"WHEAT": 30}  # well above WHEAT's 0.45 * base(25) = 11.25 threshold
    farm = _farm({})
    orders = main._build_sell_orders(shed, prices, days_remaining=20, farm=farm)
    assert orders == [["SELL", "WHEAT", 47]]


def test_build_sell_orders_holds_below_threshold():
    shed = {"MELON": 10}
    prices = {"MELON": 50}  # below 0.65 * base(250) = 162.5
    farm = _farm({})
    orders = main._build_sell_orders(shed, prices, days_remaining=20, farm=farm)
    assert orders == []


def test_build_sell_orders_holds_back_a_wheat_reserve_for_live_animals():
    """18th market-order-slot-starvation attempt (see the module docstring
    and .omc/plans/ralplan-wheat-feed-consolidation.md): with live animals
    on the farm, WHEAT sells shed-qty minus `live * min(WHEAT_SELL_RESERVE_
    DAYS, days_remaining)` instead of the full shed -- mirrors the X594
    kernel-pull reference's own reserve shape. 2 animals, days_remaining=20
    (>= WHEAT_SELL_RESERVE_DAYS=3) -> reserve = 2*3 = 6."""
    farm = _farm({(0, 0): _animal_tile(), (1, 0): _animal_tile()})
    shed = {"WHEAT": 20}
    prices = {"WHEAT": 30}
    orders = main._build_sell_orders(shed, prices, days_remaining=20, farm=farm)
    assert orders == [["SELL", "WHEAT", 14]]


def test_build_sell_orders_wheat_reserve_ignored_when_liquidating():
    """The reserve never withholds the endgame liquidation dump -- missing
    it would strand cash in unsold inventory at game end, same failure mode
    the reserve is meant to avoid for feed, not cause on the sell side."""
    farm = _farm({(0, 0): _animal_tile(), (1, 0): _animal_tile()})
    shed = {"WHEAT": 20}
    prices = {"WHEAT": 5}  # below threshold -- only liquidating forces the sell here
    orders = main._build_sell_orders(shed, prices, days_remaining=1, farm=farm)
    assert orders == [["SELL", "WHEAT", 20]]


def test_build_land_orders_blocked_below_utilization_gate():
    """See module docstring's Land utilization gate note: a replay showed
    this agent buying its 4th quadrant by day 11 while plateauing at
    ~40-48% tiles planted for the rest of the season -- don't buy the next
    quadrant until most of what's already owned is being used."""
    tiles = {(0, 0): _tile(), (1, 0): _tile()}  # 2 of 9 tiles planted (22%)
    farm = _farm(tiles)
    farm["money"] = 5000.0
    orders = main._build_land_orders(farm, days_remaining=20, money=farm["money"])
    assert orders == []


def test_build_land_orders_buys_once_utilization_gate_is_met():
    tiles = {(x, y): _tile() for x in range(3) for y in range(2)}  # 6 of 9 planted (67%)
    farm = _farm(tiles)
    farm["money"] = 5000.0
    orders = main._build_land_orders(farm, days_remaining=20, money=farm["money"])
    assert orders == [["BUY_LAND"]]


def test_build_land_orders_does_not_buy_the_4th_quadrant_under_land_quadrant_cap():
    """`_hands_target`'s formula still raises hire capacity all the way
    through 4 quadrants (see the module docstring's Real hiring pace note --
    unchanged; this isn't a hand-capacity argument), but LAND_QUADRANT_CAP
    (see the module docstring's Harvest-timing note's land-purchase-delay
    follow-up, X578/X594 lever 1 of 4) now blocks the purchase regardless,
    even at full utilization with plenty of cash -- mirrors the X594
    kernel-pull reference's own `n_unl < 3` cap, which never attempts this
    purchase at all."""
    tiles = {(x, y): _tile() for x in range(3) for y in range(3)}  # 9 of 9 planted (100%)
    farm = _farm(tiles)
    farm["unlocked_quadrants"] = ["NW", "NE", "SW"]
    farm["money"] = 50000.0
    orders = main._build_land_orders(farm, days_remaining=20, money=farm["money"])
    assert orders == []


def test_orders_cash_cost_sums_across_order_types():
    """16th market-order-slot-starvation attempt: _orders_cash_cost is the
    running same-turn budget agent() threads through its market-order-
    building sequence so a late builder (_build_animal_orders) doesn't
    request a purchase against cash earlier orders in the same list
    already committed -- see the module docstring's Same-turn cash
    competition note."""
    farm = {"unlocked_quadrants": ["NW", "NE"], "hires_today": 1}
    prices = {"WHEAT": 30}
    orders = [
        ["BUY_PRODUCT", "WHEAT", 2],
        ["BUY_SEED", "MELON", 1],
        ["BUY_ANIMAL", "COW", 1],
        ["BUY_LAND"],
        ["HIRE"],
    ]
    cost = main._orders_cash_cost(orders, farm, prices, market_inventory={})
    expected = (
        2 * 30                                            # BUY_PRODUCT WHEAT x2 @ 30
        + main.CROPS["MELON"]["seed"]                      # BUY_SEED MELON
        + main.ANIMALS["COW"]["cost"]                       # BUY_ANIMAL COW
        + main.LAND_PRICES[len(farm["unlocked_quadrants"]) - 1]  # BUY_LAND, n_extra=1
        + main.FARM_HAND_COST_MULT * main._fib(farm["hires_today"])  # HIRE, the 2nd hire today
    )
    assert cost == expected


def test_orders_cash_cost_sell_orders_are_negative_revenue():
    """SELL orders earn money rather than spend it, so they contribute a
    negative amount -- the second version of this function (see its
    docstring): leaving SELL out entirely was tried first and measured
    worse against reactive opponents (a purchase that only becomes
    affordable after this turn's own sell revenue lands would otherwise
    stay chronically starved, not just delayed one turn). Lever 4 of 4 (see
    the module docstring's Harvest-timing note's land-purchase-delay
    follow-up) walks per-unit price impact instead of a flat qty * price,
    so 4 units at WHEAT's own base price (inventory exactly at I0) nets
    slightly under 4 * 25 = 100 before SELL_PROCEEDS_DISCOUNT (0.9) too."""
    farm = {"unlocked_quadrants": ["NW"], "hires_today": 0}
    prices = {"WHEAT": 25}
    orders = [["SELL", "WHEAT", 4]]
    inventory = {"WHEAT": main.MARKET_I0}
    cost = main._orders_cash_cost(orders, farm, prices, market_inventory=inventory)
    assert cost == -(main._sell_proceeds("WHEAT", 4, main.MARKET_I0) * main.SELL_PROCEEDS_DISCOUNT)
    assert cost == -87.3  # locks the actual number so a future formula change is visible here too


def test_sell_proceeds_price_declines_across_units_in_one_order():
    """See _sell_proceeds's own docstring: each unit sold nudges the market
    price for the next, mirroring kaggriculture.py's own _commit_unit
    (inventory grows per unit sold above PRICE_FLOOR). Selling above I0
    should net LESS than a flat qty * first-unit-price estimate."""
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
    """18th market-order-slot-starvation attempt (see the module docstring
    and .omc/plans/ralplan-wheat-feed-consolidation.md). Supersedes this
    test's own prior version, which locked the *opposite* behavior (ignoring
    carried WHEAT, one order per unit of shortfall) as the 17th attempt's
    deliberately-reverted shape -- carried-inclusive accounting alone was a
    real regression back then (cut WHEAT sales 70% along with purchases,
    reactive sentinels 51.7% -> 3.3%). This time it's paired with
    `_build_sell_orders`'s own new WHEAT reserve rather than an unchanged
    sell side, mirroring the X594 kernel-pull reference's combination --
    unvalidated as of this commit, see the plan for status. 1 animal ->
    reserve = 2 (FEED_RESERVE_PER_ANIMAL); shed 0 + carried 3 already covers
    it, so no order at all now (the old shed-only accounting would have
    still requested 2)."""
    farm = _farm({(0, 0): _animal_tile()})  # 1 animal -> reserve = 2
    private = {"shed": {"WHEAT": 0}, "inventories": [{"WHEAT": 3}]}
    prices = {"WHEAT": 25}
    orders = main._build_feed_orders(farm, private, prices, money=10000.0)
    assert orders == []


def test_build_feed_orders_consolidates_a_real_shortfall_into_one_order():
    """Same accounting as above but with a genuine shortfall: 2 animals ->
    reserve = 4; shed 0 + carried 1 leaves 3 needed -- one order for the
    full amount, not three separate ['BUY_PRODUCT','WHEAT',1] orders."""
    farm = _farm({(0, 0): _animal_tile(), (1, 0): _animal_tile()})  # 2 animals -> reserve = 4
    private = {"shed": {"WHEAT": 0}, "inventories": [{"WHEAT": 1}]}
    prices = {"WHEAT": 25}
    orders = main._build_feed_orders(farm, private, prices, money=10000.0)
    assert orders == [["BUY_PRODUCT", "WHEAT", 3]]


def test_build_seed_orders_scales_reserve_up_with_an_attractive_price():
    """Locks the price-reactive seed reserve fix (see the module
    docstring's Price-reactive seed reserve note): a crop trading at
    SEED_RESERVE_PRICE_CAP times its own base price should top up to
    SEED_RESERVE_PRICE_CAP times its baseline SEED_RESERVE, not the flat
    baseline."""
    seeds = {"MELON": 0}
    base = main.MARKET_PARAMS["MELON"]["base"]
    prices = {"MELON": base * main.SEED_RESERVE_PRICE_CAP}
    orders = main._build_seed_orders(seeds, money=100000.0, days_remaining=20, prices=prices)
    melon_orders = [o for o in orders if o[1] == "MELON"]
    expected = round(main.SEED_RESERVE["MELON"] * main.SEED_RESERVE_PRICE_CAP)
    assert len(melon_orders) == expected


def test_build_seed_orders_does_not_shrink_reserve_below_baseline_on_a_dip():
    """A depressed price (below base) must not shrink the reserve below
    its normal baseline -- only ever scales up for an attractive price,
    same asymmetry as _CropPlanner's own price handling."""
    seeds = {"MELON": 0}
    base = main.MARKET_PARAMS["MELON"]["base"]
    prices = {"MELON": base * 0.3}  # well below base
    orders = main._build_seed_orders(seeds, money=100000.0, days_remaining=20, prices=prices)
    melon_orders = [o for o in orders if o[1] == "MELON"]
    assert len(melon_orders) == main.SEED_RESERVE["MELON"]


def test_build_fertilizer_orders_stockpiles_when_price_is_cheap_and_wanted():
    """Locks the price-reactive fertilizer buying fix (see the module
    docstring's Price-reactive fertilizer buying note): at a cheap price
    AND with somewhere for it to go (want_fertilizer), keep buying past
    the normal FERTILIZER_RESERVE, all the way up to
    FERTILIZER_STOCKPILE_RESERVE."""
    private = {"shed": {"FERTILIZER": main.FERTILIZER_RESERVE}, "inventories": [{}]}
    base = main.MARKET_PARAMS["FERTILIZER"]["base"]
    prices = {"FERTILIZER": base * main.FERTILIZER_CHEAP_PRICE_FRAC}  # right at the cheap cutoff
    orders = main._build_fertilizer_orders(private, prices, money=10000.0, want_fertilizer=True)
    assert orders == [["BUY_PRODUCT", "FERTILIZER", 1]]  # already past FERTILIZER_RESERVE, still buys


def test_build_fertilizer_orders_stops_at_the_normal_reserve_when_not_cheap():
    private = {"shed": {"FERTILIZER": main.FERTILIZER_RESERVE}, "inventories": [{}]}
    base = main.MARKET_PARAMS["FERTILIZER"]["base"]
    prices = {"FERTILIZER": base}  # at base -- not cheap enough to trigger the stockpile reserve
    assert main._build_fertilizer_orders(private, prices, money=10000.0, want_fertilizer=True) == []


def test_build_fertilizer_orders_stops_at_the_normal_reserve_when_nothing_needs_it():
    """Cheap price alone isn't enough -- without a fertilizable crop
    anywhere on the farm, this must not stockpile toward a real crop's
    worth of fertilizer with nowhere to deliver it."""
    private = {"shed": {"FERTILIZER": main.FERTILIZER_RESERVE}, "inventories": [{}]}
    base = main.MARKET_PARAMS["FERTILIZER"]["base"]
    prices = {"FERTILIZER": base * main.FERTILIZER_CHEAP_PRICE_FRAC}  # cheap
    assert main._build_fertilizer_orders(private, prices, money=10000.0, want_fertilizer=False) == []


def test_build_fertilizer_orders_cheap_threshold_stays_below_the_sell_threshold():
    """The stockpile-buy trigger and the sell trigger must never overlap,
    or this buys back fertilizer at a price it would also sell at -- the
    same self-inflicted round trip the reverted WHEAT sell/rebuy fix
    already found (see main.py's docstring)."""
    assert main.FERTILIZER_CHEAP_PRICE_FRAC < main.SELL_PARAMS["FERTILIZER"]["threshold"]


def test_crop_planner_balances_toward_target_share():
    """MELON's target share (0.45) is furthest from 0 among an empty farm
    (WHEAT 0.25, CARROT 0.15), so an equal seed budget should be spent on
    MELON first."""
    farm = _farm({})
    private = {"seeds": {"MELON": 5, "WHEAT": 5, "CARROT": 5}}
    planner = main._CropPlanner(private, farm, days_remaining=20)
    assert planner.choose() == "MELON"


def test_crop_planner_avoids_depressed_price_crop():
    """The bug a replay caught: MELON's price crashes hard on a big sell
    dump (its above-equilibrium shape is quadratic, data/README.md's Price
    Function) and only recovers ~1 unit/day (no town shop demands MELON --
    see data/README.md's Town Buildings table). Replanting MELON at its
    full 45% target share while the price is still below what we'd sell it
    for just re-floods a market we already flooded -- the planner should
    route the seed budget to wheat/carrot instead while that holds."""
    farm = _farm({})
    private = {"seeds": {"MELON": 5, "WHEAT": 5, "CARROT": 5}}
    prices = {"MELON": 50, "WHEAT": 25, "CARROT": 35}  # melon well below 0.65 * base(250)
    planner = main._CropPlanner(private, farm, days_remaining=20, prices=prices)
    assert planner.choose() in ("WHEAT", "CARROT")


def test_crop_planner_price_attractiveness_can_override_target_share_gap():
    """See the module docstring's Price-attractiveness-aware planting note:
    a real replay showed STRAWBERRY at 2x+ its base price still losing to
    MELON at roughly par, because _price_ok is a binary pass/fail -- once
    both crops clear it, only the target-mix-share gap decided, and
    MELON's dominant 45% share almost always has the bigger raw gap.
    STRAWBERRY (5% share) at a genuinely great price should be able to
    outrank MELON (45% share) merely at par."""
    farm = _farm({})
    private = {"seeds": {"MELON": 5, "STRAWBERRY": 5}}
    prices = {"MELON": 250, "STRAWBERRY": 240}  # melon at 1.0x its base (250),
                                                  # strawberry at 2.0x its base (120)
    planner = main._CropPlanner(private, farm, days_remaining=20, prices=prices)
    planner.counts["MELON"] = 40
    planner.counts["STRAWBERRY"] = 3
    planner.total = 100
    assert planner.choose() == "STRAWBERRY"


def test_crop_planner_falls_back_when_every_price_is_depressed():
    """Land should never sit idle just because the whole market dipped at
    once -- fall back to plain share-balancing (MELON, same as the
    no-price-data case) rather than returning None."""
    farm = _farm({})
    private = {"seeds": {"MELON": 5, "WHEAT": 5, "CARROT": 5}}
    prices = {"MELON": 1, "WHEAT": 1, "CARROT": 1}
    planner = main._CropPlanner(private, farm, days_remaining=20, prices=prices)
    assert planner.choose() == "MELON"


def test_crop_planner_excludes_crops_that_cannot_mature_in_time():
    """See main.py's Endgame liquidation note: MELON needs
    first_yield_day(10) + SEED_BUFFER_DAYS(1) = 11 days left to be worth
    planting; with fewer, only WHEAT/CARROT (first_yield_day 2) remain."""
    farm = _farm({})
    private = {"seeds": {"MELON": 5, "WHEAT": 5, "CARROT": 5}}
    planner = main._CropPlanner(private, farm, days_remaining=5)
    assert "MELON" not in planner.plantable_crops
    assert planner.choose() in ("WHEAT", "CARROT")


def _resolve_target(
    pos, farm, day, board_size, inv, planner, animal_planner, shed_tiles,
    want_fertilizer, shed_wheat_available, liquidating, animal_zone,
    quadrant_util=None, plant_priority=None,
):
    """Test helper mirroring agent()'s post-on-tile routing for a single unit -- the
    personal (inventory-dependent) candidate from _find_personal_target and the shared
    tile-job pool from _build_tile_jobs, compared together the same way _assign_tile_jobs
    would for a real turn (see the module docstring's Routing efficiency note). Replaces the
    old single-scan _find_target these tests were written against; kept as a test-only
    convenience since production code always has >=1 unit and uses _assign_tile_jobs
    directly. Returns (x, y, priority) or None."""
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
        liquidating, planner, animal_planner,
    )
    fx, fy = pos
    scored = [(x, y, prio, abs(x - fx) + abs(y - fy) + bias) for (x, y, prio, bias) in jobs]
    if personal is not None:
        scored.append(personal)  # already (x, y, prio, dist) -- no bias to add
    if not scored:
        return None
    scored.sort(key=lambda c: (c[2], c[3]))
    x, y, prio, _ = scored[0]
    return (x, y, prio)


def test_find_target_shed_return_stays_routine_below_the_carry_threshold():
    """Locks the targeted shed-return fix (see the module docstring's
    Shed-return priority, targeted second attempt note): carrying fewer
    than CARRIED_PRODUCE_URGENT_THRESHOLD units of produce must still use
    the routine (low) priority, losing to a routine animal-care chore --
    an earlier unconditional version of this fix escalated every carrying
    unit regardless of load and regressed."""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[0][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": True, "cared_today": False}  # close: dist=1
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    small_load = {"STRAWBERRY": main.CARRIED_PRODUCE_URGENT_THRESHOLD - 1}
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv=small_load,
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles={(4, 4)},  # far: dist=8
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert (target[0], target[1]) == (1, 0)  # cares for the nearby animal instead of walking home


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
    assert (target[0], target[1]) == (4, 4)  # cashes out instead


def test_find_target_wheat_below_threshold_stays_held_outside_liquidation():
    """Companion to the two tests above, but for WHEAT specifically -- see
    the module docstring's Endgame liquidation (WHEAT) note. Mid-game, a
    below-WHEAT_SHED_RETURN_THRESHOLD carried batch must NOT trigger a shed
    trip (the PICKUP/DROP livelock guard from the Carried-stock accounting
    note), so a nearby animal-care chore still wins."""
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
    assert (target[0], target[1]) == (1, 0)  # cares for the nearby animal instead of walking home


def test_find_target_wheat_below_threshold_still_stays_held_during_liquidation():
    """Same setup, but liquidating: deliberately UNCHANGED (see the module
    docstring's Endgame liquidation (WHEAT) note) -- a first attempt added
    `or liquidating` here too and it measurably regressed real money (15-seed
    pool A/B 59.3%->49.6%, our own money down even in matchups where the win
    count didn't move), because it dedicates a high-priority trip for nearly
    every unit's small stray WHEAT during the final days, crowding out
    higher-value harvesting/feeding. `_decide_unit`'s DROP gate still gets
    the liquidating escape (see the test below) so it's dropped opportunistically
    when a unit is at the shed for another reason -- just never a dedicated trip."""
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
    assert (target[0], target[1]) == (1, 0)  # still no dedicated trip -- cares for the animal instead


def test_find_target_prioritizes_a_tile_about_to_become_a_weed():
    """See the module docstring's Neglected-tile urgency note: a tile with
    consecutive_unwatered >= 1 is one missed watering from turning into a
    WEED (kaggriculture.py) -- it must win over a routine, merely-closer
    watering chore, or distant land quietly loses crops to neglect no
    matter what spatial-balancing exists elsewhere."""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    routine = _tile(watered_today=False)
    routine["consecutive_unwatered"] = 0  # watered fine yesterday, a full day of slack left
    at_risk = _tile(watered_today=False)  # _tile()'s own default: consecutive_unwatered=1
    tiles[0][1] = routine  # close: (1, 0), dist=1 from (0, 0)
    tiles[4][4] = at_risk  # far: (4, 4), dist=8 from (0, 0)
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, inv={},
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert (target[0], target[1]) == (4, 4)  # the at-risk tile wins despite being much farther


def test_find_target_prioritizes_an_at_risk_harvestable_tile_over_a_closer_one():
    """See the module docstring's Harvest-ready tiles rotting to weeds note:
    a harvestable tile is still subject to the same 2-consecutive-miss WEED
    conversion as an unharvested one, but priority alone (0, already top
    tier) can't express that urgency since every harvest ties there --
    HARVEST_AT_RISK_DIST_BIAS must make an at-risk-and-ripe tile win the
    distance tie-break over a merely-closer, not-yet-at-risk ripe tile, or
    the farther one silently rots into a WEED despite being top priority."""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    safe = _tile(crop="WHEAT", planted_day=0, watered_today=True, yield_units=4)  # consecutive_unwatered=0
    at_risk = _tile(crop="WHEAT", planted_day=0, watered_today=False, yield_units=4)  # =1, one miss from WEED
    tiles[0][1] = safe  # close: (1, 0), dist=1 from (0, 0)
    tiles[4][4] = at_risk  # far: (4, 4), dist=8 from (0, 0)
    farm = {"tiles": tiles, "hands": []}
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = _resolve_target(
        pos=[0, 0], farm=farm, day=4, board_size=board_size, inv={},  # age 4 == WHEAT's max_yield_day, fully grown
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(),
    )
    assert (target[0], target[1]) == (4, 4)  # the at-risk ripe tile wins despite being much farther
    assert target[2] == 0  # still top priority -- only the tie-break (sort_dist) changed


def test_find_target_plant_priority_loses_to_weeding_at_baseline():
    """See the module docstring's Planting priority under the gate note:
    at PLANT_PRIORITY (4), a routine weed-clear (2) always wins regardless
    of distance -- priority strictly outranks distance in the sort key
    shared by _build_tile_jobs/_assign_tile_jobs."""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[0][1] = {"kind": "WEED"}  # close: (1, 0), dist=1
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
    assert (target[0], target[1]) == (1, 0)  # weed wins even though (0,0) itself is plantable


def test_find_target_plant_priority_beats_weeding_under_the_gate():
    """Same setup as above, but with PLANT_PRIORITY_UNDER_GATE (1.5, below
    weeding's 2): planting a nearby empty tile now wins over walking to a
    farther weed -- this is the fix for land sitting under-utilized despite
    a full crew, see the module docstring's note."""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[4][4] = {"kind": "WEED"}  # far: (4, 4), dist=8
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
    assert target[2] == main.PLANT_PRIORITY_UNDER_GATE  # a nearby empty tile, not the distant weed


def test_find_target_animal_upkeep_ties_plant_priority_under_the_gate():
    """15th market-order-slot-starvation attempt: before this fix, animal
    upkeep (collect-fertilizer/care) sorted at a flat 3, so once seed
    orders flow again (the 14th attempt) a fresh PLANT_PRIORITY_UNDER_GATE
    (1.5) candidate would win outright regardless of distance, quietly
    diverting crew-turns off animal husbandry -- the traced mechanism
    behind the 14th attempt's MILK/WOOL/FERTILIZER price spillover (see
    the module docstring). Flooring animal upkeep at plant_priority under
    the gate ties them, so distance decides: a close animal-care tile
    should now win over a farther empty tile."""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[0][0] = "LOCKED"  # the unit's own tile: not itself a competing plant candidate
    tiles[0][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": True,
                    "fertilizer_available": False, "cared_today": False}  # close: (1, 0), dist=1
    tiles[4][4] = None  # far empty/plantable tile: (4, 4), dist=8
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
    assert (target[0], target[1]) == (1, 0)  # the close animal-care tile wins the tie


def test_find_target_animal_upkeep_priority_unchanged_at_baseline():
    """Outside the gate (plant_priority=PLANT_PRIORITY=4), animal upkeep
    must still sort at its original 3 -- min(3, 4) == 3 -- matching the
    pre-15th-attempt baseline exactly, since PLANT_PRIORITY (4) already
    lost to animal upkeep (3) before this fix."""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[0][1] = {"kind": "PASTURE", "animal": "COW", "fed_today": True,
                    "fertilizer_available": False, "cared_today": False}  # dist=1
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
    """Locks agent()'s own selection logic (see the module docstring's
    Planting priority under the gate note) via the shared _land_utilization
    helper _build_land_orders also uses."""
    tiles = {(0, 0): _tile()}  # 1 of 9 owned tiles planted -- well under the 0.50 gate
    farm = _farm(tiles)
    assert main._land_utilization(farm) < main.LAND_UTILIZATION_GATE


def test_find_target_prefers_the_less_planted_quadrant_on_a_distance_tie():
    """See the module docstring's spatial clustering (crops) note: a replay
    showed one quadrant planted dense while another sat mostly empty --
    the plain nearest-tile-wins selection has no reason to ever look past a
    marginally-closer chore in the already-crowded quadrant. On a distance
    tie between an empty tile in a crowded quadrant (NW, 3/4 planted) and
    one in an untouched quadrant (NE, 0/4), the untouched quadrant should
    win."""
    board_size = 4
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    for (x, y) in [(0, 0), (1, 0), (0, 1)]:
        tiles[y][x] = _tile(watered_today=True)  # tended, so not a competing water candidate
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
    assert (target[0], target[1]) == (2, 0)  # NE's empty tile, not NW's (1, 1) despite equal raw distance


def test_assign_tile_jobs_matches_globally_not_order_dependent():
    """The actual behavior difference this rework is for: two units and two tile-only jobs
    where the old per-unit sequential scan (unit 0 always gets its own single best pick
    first) would strand unit 1 with a worse pairing than the crew as a whole could achieve.
    Unit A is much closer to job 2 than unit B is, and unit B is much closer to job 1 than
    unit A is -- but job 1 (a weed) also happens to be unit A's own nearest candidate. A
    sequential scan processing unit A first would give A job 1 (its own nearest), leaving B
    stuck walking all the way to job 2. The global match must instead give each unit its
    better-fitting job, since both pairings tie on priority (weed=2 both) and only total
    distance differs."""
    jobs = [(1, 0, 2, 0), (9, 0, 2, 0)]  # two same-priority weeds, far apart
    units = [(0, (0, 0)), (1, (9, 1))]  # unit 0 near job 0; unit 1 near job 1
    assignment = main._assign_tile_jobs(units, jobs, {})
    assert assignment[0] == (1, 0)
    assert assignment[1] == (9, 0)


def test_assign_tile_jobs_personal_candidate_competes_fairly_with_the_pool():
    """See _decide_unit's own docstring: a personal candidate must NOT win outright over a
    genuinely more urgent pool job -- both compete in the same (priority, distance) sort.
    Here the personal candidate (a routine shed-return, priority 5) sits right next to the
    unit, while a much more urgent pool job (priority 0) is farther away -- the pool job must
    still win despite being farther, since priority dominates distance."""
    jobs = [(5, 5, 0, 0)]  # a harvest, far away, top priority
    personal = {0: (0, 1, 5, 1)}  # a routine shed-return, priority 5, right next door
    units = [(0, (0, 0))]
    assignment = main._assign_tile_jobs(units, jobs, personal)
    assert assignment[0] == (5, 5)  # the urgent pool job wins despite being farther


def test_animal_target_caps_at_crew_capacity_not_just_land_share():
    """See the module docstring's Animal capacity vs crop capacity note: a
    real replay showed the land-share target alone reaching 14 animal
    structures with only 7 units (farmer + 6 hands) total to also tend
    crops -- crop upkeep collapsed for the second half of the season.
    _animal_target must cap at whatever the crew can support, not just how
    much land is owned."""
    big_land_tiny_crew = {"tiles": [[None] * 10 for _ in range(10)], "hands": []}
    # land_target = round(100 * 0.18) = 18; capacity_target = round(1 * 1.0) = 1
    assert main._animal_target(big_land_tiny_crew) == 1

    small_land_big_crew = {"tiles": [[None] * 3 for _ in range(3)], "hands": [[0, 0]] * 6}
    # land_target = round(9 * 0.18) = 2; capacity_target = round(7 * 1.0) = 7 -- land wins
    assert main._animal_target(small_land_big_crew) == 2


def test_animal_zone_prefers_tiles_closest_to_shed():
    """See the module docstring's spatial clustering (animals) note: animal
    structures should claim the empty tiles nearest the shed first, not
    whichever empty tile a crop happens to reach first. Occupy the shed
    ring itself so the zone must pick from the next ring out -- if it were
    falling back to "any empty tile" (e.g. a stray reference to a far
    corner), this distance check would catch it."""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    for (sx, sy) in main._shed_adjacent_tiles(board_size):
        tiles[sy][sx] = {"kind": "PLANT"}  # occupy the shed ring itself
    farm = {"tiles": tiles}
    zone = main._animal_zone(farm, board_size, needed=1)
    ((zx, zy),) = zone
    assert main._dist_to_shed(zx, zy, board_size) == 1
    assert main._dist_to_shed(0, 0, board_size) > 1  # a far corner must lose out


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

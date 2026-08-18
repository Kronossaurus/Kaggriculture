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
    assert not main._is_harvestable(fresh, day=5)
    assert not main._is_harvestable(fresh, day=6)  # age 1, still < first_yield_day of 2


def test_is_harvestable_true_once_first_yield_day_reached():
    ripe = _tile(crop="WHEAT", planted_day=5, yield_units=1)
    assert main._is_harvestable(ripe, day=7)  # age 2 == WHEAT's first_yield_day


def test_is_harvestable_false_when_yield_units_zero():
    empty = _tile(crop="MELON", planted_day=0, yield_units=0)
    assert not main._is_harvestable(empty, day=20)  # long past first_yield_day, nothing to harvest though


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
        pos=[0, 0], inv={}, farm=farm, private={"shed": {}}, day=0, board_size=3, claimed=claimed,
        planner=planner, animal_planner=animal_planner, shed_tiles=set(), want_fertilizer=False, liquidating=False,
        animal_zone=set(), quadrant_util=_NEUTRAL_QUADRANT_UTIL, plant_priority=main.PLANT_PRIORITY,
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
        pos=[0, 0], inv={"FERTILIZER": 3}, farm=farm, private={"shed": {}}, day=0, board_size=3,
        claimed=claimed, planner=planner, animal_planner=animal_planner, shed_tiles={(0, 0)},
        want_fertilizer=False, liquidating=False, animal_zone=set(),
        quadrant_util=_NEUTRAL_QUADRANT_UTIL, plant_priority=main.PLANT_PRIORITY,
    )
    assert action == ["DROP"]


def test_decide_unit_keeps_freshly_picked_up_fertilizer_when_still_needed():
    """The regression the first version of the fix above caused (see the
    module docstring's Selling fertilizer note): a unit standing at the
    shed holding FERTILIZER it just fetched for delivery must NOT
    immediately DROP it back the moment want_fertilizer is still true, or
    every fertilizer pickup livelocks PICKUP/DROP forever -- a 10-seed pool
    comparison with that version collapsed mean money from ~$36-38k to
    ~$2,290 against `random` alone."""
    farm = _farm_with_one_fertilizable_tile()
    planner = main._CropPlanner({"seeds": {}}, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    action = main._decide_unit(
        pos=[0, 0], inv={"FERTILIZER": 1}, farm=farm, private={"shed": {}}, day=0, board_size=3,
        claimed=set(), planner=planner, animal_planner=animal_planner, shed_tiles={(0, 0)},
        want_fertilizer=True, liquidating=False, animal_zone=set(),
        quadrant_util=_NEUTRAL_QUADRANT_UTIL, plant_priority=main.PLANT_PRIORITY,
    )
    assert action == ["EAST"]  # walks toward the fertilizable tile at (2, 2), doesn't DROP


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
        pos=[0, 0], inv={"FERTILIZER": 1}, farm=farm, private={"shed": {}}, day=0, board_size=3,
        claimed=set(), planner=planner, animal_planner=animal_planner, shed_tiles={(0, 0)},
        want_fertilizer=True, liquidating=True, animal_zone=set(),
        quadrant_util=_NEUTRAL_QUADRANT_UTIL, plant_priority=main.PLANT_PRIORITY,
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
    a big harvest sells in one order, not spread across several turns."""
    shed = {"WHEAT": 47}
    prices = {"WHEAT": 30}  # well above WHEAT's 0.45 * base(25) = 11.25 threshold
    orders = main._build_sell_orders(shed, prices, days_remaining=20)
    assert orders == [["SELL", "WHEAT", 47]]


def test_build_sell_orders_holds_below_threshold():
    shed = {"MELON": 10}
    prices = {"MELON": 50}  # below 0.65 * base(250) = 162.5
    orders = main._build_sell_orders(shed, prices, days_remaining=20)
    assert orders == []


def test_build_land_orders_blocked_below_utilization_gate():
    """See module docstring's Land utilization gate note: a replay showed
    this agent buying its 4th quadrant by day 11 while plateauing at
    ~40-48% tiles planted for the rest of the season -- don't buy the next
    quadrant until most of what's already owned is being used."""
    tiles = {(0, 0): _tile(), (1, 0): _tile()}  # 2 of 9 tiles planted (22%)
    farm = _farm(tiles)
    farm["money"] = 5000.0
    orders = main._build_land_orders(farm, days_remaining=20)
    assert orders == []


def test_build_land_orders_buys_once_utilization_gate_is_met():
    tiles = {(x, y): _tile() for x in range(3) for y in range(2)}  # 6 of 9 planted (67%)
    farm = _farm(tiles)
    farm["money"] = 5000.0
    orders = main._build_land_orders(farm, days_remaining=20)
    assert orders == [["BUY_LAND"]]


def test_build_land_orders_buys_the_4th_quadrant_now_that_it_raises_capacity():
    """A replay originally caught the opposite of this: under the old
    MAX_HANDS=6, the hire formula (2*quadrants) maxed out at 3 quadrants,
    so the 4th never unlocked another hand and buying it just spread the
    same crew across more land for nothing -- see _land_worth_buying's own
    docstring. `_hands_target`'s current formula (see the module
    docstring's Real hiring pace note) was deliberately picked to keep
    raising hire capacity all the way through 4 quadrants, so the 4th
    *does* raise capacity again and should get bought at full utilization
    with plenty of cash. (Task-driven hiring -- sizing the crew off actual
    pending work instead of this flat quadrant formula -- was tried and
    reverted regardless of this cap, see the module docstring's Task-
    driven hiring note.)"""
    tiles = {(x, y): _tile() for x in range(3) for y in range(3)}  # 9 of 9 planted (100%)
    farm = _farm(tiles)
    farm["unlocked_quadrants"] = ["NW", "NE", "SW"]
    farm["money"] = 50000.0
    orders = main._build_land_orders(farm, days_remaining=20)
    assert orders == [["BUY_LAND"]]


def _animal_tile(kind="PASTURE"):
    return {"kind": kind, "animal": "COW" if kind == "PASTURE" else "GOOSE"}


def test_build_feed_orders_skipped_when_reserve_already_met():
    tiles = {(0, 0): _animal_tile()}
    farm = _farm(tiles)
    private = {"shed": {"WHEAT": 999}}
    prices = {"WHEAT": 25}
    assert main._build_feed_orders(farm, private, prices, money=10000.0) == []


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
    target = main._find_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, claimed=set(), inv={},
        planner=main._CropPlanner({"seeds": {}}, farm, days_remaining=20),
        animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(), quadrant_util=_NEUTRAL_QUADRANT_UTIL, plant_priority=main.PLANT_PRIORITY,
    )
    assert (target[0], target[1]) == (4, 4)  # the at-risk tile wins despite being much farther


def test_find_target_plant_priority_loses_to_weeding_at_baseline():
    """See the module docstring's Planting priority under the gate note:
    at PLANT_PRIORITY (4), a routine weed-clear (2) always wins regardless
    of distance -- priority strictly outranks distance in _find_target's
    sort key."""
    board_size = 5
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    tiles[0][1] = {"kind": "WEED"}  # close: (1, 0), dist=1
    farm = {"tiles": tiles, "hands": []}
    private = {"seeds": {"WHEAT": 5}}
    planner = main._CropPlanner(private, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    target = main._find_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, claimed=set(), inv={},
        planner=planner, animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(), quadrant_util=_NEUTRAL_QUADRANT_UTIL, plant_priority=main.PLANT_PRIORITY,
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
    target = main._find_target(
        pos=[0, 0], farm=farm, day=0, board_size=board_size, claimed=set(), inv={},
        planner=planner, animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(), quadrant_util=_NEUTRAL_QUADRANT_UTIL,
        plant_priority=main.PLANT_PRIORITY_UNDER_GATE,
    )
    assert target[2] == main.PLANT_PRIORITY_UNDER_GATE  # a nearby empty tile, not the distant weed


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
    _find_target's plain nearest-tile-wins selection has no reason to ever
    look past a marginally-closer chore in the already-crowded quadrant.
    On a distance tie between an empty tile in a crowded quadrant (NW, 3/4
    planted) and one in an untouched quadrant (NE, 0/4), the untouched
    quadrant should win."""
    board_size = 4
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    for (x, y) in [(0, 0), (1, 0), (0, 1)]:
        tiles[y][x] = _tile(watered_today=True)  # tended, so not a competing water candidate
    farm = {"tiles": tiles, "hands": []}
    private = {"seeds": {"WHEAT": 5}}
    planner = main._CropPlanner(private, farm, days_remaining=20)
    animal_planner = main._AnimalPlanner(farm, {}, {})
    quadrant_util = main._quadrant_utilization(farm, board_size)
    target = main._find_target(
        pos=[1, 0], farm=farm, day=0, board_size=board_size, claimed=set(), inv={},
        planner=planner, animal_planner=animal_planner, shed_tiles=set(),
        want_fertilizer=False, shed_wheat_available=False, liquidating=False,
        animal_zone=set(), quadrant_util=quadrant_util, plant_priority=main.PLANT_PRIORITY,
    )
    assert (target[0], target[1]) == (2, 0)  # NE's empty tile, not NW's (1, 1) despite equal raw distance


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

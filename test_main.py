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


def test_decide_unit_digs_a_weed_instead_of_stalling():
    """Without DIG, a WEED tile is permanently lost -- PLANT/WATER/SELL all
    no-op on it (kaggriculture.py), so it must be the top priority action
    when a unit is standing on one."""
    weed = {"kind": "WEED"}
    farm = _farm({(0, 0): weed})
    claimed = set()
    planner = main._CropPlanner({"seeds": {}}, farm, days_remaining=20)
    action = main._decide_unit(
        pos=[0, 0], inv={}, farm=farm, private={"shed": {}}, day=0, board_size=3, claimed=claimed,
        planner=planner, shed_tiles=set(), want_fertilizer=False, liquidating=False,
    )
    assert action == ["DIG"]


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


def test_crop_planner_balances_toward_target_share():
    """MELON's target share (0.5) is furthest from 0 among an empty farm, so
    an equal seed budget should be spent on MELON first."""
    farm = _farm({})
    private = {"seeds": {"MELON": 5, "WHEAT": 5, "CARROT": 5}}
    planner = main._CropPlanner(private, farm, days_remaining=20)
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

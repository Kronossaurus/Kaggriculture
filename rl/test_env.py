"""Regression tests for rl/env.py's reward-shaping and action-decoding logic --
pure functions over plain dicts, so no live kaggle_environments instance is
needed. No pytest dependency: functions are named test_* (pytest would pick
them up unmodified if the project ever adds it) but this file also runs
standalone via `uv run rl/test_env.py`.

Every function tested here caused a real bug this project hit once already
(see rl/README.md's "Update" sections): _potential's floor-fix (PLANT looking
like a shaping loss), its shed-value fix (HARVEST looking like a shaping
loss), its dynamic-price fix, and IDLE_PENALTY/NEGLECT_PENALTY's transition
conditions. These were only ever checked with throwaway `python -c` snippets
during development -- this file makes those checks permanent.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.env import BUY_SEED_OPS, FARMER_OPS, KaggricultureEnv, decode_action  # noqa: E402


def _empty_tiles() -> list:
    return [[None] * 10 for _ in range(10)]


def _make_obs(tiles=None, seeds=None, shed=None, prices=None) -> dict:
    return {
        "player": 0,
        "farms": [{"tiles": tiles or _empty_tiles()}, {"tiles": _empty_tiles()}],
        "private": {"seeds": seeds or {}, "shed": shed or {}},
        "market": {"prices": prices or {}},
    }


def _planted(crop: str, yield_units: int, watered_today: bool = False, x: int = 0, y: int = 0) -> list:
    tiles = _empty_tiles()
    tiles[y][x] = {"kind": "PLANT", "crop": crop, "yield_units": yield_units, "watered_today": watered_today}
    return tiles


# --- _potential: PLANT floor-fix -------------------------------------------------

def test_potential_plant_is_shaping_neutral():
    """Buying a wheat seed, then planting it (yield_units resets to 0), must not
    drop Phi -- that was the original bug (README: "Update (idle penalty added
    on top of the Phi floor-fix)")."""
    holding_seed = KaggricultureEnv._potential(_make_obs(seeds={"WHEAT": 1}))
    just_planted = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=0)))
    assert holding_seed == just_planted == 0.10, (holding_seed, just_planted)


def test_potential_rises_once_yield_exceeds_seed_cost():
    grown = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=4)))
    assert grown > 0.10, grown  # 4 units * base price 25 / 100 = 1.0


def test_potential_drops_when_crop_dies_from_neglect():
    """A WEED tile (dead from 2 unwatered days) is not `kind == "PLANT"`, so its
    value should be gone from Phi -- the intended, deserved penalty."""
    dead = KaggricultureEnv._potential(_make_obs(tiles=[[{"kind": "WEED"}] + [None] * 9] + _empty_tiles()[1:]))
    assert dead == 0.0, dead


# --- _potential: HARVEST / shed floor-fix ----------------------------------------

def test_potential_harvest_is_shaping_neutral():
    """HARVEST moves a tile's standing value into the shed in the same step; Phi
    must count the shed so this transition doesn't look like a shaping loss
    (README: "Update (shed value added to Phi -- HARVEST floor-fix)")."""
    before = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=4)))
    after = KaggricultureEnv._potential(_make_obs(shed={"WHEAT": 4}))
    assert before == after == 1.0, (before, after)


# --- _potential: dynamic market price --------------------------------------------

def test_potential_uses_live_market_price_over_base_price():
    base = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=4)))
    low = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=4), prices={"WHEAT": 10}))
    high = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=4), prices={"WHEAT": 50}))
    assert low < base < high, (low, base, high)


def test_potential_falls_back_to_base_price_when_unquoted():
    assert KaggricultureEnv._potential(_make_obs(shed={"WHEAT": 1}, prices={})) == 0.25


# --- _is_idle ---------------------------------------------------------------------

def test_is_idle_true_with_nothing_held():
    assert KaggricultureEnv._is_idle(_make_obs()) is True


def test_is_idle_false_holding_seed():
    assert KaggricultureEnv._is_idle(_make_obs(seeds={"WHEAT": 1})) is False


def test_is_idle_false_with_standing_crop():
    assert KaggricultureEnv._is_idle(_make_obs(tiles=_planted("WHEAT", yield_units=0))) is False


# --- _unwatered_crop_count ---------------------------------------------------------

def test_unwatered_crop_count_empty_farm():
    assert KaggricultureEnv._unwatered_crop_count(_make_obs()) == 0


def test_unwatered_crop_count_counts_only_unwatered():
    tiles = _planted("WHEAT", yield_units=1, watered_today=True, x=0)
    tiles[0][1] = {"kind": "PLANT", "crop": "CARROT", "yield_units": 1, "watered_today": False}
    assert KaggricultureEnv._unwatered_crop_count(_make_obs(tiles=tiles)) == 1


# --- decode_action ------------------------------------------------------------------

def test_decode_action_buy_seed_is_independent_of_farmer_op():
    """BUY_SEED must not consume the farmer's op slot -- a real player can move
    (or water, or plant, ...) *and* queue a buy order the same turn (README's
    v1 action scope section)."""
    obs = _make_obs()
    obs["private"]["shed"] = {}
    action = [FARMER_OPS.index("WATER"), BUY_SEED_OPS.index("WHEAT")]
    result = decode_action(action, obs)
    assert result["farmer"] == ["WATER"]
    assert ["BUY_SEED", "WHEAT", 1] in result["market"]


def test_decode_action_plant_splits_crop_from_op():
    obs = _make_obs()
    action = [FARMER_OPS.index("PLANT_CARROT"), 0]
    result = decode_action(action, obs)
    assert result["farmer"] == ["PLANT", "CARROT"]


def test_decode_action_no_buy_when_buy_idx_zero():
    obs = _make_obs()
    action = [FARMER_OPS.index("PASS"), BUY_SEED_OPS.index("NONE")]
    result = decode_action(action, obs)
    assert all(order[0] != "BUY_SEED" for order in result["market"])


def test_decode_action_sells_entire_shed():
    obs = _make_obs(shed={"WHEAT": 3, "CARROT": 0})
    action = [FARMER_OPS.index("PASS"), 0]
    result = decode_action(action, obs)
    assert ["SELL", "WHEAT", 3] in result["market"]
    assert not any(order[0] == "SELL" and order[1] == "CARROT" for order in result["market"])


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

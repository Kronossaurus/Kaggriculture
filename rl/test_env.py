"""Regression tests for rl/env.py's reward-shaping and action-decoding logic --
pure functions over plain dicts, so no live kaggle_environments instance is
needed. No pytest dependency: functions are named test_* (pytest would pick
them up unmodified if the project ever adds it) but this file also runs
standalone via `uv run rl/test_env.py`.

Every function tested here caused a real bug this project hit once already
(see rl/README.md's "Update" sections): _potential's floor-fix (PLANT looking
like a shaping loss), its inventory-value fix (HARVEST looking like a shaping
loss -- an earlier version of this fix and its test used the shed, which
turned out not to match what HARVEST actually does, see below), its dynamic-
price fix, and IDLE_PENALTY/NEGLECT_PENALTY's transition conditions. These
were only ever checked with throwaway `python -c` snippets during
development -- this file makes those checks permanent.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.env import (  # noqa: E402
    BUY_LAND_OPS,
    BUY_PRODUCT_OPS,
    BUY_SEED_OPS,
    FARMER_OPS,
    SELL_OPS,
    KaggricultureEnv,
    decode_action,
)
from rl.features import BOARD_SIZE, GRID_CHANNELS, SHED_ADJACENT_TILES, VECTOR_DIM, encode_obs  # noqa: E402


def _empty_tiles() -> list:
    return [[None] * 10 for _ in range(10)]


def _make_obs(tiles=None, seeds=None, shed=None, inventories=None, prices=None, day=0,
               money=0.0, opp_money=0.0) -> dict:
    return {
        "player": 0,
        "day": day,
        "farms": [
            {"tiles": tiles or _empty_tiles(), "money": money},
            {"tiles": _empty_tiles(), "money": opp_money},
        ],
        "private": {"seeds": seeds or {}, "shed": shed or {}, "inventories": inventories or []},
        "market": {"prices": prices or {}},
    }


def _planted(
    crop: str, yield_units: int, watered_today: bool = False, fertilized_until_day: int = -1,
    x: int = 0, y: int = 0,
) -> list:
    tiles = _empty_tiles()
    tiles[y][x] = {
        "kind": "PLANT", "crop": crop, "yield_units": yield_units, "watered_today": watered_today,
        "fertilized_until_day": fertilized_until_day,
    }
    return tiles


def _make_full_obs(farmer_pos=(0, 0), day=0, hour=0, money=1000.0, opp_money=1000.0,
                    seeds=None, shed=None, inventories=None, prices=None, market_inventory=None) -> dict:
    """A full obs dict encode_obs() can consume -- unlike _make_obs() above, which
    only builds what _potential/_is_idle/_unwatered_crop_count need."""
    return {
        "player": 0,
        "day": day,
        "hour": hour,
        "farms": [
            {"tiles": _empty_tiles(), "farmer": list(farmer_pos), "money": money, "unlocked_quadrants": ["NW"]},
            {"tiles": _empty_tiles(), "farmer": [0, 0], "money": opp_money, "unlocked_quadrants": ["NW"]},
        ],
        "town": {"unlocked_shops": []},
        "private": {"seeds": seeds or {}, "shed": shed or {}, "inventories": inventories or []},
        "market": {"prices": prices or {}, "inventory": market_inventory or {}},
    }


def _action(farmer="PASS", buy_seed="NONE", buy_product="NONE", buy_land="NONE", sell="NONE") -> list:
    """Build a 5-dim action, NONE/PASS everywhere except what's under test --
    keeps individual test cases readable and resistant to dimension reordering."""
    return [
        FARMER_OPS.index(farmer),
        BUY_SEED_OPS.index(buy_seed),
        BUY_PRODUCT_OPS.index(buy_product),
        BUY_LAND_OPS.index(buy_land),
        SELL_OPS.index(sell),
    ]


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


# --- _potential: HARVEST / inventory-value fix -----------------------------------

def test_potential_harvest_is_shaping_neutral():
    """HARVEST moves a tile's standing value into the *acting unit's personal
    inventory*, not the shed (confirmed in kaggriculture.py:
    `_inv_add(inv, tile["crop"], units)`, `inv` is the farmer's own inventory) --
    Phi must count `private["inventories"]` so this transition doesn't look like
    a shaping loss. (An earlier version of this fix/test used `shed` here, which
    doesn't match what HARVEST actually does -- see rl/README.md.)"""
    before = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=4)))
    after = KaggricultureEnv._potential(_make_obs(inventories=[{"WHEAT": 4}]))
    assert before == after == 1.0, (before, after)


def test_potential_treats_shed_and_inventory_equivalently():
    """The end-of-day auto-drop from inventory to shed must also be shaping-
    neutral -- Phi shouldn't care which side of that transfer the value is on."""
    in_inventory = KaggricultureEnv._potential(_make_obs(inventories=[{"WHEAT": 4}]))
    in_shed = KaggricultureEnv._potential(_make_obs(shed={"WHEAT": 4}))
    assert in_inventory == in_shed


def test_potential_sums_inventory_across_multiple_units():
    total = KaggricultureEnv._potential(_make_obs(inventories=[{"WHEAT": 2}, {"WHEAT": 2}]))
    assert total == KaggricultureEnv._potential(_make_obs(inventories=[{"WHEAT": 4}]))


# --- _potential: FERTILIZE floor-fix ----------------------------------------------

def test_potential_fertilize_is_shaping_neutral():
    """FERTILIZE consumes a real, Phi-valued FERTILIZER unit from inventory in
    exchange for a doubled-yield bonus that yield_units doesn't reflect until
    later WATER calls bank it -- Phi must credit the active bonus on the tile so
    this transition doesn't look like a shaping loss, same bug class as PLANT."""
    holding_fertilizer = KaggricultureEnv._potential(
        _make_obs(tiles=_planted("WHEAT", yield_units=0), inventories=[{"FERTILIZER": 1}], day=5)
    )
    just_fertilized = KaggricultureEnv._potential(
        _make_obs(tiles=_planted("WHEAT", yield_units=0, fertilized_until_day=7), day=5)
    )
    assert holding_fertilizer == just_fertilized, (holding_fertilizer, just_fertilized)


def test_potential_fertilizer_credit_expires_with_the_bonus():
    day5 = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=0, fertilized_until_day=7), day=5))
    day8 = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=0, fertilized_until_day=7), day=8))
    assert day8 < day5  # bonus lapsed (fertilized_until_day=7 < day=8), credit gone -- deserved if unused


def test_potential_fertilizer_credit_stacks_with_seed_cost_floor():
    """The fertilizer credit and the seed-cost floor are independent sunk costs
    on the same tile -- both should apply, not just the larger of the two."""
    unfertilized = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=0), day=5))
    fertilized = KaggricultureEnv._potential(
        _make_obs(tiles=_planted("WHEAT", yield_units=0, fertilized_until_day=7), day=5)
    )
    assert fertilized > unfertilized


# --- _opponent_money ---------------------------------------------------------------

def test_opponent_money_reads_the_other_farm():
    obs = _make_obs(money=500.0, opp_money=1200.0)
    assert KaggricultureEnv._opponent_money(obs) == 1200.0


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


# --- decode_action: independence of the four market-order dimensions --------------

def test_decode_action_buy_seed_is_independent_of_farmer_op():
    """BUY_SEED must not consume the farmer's op slot -- a real player can move
    (or water, or plant, ...) *and* queue a buy order the same turn (README's
    action scope section)."""
    obs = _make_obs()
    result = decode_action(_action(farmer="WATER", buy_seed="WHEAT"), obs)
    assert result["farmer"] == ["WATER"]
    assert ["BUY_SEED", "WHEAT", 1] in result["market"]


def test_decode_action_buy_product_is_independent_of_farmer_op():
    obs = _make_obs()
    result = decode_action(_action(farmer="WATER", buy_product="FERTILIZER"), obs)
    assert result["farmer"] == ["WATER"]
    assert ["BUY_PRODUCT", "FERTILIZER", 1] in result["market"]


def test_decode_action_buy_land_takes_no_item_or_quantity():
    """BUY_LAND has no target -- the game unlocks quadrants in a fixed order at
    fixed prices, so there's nothing to parameterize (kaggriculture.py's
    `_parse_order` just checks `order[0] == "BUY_LAND"`)."""
    obs = _make_obs()
    result = decode_action(_action(buy_land="BUY_LAND"), obs)
    assert ["BUY_LAND"] in result["market"]


def test_decode_action_all_four_market_orders_can_coexist():
    obs = _make_obs(shed={"WHEAT": 5})
    result = decode_action(
        _action(farmer="WATER", buy_seed="CARROT", buy_product="FERTILIZER", buy_land="BUY_LAND", sell="WHEAT"),
        obs,
    )
    assert result["farmer"] == ["WATER"]
    assert ["BUY_SEED", "CARROT", 1] in result["market"]
    assert ["BUY_PRODUCT", "FERTILIZER", 1] in result["market"]
    assert ["BUY_LAND"] in result["market"]
    assert ["SELL", "WHEAT", 5] in result["market"]


def test_decode_action_no_orders_when_everything_is_none():
    obs = _make_obs(shed={"WHEAT": 5})
    result = decode_action(_action(), obs)
    assert result["market"] == []


# --- decode_action: farmer op -------------------------------------------------------

def test_decode_action_plant_splits_crop_from_op():
    obs = _make_obs()
    result = decode_action(_action(farmer="PLANT_CARROT"), obs)
    assert result["farmer"] == ["PLANT", "CARROT"]


def test_decode_action_pickup_always_targets_fertilizer():
    """PICKUP has no item choice in the action space -- FERTILIZER is the only
    thing worth moving out of the shed in the current scope, see module
    docstring."""
    obs = _make_obs()
    result = decode_action(_action(farmer="PICKUP"), obs)
    assert result["farmer"] == ["PICKUP", "FERTILIZER", 1]


def test_decode_action_drop_takes_no_arguments():
    obs = _make_obs()
    result = decode_action(_action(farmer="DROP"), obs)
    assert result["farmer"] == ["DROP"]


# --- decode_action: SELL --------------------------------------------------------------

def test_decode_action_sell_liquidates_the_chosen_item():
    obs = _make_obs(shed={"WHEAT": 3, "CARROT": 7})
    result = decode_action(_action(sell="WHEAT"), obs)
    assert ["SELL", "WHEAT", 3] in result["market"]
    assert not any(order[0] == "SELL" and order[1] == "CARROT" for order in result["market"])


def test_decode_action_sell_is_a_no_op_when_nothing_of_that_item_is_held():
    obs = _make_obs(shed={})
    result = decode_action(_action(sell="WHEAT"), obs)
    assert not any(order[0] == "SELL" for order in result["market"])


# --- SHED_ADJACENT_TILES (features.py) ---------------------------------------------

def test_shed_adjacent_tiles_matches_the_engine():
    """Must match kaggriculture.py's _shed_access_tiles() exactly -- PICKUP/DROP
    silently no-op unless the farmer stands on one of these four tiles
    (data/README.md), and noop_audit.py found both ops failing ~100% of the
    time before this was ever exposed to the policy (see rl/README.md)."""
    half = BOARD_SIZE // 2
    assert SHED_ADJACENT_TILES == {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}
    assert SHED_ADJACENT_TILES == {(4, 4), (5, 4), (4, 5), (5, 5)}  # boardSize=10


# --- encode_obs (features.py) -------------------------------------------------------

def test_encode_obs_vector_matches_declared_dim():
    """VECTOR_DIM is a hand-maintained formula (see features.py) that has to stay
    in sync with what _encode_vector actually builds -- it's silently wrong (SB3
    would just see a mis-shaped Box) if the two ever drift, e.g. a field added
    to one but not the other."""
    vector = encode_obs(_make_full_obs())["vector"]
    assert vector.shape == (VECTOR_DIM,)


def test_encode_obs_grid_shapes():
    encoded = encode_obs(_make_full_obs())
    assert encoded["own_grid"].shape == (GRID_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert encoded["opp_grid"].shape == (GRID_CHANNELS, BOARD_SIZE, BOARD_SIZE)


def test_encode_obs_near_shed_flag_reflects_farmer_position():
    """The fix for the PICKUP/DROP gap noop_audit.py found (see rl/README.md) --
    locks the flag to the farmer's actual position, not just that the field
    exists."""
    on_shed_tile = encode_obs(_make_full_obs(farmer_pos=(4, 4)))["vector"]
    off_shed_tile = encode_obs(_make_full_obs(farmer_pos=(0, 0)))["vector"]
    near_shed_index = 6  # 6 leading scalars before it, see _encode_vector
    assert on_shed_tile[near_shed_index] == 1.0
    assert off_shed_tile[near_shed_index] == 0.0


def test_encode_obs_farmer_position_on_grid():
    encoded = encode_obs(_make_full_obs(farmer_pos=(3, 7)))
    farmer_channel = GRID_CHANNELS - 1
    assert encoded["own_grid"][farmer_channel, 7, 3] == 1.0
    assert encoded["own_grid"][farmer_channel].sum() == 1.0  # nowhere else


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

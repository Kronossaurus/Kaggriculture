"""Regression tests for rl/env.py's reward-shaping and action-decoding logic --"""

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
    HIRE_OPS,
    SELL_OPS,
    KaggricultureEnv,
    decode_action,
    encode_action,
)
from rl.features import BOARD_SIZE, GRID_CHANNELS, MAX_HANDS, SHED_ADJACENT_TILES, VECTOR_DIM, encode_obs  # noqa: E402


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
                    seeds=None, shed=None, inventories=None, prices=None, market_inventory=None,
                    hands=None) -> dict:
    """A full obs dict encode_obs() can consume -- unlike _make_obs() above, which"""
    return {
        "player": 0,
        "day": day,
        "hour": hour,
        "farms": [
            {"tiles": _empty_tiles(), "farmer": list(farmer_pos), "money": money,
             "unlocked_quadrants": ["NW"], "hands": hands or []},
            {"tiles": _empty_tiles(), "farmer": [0, 0], "money": opp_money,
             "unlocked_quadrants": ["NW"], "hands": []},
        ],
        "town": {"unlocked_shops": []},
        "private": {"seeds": seeds or {}, "shed": shed or {}, "inventories": inventories or []},
        "market": {"prices": prices or {}, "inventory": market_inventory or {}},
    }


def _action(farmer="PASS", buy_seed="NONE", buy_product="NONE", buy_land="NONE", sell="NONE",
            hire="NONE", hands=None) -> list:
    """Build a full action, NONE/PASS everywhere except what's under test -- keeps"""
    hand_ops = list(hands or [])
    hand_ops += ["PASS"] * (MAX_HANDS - len(hand_ops))
    return [
        FARMER_OPS.index(farmer),
        BUY_SEED_OPS.index(buy_seed),
        BUY_PRODUCT_OPS.index(buy_product),
        BUY_LAND_OPS.index(buy_land),
        SELL_OPS.index(sell),
        HIRE_OPS.index(hire),
        *(FARMER_OPS.index(op) for op in hand_ops),
    ]


def test_potential_plant_is_shaping_neutral():
    """Buying a wheat seed, then planting it (yield_units resets to 0), must not"""
    holding_seed = KaggricultureEnv._potential(_make_obs(seeds={"WHEAT": 1}))
    just_planted = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=0)))
    assert holding_seed == just_planted == 0.10, (holding_seed, just_planted)


def test_potential_rises_once_yield_exceeds_seed_cost():
    grown = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=4)))
    assert grown > 0.10, grown


def test_potential_drops_when_crop_dies_from_neglect():
    """A WEED tile (dead from 2 unwatered days) is not `kind == "PLANT"`, so its"""
    dead = KaggricultureEnv._potential(_make_obs(tiles=[[{"kind": "WEED"}] + [None] * 9] + _empty_tiles()[1:]))
    assert dead == 0.0, dead


def test_potential_harvest_is_shaping_neutral():
    """HARVEST moves a tile's standing value into the *acting unit's personal"""
    before = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=4)))
    after = KaggricultureEnv._potential(_make_obs(inventories=[{"WHEAT": 4}]))
    assert before == after == 1.0, (before, after)


def test_potential_treats_shed_and_inventory_equivalently():
    """The end-of-day auto-drop from inventory to shed must also be shaping-"""
    in_inventory = KaggricultureEnv._potential(_make_obs(inventories=[{"WHEAT": 4}]))
    in_shed = KaggricultureEnv._potential(_make_obs(shed={"WHEAT": 4}))
    assert in_inventory == in_shed


def test_potential_sums_inventory_across_multiple_units():
    total = KaggricultureEnv._potential(_make_obs(inventories=[{"WHEAT": 2}, {"WHEAT": 2}]))
    assert total == KaggricultureEnv._potential(_make_obs(inventories=[{"WHEAT": 4}]))


def test_potential_fertilize_is_shaping_neutral():
    """FERTILIZE consumes a real, Phi-valued FERTILIZER unit from inventory in"""
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
    assert day8 < day5


def test_potential_fertilizer_credit_stacks_with_seed_cost_floor():
    """The fertilizer credit and the seed-cost floor are independent sunk costs"""
    unfertilized = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=0), day=5))
    fertilized = KaggricultureEnv._potential(
        _make_obs(tiles=_planted("WHEAT", yield_units=0, fertilized_until_day=7), day=5)
    )
    assert fertilized > unfertilized


def test_opponent_money_reads_the_other_farm():
    obs = _make_obs(money=500.0, opp_money=1200.0)
    assert KaggricultureEnv._opponent_money(obs) == 1200.0


def test_potential_uses_live_market_price_over_base_price():
    base = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=4)))
    low = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=4), prices={"WHEAT": 10}))
    high = KaggricultureEnv._potential(_make_obs(tiles=_planted("WHEAT", yield_units=4), prices={"WHEAT": 50}))
    assert low < base < high, (low, base, high)


def test_potential_falls_back_to_base_price_when_unquoted():
    assert KaggricultureEnv._potential(_make_obs(shed={"WHEAT": 1}, prices={})) == 0.25


def test_is_idle_true_with_nothing_held():
    assert KaggricultureEnv._is_idle(_make_obs()) is True


def test_is_idle_false_holding_seed():
    assert KaggricultureEnv._is_idle(_make_obs(seeds={"WHEAT": 1})) is False


def test_is_idle_false_with_standing_crop():
    assert KaggricultureEnv._is_idle(_make_obs(tiles=_planted("WHEAT", yield_units=0))) is False


def test_unwatered_crop_count_empty_farm():
    assert KaggricultureEnv._unwatered_crop_count(_make_obs()) == 0


def test_unwatered_crop_count_counts_only_unwatered():
    tiles = _planted("WHEAT", yield_units=1, watered_today=True, x=0)
    tiles[0][1] = {"kind": "PLANT", "crop": "CARROT", "yield_units": 1, "watered_today": False}
    assert KaggricultureEnv._unwatered_crop_count(_make_obs(tiles=tiles)) == 1


def test_decode_action_buy_seed_is_independent_of_farmer_op():
    """BUY_SEED must not consume the farmer's op slot -- a real player can move"""
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
    """BUY_LAND has no target -- the game unlocks quadrants in a fixed order at"""
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


def test_decode_action_plant_splits_crop_from_op():
    obs = _make_obs()
    result = decode_action(_action(farmer="PLANT_CARROT"), obs)
    assert result["farmer"] == ["PLANT", "CARROT"]


def test_decode_action_pickup_always_targets_fertilizer():
    """PICKUP has no item choice in the action space -- FERTILIZER is the only"""
    obs = _make_obs()
    result = decode_action(_action(farmer="PICKUP"), obs)
    assert result["farmer"] == ["PICKUP", "FERTILIZER", 1]


def test_decode_action_drop_takes_no_arguments():
    obs = _make_obs()
    result = decode_action(_action(farmer="DROP"), obs)
    assert result["farmer"] == ["DROP"]


def test_decode_action_sell_liquidates_the_chosen_item():
    obs = _make_obs(shed={"WHEAT": 3, "CARROT": 7})
    result = decode_action(_action(sell="WHEAT"), obs)
    assert ["SELL", "WHEAT", 3] in result["market"]
    assert not any(order[0] == "SELL" and order[1] == "CARROT" for order in result["market"])


def test_decode_action_sell_is_a_no_op_when_nothing_of_that_item_is_held():
    obs = _make_obs(shed={})
    result = decode_action(_action(sell="WHEAT"), obs)
    assert not any(order[0] == "SELL" for order in result["market"])


def test_decode_action_hire_takes_no_item_or_quantity():
    """HIRE has no target either, same shape as BUY_LAND (data/README.md)."""
    obs = _make_obs()
    result = decode_action(_action(hire="HIRE"), obs)
    assert ["HIRE"] in result["market"]


def test_decode_action_hands_default_to_pass():
    obs = _make_obs()
    result = decode_action(_action(), obs)
    assert result["hands"] == [["PASS"]] * MAX_HANDS


def test_decode_action_hands_use_the_same_op_vocabulary_as_farmer():
    """Same FARMER_OPS decode (including the PLANT_<crop> split) applies per"""
    obs = _make_obs()
    result = decode_action(_action(farmer="PASS", hands=["WATER", "PLANT_CARROT", "PICKUP"]), obs)
    assert result["hands"][0] == ["WATER"]
    assert result["hands"][1] == ["PLANT", "CARROT"]
    assert result["hands"][2] == ["PICKUP", "FERTILIZER", 1]
    assert result["hands"][3:] == [["PASS"]] * (MAX_HANDS - 3)


def test_decode_action_hands_are_independent_of_farmer_op():
    obs = _make_obs()
    result = decode_action(_action(farmer="HARVEST", hands=["NORTH"]), obs)
    assert result["farmer"] == ["HARVEST"]
    assert result["hands"][0] == ["NORTH"]


def test_decode_action_always_emits_max_hands_entries():
    """Extra slots beyond however many hands are actually live are meant to be"""
    obs = _make_obs()
    result = decode_action(_action(), obs)
    assert len(result["hands"]) == MAX_HANDS


def test_encode_action_round_trips_through_decode_action():
    obs = _make_obs(shed={"WHEAT": 5})
    action = _action(farmer="WATER", buy_seed="CARROT", buy_product="FERTILIZER",
                      buy_land="BUY_LAND", sell="WHEAT", hire="HIRE",
                      hands=["PLANT_MELON", "HARVEST"])
    action_dict = decode_action(action, obs)
    assert list(encode_action(action_dict, obs)) == action


def test_encode_action_reads_hire_from_market_orders():
    action_dict = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}
    encoded = encode_action(action_dict, {})
    hire_idx = 5
    assert encoded[hire_idx] == HIRE_OPS.index("HIRE")


def test_encode_action_hands_shorter_than_max_hands_pad_with_pass():
    action_dict = {"farmer": ["PASS"], "hands": [["WATER"]], "market": []}
    encoded = encode_action(action_dict, {})
    hand_idxs = encoded[6:]
    assert len(hand_idxs) == MAX_HANDS
    assert hand_idxs[0] == FARMER_OPS.index("WATER")
    assert list(hand_idxs[1:]) == [FARMER_OPS.index("PASS")] * (MAX_HANDS - 1)


def test_encode_action_extra_hands_beyond_max_hands_are_dropped():
    action_dict = {"farmer": ["PASS"], "hands": [["PASS"]] * (MAX_HANDS + 5), "market": []}
    encoded = encode_action(action_dict, {})
    assert len(encoded) == 6 + MAX_HANDS


def test_shed_adjacent_tiles_matches_the_engine():
    """Must match kaggriculture.py's _shed_access_tiles() exactly -- PICKUP/DROP"""
    half = BOARD_SIZE // 2
    assert SHED_ADJACENT_TILES == {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}
    assert SHED_ADJACENT_TILES == {(4, 4), (5, 4), (4, 5), (5, 5)}


def test_encode_obs_vector_matches_declared_dim():
    """VECTOR_DIM is a hand-maintained formula (see features.py) that has to stay"""
    vector = encode_obs(_make_full_obs())["vector"]
    assert vector.shape == (VECTOR_DIM,)


def test_encode_obs_grid_shapes():
    encoded = encode_obs(_make_full_obs())
    assert encoded["own_grid"].shape == (GRID_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert encoded["opp_grid"].shape == (GRID_CHANNELS, BOARD_SIZE, BOARD_SIZE)


def test_encode_obs_near_shed_flag_reflects_farmer_position():
    """The fix for the PICKUP/DROP gap noop_audit.py found (see rl/README.md) --"""
    on_shed_tile = encode_obs(_make_full_obs(farmer_pos=(4, 4)))["vector"]
    off_shed_tile = encode_obs(_make_full_obs(farmer_pos=(0, 0)))["vector"]
    near_shed_index = 6
    assert on_shed_tile[near_shed_index] == 1.0
    assert off_shed_tile[near_shed_index] == 0.0


def test_encode_obs_farmer_position_on_grid():
    encoded = encode_obs(_make_full_obs(farmer_pos=(3, 7)))
    farmer_channel = GRID_CHANNELS - 1
    assert encoded["own_grid"][farmer_channel, 7, 3] == 1.0
    assert encoded["own_grid"][farmer_channel].sum() == 1.0


def test_encode_obs_hand_positions_on_grid():
    encoded = encode_obs(_make_full_obs(hands=[[1, 2], [8, 9]]))
    hand_channel = GRID_CHANNELS - 2
    assert encoded["own_grid"][hand_channel, 2, 1] == 1.0
    assert encoded["own_grid"][hand_channel, 9, 8] == 1.0
    assert encoded["own_grid"][hand_channel].sum() == 2.0


def test_encode_obs_hands_vector_active_slot_reflects_position_and_near_shed():
    encoded = encode_obs(_make_full_obs(hands=[[4, 4]]))["vector"]
    slot0 = encoded[-MAX_HANDS * 5 :][:5]
    assert slot0[0] == 4 / BOARD_SIZE
    assert slot0[1] == 4 / BOARD_SIZE
    assert slot0[2] == 1.0
    assert slot0[4] == 1.0


def test_encode_obs_hands_vector_inactive_slots_are_zeroed():
    encoded = encode_obs(_make_full_obs(hands=[[4, 4]]))["vector"]
    hands_vector = encoded[-MAX_HANDS * 5 :]
    slot1 = hands_vector[5:10]
    assert list(slot1) == [0.0, 0.0, 0.0, 0.0, 0.0]


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

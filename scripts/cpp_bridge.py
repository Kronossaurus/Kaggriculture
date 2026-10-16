"""Drive real Python agents (main.py, reference/*.py, or the kaggle_environments"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DLL_PATH = REPO_ROOT / "cpp_sim" / "kag_sim.dll"

OPS = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE",
       "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
       "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE"]
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
         "FERTILIZER", "GOOSE", "COW", "SHEEP"]
MOPS = ["NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"]
CROPS = ITEMS[:5]
PRODUCTS = ITEMS[:9]
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
         "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
OPI = {o: i for i, o in enumerate(OPS)}
ITI = {o: i for i, o in enumerate(ITEMS)}
MOPI = {o: i for i, o in enumerate(MOPS)}

N_ITEMS = 12
N_PRODUCTS = 9
N_CROPS = 5
MAX_UNITS = 40
BOARD = 10
MAX_SHOP_INSTANCES = 8


class Config(ctypes.Structure):
    _fields_ = [
        ("episode_steps", ctypes.c_int),
        ("board_size", ctypes.c_int),
        ("starting_money", ctypes.c_int),
        ("max_orders", ctypes.c_int),
        ("turns_per_day", ctypes.c_int),
        ("shed_capacity", ctypes.c_int),
        ("weed_chance", ctypes.c_double),
        ("shop_unlock_interval", ctypes.c_int),
        ("shop_sell_interval", ctypes.c_int),
        ("center_sell_interval", ctypes.c_int),
        ("hire_mult", ctypes.c_int),
        ("seed", ctypes.c_uint64),
    ]


def default_config(steps: int, seed: int) -> Config:
    """Competition defaults (data/README.md's Configuration Defaults table),"""
    cfg = Config(
        episode_steps=steps, board_size=10, starting_money=3000, max_orders=10,
        turns_per_day=24, shed_capacity=100, weed_chance=0.005,
        shop_unlock_interval=3, shop_sell_interval=4, center_sell_interval=24,
        hire_mult=1, seed=seed,
    )
    return cfg


class Tile(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint8),
        ("what", ctypes.c_uint8),
        ("has_animal", ctypes.c_bool),
        ("watered_today", ctypes.c_bool),
        ("fed_today", ctypes.c_bool),
        ("cared_today", ctypes.c_bool),
        ("fertilizer_available", ctypes.c_bool),
        ("consecutive_dry", ctypes.c_int8),
        ("yield_units", ctypes.c_int8),
        ("pending_care_bonus", ctypes.c_int8),
        ("planted_day", ctypes.c_int16),
        ("max_lifespan_step", ctypes.c_int32),
        ("fertilized_until_day", ctypes.c_int16),
    ]


T_EMPTY, T_LOCKED, T_WEED, T_COOP, T_PASTURE, T_PLANT = range(6)


class Farm(ctypes.Structure):
    _fields_ = [
        ("money", ctypes.c_double),
        ("tiles", (Tile * BOARD) * BOARD),
        ("pos_x", ctypes.c_int8 * MAX_UNITS),
        ("pos_y", ctypes.c_int8 * MAX_UNITS),
        ("n_units", ctypes.c_int),
        ("n_quadrants", ctypes.c_int),
        ("hires_today", ctypes.c_int),
        ("shed", ctypes.c_int16 * N_ITEMS),
        ("shed_total", ctypes.c_int),
        ("seeds", ctypes.c_int16 * N_CROPS),
        ("inv", (ctypes.c_int16 * N_ITEMS) * MAX_UNITS),
        ("inv_keys", (ctypes.c_uint8 * N_ITEMS) * MAX_UNITS),
        ("inv_nkeys", ctypes.c_uint8 * MAX_UNITS),
        ("discarded", ctypes.c_int32 * N_ITEMS),
        ("produced", ctypes.c_int32 * N_ITEMS),
        ("sold_units", ctypes.c_int32 * N_ITEMS),
        ("sell_revenue", ctypes.c_double),
        ("total_spend", ctypes.c_double),
    ]


class Market(ctypes.Structure):
    _fields_ = [
        ("inventory", ctypes.c_int32 * N_PRODUCTS),
        ("prices", ctypes.c_int32 * N_PRODUCTS),
    ]


class State(ctypes.Structure):
    _fields_ = [
        ("farms", Farm * 2),
        ("market", Market),
        ("shops", ctypes.c_uint8 * MAX_SHOP_INSTANCES),
        ("n_shops", ctypes.c_int),
        ("step", ctypes.c_int),
        ("day", ctypes.c_int),
        ("hour", ctypes.c_int),
        ("done", ctypes.c_bool),
    ]


class UnitAction(ctypes.Structure):
    _fields_ = [("op", ctypes.c_uint8), ("arg", ctypes.c_uint8), ("n", ctypes.c_int16)]


class Order(ctypes.Structure):
    _fields_ = [("op", ctypes.c_uint8), ("item", ctypes.c_uint8), ("n", ctypes.c_int32)]


class Action(ctypes.Structure):
    _fields_ = [
        ("units", UnitAction * MAX_UNITS),
        ("n_units", ctypes.c_int),
        ("orders", Order * 16),
        ("n_orders", ctypes.c_int),
    ]


def _load_dll() -> ctypes.CDLL:
    if not DLL_PATH.exists():
        raise FileNotFoundError(
            f"{DLL_PATH} not found -- build it first: see cpp_sim/README.md"
        )
    dll = ctypes.CDLL(str(DLL_PATH))
    dll.sim_new.argtypes = [ctypes.POINTER(Config)]
    dll.sim_new.restype = ctypes.c_void_p
    dll.sim_free.argtypes = [ctypes.c_void_p]
    dll.sim_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(Action), ctypes.POINTER(Action)]
    dll.sim_reward.argtypes = [ctypes.c_void_p, ctypes.c_int]
    dll.sim_reward.restype = ctypes.c_double
    dll.sim_done.argtypes = [ctypes.c_void_p]
    dll.sim_done.restype = ctypes.c_int
    dll.sim_state.argtypes = [ctypes.c_void_p]
    dll.sim_state.restype = ctypes.POINTER(State)
    for name in ("state", "farm", "tile", "market", "action", "config"):
        fn = getattr(dll, f"sim_sizeof_{name}")
        fn.restype = ctypes.c_int

    checks = [
        ("State", dll.sim_sizeof_state(), ctypes.sizeof(State)),
        ("Farm", dll.sim_sizeof_farm(), ctypes.sizeof(Farm)),
        ("Tile", dll.sim_sizeof_tile(), ctypes.sizeof(Tile)),
        ("Market", dll.sim_sizeof_market(), ctypes.sizeof(Market)),
        ("Action", dll.sim_sizeof_action(), ctypes.sizeof(Action)),
        ("Config", dll.sim_sizeof_config(), ctypes.sizeof(Config)),
    ]
    for name, cpp_size, py_size in checks:
        if cpp_size != py_size:
            raise RuntimeError(
                f"cpp_bridge.py struct mirror for {name} is out of sync with sim.hpp: "
                f"C++ sizeof={cpp_size}, ctypes sizeof={py_size}. Fix the _fields_ "
                f"list above before trusting anything this module produces."
            )
    return dll


_dll: ctypes.CDLL | None = None


def dll() -> ctypes.CDLL:
    global _dll
    if _dll is None:
        _dll = _load_dll()
    return _dll


def build_obs(state: State, player: int) -> dict:
    """Reconstruct the full obs dict (data/README.md's Observation Format)"""
    farms_out = []
    for p in range(2):
        f = state.farms[p]
        tiles = []
        for y in range(BOARD):
            row = []
            for x in range(BOARD):
                row.append(_tile_dict(f.tiles[y][x]))
            tiles.append(row)
        n_units = f.n_units
        farmer = [int(f.pos_x[0]), int(f.pos_y[0])]
        hands = [[int(f.pos_x[i]), int(f.pos_y[i])] for i in range(1, n_units)]
        nq = f.n_quadrants
        unlocked_quadrants = ["NW", "NE", "SW", "SE"][:1] if nq <= 1 else \
            ["NW"] + ["NE", "SW", "SE"][: nq - 1]
        farms_out.append({
            "money": float(f.money),
            "tiles": tiles,
            "farmer": farmer,
            "hands": hands,
            "unlocked_quadrants": unlocked_quadrants,
            "hires_today": int(f.hires_today),
        })

    market_inv = {PRODUCTS[i]: int(state.market.inventory[i]) for i in range(N_PRODUCTS)}
    market_prices = {PRODUCTS[i]: int(state.market.prices[i]) for i in range(N_PRODUCTS)}
    town_shops = [SHOPS[state.shops[i]] for i in range(state.n_shops)]

    f = state.farms[player]
    shed = {ITEMS[i]: int(f.shed[i]) for i in range(N_ITEMS) if f.shed[i] > 0}
    seeds = {CROPS[i]: int(f.seeds[i]) for i in range(N_CROPS) if f.seeds[i] > 0}
    inventories = []
    for u in range(f.n_units):
        inv = {}
        for k in range(f.inv_nkeys[u]):
            item = f.inv_keys[u][k]
            inv[ITEMS[item]] = int(f.inv[u][item])
        inventories.append(inv)

    return {
        "player": player,
        "step": int(state.step),
        "day": int(state.day),
        "hour": int(state.hour),
        "farms": farms_out,
        "market": {"inventory": market_inv, "prices": market_prices},
        "town": {"unlocked_shops": town_shops},
        "private": {"shed": shed, "seeds": seeds, "inventories": inventories},
    }


def _tile_dict(t: Tile):
    k = t.kind
    if k == T_EMPTY:
        return None
    if k == T_LOCKED:
        return "LOCKED"
    if k == T_WEED:
        return {"kind": "WEED"}
    if k == T_PLANT:
        return {
            "kind": "PLANT",
            "crop": CROPS[t.what],
            "planted_day": int(t.planted_day),
            "watered_today": bool(t.watered_today),
            "consecutive_unwatered": int(t.consecutive_dry),
            "yield_units": int(t.yield_units),
            "max_lifespan_step": int(t.max_lifespan_step),
            "fertilized_until_day": int(t.fertilized_until_day),
        }
    return {
        "kind": "COOP" if k == T_COOP else "PASTURE",
        "animal": ITEMS[t.what] if t.has_animal else None,
        "placed_day": int(t.planted_day),
        "yield_units": int(t.yield_units),
        "fed_today": bool(t.fed_today),
        "consecutive_unfed": int(t.consecutive_dry),
        "cared_today": bool(t.cared_today),
        "fertilizer_available": bool(t.fertilizer_available),
        "pending_care_bonus": int(t.pending_care_bonus),
    }


def _enc_unit(a) -> UnitAction:
    if not isinstance(a, list) or not a:
        return UnitAction(op=OPI["PASS"], arg=0, n=1)
    op = OPI.get(a[0], len(OPS))
    arg = ITI.get(a[1], 255) if len(a) >= 2 and isinstance(a[1], str) else 0
    try:
        n = int(a[2]) if len(a) >= 3 else 1
    except (TypeError, ValueError):
        n = 1
    return UnitAction(op=op, arg=arg, n=n)


def _enc_order(o) -> Order:
    if not isinstance(o, list) or not o:
        return Order(op=0, item=0, n=0)
    op = MOPI.get(o[0], 0)
    if op in (1, 2):
        return Order(op=op, item=0, n=1)
    if len(o) < 3:
        return Order(op=0, item=0, n=0)
    return Order(op=op, item=ITI.get(o[1], 255), n=int(o[2]))


def encode_action(action_dict: dict) -> Action:
    a = Action()
    farmer = (action_dict or {}).get("farmer") or ["PASS"]
    hands = (action_dict or {}).get("hands") or []
    units = [farmer] + list(hands)
    a.n_units = min(len(units), MAX_UNITS)
    for i in range(a.n_units):
        a.units[i] = _enc_unit(units[i])
    orders = (action_dict or {}).get("market") or []
    a.n_orders = min(len(orders), 16)
    for i in range(a.n_orders):
        a.orders[i] = _enc_order(orders[i])
    return a


_BUILTIN_MODULE = None


def _builtins():
    global _BUILTIN_MODULE
    if _BUILTIN_MODULE is None:
        from kaggle_environments.envs.kaggriculture import kaggriculture as m
        _BUILTIN_MODULE = m
    return _BUILTIN_MODULE


_agent_cache: dict[str, object] = {}


def load_agent(spec: str):
    """Return a callable agent(obs) -> dict, matching run_local.resolve_agent's"""
    if spec in _agent_cache:
        return _agent_cache[spec]
    if spec in {"random", "pass", "starter"}:
        fn = _builtins().agents[spec]
    else:
        path = None
        for base in (Path.cwd(), REPO_ROOT):
            candidate = base / spec
            if candidate.exists():
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(f"Agent file not found: {spec}")
        mod_name = f"_cpp_bridge_agent_{abs(hash(str(path)))}"
        spec_obj = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[mod_name] = module
        spec_obj.loader.exec_module(module)
        fn = module.agent
    _agent_cache[spec] = fn
    return fn


def play_fast(agent_spec: str, opponent_spec: str, seed: int, steps: int = 720) -> tuple[float, float]:
    """Same contract as pool_winrate.play(): returns (own_money, opp_money)."""
    d = dll()
    cfg = default_config(steps, seed)
    sim = d.sim_new(ctypes.byref(cfg))
    try:
        agents = [load_agent(agent_spec), load_agent(opponent_spec)]
        while not d.sim_done(sim):
            state = d.sim_state(sim)[0]
            obs0 = build_obs(state, 0)
            obs1 = build_obs(state, 1)
            a0 = encode_action(agents[0](obs0))
            a1 = encode_action(agents[1](obs1))
            d.sim_step(sim, ctypes.byref(a0), ctypes.byref(a1))
        return d.sim_reward(sim, 0), d.sim_reward(sim, 1)
    finally:
        d.sim_free(sim)

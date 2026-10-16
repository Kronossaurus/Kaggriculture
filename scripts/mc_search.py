"""Random (Monte Carlo) search over main.py's tunable constants, scored by"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from pool_winrate import play  # noqa: E402
from run_local import resolve_agent  # noqa: E402

DEFAULT_SEARCH_POOL = [
    "reference/x492_nah_i_d_win.py",
    "reference/main_fertilizer_selling_submitted_20260817.py",
]

TRIAL_DIR = REPO_ROOT / "analysis" / "mc_search_trials"
RESULTS_PATH = REPO_ROOT / "analysis" / "mc_search_results.jsonl"

PARAM_SPACE = {
    "HANDS_BASE": ("int", 1, 6),
    "HANDS_PER_QUADRANT": ("int", 1, 5),
    "MAX_HANDS": ("int", 8, 16),
    "LAND_UTILIZATION_GATE": ("float", 0.35, 0.65),
    "PRICE_ATTRACTIVENESS_WEIGHT": ("float", 0.0, 0.3),
    "WHEAT_SELL_THRESHOLD": ("float", 0.45, 2.0),
    "FEED_MAX_PRICE_FRAC": ("float", 1.0, 5.0),
    "SAFETY_RESERVE": ("int", 0, 800),
    "FERTILIZER_MAX_PRICE_FRAC": ("float", 1.05, 2.0),
    "SHED_OVERFLOW_GUARD": ("float", 0.6, 0.95),
    "HANDS_TARGET_Q1": ("int", 3, 9),
    "HARVEST_SAFE_PRIORITY": ("float", 0.51, 1.9),
    "WATER_YIELD_INERT_PRIORITY": ("float", 1.1, 3.9),
    "STRAWBERRY_LATE_SHARE": ("float", 0.35, 0.60),
    "CROP_MIX_RAMP_START_DAY": ("int", 2, 10),
    "CROP_MIX_RAMP_DURATION": ("int", 2, 10),
    "STRAWBERRY_RESERVE_LATE": ("int", 3, 8),
    "WATER_ROUTINE_PRIORITY": ("float", -0.5, 3.0),
    "ANIMAL_TARGET_SHARE_COW": ("float", 0.35, 0.85),
    "ANIMAL_TARGET_SHARE_SHEEP": ("float", 0.15, 0.40),
    "WOOL_SELL_THRESHOLD": ("float", 0.10, 0.65),
}

_CUSTOM_PATCH_PATTERNS: dict[str, re.Pattern] = {}


def sample_params(rng: random.Random, active: dict) -> dict[str, float | int]:
    params: dict[str, float | int] = {}
    for name, (kind, low, high) in active.items():
        params[name] = rng.randint(low, high) if kind == "int" else round(rng.uniform(low, high), 3)
    if "MAX_HANDS" in params and "HANDS_BASE" in params:
        params["MAX_HANDS"] = max(params["MAX_HANDS"], params["HANDS_BASE"])
    return params


def patch_source(src: str, params: dict[str, float | int]) -> str:
    for name, value in params.items():
        if name in _CUSTOM_PATCH_PATTERNS:
            pattern = _CUSTOM_PATCH_PATTERNS[name]
            if not pattern.search(src):
                raise ValueError(f"pattern for {name!r} not found in main.py -- did it get restructured?")
            src = pattern.sub(rf"\g<1>{value}\g<2>", src, count=1)
            continue
        pattern = re.compile(rf"^{name} = [^#\n]+", re.MULTILINE)
        if not pattern.search(src):
            raise ValueError(f"constant {name!r} not found in main.py -- did it get renamed?")
        src = pattern.sub(f"{name} = {value}", src, count=1)
    return src


def make_trial_agent(base_src: str, params: dict[str, float | int], trial_id: int) -> Path:
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    path = TRIAL_DIR / f"trial_{trial_id}.py"
    path.write_text(patch_source(base_src, params), encoding="utf-8")
    return path


def evaluate(agent_path: str, pool: list[str], seeds: int, steps: int, play_fn=play) -> dict:
    wins = losses = ties = 0
    margin_sum = 0.0
    n = 0
    for opponent_path in pool:
        for seed in range(1, seeds + 1):
            own, opp = play_fn(agent_path, opponent_path, seed, steps)
            n += 1
            margin_sum += own - opp
            if own > opp:
                wins += 1
            elif own < opp:
                losses += 1
            else:
                ties += 1
    return {"win_rate": wins / n, "mean_margin": margin_sum / n, "w": wins, "l": losses, "t": ties}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="main.py", help="Base agent to patch constants into (default: main.py)")
    parser.add_argument("--trials", type=int, default=30, help="Number of random combinations to try (default: 30)")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per pool opponent per trial (default: 5)")
    parser.add_argument("--steps", type=int, default=720, help="episodeSteps per episode (default: 720)")
    parser.add_argument(
        "--pool", nargs="+", default=DEFAULT_SEARCH_POOL,
        help=f"Opponents for the search phase (default: {' '.join(DEFAULT_SEARCH_POOL)})",
    )
    parser.add_argument("--rng-seed", type=int, default=0, help="Seed for the search's own sampling (default: 0)")
    parser.add_argument(
        "--params", nargs="+", default=None, choices=list(PARAM_SPACE),
        help="Subset of PARAM_SPACE to vary (default: all of it)",
    )
    parser.add_argument(
        "--engine", choices=["python", "fast"], default="python",
        help="'fast' drives episodes through cpp_sim/'s C++ port via cpp_bridge.py instead of "
             "kaggle_environments -- ~4.7x faster in practice for main.py (see cpp_sim/README.md; "
             "not the notebook's headline number, which is environment-only). Validated bit-exact "
             "against the real environment for deterministic agents (main.py, reference/*.py); "
             "needs cpp_sim/kag_sim.dll built first (see cpp_sim/README.md).",
    )
    args = parser.parse_args()

    base_path = resolve_agent(args.base)
    base_src = Path(base_path).read_text(encoding="utf-8")
    pool = [resolve_agent(spec) for spec in args.pool]
    rng = random.Random(args.rng_seed)
    active = {k: PARAM_SPACE[k] for k in (args.params or PARAM_SPACE)}

    play_fn = play
    if args.engine == "fast":
        from cpp_bridge import play_fast
        play_fn = play_fast

    print(f"=== MC search: {args.trials} trials, {args.seeds} seeds x {len(pool)} opponents each, "
          f"varying {list(active)}, engine={args.engine} ===\n")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    search_start = time.monotonic()
    with RESULTS_PATH.open("w", encoding="utf-8") as out:
        for trial_id in range(1, args.trials + 1):
            trial_start = time.monotonic()
            params = sample_params(rng, active)
            trial_path = make_trial_agent(base_src, params, trial_id)
            score = evaluate(str(trial_path), pool, args.seeds, args.steps, play_fn=play_fn)
            trial_elapsed = time.monotonic() - trial_start
            row = {"trial": trial_id, "params": params, "elapsed_s": round(trial_elapsed, 1), **score}
            results.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            elapsed_total = time.monotonic() - search_start
            eta = elapsed_total / trial_id * (args.trials - trial_id)
            print(
                f"[{trial_id:3d}/{args.trials}] win_rate={score['win_rate']:.2f} "
                f"margin=${score['mean_margin']:>9,.0f}  {trial_elapsed:5.1f}s/trial  "
                f"elapsed={elapsed_total / 60:5.1f}m  eta={eta / 60:5.1f}m  {params}"
            )

    total_elapsed = time.monotonic() - search_start
    results.sort(key=lambda r: (r["win_rate"], r["mean_margin"]), reverse=True)
    print(f"\n=== Top 5 (full results in {RESULTS_PATH.relative_to(REPO_ROOT)}) === "
          f"total search time: {total_elapsed / 60:.1f}m ({total_elapsed / args.trials:.1f}s/trial avg)")
    for row in results[:5]:
        print(f"win_rate={row['win_rate']:.2f}  margin=${row['mean_margin']:>9,.0f}  "
              f"{row['elapsed_s']:5.1f}s  {row['params']}")
    print(
        "\nThese are search-phase results (small pool, few seeds) -- do NOT trust a "
        "single pool_winrate.py run to confirm the winner. A 10-seed comparison has a "
        "real noise floor of about +/-2 seeds even with identical code (see the "
        "pool-winrate-noise-floor memory note / main.py's own docstring); run the "
        "winner and the current baseline back-to-back in the same process at ~30 "
        "seeds each before deciding to keep anything found here."
    )


if __name__ == "__main__":
    main()

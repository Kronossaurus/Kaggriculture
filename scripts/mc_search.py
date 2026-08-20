"""Random (Monte Carlo) search over main.py's tunable constants, scored by
the same pool-based win rate `pool_winrate.py` already uses -- see that
script's own docstring for why win rate against a field, not margin
against one baseline, is the metric that matters here.

Every constant in this project has so far been tuned one at a time by
hand: pick a value, run `pool_winrate.py`, eyeball win rate, keep or
revert (see main.py's module docstring for the whole history of that
process). That's slow and can't find interactions between constants --
e.g. the neglected-tile-urgency fix only paid off once paired with the
hire-formula change, something a one-at-a-time search would likely have
missed. This script instead samples random combinations of several
constants at once, patches a temp copy of main.py for each, and scores it
against a small, cheap pool -- a search phase, not a final verdict. The
winning candidate(s) still need a full `pool_winrate.py` run (default
pool, default seed count) before being trusted, exactly like every other
change in this project; this script does not skip that gate, it just
narrows down what's worth spending a full run on.

Deliberately NOT here: MCTS or any other per-turn/in-episode search --
the 720-turn horizon and per-turn branching factor (farmer + hands +
up to 10 market orders) make that a poor fit for this problem, and the
existing priority-based heuristics in main.py already handle per-turn
decisions well. This script only tunes the numeric constants layered on
top of that heuristic, offline, between games.

Usage:
    uv run scripts/mc_search.py                       # 30 trials, 5 seeds/opponent, all params
    uv run scripts/mc_search.py --trials 50 --seeds 8
    uv run scripts/mc_search.py --pool reference/x492_nah_i_d_win.py
    uv run scripts/mc_search.py --rng-seed 1           # reproduce a previous search
    uv run scripts/mc_search.py --params WHEAT_SELL_THRESHOLD FEED_MAX_PRICE_FRAC
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from pool_winrate import play  # noqa: E402
from run_local import resolve_agent  # noqa: E402

# Cheap-but-discriminating default: a strong external agent (x492) and a
# frozen submission snapshot of our own past self -- the two pool entries
# most likely to actually separate a good combination from a mediocre one.
# See pool_winrate.py's docstring for why random/starter were dropped from
# its own default pool (no longer discriminating); the same reasoning
# applies here, doubly so since this pool is walked `--trials` times over.
DEFAULT_SEARCH_POOL = [
    "reference/x492_nah_i_d_win.py",
    "reference/main_fertilizer_selling_submitted_20260817.py",
]

TRIAL_DIR = REPO_ROOT / "analysis" / "mc_search_trials"  # analysis/ is gitignored
RESULTS_PATH = REPO_ROOT / "analysis" / "mc_search_results.jsonl"

# name -> ("int" | "float", low, high). Kept intentionally small: this
# project's own history (dedicated-hand allocation, animal structure
# buffer) shows more knobs doesn't reliably mean a better search, just a
# slower one. Animal-*ownership* constants are deliberately left out until
# the token-stranding bug (see main.py's `_build_animal_orders` docstring)
# is actually resolved -- tuning around a known-buggy subsystem risks the
# search "fixing" the bug's symptom via unrelated constants. WHEAT_SELL_
# THRESHOLD/FEED_MAX_PRICE_FRAC are feed-*pricing* knobs, not ownership,
# so they're fair game -- see main.py's Sell/rebuy round-trip docstring
# note: top-ladder replays never buy WHEAT above ~2.1x base or sell below
# ~1.1x, both far tighter than this agent's current 5.0x / 0.45x.
PARAM_SPACE = {
    "HANDS_BASE": ("int", 1, 6),
    "HANDS_PER_QUADRANT": ("int", 1, 5),
    "MAX_HANDS": ("int", 8, 16),
    "LAND_UTILIZATION_GATE": ("float", 0.35, 0.65),
    "PRICE_ATTRACTIVENESS_WEIGHT": ("float", 0.0, 0.3),
    "WHEAT_SELL_THRESHOLD": ("float", 0.45, 2.0),
    "FEED_MAX_PRICE_FRAC": ("float", 1.0, 5.0),
}

# WHEAT_SELL_THRESHOLD isn't a top-level `NAME = value` constant -- it's
# nested in SELL_PARAMS["WHEAT"]["threshold"]. Patched with its own regex
# instead of forcing every constant through the generic top-level pattern.
_CUSTOM_PATCH_PATTERNS = {
    "WHEAT_SELL_THRESHOLD": re.compile(r'("WHEAT":\s*\{"threshold":\s*)[^}]+(\})'),
}


def sample_params(rng: random.Random, active: dict) -> dict[str, float | int]:
    params: dict[str, float | int] = {}
    for name, (kind, low, high) in active.items():
        params[name] = rng.randint(low, high) if kind == "int" else round(rng.uniform(low, high), 3)
    # A cap below the base hire count is nonsensical -- clamp instead of
    # discarding the whole sample, cheaper than rejection sampling.
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
    with RESULTS_PATH.open("w", encoding="utf-8") as out:
        for trial_id in range(1, args.trials + 1):
            params = sample_params(rng, active)
            trial_path = make_trial_agent(base_src, params, trial_id)
            score = evaluate(str(trial_path), pool, args.seeds, args.steps, play_fn=play_fn)
            row = {"trial": trial_id, "params": params, **score}
            results.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(
                f"[{trial_id:3d}/{args.trials}] win_rate={score['win_rate']:.2f} "
                f"margin=${score['mean_margin']:>9,.0f}  {params}"
            )

    results.sort(key=lambda r: (r["win_rate"], r["mean_margin"]), reverse=True)
    print(f"\n=== Top 5 (full results in {RESULTS_PATH.relative_to(REPO_ROOT)}) ===")
    for row in results[:5]:
        print(f"win_rate={row['win_rate']:.2f}  margin=${row['mean_margin']:>9,.0f}  {row['params']}")
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

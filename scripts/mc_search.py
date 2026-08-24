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
import time
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
    # Added 2026-08-20 after the first 7-param sweep found nothing that beat
    # baseline on the full pool (see the next-steps-2026-08-19 memory note's
    # item 1) -- these three are, like the pair above, cash-timing/threshold
    # knobs rather than land/crop/animal ownership, so the same "fair game"
    # reasoning applies. Bounds centered on main.py's current value with
    # headroom both directions, not blind ranges.
    "SAFETY_RESERVE": ("int", 0, 800),  # main.py: 300
    "FERTILIZER_MAX_PRICE_FRAC": ("float", 1.05, 2.0),  # main.py: 1.2
    "SHED_OVERFLOW_GUARD": ("float", 0.6, 0.95),  # main.py: 0.85
    # Added 2026-08-22 after three hand-picked single-value attempts each regressed (see
    # hands-target-q1-reduction-attempt / water-yield-inert-deprioritization-tradeoff /
    # harvest-safe-deprioritization-tradeoff memory notes) -- exactly the "can't find
    # interactions between constants" gap this script's own docstring describes, since all
    # three plausibly interact (crew size changes how much contention the two priority tiers
    # actually see). HARVEST_SAFE_PRIORITY's range deliberately starts just above 0.5 (at-risk
    # water's own priority), not at 0.5 or below -- the first attempt tied them exactly and
    # regressed by turning a decisive ordering into an ambiguous distance-only contention.
    "HANDS_TARGET_Q1": ("int", 3, 9),  # main.py: 8 (== the un-overridden formula's own value)
    "HARVEST_SAFE_PRIORITY": ("float", 0.51, 1.9),  # main.py: 0.75
    "WATER_YIELD_INERT_PRIORITY": ("float", 1.1, 3.9),  # main.py: 3
    # Added 2026-08-22 after the time-varying crop-mix ramp (see strawberry-shift-throughput-gap
    # memory note's final update) hand-picked at STRAWBERRY_LATE_SHARE=0.45/RAMP_START_DAY=10/
    # RAMP_DURATION=10/RESERVE_LATE=6 already cleared +9.5pp on the full pool (52.9%->62.4%,
    # every opponent improved or tied, none regressed) with zero tuning -- searching around
    # these hand-picked values to see if there's more room, not because they looked wrong.
    "STRAWBERRY_LATE_SHARE": ("float", 0.35, 0.60),  # main.py: 0.45
    "CROP_MIX_RAMP_START_DAY": ("int", 2, 10),  # main.py: 10 -- narrowed after a first pass
    # showed low start days (3-7) and short durations (3-8) consistently beating the initial
    # hand-picked 10/10 by a wide margin (up to 0.82 vs ~0.5-0.6 win rate on the search pool)
    "CROP_MIX_RAMP_DURATION": ("int", 2, 10),  # main.py: 10
    "STRAWBERRY_RESERVE_LATE": ("int", 3, 8),  # main.py: 6
    # Added 2026-08-22: a top-ladder replay comparison found this agent's real end-of-day
    # watering coverage averaging 71.1% (days 10-27) vs a top player's 86.1% -- a real ~15pp
    # gap, unlike DIG rate or MOVE fraction (both comparable). Three targeted fixes elsewhere
    # (crew size, fertilizer batching/delivery gating) all failed to move it -- testing whether
    # routine watering's own priority is simply not urgent enough relative to what it competes
    # against (animal upkeep, weed-clearing, routine planting). Range extends below 0 (more
    # urgent than any at-risk-harvest/at-risk-water tier) and up to 3 (roughly WATER_YIELD_
    # INERT_PRIORITY's own range) to let the search find either direction.
    "WATER_ROUTINE_PRIORITY": ("float", -0.5, 3.0),  # main.py: 1
    # Added 2026-08-23, user-requested: ANIMAL_TARGET_SHARE {"COW": 0.60, "SHEEP": 0.40} was
    # hand-picked, never searched. Now paired with SHEEP_MAX=6 (a hard ceiling added the same
    # session after a real replay showed 7 live SHEEP tanking WOOL's own price), so the
    # proportional share and the hard cap now interact -- worth re-tuning together rather than
    # trusting the original hand-picked split still holds. GOOSE stays fixed at 0 (5/5 real
    # top-ladder replays never build one, see that constant's own comment). Ranges centered on
    # the current values with headroom both directions, not required to sum to 1 (the ranking
    # logic only needs relative gaps, doesn't strictly require it). That first joint search
    # validated COW=0.629/SHEEP=0.456 (main.py, committed) -- annotations below reflect that.
    "ANIMAL_TARGET_SHARE_COW": ("float", 0.35, 0.85),  # main.py: 0.629
    # SHEEP's upper bound was 0.65 in the search above -- narrowed here. _AnimalPlanner.
    # choose_structure() (main.py:1062) balances live COW/SHEEP toward these shares out of a
    # land-share total of ~14 (3 quadrants x 25 tiles x ANIMAL_LAND_SHARE=0.18); at the COW
    # value this now centers on (0.629), any SHEEP share above ~0.40 already implies a live
    # SHEEP count past SHEEP_MAX=6 (main.py:1056 _at_hard_cap), which silently truncates it --
    # the hard cap and the proportional share then produce the *identical* final SHEEP count,
    # so trials sampled above ~0.40 mostly differ only in COW/SHEEP build *order*, not outcome,
    # wasting search budget on a behavioral plateau. Capped at 0.40 so the range stays
    # informative about the actual (uncapped) split; revisit together with SHEEP_MAX itself
    # (not currently in this PARAM_SPACE) if a future search wants to explore the cap jointly.
    "ANIMAL_TARGET_SHARE_SHEEP": ("float", 0.15, 0.40),  # main.py: 0.456 (already above the
                                                          # new ceiling -- kept as the baseline
                                                          # value patched from, just outside the
                                                          # searched range now)
    # Added 2026-08-23, user-requested follow-up to SHEEP_MAX: 5 downloaded top-ladder replays
    # cross-referencing own SHEEP count, opponent SHEEP count, and WOOL shed stockpile against
    # WOOL price found count (own or combined) does NOT predict a price crash -- a game with only
    # 5-7 combined SHEEP crashed as hard as one with 29. What differed: the top player with the
    # healthiest/most stable WOOL price (Ryo, #1 ladder) held WOOL shed stock at 0 the entire
    # game -- sold every turn regardless of price -- while the two worst games showed WOOL
    # stockpiling to 16-24 units precisely while price stayed pinned near $1. Our own
    # SELL_PARAMS["WOOL"]["threshold"]=0.60 makes this agent do the opposite of the pattern that
    # correlated with recovery (hold out for a better price instead of dumping). Range extends
    # well below the current value to test that hypothesis, and up to it to let the search find
    # the null result too.
    "WOOL_SELL_THRESHOLD": ("float", 0.10, 0.65),  # main.py: 0.60
}

# Was needed for WHEAT_SELL_THRESHOLD/WOOL_SELL_THRESHOLD/ANIMAL_TARGET_SHARE_COW/_SHEEP, which
# used to live nested inside dict literals (SELL_PARAMS["WHEAT"]["threshold"], etc.) with no
# top-level `NAME = value` line the generic pattern below could match. main.py's 2026-08-23
# constants simplification (user-requested, "para facilitar futuros mc") promoted every one of
# those to a real top-level constant that the dict is now assembled *from* -- see main.py's
# CROP_TARGET_SHARE comment for the full rationale. Nothing needs a custom pattern any more, but
# the mechanism stays available (empty) in case some future constant genuinely can't be promoted
# the same way.
_CUSTOM_PATCH_PATTERNS: dict[str, re.Pattern] = {}


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

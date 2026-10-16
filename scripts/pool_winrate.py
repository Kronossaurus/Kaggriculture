"""Evaluate an agent's win rate against a field of opponents ("the pool"),"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kaggle_environments import make

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_local import resolve_agent  # noqa: E402

DEFAULT_POOL = [
    "reference/x492_nah_i_d_win.py",
    "reference/main_land_gate_60pct.py",
    "reference/main_fertilizer_selling_submitted_20260817.py",
    "reference/main_plant_priority_submitted_20260818.py",
    "reference/main_price_reactive_seed_reserve_submitted_20260819.py",
    "reference/main_harvest_timing_submitted_20260821.py",
    "reference/main_mc_tuned_priority_submitted_20260822.py",
    "reference/main_strawberry_ramp_submitted_20260822.py",
    "reference/main_water_priority_submitted_20260823.py",
    "reference/main_wheat_feed_futile_submitted_20260824.py",
]


def play(agent_path: str, opponent_path: str, seed: int, steps: int) -> tuple[float, float]:
    """Returns (agent_money, opponent_money) for one episode, agent as P1."""
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    env.run([agent_path, opponent_path])
    final = env.steps[-1]
    return final[0].reward, final[1].reward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="main.py", help="Agent under evaluation (default: main.py)")
    parser.add_argument(
        "--pool", nargs="+", default=DEFAULT_POOL,
        help=f"Opponents to face (default: {' '.join(DEFAULT_POOL)})",
    )
    parser.add_argument("--seeds", type=int, default=20, help="Episodes per opponent, seeds 1..N (default: 20)")
    parser.add_argument("--steps", type=int, default=720, help="episodeSteps per episode (default: 720)")
    parser.add_argument(
        "--engine", choices=["python", "fast"], default="python",
        help="'fast' drives episodes through cpp_sim/'s C++ port via cpp_bridge.py instead of "
             "kaggle_environments (see cpp_sim/README.md); needs cpp_sim/kag_sim.dll built first.",
    )
    args = parser.parse_args()

    play_fn = play
    if args.engine == "fast":
        from cpp_bridge import play_fast
        play_fn = play_fast

    agent_path = resolve_agent(args.agent)
    pool = [(spec, resolve_agent(spec)) for spec in args.pool]

    print(f"=== {args.agent} vs pool ({args.seeds} seeds each, {args.steps} steps, engine={args.engine}) ===\n")

    total_w = total_l = total_t = 0
    for label, opponent_path in pool:
        w = l = t = 0
        own_sum = opp_sum = 0.0
        for seed in range(1, args.seeds + 1):
            own, opp = play_fn(agent_path, opponent_path, seed, args.steps)
            own_sum += own
            opp_sum += opp
            if own > opp:
                w += 1
            elif own < opp:
                l += 1
            else:
                t += 1
        total_w, total_l, total_t = total_w + w, total_l + l, total_t + t
        n = args.seeds
        print(
            f"{label:32s}  {w:2d}W-{l:2d}L-{t:2d}T  ({w / n * 100:5.1f}%)   "
            f"mean ${own_sum / n:>8,.0f}  vs  ${opp_sum / n:>8,.0f}"
        )

    n_total = total_w + total_l + total_t
    print(f"\n{'POOL TOTAL':32s}  {total_w:2d}W-{total_l:2d}L-{total_t:2d}T  ({total_w / n_total * 100:5.1f}%)")


if __name__ == "__main__":
    main()

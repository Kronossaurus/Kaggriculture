"""Evaluate an agent's win rate against a field of opponents ("the pool"),
scored and reported per-opponent as well as aggregated -- see rl/README.md's
Guiding Principles: "rank by win rate against a field, never margin against
one baseline." Every prior evaluation in this project (`run_local.py`,
`rl/analyze_actions.py`) only ever compared against a single fixed opponent;
this is the missing piece.

Default pool is every scripted reference agent in this repo worth
tracking (`checkpoints/v0/main.py`, the original one-tile wheat loop;
`reference/melon_maxxer.py`, Kaggle's own official single-crop example;
`reference/x492_nah_i_d_win.py`, a strong external agent pulled from Kaggle
Kernels for comparison only -- not ours, never submit it; and
`reference/main_land_gate_60pct.py`, a frozen ablation of main.py itself --
see its own docstring for the land-utilization-gate trade-off it tracks;
`reference/main_fertilizer_selling_submitted_20260817.py`, a frozen
snapshot of the exact main.py submitted to Kaggle on that date -- see
main.py's own docstring's Selling fertilizer note; and
`reference/main_plant_priority_submitted_20260818.py`, a frozen snapshot
of the exact main.py submitted to Kaggle on that date -- see main.py's
own docstring's Planting priority under the gate note. Standing rule:
every Kaggle submission gets snapshotted here so future runs automatically
check new work against what was actually shipped, not just hand-picked
ablations.
`kaggle_environments`' own `random`/`starter` built-ins were dropped from
the default (2026-08-18): every main.py revision this session has beaten
both 10/10 seeds by a wide margin, so they no longer discriminate between
changes -- pass `--pool random starter` explicitly if a fresh sanity check
against them is ever needed again.
No RL checkpoint is included -- see rl/README.md, none survive the current
action space.

Usage:
    uv run scripts/pool_winrate.py                              # main.py vs the default pool, 20 seeds each
    uv run scripts/pool_winrate.py --agent reference/melon_maxxer.py
    uv run scripts/pool_winrate.py --pool random starter --seeds 10
    uv run scripts/pool_winrate.py --agent checkpoints/v0/main.py --pool reference/melon_maxxer.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kaggle_environments import make

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_local import resolve_agent  # noqa: E402

DEFAULT_POOL = [
    "checkpoints/v0/main.py",
    "reference/melon_maxxer.py",
    "reference/x492_nah_i_d_win.py",
    "reference/main_land_gate_60pct.py",
    "reference/main_fertilizer_selling_submitted_20260817.py",
    "reference/main_plant_priority_submitted_20260818.py",
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
    args = parser.parse_args()

    agent_path = resolve_agent(args.agent)
    pool = [(spec, resolve_agent(spec)) for spec in args.pool]

    print(f"=== {args.agent} vs pool ({args.seeds} seeds each, {args.steps} steps) ===\n")

    total_w = total_l = total_t = 0
    for label, opponent_path in pool:
        w = l = t = 0
        own_sum = opp_sum = 0.0
        for seed in range(1, args.seeds + 1):
            own, opp = play(agent_path, opponent_path, seed, args.steps)
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

"""No-op audit: instrument the engine, not the agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.kaggriculture import kaggriculture as K  # noqa: E402

from rl.checkpoint import resolve_agent  # noqa: E402


def noop_audit(agents, seed: int, episode_steps: int) -> dict[str, list[int]]:
    """Run one live game, hooking kaggriculture's `_apply_unit_action` to count"""
    original = K._apply_unit_action
    stats: dict[str, list[int]] = {}

    def hooked(farm, private, idx, action, *args, **kwargs):
        before = (json.dumps(farm, sort_keys=True), json.dumps(private, sort_keys=True))
        result = original(farm, private, idx, action, *args, **kwargs)
        after = (json.dumps(farm, sort_keys=True), json.dumps(private, sort_keys=True))
        name = action[0] if isinstance(action, (list, tuple)) and action else str(action)
        row = stats.setdefault(name, [0, 0])
        row[0] += 1
        row[1] += (before == after)
        return result

    K._apply_unit_action = hooked
    try:
        make("kaggriculture", configuration={"seed": seed, "episodeSteps": episode_steps}, debug=False).run(agents)
    finally:
        K._apply_unit_action = original
    return stats


def report(label: str, stats: dict[str, list[int]]) -> None:
    real = {k: v for k, v in stats.items() if k != "PASS"}
    total = sum(v[0] for v in real.values())
    wasted = sum(v[1] for v in real.values())
    print(f"{label:30s} {total:5d} real actions   {wasted:5d} wasted   "
          f"{100 * wasted / max(1, total):5.1f}%")
    for name, (n, bad) in sorted(real.items(), key=lambda kv: -kv[1][1]):
        if bad:
            print(f"    {name:16s} {n:5d} issued  {bad:5d} no-op  {100 * bad / n:5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent", help="Built-in name (starter/random/pass), SB3 checkpoint .zip, or a main.py-style agent file")
    parser.add_argument("--opponent", default=None,
                         help="Same spec options as `agent`; defaults to auditing `agent` against "
                              "itself (both seats), matching how this technique is normally used")
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--steps", type=int, default=720, help="Episode length (episode_steps)")
    parser.add_argument("--stochastic", action="store_true",
                         help="Sample from the audited checkpoint's action distribution instead of "
                              "argmax -- audits the same distribution rl/analyze_actions.py's "
                              "--stochastic reports on, not just the (often near-idle) greedy policy. "
                              "Only affects `agent` if it's a checkpoint .zip; the opponent, if a "
                              "separate checkpoint, stays deterministic.")
    args = parser.parse_args()

    agent = resolve_agent(args.agent, deterministic=not args.stochastic)
    opponent = resolve_agent(args.opponent) if args.opponent else agent
    label = f"{args.agent} vs {args.opponent or args.agent} (seed={args.seed})"

    print(f"Auditing {label} ...")
    stats = noop_audit([agent, opponent], args.seed, args.steps)
    report(label, stats)


if __name__ == "__main__":
    main()

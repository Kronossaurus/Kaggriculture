"""Run a local Kaggriculture episode and optionally dump a replay JSON.

Usage:
    uv run scripts/run_local.py                          # main.py vs random
    uv run scripts/run_local.py --p2 starter              # main.py vs starter baseline
    uv run scripts/run_local.py --p1 random --p2 random   # baseline vs baseline
    uv run scripts/run_local.py --steps 200 --replay out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_environments import make

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_agent(spec: str) -> str:
    """Pass through built-in names ("random", "pass", "starter") or a .py path.

    Path is checked relative to the current working directory first, then
    relative to the repo root, so this works regardless of where the script
    is invoked from (repo root, scripts/, an IDE run config, etc).
    """
    if spec in {"random", "pass", "starter"}:
        return spec
    for base in (Path.cwd(), REPO_ROOT):
        path = base / spec
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"Agent file not found: {spec} (looked in cwd={Path.cwd()} and repo root={REPO_ROOT})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", default="main.py", help="Agent 1 (default: main.py)")
    parser.add_argument("--p2", default="random", help="Agent 2 (default: random)")
    parser.add_argument("--steps", type=int, default=720, help="episodeSteps (default: 720, the full season)")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic episode seed")
    parser.add_argument("--replay", default=None, help="Path to dump replay JSON, e.g. replays/last.json")
    parser.add_argument("--debug", action="store_true", default=True)
    args = parser.parse_args()

    config = {"episodeSteps": args.steps}
    if args.seed is not None:
        config["seed"] = args.seed

    env = make("kaggriculture", configuration=config, debug=args.debug)
    env.run([resolve_agent(args.p1), resolve_agent(args.p2)])

    final = env.steps[-1]
    print(f"\n=== Result ({args.p1} vs {args.p2}, {args.steps} steps) ===")
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")

    if args.replay:
        out = Path(args.replay)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(env.toJSON()))
        print(f"Replay written to {out}")


if __name__ == "__main__":
    main()

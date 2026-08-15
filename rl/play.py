"""Run a trained RL checkpoint against an opponent locally -- the RL equivalent
of scripts/run_local.py, but for an SB3 checkpoint (.zip) instead of a
main.py-style agent file.

Usage:
    uv run rl/play.py checkpoints/v1/best_model.zip
    uv run rl/play.py checkpoints/v1/best_model.zip --opponent random --steps 200
    uv run rl/play.py checkpoints/v1/best_model.zip --opponent checkpoints/v0/main.py
    uv run rl/play.py checkpoints/v1/best_model.zip --replay replays/v1_vs_starter.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_environments import make  # noqa: E402

from rl.checkpoint import load_agent, resolve_opponent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", default="checkpoints/v1/best_model.zip", help="Path to an SB3 checkpoint .zip (e.g. checkpoints/v1/best_model.zip)")
    parser.add_argument("--opponent", default="starter", help="random|pass|starter or a main.py-style agent file")
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--replay", default=None, help="Path to dump replay JSON, e.g. replays/last.json")
    args = parser.parse_args()

    agent = load_agent(args.checkpoint)
    opponent = resolve_opponent(args.opponent)

    config = {"episodeSteps": args.steps}
    if args.seed is not None:
        config["seed"] = args.seed

    env = make("kaggriculture", configuration=config, debug=True)
    env.run([agent, opponent])

    final = env.steps[-1]
    print(f"\n=== Result ({args.checkpoint} vs {args.opponent}, {args.steps} steps) ===")
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")

    if args.replay:
        out = Path(args.replay)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(env.toJSON()))
        print(f"Replay written to {out}")


if __name__ == "__main__":
    main()

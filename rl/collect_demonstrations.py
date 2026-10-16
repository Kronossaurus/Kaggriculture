"""Record (obs, action) demonstration pairs from a scripted Kaggriculture agent"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.kaggriculture.kaggriculture import starter_agent  # noqa: E402

from rl.env import encode_action  # noqa: E402
from rl.features import encode_obs  # noqa: E402


def collect_episode(opponent: str, episode_steps: int, seed: int):
    """Play one episode with starter_agent driving player 0 (recorded) and"""
    kaggle_env = make("kaggriculture", configuration={"episodeSteps": episode_steps, "seed": seed})
    trainer = kaggle_env.train([None, opponent])
    raw_obs = trainer.reset()
    obs_list, action_list = [], []
    done = False
    while not done:
        action_dict = starter_agent(raw_obs)
        obs_list.append(encode_obs(raw_obs))
        action_list.append(encode_action(action_dict, raw_obs))
        raw_obs, _reward, done, _info = trainer.step(action_dict)
    return obs_list, action_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--opponent", default="starter", help="Opponent driving player 1 -- random|pass|starter (default: starter, matches rl/train.py's default)")
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", default="rl/demos/starter_v1.npz")
    args = parser.parse_args()

    all_obs, all_actions = [], []
    t0 = time.time()
    for i in range(args.episodes):
        obs_list, action_list = collect_episode(args.opponent, args.episode_steps, args.seed_start + i)
        all_obs.extend(obs_list)
        all_actions.extend(action_list)
        print(f"episode {i + 1}/{args.episodes}: {len(obs_list)} steps ({time.time() - t0:.1f}s elapsed)")

    own_grid = np.stack([o["own_grid"] for o in all_obs])
    opp_grid = np.stack([o["opp_grid"] for o in all_obs])
    vector = np.stack([o["vector"] for o in all_obs])
    actions = np.stack(all_actions)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, own_grid=own_grid, opp_grid=opp_grid, vector=vector, actions=actions)
    print(f"Saved {len(actions)} demonstration steps from {args.episodes} episodes to {out_path}")


if __name__ == "__main__":
    main()

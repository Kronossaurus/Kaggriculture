"""Behavioral-cloning pretraining: warm-start a PPO policy by supervised"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from rl.env import KaggricultureEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demos", default="rl/demos/starter_v1.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", default="checkpoints/bc_v1")
    parser.add_argument("--device", default="cpu", help="cpu is the right default here too -- see rl/train.py's perf notes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    demos_path = Path(args.demos)
    if not demos_path.is_absolute():
        demos_path = REPO_ROOT / demos_path
    data = np.load(demos_path)
    own_grid, opp_grid, vector, actions = data["own_grid"], data["opp_grid"], data["vector"], data["actions"]
    n = len(actions)
    print(f"Loaded {n} demonstration steps from {demos_path}")

    dummy_env = DummyVecEnv([lambda: KaggricultureEnv(opponent="starter")])
    model = PPO("MultiInputPolicy", dummy_env, verbose=0, device=args.device, seed=args.seed)
    policy = model.policy
    optimizer = policy.optimizer
    for g in optimizer.param_groups:
        g["lr"] = args.lr

    device = policy.device
    actions_t_all = torch.as_tensor(actions, dtype=torch.long, device=device)

    policy.set_training_mode(True)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            obs_batch = {
                "own_grid": torch.as_tensor(own_grid[idx], dtype=torch.float32, device=device),
                "opp_grid": torch.as_tensor(opp_grid[idx], dtype=torch.float32, device=device),
                "vector": torch.as_tensor(vector[idx], dtype=torch.float32, device=device),
            }
            actions_batch = actions_t_all[idx]
            _values, log_prob, _entropy = policy.evaluate_actions(obs_batch, actions_batch)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch + 1}/{args.epochs}  mean NLL loss = {total_loss / n_batches:.4f}")
    policy.set_training_mode(False)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.zip")
    print(f"Saved BC-pretrained policy to {out_dir / 'model.zip'}")


if __name__ == "__main__":
    main()

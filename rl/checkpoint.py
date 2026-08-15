"""Load a saved SB3 checkpoint as a plain `obs -> action dict` callable, so it can
be used anywhere kaggle_environments expects an agent: `env.run([...])`,
`env.train([None, ...])`, `kaggle_environments`' renderer, etc.

Shared by rl/train.py (self-play opponents) and rl/play.py (local evaluation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO

from .env import decode_action
from .features import encode_obs

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_opponent(spec: str) -> str:
    """Resolve an --opponent CLI spec into what kaggle_environments' env.run()/
    env.train() accepts: a built-in name as-is, or an agent .py path checked
    relative to cwd then repo root. Shared by rl/play.py and rl/analyze_actions.py
    (rl/train.py has its own make_opponent(), which additionally supports loading
    a checkpoint .zip for self-play)."""
    if spec in {"random", "pass", "starter"}:
        return spec
    for base in (Path.cwd(), REPO_ROOT):
        path = base / spec
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Opponent agent not found: {spec}")


def load_agent(path: str | Path) -> Callable[[dict], dict]:
    """Load a PPO checkpoint and wrap it as an agent function.

    `deterministic=True` (argmax over the policy's action distribution) since
    this is for evaluation/opponent play, not exploration.
    """
    model = PPO.load(path)

    def agent(obs) -> dict:
        action, _ = model.predict(encode_obs(obs), deterministic=True)
        return decode_action(action, obs)

    return agent

"""Load a saved SB3 checkpoint as a plain `obs -> action dict` callable, so it can"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from stable_baselines3 import PPO

from .env import decode_action
from .features import encode_obs

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_opponent(spec: str) -> str:
    """Resolve an --opponent CLI spec into what kaggle_environments' env.run()/"""
    if spec in {"random", "pass", "starter"}:
        return spec
    for base in (Path.cwd(), REPO_ROOT):
        path = base / spec
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Opponent agent not found: {spec}")


def resolve_agent(spec: str, deterministic: bool = True):
    """Resolve an --opponent/agent CLI spec into whatever kaggle_environments'"""
    if spec in {"random", "pass", "starter"}:
        return spec
    path = Path(spec)
    if not path.is_absolute():
        path = REPO_ROOT / spec
    if path.suffix == ".zip":
        return load_agent(path, deterministic=deterministic)
    return resolve_opponent(spec)


def load_agent(path: str | Path, deterministic: bool = True) -> Callable[[dict], dict]:
    """Load a PPO checkpoint and wrap it as an agent function."""
    model = PPO.load(path)

    def agent(obs) -> dict:
        action, _ = model.predict(encode_obs(obs), deterministic=deterministic)
        return decode_action(action, obs)

    return agent

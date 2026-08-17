"""Load a saved SB3 checkpoint as a plain `obs -> action dict` callable, so it can
be used anywhere kaggle_environments expects an agent: `env.run([...])`,
`env.train([None, ...])`, `kaggle_environments`' renderer, etc.

Shared by rl/train.py (self-play opponents) and rl/play.py (local evaluation).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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


def resolve_agent(spec: str, deterministic: bool = True):
    """Resolve an --opponent/agent CLI spec into whatever kaggle_environments'
    env.run()/env.train() accepts: a built-in name as-is, an SB3 checkpoint .zip
    (loaded via load_agent), or an agent .py path (resolve_opponent's cwd-then-
    repo-root rule). Shared by train.py (self-play opponents) and
    noop_audit.py (both the audited agent and its opponent) -- unlike
    resolve_opponent, this one also handles the .zip case, since both of those
    callers need to point either side at a trained checkpoint, not just a
    scripted/built-in opponent."""
    if spec in {"random", "pass", "starter"}:
        return spec
    path = Path(spec)
    if not path.is_absolute():
        path = REPO_ROOT / spec
    if path.suffix == ".zip":
        return load_agent(path, deterministic=deterministic)
    return resolve_opponent(spec)  # .py agent: cwd then repo root


def load_agent(path: str | Path, deterministic: bool = True) -> Callable[[dict], dict]:
    """Load a PPO checkpoint and wrap it as an agent function.

    `deterministic=True` (argmax over the policy's action distribution) by
    default since this is normally for evaluation/opponent play, not
    exploration -- pass `deterministic=False` to sample instead (e.g.
    rl/noop_audit.py's --stochastic, to audit the same distribution
    rl/analyze_actions.py's --stochastic reports on).
    """
    model = PPO.load(path)

    def agent(obs) -> dict:
        action, _ = model.predict(encode_obs(obs), deterministic=deterministic)
        return decode_action(action, obs)

    return agent

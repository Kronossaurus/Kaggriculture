"""Train a Kaggriculture PPO agent (v1 action scope -- see rl/README.md)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize  # noqa: E402

from rl.checkpoint import resolve_agent  # noqa: E402
from rl.env import KaggricultureEnv  # noqa: E402

GAMMA = 0.99


def make_env_fn(opponent, episode_steps: int, seed: int | None):
    """Factory for a single Monitor-wrapped KaggricultureEnv, picklable so"""

    def _init():
        return Monitor(KaggricultureEnv(opponent=opponent, episode_steps=episode_steps, seed=seed))

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opponent", default="starter", help="random|pass|starter, a .py agent, or a checkpoint .zip")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--out", default="checkpoints/v1", help="Where to save model.zip / best_model.zip")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Timesteps between evals (not vec-env steps)")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel envs via SubprocVecEnv (1 = single process)")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda -- see perf notes in this file's docstring")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                         help="Entropy bonus coefficient (SB3 PPO default 0.0 = no exploration bonus). "
                              "Try 0.01-0.02 if the policy collapses onto movement/PASS before ever "
                              "discovering the plant->water->harvest->sell loop -- see rl/README.md.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-from", default=None,
                         help="Path to an existing checkpoint .zip to continue training from (e.g. "
                              "checkpoints/v2/model.zip) instead of initializing a fresh policy. "
                              "--timesteps is how many *additional* steps to run. --ent-coef (and other "
                              "PPO hyperparams passed here) are re-applied on top of the loaded model. "
                              "Also resumes vecnormalize.pkl from next to this path if present -- see "
                              "this file's docstring. Note: EvalCallback's best-score tracking doesn't "
                              "persist across runs, so the first eval of the resumed run may overwrite "
                              "best_model.zip even if it's worse than the previous run's best.")
    args = parser.parse_args()

    opponent = resolve_agent(args.opponent)
    env_fns = [make_env_fn(opponent, args.episode_steps, args.seed) for _ in range(args.n_envs)]
    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    train_env = vec_env_cls(env_fns)
    eval_env = DummyVecEnv([make_env_fn(opponent, args.episode_steps, args.seed)])

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = REPO_ROOT / resume_path

    vecnorm_path = resume_path.parent / "vecnormalize.pkl" if resume_path else None
    if vecnorm_path and vecnorm_path.exists():
        train_env = VecNormalize.load(str(vecnorm_path), train_env)
        print(f"Resuming reward-normalization stats from {vecnorm_path}")
    else:
        if resume_path:
            print(f"No vecnormalize.pkl next to {resume_path} -- starting fresh reward-normalization stats.")
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=GAMMA)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, gamma=GAMMA, training=False)

    if resume_path:
        print(f"Resuming from {resume_path} ...")
        model = PPO.load(
            resume_path, env=train_env, device=args.device, ent_coef=args.ent_coef, gamma=GAMMA,
            tensorboard_log=str(REPO_ROOT / "runs"),
        )
        reset_num_timesteps = False
    else:
        model = PPO(
            "MultiInputPolicy", train_env, verbose=1, tensorboard_log=str(REPO_ROOT / "runs"),
            device=args.device, ent_coef=args.ent_coef, gamma=GAMMA,
        )
        reset_num_timesteps = True

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, reset_num_timesteps=reset_num_timesteps)

    final_path = out_dir / "model.zip"
    model.save(final_path)
    vecnorm_out = out_dir / "vecnormalize.pkl"
    train_env.save(str(vecnorm_out))
    print(f"Saved final model to {final_path}")
    print(f"Saved reward-normalization stats to {vecnorm_out}")
    print(f"Best eval checkpoint (if any improvement found) at {out_dir / 'best_model.zip'}")


if __name__ == "__main__":
    main()

"""Train a Kaggriculture PPO agent (v1 action scope -- see rl/README.md).

Usage:
    uv run rl/train.py                                            # vs starter, 200k steps
    uv run rl/train.py --opponent random --timesteps 50000        # quick smoke test
    uv run rl/train.py --n-envs 8 --timesteps 1000000             # parallel rollout collection
    uv run rl/train.py --opponent checkpoints/v1/model.zip --out checkpoints/v2  # self-play
    uv run rl/train.py --ent-coef 0.01 --n-envs 8 --timesteps 2000000            # more exploration
    uv run rl/train.py --resume-from checkpoints/v2/model.zip --out checkpoints/v2 --timesteps 500000  # continue training

Writes the best-eval checkpoint (best_model.zip), the final model (model.zip),
and reward-normalization stats (vecnormalize.pkl, see below) to --out, plus
tensorboard logs under runs/.

Performance notes:
- The bottleneck here is Kaggriculture's own single-threaded Python simulator
  (env.step()), not the policy network -- a small flattened MLP. `--n-envs`
  (SubprocVecEnv, one game per CPU process) is the lever that actually
  matters; it scales rollout collection close to linearly with cores.
- `--device cuda` is available but unlikely to help *this* policy size/env --
  GPU transfer overhead on tiny per-step batches tends to wash out any
  compute win. Kept around for when the policy grows (e.g. a real CNN over
  the grid observations instead of the current flatten -- see rl/README.md).
- Self-play (`--opponent <checkpoint>.zip`) combined with `--n-envs > 1` is
  untested: SubprocVecEnv pickles the opponent closure (which holds a loaded
  PPO model) into each worker process, which may be slow or fragile. Prefer
  `--n-envs 1` for self-play runs until this is verified.

Reward normalization (`VecNormalize`, `norm_reward=True`): rl/env.py's reward
mixes several hand-tuned terms at very different scales -- most events are
~0.01-1.0, but BUY_LAND is a one-time -10 to -40 hit (see rl/env.py's module
docstring), a scale mismatch flagged as a training-stability risk but never
addressed. `VecNormalize` divides the reward by a running estimate of its
discounted-return std instead of a hand-picked constant, so it adapts as
training progresses rather than needing the /100 scale (or a BUY_LAND-specific
fudge factor) to be guessed right upfront. Deliberately `norm_obs=False`:
features.py's encoder already hand-normalizes every observation field to a
small range, and reward scale was the specific problem being chased here, not
observation scale -- changing both at once would make it harder to tell which
change did what. Reward normalization only affects what the *value function*
is fit against, not the actions a trained policy outputs, so it's invisible
to inference: rl/checkpoint.py, rl/play.py, and rl/analyze_actions.py never
touch reward, only `model.predict(obs)`, and need no changes.

`vecnormalize.pkl` in --out holds the running reward statistics at the end of
training (matching final `model.zip` more precisely than `best_model.zip`,
which may have been saved earlier -- EvalCallback doesn't checkpoint
VecNormalize stats per-eval, only the model). `--resume-from` looks for a
`vecnormalize.pkl` next to the given checkpoint and resumes those statistics
too if found; otherwise it starts fresh and says so.
"""

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

# Must match PPO's own `gamma` (passed explicitly below, not left at SB3's default)
# and rl/env.py's KaggricultureEnv.GAMMA -- VecNormalize's reward normalization,
# PPO's advantage estimation, and env.py's potential-based shaping all need to
# agree on the discount rate for their respective guarantees/behavior to hold.
GAMMA = 0.99


def make_env_fn(opponent, episode_steps: int, seed: int | None):
    """Factory for a single Monitor-wrapped KaggricultureEnv, picklable so
    SubprocVecEnv can ship it to worker processes. Monitor sits *inside* the
    later VecNormalize wrapping (see main()), so it logs true, un-normalized
    episode returns -- tensorboard's rollout/ep_rew_mean keeps meaning what it
    always has, regardless of reward normalization."""

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
    # Always DummyVecEnv (single process): eval doesn't need --n-envs's parallelism
    # (n_eval_episodes defaults to 5, sequential is fine), just needs to be *some*
    # VecEnv for VecNormalize to wrap below.
    eval_env = DummyVecEnv([make_env_fn(opponent, args.episode_steps, args.seed)])

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = REPO_ROOT / resume_path

    # See this file's docstring for why norm_obs=False and why Monitor (inside
    # train_env/eval_env already) sits underneath this wrapping.
    vecnorm_path = resume_path.parent / "vecnormalize.pkl" if resume_path else None
    if vecnorm_path and vecnorm_path.exists():
        train_env = VecNormalize.load(str(vecnorm_path), train_env)
        print(f"Resuming reward-normalization stats from {vecnorm_path}")
    else:
        if resume_path:
            print(f"No vecnormalize.pkl next to {resume_path} -- starting fresh reward-normalization stats.")
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=GAMMA)
    # Frozen (training=False): apply train_env's current stats, don't let eval
    # episodes update them. EvalCallback calls sync_envs_normalization() each
    # eval to copy train_env's latest stats here automatically -- no manual
    # syncing needed.
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
        # EvalCallback counts calls to _on_step, i.e. one per vectorized step
        # (n_envs timesteps) -- divide so --eval-freq stays in timestep units
        # regardless of --n-envs.
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

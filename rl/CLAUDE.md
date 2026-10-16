# CLAUDE.md (rl/)

Guidance specific to the `rl/` PPO training track. See the project root `CLAUDE.md` for the
Kaggriculture agent contract and general repo conventions; this file covers only what's specific
to reinforcement-learning work under this directory.

## Commands

```bash
uv run rl/collect_demonstrations.py --episodes 30 --out rl/demos/starter_v1.npz  # record starter_agent demos
uv run rl/pretrain_bc.py --demos rl/demos/starter_v1.npz --out checkpoints/bc_v1 # behavioral-clone warm start
uv run rl/train.py --resume-from checkpoints/bc_v1/model.zip --opponent starter --timesteps 300000  # PPO fine-tune
uv run rl/train.py --opponent starter --timesteps 200000          # PPO training run (from scratch -- expect collapse, see below)
uv run rl/train.py --opponent random --timesteps 20000            # quick smoke test
uv run rl/train.py --n-envs 14 --device cpu --timesteps 2000000   # parallel envs (the real speed lever) -- cpu beats cuda here, see rl/train.py's perf notes
uv run rl/train.py --opponent checkpoints/v2/model.zip --out checkpoints/v3  # self-play
uv run rl/play.py checkpoints/v2/best_model.zip --opponent starter  # run a saved checkpoint locally
uv run rl/analyze_actions.py checkpoints/v2/best_model.zip --stochastic  # action-density report -- always check --stochastic too, see below
uv run rl/noop_audit.py checkpoints/v2/best_model.zip --opponent starter  # how much of the policy's action is wasted
uv run python rl/test_env.py                                        # reward/decode_action/encode_action regression tests
```

Use short training cycles (~300k timesteps, ~5-6 min at `--n-envs 14 --device cpu`) and check
`analyze_actions.py`/`noop_audit.py` in between rather than committing to one long unsupervised
run — this environment is prone to the policy collapsing onto a single always-legal action (see
below), and a short feedback loop catches that quickly instead of burning a long run on it.
`analyze_actions.py`'s *deterministic* mode can look totally collapsed right after behavioral
cloning even when the policy is fine — BC's demonstrations are class-imbalanced (mostly PASS),
so the argmax favors the majority class while the *stochastic* distribution still matches the
demonstrator's real per-action frequencies closely; PPO's own rollout collection samples
stochastically, so that's the distribution that actually matters for training, not the greedy
eval. Also check `model.zip` (final state), not just `best_model.zip` — `EvalCallback` only
overwrites `best_model.zip` on a *new* best score, so if the very first eval fires before the
first policy-gradient update completes, `best_model.zip` can silently stay the untouched
starting checkpoint for an entire run.

## Architecture

Separate track from `main.py`, not shipped to Kaggle. `rl/env.py` wraps `kaggle_environments`'
own `env.train([None, opponent])` trainer as a Gymnasium `Env` — but computes its own per-turn
money-diff reward rather than trusting the trainer's: Kaggriculture's interpreter only writes
`state.reward` on the terminal step, so the trainer's "per-step diff" is silently zero for
719/720 turns (confirmed root cause of `checkpoints/v1` collapsing to a do-nothing policy — see
`rl/README.md`). `rl/features.py` encodes observations into a `Dict` space (`own_grid`/`opp_grid`
per-tile channels + a flat `vector`); `rl/train.py` runs PPO (`stable-baselines3`) against a
configurable opponent (built-in name, agent `.py`, or another checkpoint for self-play). The
policy controls farmer movement/tile ops, buying (seeds, fertilizer, land), selling, hiring
(`HIRE`), and up to `MAX_HANDS` (features.py) hands' per-turn ops (same op vocabulary as the
farmer, one fixed action slot per hand — see `rl/env.py`'s module docstring) as independent
per-turn choices — animals are still unavailable. `rl/collect_demonstrations.py` +
`rl/pretrain_bc.py` behavioral-clone a policy from `kaggle_environments`' own `starter_agent`
before PPO fine-tuning (`rl/train.py --resume-from`), which in this project's experience is
needed to escape a total action-collapse PPO-from-scratch falls into on this environment — see
`rl/README.md`'s checkpoint-status history. See `rl/README.md` for the exact scope, the
reward-bug writeup, and its Open questions section for what's unresolved (notably: no export
path yet from an SB3 checkpoint to a submittable `main.py`).

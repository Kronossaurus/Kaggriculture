# CLAUDE.md (rl/)

Guidance specific to the `rl/` PPO training track. See the project root `CLAUDE.md` for the
Kaggriculture agent contract and general repo conventions; this file covers only what's specific
to reinforcement-learning work under this directory.

## Commands

```bash
uv run rl/train.py --opponent starter --timesteps 200000          # PPO training run
uv run rl/train.py --opponent random --timesteps 20000            # quick smoke test
uv run rl/train.py --n-envs 8 --device cuda --timesteps 2000000   # parallel envs (the real speed lever) + GPU
uv run rl/train.py --opponent checkpoints/v2/model.zip --out checkpoints/v3  # self-play
uv run rl/play.py checkpoints/v2/best_model.zip --opponent starter  # run a saved checkpoint locally
uv run rl/analyze_actions.py checkpoints/v2/best_model.zip          # action-density report
uv run python rl/test_env.py                                        # reward/decode_action regression tests
```

## Architecture

Separate track from `main.py`, not shipped to Kaggle. `rl/env.py` wraps `kaggle_environments`'
own `env.train([None, opponent])` trainer as a Gymnasium `Env` — but computes its own per-turn
money-diff reward rather than trusting the trainer's: Kaggriculture's interpreter only writes
`state.reward` on the terminal step, so the trainer's "per-step diff" is silently zero for
719/720 turns (confirmed root cause of `checkpoints/v1` collapsing to a do-nothing policy — see
`rl/README.md`). `rl/features.py` encodes observations into a `Dict` space (`own_grid`/`opp_grid`
per-tile channels + a flat `vector`); `rl/train.py` runs PPO (`stable-baselines3`) against a
configurable opponent (built-in name, agent `.py`, or another checkpoint for self-play). The
policy controls farmer movement/tile ops plus buying (seeds, fertilizer, land) and selling as
independent per-turn choices — hands and animals are still unavailable. See `rl/README.md` for
the exact scope, the reward-bug writeup, and its Open questions section for what's unresolved
(notably: no export path yet from an SB3 checkpoint to a submittable `main.py`).

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An agent for the [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) Kaggle
competition: a 1v1 farming-sim simulation run via `kaggle-environments`. Two players compete
for 720 turns (24 turns/day × 30 days) to end with the most money. The full game rules (crop
tables, market price function, turn processing order, town shops, observation schema) live in
`data/README.md` — read it before changing agent logic, it is not duplicated here. `data/AGENTS.md`
covers the competition workflow (local testing, CLI submission, episodes/replays/logs).

## Commands

```bash
uv sync                                              # install/sync deps into .venv

uv run scripts/run_local.py                          # main.py vs random, full season (720 steps)
uv run scripts/run_local.py --p2 starter --steps 200 # main.py vs the official baseline, shorter episode
uv run scripts/run_local.py --p1 random --p2 random  # baseline vs baseline
uv run scripts/run_local.py --seed 42                # deterministic episode
uv run scripts/run_local.py --replay replays/last.json  # dump replay JSON for the visualizer
uv run scripts/pool_winrate.py                          # main.py win rate vs a field of opponents, not just one
uv run python test_main.py                               # main.py regression tests (real bugs this session hit)
uv run ruff check .                                      # lint suite

uv run kaggle competitions submit kaggriculture -f main.py -m "message"
uv run kaggle competitions submissions kaggriculture
uv run kaggle competitions leaderboard kaggriculture -s

uv run rl/train.py --opponent starter --timesteps 200000          # PPO training run
uv run rl/train.py --opponent random --timesteps 20000            # quick smoke test
uv run rl/train.py --n-envs 8 --device cuda --timesteps 2000000   # parallel envs (the real speed lever) + GPU
uv run rl/train.py --opponent checkpoints/v2/model.zip --out checkpoints/v3  # self-play
uv run rl/play.py checkpoints/v2/best_model.zip --opponent starter  # run a saved checkpoint locally
uv run rl/analyze_actions.py checkpoints/v2/best_model.zip          # action-density report
uv run python rl/test_env.py                                        # reward/decode_action regression tests
```

`scripts/run_local.py --p1`/`--p2` accept the built-in names `random`, `pass`, `starter`, or a
path to an agent `.py` file (resolved relative to cwd, then repo root — works from any directory).

`uv run ruff check .` is the lint suite (config in `pyproject.toml`'s `[tool.ruff]`; default
rule set plus E402, since several `rl/` scripts deliberately import after a `sys.path.insert()`
and mark it `# noqa: E402` — E402 has to be enabled for those to mean anything). For agent
correctness, there's no assertion-based suite beyond `test_main.py`; `run_local.py` against
`random`/`starter` and `pool_winrate.py` for win rate against a field are the main checks.

### Environment

- Python is pinned to **3.12** via `.python-version` — do not bump to 3.13/3.14; `kaggle-environments`
  pulls in `pettingzoo`/`open_spiel` and related deps that don't build cleanly on newer CPython yet.
- Kaggle API auth is expected via the `KAGGLE_API_TOKEN` env var (already configured on this machine).
  `data/AGENTS.md` documents alternatives (token file, `kaggle auth login`).
- You'll see `OpenSpiel exception: Unknown game 'universal_poker'...` noise on every
  `import kaggle_environments` — it's `open_spiel` failing to register poker variants this build
  wasn't compiled with, unrelated to Kaggriculture. Harmless, safe to ignore.
- You may also see a "Gym has been unmaintained since 2022..." deprecation warning. It's the
  legacy `gym` package (pulled in transitively via `kaggle-environments` → `gymnax` → `gym`, not
  something we import — everything under `rl/` uses `gymnasium`). Harmless, safe to ignore.
- `torch` installs from a CUDA-specific index (`pytorch-cu128`), pinned via `[tool.uv.sources]` /
  `[[tool.uv.index]]` in `pyproject.toml` with `explicit = true` — that keeps the extra index from
  being considered for unrelated packages (adding it globally previously broke resolution for
  `litellm`/`importlib-metadata`). If `uv sync` ever fails to resolve `torch`, check that index's
  available versions before loosening the `torch` version constraint, not the other way around.

## Architecture

**`main.py`** is the actual submission artifact — Kaggle's runner loads this file directly and
calls `agent(obs)` once per turn for whichever player index we're assigned. Keep it self-contained;
if the agent grows beyond one file, bundle extra modules into a `submission.tar.gz` with `main.py`
at the root (per `data/AGENTS.md`), since the runner ships this file standalone, not the repo.

**`scripts/run_local.py`** is the harness for iterating without hitting Kaggle: it wraps
`kaggle_environments.make("kaggriculture", ...)` + `env.run([...])`, printing final rewards and
optionally dumping a replay JSON. Use it to A/B agent versions against `random`/`starter` before
submitting.

### Agent contract (the part you need without reading the full rules doc)

`agent(obs)` returns a dict every turn:

```py
{
  "farmer": [op, ...args],          # one op for the main farmer this turn
  "hands":  [[op, ...args], ...],   # one op per hired hand, in hands order
  "market": [[op, ...args], ...],   # ordered market orders, capped at maxMarketOrdersPerTurn (default 10)
}
```

Key `obs` fields: `obs["player"]` (0/1), `obs["day"]`/`obs["hour"]`, `obs["farms"][player]` (own
public state: `money`, `tiles[y][x]`, `farmer`, `hands`, `unlocked_quadrants`, `hires_today`),
`obs["private"]` (own-only: `shed`, `seeds`, `inventories` — opponent's private state is hidden),
`obs["market"]` (shared `inventory`/`prices`), `obs["town"]["unlocked_shops"]`.

A `tile` is `None` (empty), `"LOCKED"` (unbought quadrant), or a dict — `kind: "PLANT"` (crop,
watering/fertilizer state, yield), `kind: "WEED"`, or `kind: "COOP"|"PASTURE"` (animal structure).
Invalid actions are silent no-ops, so agent logic tends to be written defensively (check state
before acting rather than relying on errors). See `data/README.md`'s Observation Format section
for the complete schema, and the Object Types / Price Function / Turn Processing Order sections
for the economic model driving strategy (dynamic per-resource market prices, crop/animal yield
curves, town demand growth over the season).

`main.py`'s current strategy is a minimal "wheat loop" (buy seed → plant → water → harvest →
sell, repeat) — a floor to beat, not a target design. Stronger play means reasoning about land
expansion (`BUY_LAND`), hiring farm hands, higher-value/ongoing crops and animals, and timing
sells against the dynamic market rather than dumping produce immediately.

### RL training (`rl/`)

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
the exact scope, the reward-bug
writeup, and its Open questions section for what's unresolved (notably: no export path yet from
an SB3 checkpoint to a submittable `main.py`).

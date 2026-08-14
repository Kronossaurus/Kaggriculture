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

uv run kaggle competitions submit kaggriculture -f main.py -m "message"
uv run kaggle competitions submissions kaggriculture
uv run kaggle competitions leaderboard kaggriculture -s
```

`scripts/run_local.py --p1`/`--p2` accept the built-in names `random`, `pass`, `starter`, or a
path to an agent `.py` file (resolved relative to cwd, then repo root — works from any directory).

There is no lint/test suite yet; `run_local.py` against `random`/`starter` is the current
correctness check for agent changes.

### Environment

- Python is pinned to **3.12** via `.python-version` — do not bump to 3.13/3.14; `kaggle-environments`
  pulls in `pettingzoo`/`open_spiel` and related deps that don't build cleanly on newer CPython yet.
- Kaggle API auth is expected via the `KAGGLE_API_TOKEN` env var (already configured on this machine).
  `data/AGENTS.md` documents alternatives (token file, `kaggle auth login`).
- You'll see `OpenSpiel exception: Unknown game 'universal_poker'...` noise on every
  `import kaggle_environments` — it's `open_spiel` failing to register poker variants this build
  wasn't compiled with, unrelated to Kaggriculture. Harmless, safe to ignore.

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

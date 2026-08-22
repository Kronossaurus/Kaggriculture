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
```

RL training commands live in `rl/CLAUDE.md`.

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

## Investigation methodology: revert vs. tune

A win-rate regression in a 30-seed A/B is **not** by itself grounds to revert a fix — it
gates Kaggle submission (don't submit a build that regresses the field), not whether the
underlying code change survives in the repo. Before reverting anything that fixed a real,
demonstrated bug (an accounting error, a livelock, a structural gap), root-cause the
regression with a paired-seed replay trace (`scripts/analyze_replay.py --orders`/
`--purchase-trace RESOURCE`, or a money-flow-by-resource trace like the ones in
[[feed-orders-starve-market-slots]]'s 14th/17th-attempt notes) before deciding:

- **Companion constant miscalibrated for the new behavior** (the fix changed some dynamic —
  order volume, timing, crew allocation — that a fixed constant elsewhere was implicitly
  tuned around) → re-tune that constant using the trace's own signal, not a blind guess, and
  re-measure. Only give up on this path once a *specific* retuning hypothesis, motivated by
  the trace, has actually been tried and failed — not on suspicion alone.
- **Structural economic trade-off** (the fix genuinely reduces volume/revenue on something
  that pays for a large share of the agent's economy, measured directly via the trace, the
  way the 17th feed/seed-starvation attempt measured WHEAT throughput as +$35k of a $57k
  final) → revert, but document the measured mechanism, not just the win/loss number.

A fix that's mechanically validated (behaves as intended in a replay trace) but not yet
tuned can sit on its own branch/commit while a tuning pass continues, rather than being
reverted outright the moment the first A/B comes back worse — only promote to `main` (and
therefore to Kaggle-submission eligibility) once the 30-seed A/B matches or beats baseline
against the full pool *and* the two reactive self-play sentinels specifically (see
[[feed-orders-starve-market-slots]] for why a non-reactive tape like `tape_kiro` alone is a
biased, oversensitive signal for this kind of change — it rewards almost any reduction in
this agent's own market activity, real improvement or not).

### RL training (`rl/`)

Separate track from `main.py`, not shipped to Kaggle — see `rl/CLAUDE.md` for its architecture,
scope, and commands.

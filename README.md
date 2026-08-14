# Kaggriculture

Agent for the [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) Kaggle
competition — a 1v1 farming-sim simulation ($50,000 prize pool, deadline 2026-09-30).

Official rules and reference docs are in `data/README.md` and `data/AGENTS.md` (downloaded
from the competition). Read those first — this file only covers project setup.

## Setup

Environment is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Kaggle API auth: the `KAGGLE_API_TOKEN` env var is expected to be set (see
`data/AGENTS.md` for alternatives — token file or `kaggle auth login`).

## Project layout

- `main.py` — the submission entry point. Must expose an `agent(obs)` function; this
  is what `kaggle competitions submit` ships. Bundle extra modules into a
  `submission.tar.gz` with `main.py` at the root if the agent grows beyond one file.
- `scripts/run_local.py` — run local episodes against `random`/`pass`/`starter` or another
  agent file, with optional replay dump for the visualizer.
- `data/` — competition docs and downloaded assets (`README.md` rules, `AGENTS.md` workflow
  guide).
- `replays/` — local replay JSON dumps (gitignored).

## Local testing

```bash
uv run scripts/run_local.py                        # main.py vs random, full season
uv run scripts/run_local.py --p2 starter --steps 200
uv run scripts/run_local.py --replay replays/last.json
```

## Submitting

```bash
uv run kaggle competitions submit kaggriculture -f main.py -m "v1"
uv run kaggle competitions submissions kaggriculture
uv run kaggle competitions leaderboard kaggriculture -s
```

See `data/AGENTS.md` for the full workflow (episodes, replays, logs).

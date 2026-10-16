# Kaggriculture

My submission to the [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) Kaggle
competition, a 1v1 farming-sim simulation. **The competition has ended** (deadline 2026-09-30);
this repo is published as a record of the agent and the tooling built around it, not as an
active project.

Two players farm for 720 turns (24 turns/day × 30 days) and the one with the most money wins.
Each turn an agent controls a main farmer, any hired hands, and a list of market orders. The
official rules are in `data/README.md` and the submission workflow in `data/AGENTS.md`, both
downloaded from the competition.

## What's here

- `main.py`: the submitted agent, a single self-contained file exposing `agent(obs)`. It is a
  rule-based planner: it assigns crew to tile jobs (plant, water, harvest, weed, animal care),
  decides how much land, crop mix, animals and hands to scale up to over the season, and times
  market buys and sells against the dynamic price function.
- `test_main.py`: regression tests for bugs found along the way (`uv run python test_main.py`).
- `scripts/`: local tooling.
  - `run_local.py`: play episodes against `random`/`pass`/`starter` or another agent file.
  - `pool_winrate.py`: win rate against a pool of opponents rather than a single one.
  - `mc_search.py`: Monte Carlo search over the agent's tunable constants.
  - `analyze_replay.py`, `build_replay_viewer.py`, `extract_replay_tape.py`: replay analysis and
    an offline HTML viewer built on the official visualizer.
  - `cpp_bridge.py`: runs the agent against the C++ simulator below.
- `cpp_sim/`: a faster C++ episode engine. `sim.hpp` and `pyrandom.hpp` are an unmodified import
  of a public Kaggle notebook (credited in `cpp_sim/README.md`); `capi.cpp` and the bridge are
  mine.
- `rl/`: an experimental PPO / behavioral-cloning track. It never shipped and is not part of the
  submission; see `rl/README.md` for what was tried and why it did not work.
- `reference/`: frozen snapshots of my own earlier submissions, used as sparring opponents, plus
  a few agents downloaded from public Kaggle notebooks for comparison (e.g. `melon_maxxer.py`
  from the official getting-started notebook). Third-party files belong to their original authors.
- `checkpoints/`: early baseline snapshot (`v0`, a plain wheat loop).

## Setup

Python 3.12 (pinned in `.python-version`) and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

`import kaggle_environments` prints an `OpenSpiel exception: Unknown game 'universal_poker'`
message and a `gym` deprecation warning. Both come from transitive dependencies and are harmless.

## Running it

```bash
uv run scripts/run_local.py                           # main.py vs random, full season
uv run scripts/run_local.py --p2 starter --steps 200  # vs the official baseline, shorter
uv run scripts/run_local.py --seed 42                 # deterministic episode
uv run scripts/pool_winrate.py                        # win rate against a field of opponents
uv run python test_main.py                            # regression tests
uv run ruff check .                                   # lint
```

Submitting is no longer possible now that the competition is over; the original command was
`kaggle competitions submit kaggriculture -f main.py -m "message"`.

## License

See `LICENSE`.

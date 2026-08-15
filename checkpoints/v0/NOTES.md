# v0 — wheat loop (scripted baseline)

Frozen copy of `main.py` as of the pre-RL baseline: buy wheat seed, plant, water,
harvest, sell, repeat on the farmer's current tile. No land expansion, no hands,
no other crops/animals, no market timing (sells immediately on harvest).

This is the floor every future checkpoint (RL or heuristic) should beat.

## Results (local, `scripts/run_local.py`, starting money $3000)

| Matchup | Steps | v0 | Opponent |
| --- | --- | --- | --- |
| v0 vs `random` | 200 | 3022 | 2040 |
| v0 vs `random` | 100 | 2987 | 2650 |
| v0 vs `starter` | 200 | 3022 | 3068 |

Reproduce: `uv run scripts/run_local.py --p1 checkpoints/v0/main.py --p2 starter --steps 200`

# cpp_sim — fast C++ Kaggriculture simulator

## Provenance

`pyrandom.hpp` and `sim.hpp` are an **unmodified import** of the C++ port
published in
[nikital7/4000x-environment-speedup-kaggriculture](https://www.kaggle.com/code/nikital7/4000x-environment-speedup-kaggriculture)
(Kaggle notebook, pulled via `kaggle kernels pull` on 2026-08-20). It is not
this project's own code — see that notebook for the author's own writeup of
the port's design and the three subtle correctness traps it had to get right
(CPython's exact Mersenne Twister construction, per-unit inventory
insertion-order, and market-order slot-index resolution across both
players).

`capi.cpp` and `scripts/cpp_bridge.py` **are** this project's own code,
written to actually use the port: the notebook is deliberately
"environment only, no strategy code" and only replays pre-recorded action
sequences (`validate.cpp`) or benchmarks raw throughput (`bench.cpp`) — it
doesn't drive a live Python agent against the simulator. `capi.cpp` exposes
a flat C API (`extern "C"`, plain cdecl, no Python C API dependency) so
`cpp_bridge.py` can load it with `ctypes` and, each turn, reconstruct the
obs dict an agent expects from the C++ `Sim`'s internal state, call the
agent, and encode its action back into the C++ `Action` struct.

## Build

Needs a C++17 compiler. This machine has none in PATH by default; a portable
one was installed via `winget install --id BrechtSanders.WinLibs.POSIX.UCRT
--scope user` (no admin required — chocolatey's default install path needs
elevation and failed here). Rebuild `kag_sim.dll` after any change to
`capi.cpp` or `sim.hpp`:

```bash
g++ -O3 -march=native -std=c++17 -static -shared -o kag_sim.dll capi.cpp
```

`-static` matters: without it the DLL depends on `libstdc++-6.dll` /
`libgcc_s_seh-1.dll` from the compiler's own bin directory, which won't be
on PATH wherever `cpp_bridge.py` runs from.

`kag_sim.dll` itself is **not** committed — `analysis/`-style build output,
rebuild it locally (one command, a few seconds).

## Validation

Two independent checks, both passing as of 2026-08-20:

1. **Trace replay** (the port's own method, `validate.cpp` +
   `export_trace.py`): 39/39 recorded episodes reproduced bit-exact
   (identical money and market inventory at every step) — 15 with the
   notebook's own starter/random/pass pairings, 24 with `main.py` itself
   (vs starter, vs random, self-play, both seats), 8 seeds each. Confirms
   the imported simulator core is correct for this project's own agent.

2. **Live bridge** (`cpp_bridge.py`'s own obs-reconstruction +
   action-encoding, which the port doesn't provide): ran `play_fast()`
   end-to-end against `kaggle_environments`' real environment, same seeds,
   and compared final money. **9/9 exact matches** across main.py vs
   x492 (both seats), self-play, and starter vs random — real strategies
   reading a wide range of obs fields (tiles, shed, seeds, inventories,
   market prices, hires_today, unlocked_quadrants), not just main.py's own
   narrower usage.
   - **Known non-goal**: comparing against the `random` built-in directly
     mismatches (checked, understood, not a bug) — `random_agent` in
     `kaggle_environments`' own kaggriculture.py calls `random.Random()`
     fresh, unseeded, every invocation, so its behavior isn't reproducible
     between two separate process runs even through the real environment.
     Two live runs of `starter vs random` at the same seed already diverge
     in `kaggle_environments` itself; this has nothing to do with the C++
     bridge. Not a concern for actual use — the search pools this bridge
     is for (`mc_search.py`'s `DEFAULT_SEARCH_POOL`, `pool_winrate.py`'s
     `DEFAULT_POOL`) use deterministic strategies (`main.py`,
     `reference/*.py`), never the `random` built-in.

Struct-layout safety net: `cpp_bridge.py`'s ctypes mirrors of `Tile` /
`Farm` / `Market` / `State` / `Action` / `Config` must match `sim.hpp`
field-for-field (same order, same integer widths) for direct memory access
to mean anything. `sim_sizeof_*()` in `capi.cpp` exports each struct's real
`sizeof()`; `cpp_bridge.py` asserts these against `ctypes.sizeof()` of its
own mirrors at import time, so a future edit to `sim.hpp` that isn't
mirrored here fails loudly on the next run instead of silently misreading
state.

## Measured speedup (this project's actual workload, not the notebook's
noop-agent benchmark)

The notebook's headline 2,700–4,500x is environment-only (do-nothing
agents) — the honest number for `main.py`, whose own per-turn heuristics
(`_find_target`/`_decide_unit`, looping the whole tile grid) are real
compute, not overhead:

| path | main.py vs starter, 8 seeds, 720 steps |
|---|---|
| `kaggle_environments` (warm loop) | 0.56 eps/sec |
| `cpp_bridge.play_fast` | 2.61 eps/sec |

**~4.7x**, not 1000x+ — most of the remaining cost is `main.py`'s own
Python decision logic (identical either way, cProfile shows `agent()`/
`_find_target`/`_decide_unit` dominate `play_fast`'s own wall time, not
`build_obs`/`encode_action`), not something a faster environment can
remove. Still a real, validated win for anything seed-count-bound: `mc_search.py`'s
default 30-trial/5-seed/2-opponent search (~300 episodes) drops from
~18-20 min to well under 5.

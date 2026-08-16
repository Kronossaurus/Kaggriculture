# RL training (v1 scaffold)

Training-time infra for a learned Kaggriculture agent. Not shipped to Kaggle —
`main.py` stays the plain, dependency-free submission entry point. Turning a
trained checkpoint into something submittable is unsolved (see Open questions).

## Guiding principles (learned the hard way)

- **When a search looks converged, check what you're measuring before tuning
  harder.** Every real plateau/oscillation hit in this project's history
  turned out to be a broken measurement, not a hyperparameter problem: the
  terminal-only reward bug, the Phi floor-fix, the HARVEST-inventory gap, the
  FERTILIZE credit, the shed-adjacency gap. `--ent-coef` and timesteps were
  tuned repeatedly around several of these before the actual cause was found
  — check the environment/reward/observation modeling first, the knobs
  second.
- **Rank by win rate against a field, never margin against one baseline.**
  `analyze_actions.py` now reports win/loss/tie (see its checkpoint-status
  update), which is the "never margin" half — but every evaluation in this
  project (`train.py`'s `EvalCallback`, `analyze_actions.py`'s default)
  still scores against a single opponent (`starter`). "Field" (several
  distinct opponents, scored and ranked separately) isn't built yet — a
  checkpoint that's learned to beat `starter` specifically isn't necessarily
  stronger in general, just better at beating that one baseline's particular
  patterns. `reference/melon_maxxer.py` (a second scripted, zero-waste --
  confirmed with `noop_audit.py` -- but narrower agent than `starter`,
  downloaded from Kaggle's official `bovard/kaggriculture-getting-started`
  notebook via `kaggle kernels pull`) plus `checkpoints/v0/main.py` give
  three distinct non-self opponents to build that field from, once it's
  built.
- **Get at least one agent you didn't write into the evaluation loop, early.**
  Already true here, incidentally: `starter` is `kaggle_environments`' own
  built-in agent, not ours, and has been the default opponent since the
  start — but see the point above, one such agent isn't a field.
- **Verify which engine you're actually playing by checking source, not just
  docs.** Every non-obvious mechanic documented in this file (HARVEST landing
  in personal inventory not the shed, PICKUP/DROP's shed-adjacency
  requirement, BUY_LAND's fixed unlock order) was confirmed by reading
  kaggriculture.py directly, not just data/README.md -- worth keeping up
  as a habit, matching kaggle_environments version changes could silently
  invalidate assumptions otherwise.
- **Instrument the engine to find silent no-ops, then test the fix -- don't
  assume a measured waste is a recoverable one.** `noop_audit.py` (see
  Pieces) does the first half. Some waste is structural, not a bug (e.g. an
  exploring stochastic policy calling WATER on an already-watered tile isn't
  a modeling gap to fix, it's what exploration costs) -- 0% waste isn't the
  right target, confirming a *specific* fix actually moved a *specific*
  op's waste rate is. Re-run `noop_audit.py` after any training run that
  follows an observation/reward fix motivated by an earlier audit, don't
  just trust that the code change looks correct.

## Pieces

- `features.py` — encodes a raw Kaggriculture observation into the RL env's
  `Dict` observation space: `own_grid`/`opp_grid` (`(13, 10, 10)` per-tile
  channels: locked, empty, weed, is-plant, 5× crop one-hot, watered, yield,
  fertilized, farmer-here) + `vector` (money, day/hour, seed counts, shed
  contents, held-in-inventory contents, market prices/inventory, a near-shed
  flag — all normalized). Symmetric in `obs["player"]`, so the same function
  encodes observations for whichever side is asking, which is what lets a
  frozen checkpoint act as an opponent.
- `env.py` — a Gymnasium `Env` wrapping `kaggle_environments`' own
  `env.train([None, opponent])` trainer. **Does not use the trainer's own
  `reward`** — Kaggriculture's interpreter only ever writes `state.reward` on
  the terminal step (see `kaggriculture.py:947-950`), so the trainer's
  documented "per-step diff" is actually zero for 719/720 turns and then a
  single terminal jump equal to raw final money, not a real gain. `env.py`
  computes its own per-turn money diff straight from the observation instead
  — see the module docstring there for how this was found (it's the reason
  v1 collapsed, below). `decode_action` turns a 5-tuple (farmer op, buy-seed,
  buy-product, buy-land, sell) into a full Kaggriculture action dict.
- `train.py` — PPO (`stable-baselines3`, `MultiInputPolicy`) training loop with
  periodic eval against the same opponent, saving to a checkpoint dir.
- `checkpoint.py` — loads a saved `.zip` as a plain `obs -> action dict` agent
  function, shared by `train.py` (self-play opponents) and `play.py`.
- `play.py` — run a saved checkpoint locally against an opponent, mirroring
  `scripts/run_local.py` (result printout, optional replay dump).
- `analyze_actions.py` — run episodes and report the *density* of the policy's
  choices: per-op counts, Shannon entropy (a single "how diverse are the
  choices" number), longest identical-action streak, a spatial heatmap of
  farmer position, and op mix over the season. This is the tool that would
  have caught the `checkpoints/v1` collapse below at a glance (entropy ≈ 0,
  streak ≈ episode length) instead of needing a traced-episode diff.
- `noop_audit.py` — a different question from `analyze_actions.py`: not *which*
  op did the policy pick, but did the engine actually do anything with it.
  Hooks kaggriculture.py's own `_apply_unit_action` (the actual mutation
  point, not our `decode_action`) and diffs farm+private state either side of
  every call, so it catches waste `analyze_actions.py` structurally can't --
  see the checkpoint-status update below for what it already found.

## Action scope

The policy picks **five independent choices per turn** (a `MultiDiscrete`
action space, not one combined op):

```
farmer op:    PASS, NORTH, SOUTH, EAST, WEST, WATER, HARVEST, FERTILIZE, DIG,
              PICKUP, DROP,
              PLANT_WHEAT, PLANT_CARROT, PLANT_TOMATO, PLANT_STRAWBERRY, PLANT_MELON

buy seed:     NONE, WHEAT, CARROT, TOMATO, STRAWBERRY, MELON
buy product:  NONE, FERTILIZER
buy land:     NONE, BUY_LAND
sell:         NONE, WHEAT, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL, FERTILIZER
```

They're separate dimensions because they're separate in the game itself — a
turn's action dict has independent `farmer` and `market` keys (see
`data/README.md`), so a real player can move (or water, or harvest, ...) *and*
queue a `BUY_SEED` order *and* a `BUY_PRODUCT` order *and* a `BUY_LAND` order
*and* a `SELL` order all the same turn. An earlier iteration modeled
`BUY_SEED_<crop>` as one more option competing with movement/tile ops for a
single discrete choice, which meant picking it cost the farmer its whole turn
— not how the game works, and an unnecessary extra tax on top of the
credit-assignment problem below. `PLANT_<crop>` with no seed on hand is just a
silent no-op like any other invalid action, same as `BUY_SEED` if
unaffordable — the policy has to learn the buy-then-plant sequence itself,
nothing auto-buys a seed for it.

A grouped single `PLANT` op (crop inferred from whichever seed is held, sized
down from 5 ops to 1) was tried and **reverted** — it removes the policy's
ability to choose *which* held seed to plant when it's holding more than one
type at once, which matters even though a same-turn `BUY_SEED` can never be
the seed a same-turn `PLANT` consumes (farmer actions are recorded before
market actions each turn, per data/README.md's Turn Processing Order).

**`FERTILIZE`, `PICKUP`, and `BUY_PRODUCT` complete the fertilizer pipeline**
(previously `FERTILIZE` was listed but always a no-op — a scope gap, not a
reward bug, see the "Update" entries below for the full history). Per
data/README.md and confirmed in kaggriculture.py: `BUY_PRODUCT FERTILIZER`
lands fertilizer in the shed; `FERTILIZE` consumes it from the *active
farmer's inventory*, not the shed; so `PICKUP` (shed → inventory) is required
in between. `PICKUP` has no item choice in the action space — FERTILIZER is
the only thing worth moving out of the shed in the current scope.

**`DROP`** dumps the farmer's entire inventory into the shed on demand.
Without it, harvested produce (which HARVEST puts in the farmer's personal
inventory, not the shed — see data/README.md and kaggriculture.py) only
reaches the shed, and becomes sellable at all, via the automatic end-of-day
drop, up to `turnsPerDay - 1` turns later. `DROP` lets the policy shorten
that gap to whatever it wants.

**`SELL` is now a learned per-turn choice** (one product per turn, fully
liquidated) instead of auto-selling the entire shed every turn the way
`checkpoints/v0` and earlier RL checkpoints did. This reopens the same
credit-assignment question `BUY_SEED` had before `IDLE_PENALTY` — does the
policy connect "chose SELL" to "money went up"? — untested at the scale that
took. No new penalty term was added to push it, on the theory that
Phi(terminal) = 0 (see `env.py`'s `step()`) already creates real pressure to
sell before the episode ends, since unsold value held at the very end is an
uncompensated loss in the shaped reward.

**`BUY_LAND`** unlocks the next quadrant. It takes no target -- the game
always unlocks NE, then SW, then SE, at fixed prices ($1k/$2k/$4k) -- so it's
a plain per-turn toggle, no design ambiguity.

Farm hands, hiring (`HIRE`), and animals are still not in the action space at
all. Play is confined to a single farmer, though `BUY_LAND` above means it can
now range across all four quadrants rather than just the starting NW one.

This is the "macro RL + scripted micro" split discussed for the project: get
the wrapper/encoder/PPO pipeline actually training before growing the action
space. Remaining natural next steps:

1. Farm hands (multiplies actions-per-turn — needs either a per-hand action
   head or a shared policy applied per-unit).
2. Animals (coops/pastures, `BUILD_COOP`/`BUILD_PASTURE`,
   `PLACE`/`FEED`/`CARE`/`COLLECT_FERTILIZER`, `BUY_ANIMAL`).

## Usage

```bash
uv run rl/train.py --opponent starter --timesteps 200000       # baseline run
uv run rl/train.py --opponent random --timesteps 20000         # quick smoke test
uv run rl/train.py --n-envs 8 --device cuda --timesteps 2000000   # parallel + GPU

# Self-play: opponent is a frozen earlier checkpoint
uv run rl/train.py --opponent checkpoints/v1/model.zip --out checkpoints/v2
```

`--n-envs N` runs N Kaggriculture games in parallel processes (`SubprocVecEnv`)
for rollout collection — this is the real speed lever (the simulator, not the
policy net, is the bottleneck; ~3x fps going from 1 to 4 envs on a 16-core
machine in testing). `--device cuda` is available too but expect little or no
gain at the current policy size (small flattened MLP) — see the perf notes at
the top of `rl/train.py`. Self-play + `--n-envs > 1` together is untested.

Outputs land in `--out` (default `checkpoints/v1/`): `best_model.zip` (best
eval score seen) and `model.zip` (final). Tensorboard logs go to `runs/`
(`tensorboard --logdir runs`).

Run a checkpoint locally to see how it actually plays:

```bash
uv run rl/play.py checkpoints/v1/best_model.zip                       # vs starter, full season
uv run rl/play.py checkpoints/v1/best_model.zip --opponent random --steps 200
uv run rl/play.py checkpoints/v1/best_model.zip --opponent checkpoints/v0/main.py
uv run rl/play.py checkpoints/v1/best_model.zip --replay replays/v1.json
```

Analyze how diverse (or collapsed) a checkpoint's choices are:

```bash
uv run rl/analyze_actions.py checkpoints/v1/best_model.zip
uv run rl/analyze_actions.py checkpoints/v1/best_model.zip --opponent random --episodes 5
```

Prints a win/loss/tie record + mean own-vs-opponent money, an action-count
table, Shannon entropy, and longest identical-action streak to the console, and
writes `action_distribution.png`, `buy_seed_distribution.png`,
`buy_product_distribution.png`, `buy_land_distribution.png`,
`sell_distribution.png`, `position_heatmap.png`, `action_over_time.png`, and a
raw `log.csv` to `analysis/<checkpoint stem>/` (or `--out`). `--seed` defaults
to a fixed value (not random) -- see the checkpoint-status update below for
why comparing runs on different seeds is misleading here.

Check whether a checkpoint's chosen actions actually *did* anything (not just
which ones it picked -- see the Pieces section above):

```bash
uv run rl/noop_audit.py starter                                    # sanity check the tool itself
uv run rl/noop_audit.py checkpoints/v2/best_model.zip --opponent starter
uv run rl/noop_audit.py checkpoints/v2/best_model.zip --stochastic --opponent starter
```

Prints total wasted (silently no-op) actions and a per-op breakdown sorted by
waste -- the breakdown is what points at *which* rule the policy hasn't
learned, not just that something's off.

## v1 checkpoint status: collapsed, invalidated by the reward bug above

The first real `checkpoints/v1` training run (200k timesteps vs `starter`) converged to a
degenerate policy that spams a single movement op for the entire episode, every episode,
regardless of opponent or observation (verified: 719/719 identical actions in a traced
episode). Root cause was the reward bug described above — PPO only ever saw a nonzero
signal on the last of 720 turns, so any multi-step profitable sequence (buy → wait →
water → harvest → sell) looked exactly as good as doing nothing unless it happened to
also net exactly $3000, and doing nothing was the safer bet under that signal. Now that
`env.py` computes real per-turn reward, retrain before trusting any `checkpoints/v1`
artifact — `best_model.zip`/`model.zip` there predate the fix.

Even with real per-turn money reward, a training run afterward still showed the policy
avoiding `PLANT`/`WATER`/`HARVEST` entirely (all `PASS`/movement — see `analyze_actions.py`
above). Root cause: the delay between `PLANT`'s upfront seed cost and the `SELL` payoff
~2 days/~48 turns later (wheat, per `main.py`'s `WHEAT_FIRST_YIELD_DAY`) was enough for
early PPO updates to suppress those ops before the value function learned they pay off,
compounded by `ent_coef=0.0` (SB3 PPO's default — no exploration bonus) letting that
suppression stick. Two mitigations added:

- `env.py` now adds **potential-based reward shaping** (Ng, Harada & Russell 1999) on top
  of the real money-diff reward: `Phi(s)` = value of standing crops' `yield_units` at base
  market price, `shaped = raw + gamma*Phi(s') - Phi(s)`. This is provably policy-invariant
  (same optimal policy as the unshaped reward) but pays out turn by turn as a crop grows
  instead of one lump sum at `SELL`. `KaggricultureEnv.GAMMA` must be kept in sync with
  PPO's `gamma` (SB3 default 0.99, not currently overridden) for that guarantee to hold.
- `train.py` gained `--ent-coef` (default `0.0`, matching SB3) to keep exploring ops that
  look bad short-term for longer — try `0.01`–`0.02` first.

**Update (after 500k → 1M → 1.5M timesteps of continued fine-tuning, each round worse than the
last):** the deterministic policy went from safely inert (500k, $3000 flat) to losing most of
its money (1M, ~$60 left) to losing all of it (1.5M, ~$0), while the *stochastic* distribution
stayed reasonably diverse and safe throughout. Root cause found in `_potential` itself, not the
training hyperparameters: scoring a standing crop at raw `yield_units * price` meant `PLANT`
looked like a shaping *loss* the instant it happened (yield_units is 0 right after planting, so
Phi drops from the seed's held value to ~0) -- an accidental penalty on the very op the shaping
was supposed to encourage. That's why the policy kept learning to buy seeds and abandon them
instead of watering them through to harvest. Fixed by flooring a standing crop's value at its
own seed cost (see `_potential`'s docstring in `rl/env.py`) so `PLANT` is shaping-neutral like
`BUY_SEED`, Phi only rises for real once growth passes that floor, and letting the crop die from
neglect drops Phi from the floor to 0 -- a real penalty for abandoning a started flow. Retraining
from scratch under this fix (not resuming `checkpoints/v2`, which was shaped under the buggy
`Phi` throughout) -- see checkpoint status below for the result once it's in.

**Update (idle penalty added on top of the Phi floor-fix):** even after the floor-fix made
`BUY_SEED`/`PLANT` shaping-*neutral*, the `v2` checkpoint above (500k timesteps, fresh from
scratch, `ent_coef=0.01`) still never learned to buy a seed at all in deterministic play —
`analyze_actions.py` showed `best_model.zip` spamming `WATER` on an empty tile 100% of the
time and `model.zip` toggling `FERTILIZE`/`PLANT_WHEAT` with `BUY_SEED` stuck at `NONE` for
all 3595 logged turns in both, money flat at the $3000 starting balance the whole episode
(neutral, but never actually playing). Stochastic sampling looked deceptively healthy —
entropy at 3.80/3.81 bits, ~100% of max — but that turned out to mean the policy's logits
were still nearly *uniform*, not that it had learned a real preference for the profitable
sequence; money under stochastic sampling dropped to ~$0 in 4/5 episodes as random buys got
abandoned before harvest.

Diagnosis: "shaping-neutral" only removes the disincentive to buy a seed, it doesn't add any
incentive *to* buy one — and potential-based shaping structurally can't add that pull either,
since by construction (telescoping) it's invariant to how long the policy dwells in a state
and only pays out on transitions. `env.py` now adds `IDLE_PENALTY`, a flat per-turn cost while
holding no seed and no standing crop (`KaggricultureEnv._is_idle`) — deliberately *not*
potential-based, so it accumulates for real the longer the loop sits empty, giving `BUY_SEED`
something to escape rather than just something that no longer hurts. Not yet validated with a
full training run — see checkpoint status below once `checkpoints/v3` (or a fresh `v2` re-run)
is in.

**Update (idle penalty validated, second gap found -- watering):** retrained `checkpoints/v2`
from scratch with `IDLE_PENALTY` (500k timesteps, `ent_coef=0.01`), then continued it +500k more
(1M total, `checkpoints/v2_1M_idle_penalty_only_snapshot`). The trend was real:
`rollout/ep_rew_mean` went from -173.6 (start) to -23.9 (1M), and `analyze_actions.py` showed the
deterministic policy actually attempting `BUY_SEED=WHEAT` and then `PLANT_WHEAT` (78% of turns) for
the first time -- previously `BUY_SEED` never left `NONE` in greedy play. But tracing the
`log.csv` showed the crop, once planted, was never watered or harvested for the rest of the
719-turn episode -- `WATER`/`HARVEST` sat at 0% in deterministic play the whole time, so the
planted wheat just died from neglect (2 unwatered days -> weed, per data/README.md). Root cause:
the idle penalty stops applying the instant a seed is planted (there's a standing crop, so
`_is_idle` is `False`), but nothing else creates urgency to water it *today* -- Phi's floor-fix
only makes watering non-negative, same gap as it left for BUY_SEED before IDLE_PENALTY was added.
Added `NEGLECT_PENALTY`, a per-tile per-turn tax for each standing crop not watered today
(`KaggricultureEnv._unwatered_crop_count`), same non-potential-based reasoning as `IDLE_PENALTY`
one link further down the loop.

**Update (NEGLECT_PENALTY result: real progress, then a training-instability trap, then a lost
checkpoint):** retrained `checkpoints/v2` from scratch with both `IDLE_PENALTY` and
`NEGLECT_PENALTY` (`ent_coef=0.01`), continued +500k twice (500k -> 1M -> 1.5M total). At 1M,
`analyze_actions.py` showed the best result yet: deterministic play bought seeds, planted wheat,
and stochastic play showed `WATER` (14%) and `HARVEST` (9.1%) at real, non-uniform rates for the
first time -- all four loop stages present together, not just one at a time. At 1.5M both
`best_model.zip` and `model.zip` had regressed back to mostly `PASS`/movement with `BUY_SEED`
stuck near `NONE` -- worse than the 1M snapshot, and **that 1M checkpoint was not saved before
continuing, so it's gone** (only its `analyze_actions.py` logs/plots survive, under
`analysis/v2_neglect_1M_*` -- not the weights). This was avoidable: the exact
500k-better-1M-worse-1.5M pattern had already happened once before (the Phi-bug saga above), and
that time a snapshot was taken before each continue; this time it wasn't.

**Update (shed value added to Phi -- HARVEST floor-fix):** noticed a third instance of the same
bug class the PLANT floor-fix addressed: `decode_action` builds each turn's auto-SELL orders from
the *pre-action* obs, so a turn that calls HARVEST doesn't also sell what it just harvested --
the produce sits in the shed until the following turn's auto-sell. `_potential` didn't count shed
inventory at all, so HARVEST (especially the final harvest of a one-time crop, which removes the
tile) made Phi drop a full turn before the matching money-diff reward arrived -- the same
shaping-penalty-on-the-op-we-want-to-encourage bug PLANT had, just one step later in the chain.
Fixed by adding shed inventory to `_potential` at full market price (no flooring needed, it's
already realized). Verified with a synthetic before/after-harvest state (Phi unchanged across the
transition, as intended). `checkpoints/v2` retrained from scratch with this fix on top of
`IDLE_PENALTY`/`NEGLECT_PENALTY` (previous checkpoint snapshotted first this time, as
`v2_1.5M_idle_neglect_oscillating_snapshot`).

**Update (Phi's "market price" was static, should be live):** both the standing-crop value and
the new shed value above were priced off `BASE_PRICE`, the fixed table anchor -- but
`data/README.md` documents that sell prices move dynamically per resource and persist across
days, and that live quote is already in the observation (`obs["market"]["prices"]`, already used
by `features.py`'s encoder), just not wired into `_potential`. Switched both lookups to the live
price with a `BASE_PRICE` fallback for anything missing from the quote. Buy-side seed cost is
*not* affected -- the market only moves sell prices, seed purchase prices are fixed per
data/README.md, so `SEED_COST` staying static is correct as-is. Low-impact in practice (every
product starts with `I0=10,000` inventory, "far above any single game's realistic production
volume" per data/README.md, so price should stay close to base for most of a game) but strictly
more correct, and doesn't touch the shaping's policy-invariance guarantee (market price is just
another part of the observable state Phi is a function of). Verified with a synthetic
low/high-price test (Phi scales with the quoted price, falls back to `BASE_PRICE` when unquoted).
See checkpoint status below once a training run with this fix is in.

Root cause of the regression, from the full `rollout/ep_rew_mean` curve (not just start/end):
the reward oscillates strongly throughout the *entire* 1.5M run, repeatedly swinging between
peaks around -10 and valleys around -230, not converging smoothly and not merely unlucky at the
very end. `model.learn()`'s final timestep happened to land in a valley. Working hypothesis:
`ent_coef=0.01` is high enough to keep perturbing a policy that's already found a good sequence,
so gains don't stick. Next attempt should back off `ent_coef` (e.g. 0.003-0.005) rather than just
adding more raw timesteps, **and snapshot the checkpoint dir before every continued run**,
without exception, given this is now the second time skipping that step lost a good result.
Paused here without further training -- see Open questions.

`env.py`'s action space also changed from `Discrete` to `MultiDiscrete` (see v1 action scope
above) to give the policy a real, independent `BUY_SEED` choice — this **breaks
compatibility** with every checkpoint trained before this change (`checkpoints/v1`,
`checkpoints/v2`): their saved policy only ever outputs a single scalar (0-d array) action,
which `decode_action`'s `(farmer_op, buy_seed)` unpacking can't consume (`TypeError:
iteration over a 0-d array`). `PPO.load()` itself still succeeds, but calling the loaded
agent — via `rl/play.py`, as a self-play opponent, or through `rl/checkpoint.py` at all —
throws on every turn; `kaggle_environments` swallows that into a per-player `ERROR` status
instead of crashing the run, so it can look like it "ran" while actually doing nothing the
entire episode (confirmed with `debug=True`). Treat `checkpoints/v1`/`v2` as historical
artifacts only, not resumable or playable checkpoints.

**Update (game-modeling audit: fertilizer completed, land/sell-timing added, a real HARVEST gap
found):** requested review of what else the action space was missing against data/README.md and
kaggriculture.py's actual source, not just the doc. Found and fixed:

- **Fertilizer completed**: `BUY_PRODUCT` (buy-product dimension, FERTILIZER only) and `PICKUP`
  (farmer op, fixed target FERTILIZER) added, so `FERTILIZE` -- present in `FARMER_OPS` since
  early on but always a no-op -- can finally do something.
- **`DROP` added**, and with it a real, previously-undiagnosed gap: HARVEST was assumed (by an
  earlier `_potential` fix, see the shed-value update above) to deposit produce straight into the
  shed. It doesn't -- confirmed in kaggriculture.py, HARVEST calls `_inv_add(inv, ...)` where
  `inv` is the *acting unit's personal inventory*, and `_drop_inventories_to_shed` (the only thing
  that ever moves it to the shed) runs once per day, in `_end_of_day`. `SELL` explicitly checks
  `private["shed"]` and nothing else, so harvested produce is provably unsellable for up to
  `turnsPerDay - 1` turns after HARVEST -- not a wrapper bug, an actual game mechanic, but one the
  action space had no lever for. `DROP` gives the policy a way to force that transfer early.
- **`_potential` fixed to match**: it counted `shed` but never `private["inventories"]`, so Phi
  still dropped a tile's value at HARVEST with nothing counted to replace it until the automatic
  shed-drop -- the earlier "HARVEST floor-fix" was validated against a synthetic test that put the
  harvested amount directly in `shed`, which doesn't reflect what the real game does. Now sums
  `inventories` the same way as `shed`. `features.py`'s observation vector had the identical gap
  (never encoded `private["inventories"]` at all -- the policy was blind to what it had just
  harvested until the day-end transfer) and is now fixed too, `VECTOR_DIM` grew accordingly.
- **`BUY_LAND` added** as a plain toggle (buy-land dimension) -- it takes no target quadrant, the
  game unlocks NE → SW → SE automatically at fixed prices, so there was no design question here.
- **Learned `SELL` added**, replacing the auto-sell-everything-every-turn behavior every previous
  checkpoint trained under. One product per turn, chosen explicitly and fully liquidated -- not a
  blanket hold/sell toggle, so the policy keeps real per-item control, consistent with the
  `PLANT_<crop>`-not-grouped decision above.

Action space grew from `MultiDiscrete([14, 6])` to `MultiDiscrete([16, 6, 2, 2, 10])` (14→16
farmer ops for `PICKUP`/`DROP`, plus three new dimensions) and the observation vector grew for the
inventory fix -- **breaks compatibility** with every checkpoint before this update, same failure
mode as the earlier `Discrete`→`MultiDiscrete` and 14→10-op breaks (silent `PPO.load()` without
`env=`, then an `IndexError`/shape mismatch partway through `predict`/`decode_action`). All prior
checkpoints are historical-only artifacts.

**Update (fairness audit of the reward function against the new ops):** requested a pass to make
sure nothing new was being unfairly penalized, same bug class as the PLANT/HARVEST fixes above.
Checked every new op against "does it spend something real now for a benefit Phi doesn't credit
until later":

- `BUY_PRODUCT`, `PICKUP`, `DROP` — all fine already, by construction. Shed and inventory are
  valued identically in `_potential`, so moving value between them (`PICKUP`/`DROP`) is exactly
  neutral, and buying fertilizer is the same near-neutral trade `BUY_SEED` already is.
- `SELL` — fine already. The item's Phi value and its realized money price are (approximately) the
  same live market quote, so selling converts Phi-value into money-reward at close to 1:1 (a tiny
  gamma-discount drag, same as every other transition).
- **`FERTILIZE` — found and fixed.** It consumes a real, Phi-valued `FERTILIZER` unit from
  inventory for a doubled-yield bonus that `yield_units` doesn't reflect until later `WATER` calls
  bank it (data/README.md: doubles the per-day yield bonus for 3 days) -- the exact PLANT bug,
  one link further down the chain. Fixed the same way: `_potential` now credits a tile with
  `fertilized_until_day >= day` at one `FERTILIZER` unit's price, added on top of the existing
  seed-cost floor (independent sunk costs on the same tile, not a `max()` -- both apply). The
  credit lapses when the bonus does (3 days later); by then the doubled watering should have
  grown `yield_units` past it for a smooth handoff, or if the window was wasted, Phi drops for
  real -- a deserved penalty, same as letting a crop die from neglect.
- **`BUY_LAND` — audited, deliberately left alone, flagged instead of fixed.** Its $1k/$2k/$4k
  cost has no Phi credit, but this isn't the same bug: seeds/crops/produce/fertilizer all have
  concrete present value Phi was just failing to count, while land's payoff is genuinely
  prospective (spread across many future turns) with no honest way to price it today -- crediting
  it would be guessing, not fixing a measurement artifact, and a real player faces the identical
  up-front-cost-vs-uncertain-future trade-off. Flagged instead: at `/100` scale that's a -10 to
  -40 reward swing in one turn, 10-40x any other single event in this reward function -- a real
  risk to training stability distinct from "fairness," untested at this scale. Revisit if
  `BUY_LAND` never gets explored in practice.

Not yet validated with a training run -- see checkpoint status above once one is in.

**Update (terminal relative-outcome bonus -- reward now knows about winning and losing):**
every term in the reward up to this point is purely about our own money -- the actual win
condition (data/README.md: "whoever has the most money in the bank at the end... wins") is
relative to the opponent, and nothing compares to them anywhere. Checked whether Kaggriculture's
own discarded `state.reward` could help here instead: confirmed in kaggriculture.py it's just
`float(own money)` on the terminal step (`s.reward = float(obs0.farms[s.observation.player]
["money"])`) -- the same absolute quantity `env.py` already computes per-turn, no opponent
comparison baked in anywhere in the game's own reward either. Added `RELATIVE_OUTCOME_SCALE *
(own money - opponent's money) / 100` as a new term, applied only on the terminal step -- positive
if the episode ends as a win, negative if a loss, magnitude scaled by margin. This does *not* mean
the policy observes or imitates the opponent's actions (still not observed, see the imitation-
learning open question below) -- it means the actual outcome of the match now shapes the reward,
and standard RL credit assignment (the value function, via GAE) propagates "this trajectory led to
a win" back across the whole episode without needing behavioral cloning. Doesn't touch the dense
per-turn shaping built up over every update above -- purely additive, only nonzero once per
episode. Not yet validated with a training run.

**Update (evaluation methodology fix -- absolute money isn't comparable across episodes, and a
self-play trap to remember before ever trying it):** every `analyze_actions.py` invocation in this
project's history used `--seed None` (random), and every "final money ~X" comparison between
checkpoints implicitly assumed that number reflected policy quality. It doesn't, fully: per
data/README.md, town shops unlock on a random per-episode draw, and total demand (so attainable
money) varies a lot with that draw -- comparing two runs that faced different draws is
confounded by luck, not just skill. Fixed `analyze_actions.py`'s `--seed` to default to a fixed
value (`0`, not random) and added a win/loss/tie + mean-money summary computed *within* each
episode (own money vs. the same-episode opponent, the only comparison the random draw doesn't
poison) -- `collect()` now returns per-episode outcomes alongside the row log. Past "final money
~X" figures elsewhere in this file should be read as directional, not exact -- they weren't
seed-controlled.

Also documented (not yet relevant -- `--opponent` still defaults to `starter`, not a checkpoint,
so this project isn't doing self-play): a known self-play failure mode worth remembering before
ever pointing `--opponent` at a checkpoint. In a shared market, both copies of a self-play pair
can raise their *absolute* score by producing less (less supply relieves both copies' own price
pressure) while one of them is actually losing the ladder -- rank self-play checkpoints by win
rate, never by absolute self-play score, and treat "the opponent's absolute score also rose"
during training as a warning sign of the same pathology, not a good sign.

**Update (VecNormalize for reward scale -- the untuned-guess open question below, finally
addressed):** the "Reward scale" open question (below) flagged the `/100` divisor and lack of
`VecNormalize` as untuned guesses from the start. Given evidence since then (`BUY_LAND`'s -10 to
-40 one-turn hit vs. ~0.01-1.0 for everything else), added `VecNormalize(norm_obs=False,
norm_reward=True)` around both `train_env` and `eval_env` in `train.py`. Reward is now divided by
a running estimate of its discounted-return std instead of the fixed `/100`, so it adapts instead
of needing every term's scale guessed right upfront (including future ones -- animals, hands).
Deliberately left `norm_obs=False`: `features.py` already hand-normalizes every observation field,
and reward scale was the specific, evidenced problem -- changing both at once would make it
impossible to tell which change did what. Reward normalization doesn't touch what a trained policy
outputs (only what the value function is fit against during training), so it's invisible to
inference -- `checkpoint.py`/`play.py`/`analyze_actions.py` needed no changes, verified by running
`play.py` against a `VecNormalize`-trained checkpoint. `train.py` now also saves/loads
`vecnormalize.pkl` alongside the model (`--resume-from` picks it up automatically if present next
to the given checkpoint, otherwise starts fresh and says so -- verified both paths with a smoke
test). `Monitor` still sits *inside* the `VecNormalize` wrapping, so tensorboard's
`rollout/ep_rew_mean` keeps logging true, un-normalized reward -- every `ep_rew_mean` figure
elsewhere in this file stays comparable to whatever comes out of a `VecNormalize` run. Not yet
validated with a training run at scale.

**Update (no-op audit -- a new tool finds a real, previously invisible gap):** built
`noop_audit.py`, which hooks kaggriculture.py's own `_apply_unit_action` and diffs farm+private
state either side of every call -- a fundamentally different question from `analyze_actions.py`
("did the engine actually do anything", not "which op did the policy pick"). Sanity-checked
against the two built-in agents first: `starter` wastes 0% (it only ever issues actions it's
already verified), `random` wastes ~53%, heavily concentrated in `PLANT`/`WATER`/`HARVEST`
(~95-100% no-op each) -- both numbers line up with what's expected from a scripted-verified vs.
a blind-random agent, so the tool itself checks out before trusting it on our own checkpoints.

Run against `checkpoints/v2` (stochastic, vs. `starter`): 58.1% of all real (non-`PASS`) actions
did nothing, and the breakdown pointed at something specific -- `DROP` 100% no-op (46/46),
`PICKUP` 97.9% no-op (46/47), both far higher than `PLANT`/`HARVEST`/`WATER`'s already-bad
75-85%. Traced to a positional precondition neither op's no-op rate had any other explanation
for: confirmed in kaggriculture.py, both `PICKUP` and `DROP` call `_is_shed_adjacent((fx, fy),
board_size)` and silently return if the farmer isn't standing on one of the four fixed
shed-access tiles (`(half-1,half-1)`, `(half,half-1)`, `(half-1,half)`, `(half,half)` for
`half = board_size // 2` -- data/README.md, `_shed_access_tiles()`) -- and nothing in the
observation ever exposed whether the farmer was on one of those tiles. The policy had no way to
learn "go here first" because it couldn't see where "here" was relative to its own position, only
infer it indirectly (if at all) from having tried `PICKUP`/`DROP` there before. Fixed by adding a
`SHED_ADJACENT_TILES` constant (computed with the same formula, so it stays correct if
`boardSize` is ever configured away from 10) and a `near_shed` boolean scalar to
`_encode_vector` -- a single flag, not a spatial grid channel, since the four tiles are static
and the question ("can I `PICKUP`/`DROP` right now") is exactly a scalar, not something that
needs per-tile resolution. `VECTOR_DIM` grew by 1 -- another observation-space break, same
failure mode as prior ones. Not yet validated with a training run; the currently-running
`VecNormalize` run above predates this fix and will still show the same `PICKUP`/`DROP` gap.

**Update (tested per the guiding principle above -- fix confirmed wired in, not yet confirmed
learned):** retrained `checkpoints/v2` from scratch with `VecNormalize` and the `near_shed` fix
both in place (500k timesteps, `ent_coef=0.005`; `rollout/ep_rew_mean` -142 -> -140, a real if
small recovery from a mid-run dip to -195, better trajectory than the `VecNormalize`-only run
above which ended at -209 without ever recovering). Re-ran `noop_audit.py` (stochastic, vs.
`starter`) as promised rather than assuming the code change was enough: `DROP` still 100% no-op
(69/69), `PICKUP` only marginally better (94.1% vs. the earlier 97.9%) -- total waste actually
rose to 67.8% (from 58.1%). The fix is correctly wired (`near_shed` is in the observation,
confirmed by `test_shed_adjacent_tiles_matches_the_engine` and the smoke tests when it was
built) but 500k timesteps from scratch isn't enough for the policy to have learned to navigate to
a shed-access tile *before* attempting `PICKUP`/`DROP` -- the same pattern `BUY_SEED` showed
before `IDLE_PENALTY`, and that even `IDLE_PENALTY` needed real timesteps (not just the
mechanism existing) to turn into an actual preference. Not a failure of the fix, just not yet
enough data to confirm it helps -- next continued-training round is the actual test.

## Open questions / not yet solved

- **Submission format**: Kaggle's runner loads `main.py` standalone; it's not
  guaranteed to have `torch`/`stable-baselines3` available. Once a checkpoint
  is worth submitting, it needs an export step — e.g. dump the policy's
  weights and hand-write a numpy-only forward pass in `main.py`, or bundle a
  minimal torch-only inference path via `submission.tar.gz`. Not built yet.
- **Reward scale**: the `/100` divisor in `env.py` is still a hand-picked constant for the
  per-turn money term specifically (unchanged), but `VecNormalize` now handles the *overall*
  scale adaptively on top of it (see the checkpoint-status update above) — revisit the `/100`
  itself only if `VecNormalize`'s stats take unreasonably long to stabilize around it.
- **Self-play stability**: `make_opponent` in `train.py` supports pointing at
  a previous checkpoint, but there's no opponent pool/league yet — training
  against a single frozen past self can still cycle/collapse.
- **Imitation learning from `starter_agent`**: `kaggle_environments`' built-in
  `starter` opponent (already the default `--opponent`, see
  `.venv/.../kaggriculture/kaggriculture.py:1041`) is a working scripted
  buy→plant→water→harvest→sell loop (a "carrot loop"). Just training *against*
  it doesn't transfer that behavior — PPO never observes the opponent's chosen
  actions, only the shared environment outcome. Behavioral cloning (record
  `starter_agent`'s (obs, action) pairs, supervised-pretrain the policy to
  imitate them, then PPO fine-tune from that checkpoint) would sidestep the
  exploration/credit-assignment struggle the reward-shaping updates above have
  been fighting, but hasn't been tried — deliberately deferred in favor of
  continuing the pure-reward-shaping approach, revisit if that keeps stalling.
- **Training instability at `ent_coef=0.01`**: `rollout/ep_rew_mean` oscillates
  strongly (peaks ~-10, valleys ~-230) throughout long continued runs instead of
  converging — see the `NEGLECT_PENALTY` update above. Untried: lower `ent_coef`
  (0.003–0.005) for continued runs once a decent checkpoint is reached, to let
  gains stick instead of getting perturbed away.

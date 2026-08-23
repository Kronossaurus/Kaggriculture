"""Kaggriculture submission entry point.

Kaggle loads this file and calls `agent(obs)` (or `agent(obs, config)`) once
per turn for whichever player index we've been assigned. Keep this file
self-contained (or bundle extra modules into a tar.gz with this at the root
per AGENTS.md) since it's shipped standalone to the competition runner.

Strategy ("farm maxxer"): a scripted multi-unit farm loop built on top of
`reference/melon_maxxer.py` (Kaggle's own official example agent), fixing its
three big gaps -- it never expands land or hires hands (this agent does
both continuously; hiring is fib(n) dollars for the n-th hire that day,
essentially free), it only ever grows MELON (this agent grows a mix, see
Crop mix below), and it never fertilizes (this agent fertilizes every
one-time crop, except MELON, whose unfertilized yield cap is already
reachable by watering alone).

This docstring is a condensed index of current design decisions and the
ideas already tried and ruled out, so they aren't blindly retried. Exact
A/B numbers, replay traces, and full attempt-by-attempt narratives live in
git history and this project's session memory (see MEMORY.md), not here --
this only tracks what still affects how the code should be read or changed.

**Methodology, load-bearing for everything below.** A single 10-seed
`pool_winrate.py` run has a real noise floor around +/-2 seeds out of 10.
Trust a 30-seed run (or a same-process A/B) over any single small run.
Money margin against one opponent isn't the same signal as win rate --
validate both, against a panel with genuinely close matchups, not just
lopsided ones. Many "obviously safe" tweaks in this project's history
measured as real regressions despite looking correct on paper; nothing
here is kept on faith. When a fix regresses, root-cause it with a replay
trace before reverting -- see the ledger below for how often the real
mechanism turned out to be something other than the obvious guess.

## Current mechanisms

**Selling** (`_build_sell_orders`): dumps the full shed once a crop's price
clears its sell threshold (a per-turn sell cap was tried and lost 19/20
seeds -- delayed reinvestment costs more than the occasional price dip).
FERTILIZER is also sellable (a real revenue source against animal-heavy
opponents). `SHED_OVERFLOW_GUARD` bypasses the price threshold before the
shed overflows and produce gets discarded.

**Animals** (`_AnimalPlanner`, GOOSE/COW/SHEEP): every unit shares one
decision function across crop and animal work, FEED as the single highest
priority (a missed feed costs the whole animal). `ANIMAL_LAND_SHARE`
targets owned-land share (matching real top-ladder players), capped by
`ANIMAL_MAX_PER_UNIT` so the target never outgrows the crew tending it.
`ANIMAL_TARGET_SHARE` favors COW/SHEEP over GOOSE (better product-value per
production-interval; real top-ladder replays rarely buy GOOSE either).
`_animal_zone` reserves the tiles closest to the shed for animal structures
so they cluster tight instead of competing tile-by-tile with crops.

**Price-aware planting** (`_CropPlanner`): a crop priced below its own sell
threshold is skipped for new planting (`_price_ok`), falling back to
ignoring price only if every plantable crop is depressed. `_score` also
weights each candidate by how far above base its current price sits
(`PRICE_ATTRACTIVENESS_WEIGHT`), so a crop trading well pulls ahead of
target-mix-share gap alone.

**Land / hands**: `_build_land_orders` gates each `BUY_LAND` on
`LAND_UTILIZATION_GATE` of currently-owned land already being planted, so
cash doesn't buy land faster than it can be staffed (tuned between two
failure modes found in real replays: too high deadlocks below this agent's
own achievable utilization ceiling, too low regresses money by enabling
oversupply before the crew can work the new land). `LAND_QUADRANT_CAP = 3`
-- this agent never buys the 4th quadrant at all (matches the pool's
strongest opponent and a top-ladder replay pattern; see the ledger's
harvest-timing land-purchase entry for why). `_hands_target` uses
`HANDS_BASE + HANDS_PER_QUADRANT * quadrants`, calibrated off real
top-ladder hiring curves rather than a flat formula. `_quadrant_utilization`
inflates a fresh PLANT candidate's effective distance by how built-up its
quadrant already is, so a same-distance tie favors the neglected quadrant
(crops were clustering near the shed otherwise).

**Crop mix** (`_CropPlanner`, `_crop_target_share`): `CROP_TARGET_SHARE` is
MELON-heavy at the season's start (MELON's high base price dwarfs the
others, with the price-crash risk already handled by price-aware planting
above), but STRAWBERRY's own target share and seed reserve now *ramp up*
over time (`STRAWBERRY_LATE_SHARE`/`CROP_MIX_RAMP_START_DAY`/
`CROP_MIX_RAMP_DURATION`/`STRAWBERRY_RESERVE_LATE`, see
`_crop_target_share`/`_strawberry_seed_reserve`) -- real top-ladder players
lean heavily on STRAWBERRY (an "ongoing" crop, no replanting between
harvests) but a flat season-long share can't fund that without starving
early cash flow (STRAWBERRY's seed costs 10x WHEAT's with a 5x longer
payback delay); ramping the target up only once the early game has had
time to compound cash cleared the biggest single validated win of this
project (see the ledger). TOMATO/STRAWBERRY are "ongoing" crops (`_last_
yield_day` extends their fertilization window to their real last
production day, not just the placeholder `max_yield_day`).

**Endgame liquidation**: as `days_remaining` shrinks, this agent stops
buying seeds/land that can't pay back in time, prioritizes walking carried
produce home over watering a crop that won't finish, and drops the
sell-price threshold entirely (unsold inventory doesn't count toward final
score). Carried FERTILIZER stops being re-fetched once liquidating and
counts as sellable produce instead. WHEAT's feed reserve is exempted --
missing a feed this late still loses the whole animal. A carried WHEAT
batch below `WHEAT_SHED_RETURN_THRESHOLD` DROPs at the shed once liquidating
*unless* some live animal still needs it (`_any_unfed_animal`) -- two
earlier, less careful versions of this exact rule regressed (see the
ledger).

**Neglected-tile urgency / harvest-ready tiles rotting to weeds**: a tile
that missed watering once (`consecutive_unwatered >= 1`) scores priority 0.5
in `_build_tile_jobs`, above routine chores but below at-risk harvest --
two consecutive misses turns a PLANT into a WEED, and kaggriculture.py's
daily refresh doesn't exempt already-ripe plants from that clock either.
`HARVEST_AT_RISK_DIST_BIAS` biases distance (not priority, already top
tier) so an at-risk harvestable tile is served ahead of a merely-closer,
non-urgent one. `LAST_CHANCE_HOURS` escalates at-risk water to top priority
only in the last few hours of the day (a full-day tie regressed by
competing with genuinely urgent harvests all day; a last-turn-only version
was briefly mistaken for a reachability ceiling but was just a zero-buffer
test artifact). `HARVEST_SAFE_PRIORITY`/`WATER_YIELD_INERT_PRIORITY`
(MC-searched together with `HANDS_TARGET_Q1`) deprioritize a currently-safe
harvestable tile and a still-growing crop before its yield-bonus window,
respectively -- both regressed when tried alone or exactly tied with
at-risk water's own priority (ambiguous contention, not a bad idea); a
combination search with genuinely different values found a clean win (see
the ledger).

**Shed-return priority**: normally the worst tier (below routine watering/
weeding) -- fine under a diversified mix, but a unit carrying produce under
a dense crop never won the race home against a steady stream of harvest
candidates (see the STRAWBERRY ledger entry). `SHED_RETURN_URGENT_PRIORITY`
escalates only once a unit carries at least `CARRIED_PRODUCE_URGENT_
THRESHOLD` units -- a light load still finishes a nearby chore first, a
real backlog heads home (an unconditional escalation regressed: see the
ledger).

**Seed reserve** (`_build_seed_orders`): `SEED_RESERVE` scales up per-crop
by `SEED_RESERVE_PRICE_CAP` when that crop's own price sits above base
(mirrors the crop planner's own price-attractiveness logic, applied to how
far ahead of need to buy). STRAWBERRY's own reserve additionally ramps with
its target share (see Crop mix above).

**Fertilizer buying** (`_build_fertilizer_orders`): a flat `FERTILIZER_
RESERVE` most of the time, stocking up to `FERTILIZER_STOCKPILE_RESERVE`
whenever price drops to `FERTILIZER_CHEAP_PRICE_FRAC` of base (below
`SELL_PARAMS["FERTILIZER"]`'s own sell threshold, so it never buys back at
a price it would also sell at). Kept as a small, gated, provably-safe
change even though it's measured as a no-op so far (the stockpile order
itself is a casualty of the market-order-slot-starvation mechanism below).

**Harvest timing** (`_is_harvestable`): a one-time crop keeps growing
`yield_units` for every watered day from `first_yield_day` through
`max_yield_day` (confirmed directly against kaggriculture.py, not just
data/README.md) -- harvesting the instant it's legal used to forfeit up to
4-6x the yield on WHEAT specifically, this agent's own profit centre (see
the ledger's WHEAT-throughput entry). `_is_harvestable` now defers a
one-time crop past `first_yield_day` until `age >= max_yield_day`,
`yield_units` already at cap, or `liquidating`, reusing the existing
WATER/FERTILIZE fallback branches unchanged so a deferred tile stays
tended while it waits. MELON is unaffected in practice (its own bonus
window already saturates at or before `first_yield_day` under normal
watering).

**Routing efficiency** (`_build_tile_jobs`/`_assign_tile_jobs`): tile-only
candidates (harvest/water/weed/animal-upkeep/plant/build -- anything any
unit could do) are built once per turn as a shared pool, then matched
globally against every still-unresolved unit, instead of each unit
rescanning the whole board and claiming its own best candidate in a fixed,
order-dependent sequence. Inventory-dependent candidates (feed/fertilizer-
delivery/animal-token-placement/shed-return) stay per-unit
(`_find_personal_target`) but compete fairly in the same sort as pool jobs.
A real, measured efficiency gain (crew work-time share rose meaningfully,
confirmed via an action-mix trace, not just assumed) -- see the ledger for
why this alone didn't close a separate reactive-opponent gap it was hoped
to fix.

**Cash-budget threading** (`_orders_cash_cost`): the game engine checks
`farm["money"] < price` live, per unit, in submitted order-list order --
not a static per-turn snapshot -- so a builder late in `agent()`'s own
sell/land/hire/feed/seed/fertilizer/animal sequence checking the raw
balance could request a purchase the engine would silently reject once
earlier orders had already drawn it down. A running `cash_reserved`
accumulator (SELL orders subtract, i.e. add to budget; BUY-side orders use
a real per-unit price-walk estimate via `_sell_proceeds`/`_buy_cost`, not a
flat `qty * price`) is threaded through every builder from feed onward, and
now covers land purchases too. `HIRE`'s own order-building only requests
what it can actually afford, mirroring the same pattern.

## Don't retry without new evidence

Compact ledger of ideas already tried and ruled out (or narrowly kept) --
see git history and this project's session memory for the full numbers.

- **Animal placement**: dedicating fixed hand capacity to animals only
  (severe regression); reserving more near-shed land ahead of strict
  animal-structure need, at two magnitudes (structural cost, not a tuning
  miss -- crop planting loses more than drift saves); raising `ANIMAL_
  LAND_SHARE`/`ANIMAL_MAX_PER_UNIT` to chase a real animal-share/win-rate
  correlation in opponent replays (correlation real, not simply causal).
- **Hiring**: sizing hand count off a live actionable-tile count instead
  of the flat quadrant formula (over-hires during backlog spikes); a
  competitor-inspired aggressive-then-workload-scaled dynamic formula
  (crashes crew right when the single-quadrant opening still needs full
  staffing) -- three independent formulas, same failure mode, this agent
  can't run its opening on meaningfully less crew than the current flat
  formula.
- **Crop mix**: shifting `CROP_TARGET_SHARE` hard toward STRAWBERRY as a
  flat, season-long constant, twice (produce piling up in carried
  inventory unable to reach the shed, then -- after fixing that -- a
  capital-allocation/cash-flow-timing problem a flat share can't express).
  **Resolved**: a time-varying ramp (see Crop mix above) fixed it cleanly,
  the biggest validated win of the project so far.
- **WHEAT market-order-slot starvation** (`_build_feed_orders` routinely
  crowding out `BUY_SEED`/fertilizer orders out of the shared 10-order/turn
  cap): eighteen attempts across many sessions. Order consolidation, a
  per-turn quantity cap, list reordering, and a flat sell-side WHEAT
  reserve (three sizes) all regressed via a "standing market price" effect
  -- reducing this agent's own WHEAT buy/sell volume raises WHEAT's price
  against itself more than it saves. Directly including WHEAT in the
  normal carried-produce shed-return mechanic (removing its special-case
  exclusion) regressed via a different mechanism: weed count exploded, as
  nearly every unit carries a little WHEAT nearly always. A consolidated
  feed order alone (the mechanistically correct fix) freed the market
  slots exactly as designed but exposed a second bottleneck: it reduces
  this agent's own animal-byproduct supply just enough to raise MILK/WOOL/
  FERTILIZER prices, which favors animal-heavy real opponents more than it
  helps this agent. Three independent rescue mechanisms for that
  spillover -- a crew-priority floor for animal upkeep, and two cash-budget
  threading variants (one with an accounting bug that accidentally helped
  by re-throttling WHEAT activity, one correctly fixed) -- each failed to
  rescue the consolidation against reactive opponents, though each
  survives unpaired as a genuine narrow correctness fix (the priority
  floor, and the corrected cash-budget threading, both kept). A carried-
  inventory-inclusive WHEAT accounting fix (closing a real bug: feed
  reserve was compared only against shed stock, not carried inventory too)
  mechanically worked better than any prior attempt but regressed hardest
  of all once measured against reactive opponents: WHEAT market throughput
  turned out to be this agent's single largest profit centre by a wide
  margin, so cutting the over-buying that funds it is a real, direct loss,
  not a side effect any scheduling fix can route around. **Settled**: the
  cost is WHEAT/seed volume itself moving the shared market whenever an
  opponent can react to it, not a side-channel resource-contention effect
  -- any future attempt needs a lever that changes how much aggregate
  WHEAT/seed activity increases, not just how the freed capacity gets
  reallocated. A final attempt (X594-kernel-inspired: standing sell-side
  reserve + carried-inclusive consolidated feed order together, the exact
  combination the first seventeen attempts only ever tried individually)
  still regresses against reactive opponents specifically -- **kept in the
  codebase anyway per explicit user direction**, an open thread, not
  settled; see this project's session memory before touching this area
  again.
- **Routing, following from the harvest-timing/weed-loss problem**: a
  per-hand territory-anchor distance bias measured as noisy and
  non-robust (removed, replaced by the global assignment above); the
  global assignment itself is a real, robust efficiency gain but did not
  close a separate near-total collapse against one reactive sentinel --
  strong evidence that specific gap isn't really about routing at all (see
  the WHEAT-consolidation entry above for the likely mechanism). A
  watering-capacity throttle (competitor-inspired, cap standing crop tiles
  by crew size) is a total no-op at every value except one that also
  regresses -- this agent's tiles/crew ratio never binds against it. A
  soft, threshold-gated target-commitment bias (protect a unit's in-
  progress target once several turns are invested) regressed -- this
  agent's routing needs full per-turn reactivity, not commitment, even for
  the minority of switches that look like wasted investment from outside
  the simulation.
- **Quadrant-balance for planting** (the 3rd-bought quadrant plateaus at
  roughly half the 2nd-bought one's planted-tile occupancy, for its whole
  remaining lifetime, with a *lower* weed rate ruling out neglect):
  scaling up the existing anti-clustering distance bias regressed on the
  standard protocol despite promising small-pool search results, and a
  paired-seed trace falsified both obvious follow-up hypotheses (hired
  hands do not carry positional advantage between days -- they respawn
  symmetrically at the shed every morning; seed inventory stays healthy in
  both runs) -- no clean mechanism found yet, most likely just the
  inherent sensitivity of the greedy per-turn scorer to any single
  constant change compounding over many days.
- **Land-purchase timing** (after the harvest-timing fix delayed this
  agent's own cash flow, delaying the 4th-quadrant purchase): loosening
  `LAND_UTILIZATION_GATE` did nothing (confirms money, not utilization, was
  the real constraint); counting standing (unharvested) crop value as
  spendable cash for the land gate regressed (commits to a purchase against
  value that isn't liquid yet); threading land into the same-turn cash
  budget alone regressed further, in the opposite direction than expected
  (land now also wins the race against hire/seed for arriving cash, a slow
  cumulative drag, not a single dramatic loss). **Fixed instead by
  sidestepping it**: `LAND_QUADRANT_CAP = 3` (never buy the 4th quadrant),
  borrowed from a competitor reference agent and a top-ladder replay
  pattern -- removes the purchase that was straining cash flow rather than
  fixing the strain itself, kept alongside three smaller, principled (if
  measured-inert) precision improvements it made room for (an operation-
  scaled cash reserve for the land gate, hiring built before land in
  `agent()`'s sequence, and the per-unit sell-proceeds estimate above
  extended to land's own affordability check).

Still out of scope: any opponent modeling -- this agent only ever reacts
to its own farm/market state. See rl/README.md for the parallel (currently
stalled) attempt at a learned policy.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_I0,
    MARKET_PARAMS,
    PRICE_FLOOR,
    market_price,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

CASH_CROPS = ("MELON", "WHEAT", "CARROT", "TOMATO", "STRAWBERRY")
CROP_TARGET_SHARE = {"MELON": 0.45, "WHEAT": 0.25, "CARROT": 0.15, "TOMATO": 0.10, "STRAWBERRY": 0.05}
# [[strawberry-shift-throughput-gap]]: two straight attempts to shift this flat, season-long dict
# toward real-ladder players' ~64% STRAWBERRY share regressed hard (47.8%->25.6% on the second,
# post-throughput-fix attempt) via a capital-allocation/cash-flow-timing mechanism a *flat* share
# can't express -- STRAWBERRY's $100 seed (10x WHEAT's $10) plus its first_yield_day=10 payback
# delay (vs WHEAT's 2) means committing to it early starves the fast-compounding-cash engine this
# agent's early game depends on. STRAWBERRY_LATE_SHARE/CROP_MIX_RAMP_START_DAY/
# CROP_MIX_RAMP_DURATION (below) let the target ramp up *after* the early game has had time to
# compound cash, instead of holding one value the whole season. Hand-picked first (0.45/day
# 10/10-day ramp) already cleared +9.5pp on the full 30-seed pool with every opponent improving
# or tying, none regressing (52.9%->62.4%) -- then MC-searched (scripts/mc_search.py) for a
# better combination; the search consistently favored an *earlier* ramp start (day 2-10) and a
# *short* duration (2-8 days) over the hand-picked 10/10, up to 0.84 win rate on the search pool
# vs. ~0.5-0.6 for later/slower starts.
STRAWBERRY_LATE_SHARE = 0.478  # STRAWBERRY's target share once the mix ramp fully completes --
    # the other 4 crops' shares scale down proportionally to keep the mix summing to 1 (see
    # _crop_target_share), same relative balance among themselves as CROP_TARGET_SHARE's baseline.
CROP_MIX_RAMP_START_DAY = 8  # day the mix starts shifting toward STRAWBERRY_LATE_SHARE --
    # MC-searched; the region 2-10 all scored well, so this isn't a sharp optimum, just the top
    # of the search (see the constant block's comment above for the full result).
CROP_MIX_RAMP_DURATION = 2  # days over which the shift completes, linearly, starting at
    # CROP_MIX_RAMP_START_DAY -- 0 means an instant step at that day instead of a ramp.
STRAWBERRY_RESERVE_LATE = 3  # SEED_RESERVE["STRAWBERRY"] ramps toward this on the same schedule
    # as the target-share ramp above -- without also raising the seed *reserve*,
    # _build_seed_orders would never stock enough STRAWBERRY seed to actually reach a higher
    # target share regardless of what _CropPlanner.choose() wants to plant. Barely above the
    # unramped baseline (SEED_RESERVE["STRAWBERRY"]==2) -- the search found low reserve values
    # winning consistently, likely because a small standing reserve restocked continuously each
    # day still sustains high throughput over a season, without tying up cash in idle seed stock.
FERTILIZE_CROPS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY"}
PRICE_ATTRACTIVENESS_WEIGHT = 0.26  # was 0.1; how much a crop's current price (as a ratio
                                     # of its base) pulls _CropPlanner.choose() toward it,
                                     # on top of the target-mix-share gap -- see the module
                                     # docstring's Price-aware planting note for the
                                     # original tuning, and its Harvest-timing note's
                                     # MC-search-tuning follow-up for this value

PLANT_PRIORITY = 4  # _find_target's base priority for planting an empty tile -- see
                     # the module docstring's Land / hands note
PLANT_PRIORITY_UNDER_GATE = 1.5  # used instead of PLANT_PRIORITY while owned land is
                                  # under LAND_UTILIZATION_GATE -- still below harvest (0)
                                  # and at-risk water (0.5) so a dying crop is never
                                  # sacrificed for a new planting, but now competes with
                                  # routine water (1) and beats weeding (2) and fertilizer
                                  # care (3), instead of always losing to them
SHED_RETURN_URGENT_PRIORITY = 2  # _find_target's priority for walking carried produce home
                                  # once carrying at least CARRIED_PRODUCE_URGENT_THRESHOLD
                                  # units -- see the module docstring's Shed-return priority
                                  # note. An earlier unconditional
                                  # version of this (every carrying unit, not just a real
                                  # backlog) regressed; escalating only once a real backlog has
                                  # piled up should let a unit carrying just 1-2 units finish a
                                  # nearby routine chore first (the case that already worked
                                  # fine) while still capping how much value piles up in carry
CARRIED_PRODUCE_URGENT_THRESHOLD = 3  # see SHED_RETURN_URGENT_PRIORITY
HARVEST_AT_RISK_DIST_BIAS = 1000  # subtracted from a harvestable tile's sort_dist when it's one
    # missed watering away from turning into a WEED (see _find_target's HARVEST branch) --
    # large enough to strictly dominate any real on-board distance, so an about-to-expire ripe
    # tile is always served before a merely-closer one still within priority 0. See the module
    # docstring's Harvest-ready tiles rotting to weeds note.
HARVEST_SAFE_PRIORITY = 1.206  # MC-searched (scripts/mc_search.py) -- a NOT-at-risk harvestable
    # tile's priority (at-risk stays 0, unchanged). Deliberately kept > 0.5 (at-risk water's own
    # priority) by construction here so the two never tie -- see
    # [[harvest-safe-deprioritization-tradeoff]] for why an exact tie (the first, hand-picked
    # attempt) regressed by turning a decisive ordering into an ambiguous distance-only
    # contention. Re-searched 2026-08-22 jointly with WATER_YIELD_INERT_PRIORITY and
    # WATER_ROUTINE_PRIORITY below (see that constant's own comment for what motivated adding
    # it) using the two reactive sentinels + main_mc_tuned_priority as the search pool: this
    # combination scored 0.93 there, and confirmed at the full 30-seed default-pool protocol:
    # 71.2%->77.1% overall, every opponent improving or tying, none regressing
    # (main_plant_priority +16.6pp, main_price_reactive_seed_reserve +13.4pp,
    # main_mc_tuned_priority +10.0pp) -- also cut mean weed% on a 20-replay sample 6.8%->4.4%,
    # the actual metric this round of tuning was chasing, not just win rate.
WATER_YIELD_INERT_PRIORITY = 1.387  # MC-searched alongside HARVEST_SAFE_PRIORITY above --
    # priority for a not-yet-at-risk tile whose watering wouldn't add any yield_units yet
    # (_is_yield_inert). See [[water-yield-inert-deprioritization-tradeoff]] for the first,
    # hand-picked attempt at this (value 3 -- barely different in magnitude, but paired with an
    # exactly-tied HARVEST_SAFE_PRIORITY there instead of a properly searched one).
WATER_ROUTINE_PRIORITY = 2.294  # added 2026-08-22: routine (not-yet-at-risk) watering's own
    # priority in _build_tile_jobs (liquidating still forces a flat 5, unrelated). A top-ladder
    # replay comparison found this agent's real end-of-day watering coverage averaging 71.1%
    # (days 10-27) vs a top player's 86.1% -- a real ~15pp gap, unlike DIG rate or MOVE fraction
    # (both comparable) -- while three targeted fixes elsewhere (more crew, fertilizer batching/
    # delivery gating, see [[hands-per-animal-fibonacci-cost-trap]]/
    # [[fertilizer-delivery-gate-inert]]) all failed to move it. Two single-variable MC searches
    # found negative values catastrophic (win rate collapses to ~0.27-0.42 regardless of the
    # other two priorities above -- a hard floor, not tuning noise) and the landscape above 0
    # non-monotonic (hand-picked 1.5/2.6 each showed real trade-offs against one reactive
    # sentinel or the other at the full-pool level). A joint search alongside HARVEST_SAFE_
    # PRIORITY/WATER_YIELD_INERT_PRIORITY found this combination with no such trade-off -- see
    # HARVEST_SAFE_PRIORITY's own comment for the validated result.
WHEAT_SHED_RETURN_THRESHOLD = 5  # minimum carried WHEAT before it's worth a dedicated trip to
    # the shed (_find_personal_target) or before it's eligible to DROP once already there
    # (_decide_unit) -- 12th market-order-slot-starvation attempt. Must exceed the 4-unit cap
    # on a feed PICKUP (see _decide_unit's own PICKUP WHEAT call) or a freshly-picked-up
    # feed batch would immediately qualify for DROP and livelock right back into the shed
    # -- see _decide_unit's carried_produce comment for the full story (this was the actual
    # bug behind the 11th/12th attempts' weed-count explosion, not a trip-generation issue)

SEED_RESERVE = {"MELON": 3, "WHEAT": 3, "CARROT": 3, "TOMATO": 2, "STRAWBERRY": 2}  # baseline,
    # used as-is while a crop's price isn't currently attractive -- see the module
    # docstring's Seed reserve note
SEED_RESERVE_PRICE_CAP = 2.0  # max multiplier on SEED_RESERVE when a crop's current price is
    # attractively above its own base -- see _build_seed_orders. Doubling SEED_RESERVE flatly
    # for every crop was tried and reverted (catastrophic: it blew up the WHEAT sell/rebuy
    # round trip by volume, see the module docstring's WHEAT sell/rebuy round trip note);
    # scaling by price instead means
    # WHEAT -- whose price rarely swings far above base -- stays close to its own baseline
    # reserve, while a crop that's actually worth growing more of (MELON/STRAWBERRY trading
    # well above base) gets more room to stock up
FERTILIZER_RESERVE = 2
FERTILIZER_STOCKPILE_RESERVE = 10  # reserve target while price is cheap -- see the module
                                    # docstring's Price-reactive fertilizer buying note
FERTILIZER_CHEAP_PRICE_FRAC = 0.35  # price <= this fraction of base triggers stockpiling.
                                     # Deliberately well below SELL_PARAMS["FERTILIZER"]'s own
                                     # 0.50 sell threshold -- see the same docstring note
FERTILIZER_MAX_PRICE_FRAC = 1.92  # don't buy fertilizer more than ~92% over base -- was
                                   # 1.147 (before that 1.2; see below for that history). MC
                                   # search (2026-08-20, scripts/mc_search.py, paired jointly
                                   # with SAFETY_RESERVE/SHED_OVERFLOW_GUARD) found 1.147,
                                   # validated at full-pool 30-seed A/B: 71.9%->74.8% win
                                   # rate, +$1.2-1.5k mean margin, improved on both reactive
                                   # self-play sentinels, regressed nowhere. That search
                                   # predates the harvest-timing fix; a follow-up MC search
                                   # (see the module docstring's Harvest-timing note) re-tuned
                                   # this and 9 other constants together on top of the fix +
                                   # all 4 land-purchase levers, found 1.92, validated at
                                   # full-pool 30-seed A/B: 41.3%->46.7%, improves on both
                                   # reactive sentinels again (1W->2W, 3W->8W), regresses
                                   # nowhere -- see that note for the full result table.

SELL_PARAMS = {
    # threshold: fraction of base price required to sell at all -- see
    # _build_sell_orders for why there's no per-turn cap. Shared by
    # _CropPlanner and _AnimalPlanner's own _price_ok, so a crop/animal
    # kind's plant/build eligibility always matches its sell eligibility.
    "MELON":      {"threshold": 0.65},
    "WHEAT":      {"threshold": 0.854},  # was 0.45 -- see the module docstring's Harvest-
                                          # timing note's MC-search-tuning follow-up. A much
                                          # higher bar to sell at all now that a WHEAT
                                          # harvest is worth 4-6x more per tile (deferred to
                                          # max_yield_day): holding out for a better price is
                                          # worth more per unit than it used to be.
    "CARROT":     {"threshold": 0.45},
    "TOMATO":     {"threshold": 0.50},
    "STRAWBERRY": {"threshold": 0.55},
    "MILK":       {"threshold": 0.50},
    "WOOL":       {"threshold": 0.60},  # WOOL's above-equilibrium price shape is nearly
                                         # as steep as MELON's, see the Price-aware
                                         # planting note
    "EGG":        {"threshold": 0.35},
    "FERTILIZER": {"threshold": 0.50},  # collected free from animal structures (COLLECT_FERTILIZER)
                                         # in excess of what crops currently need -- see the module
                                         # docstring's Selling fertilizer note
}

SHED_CAPACITY = 100  # data/README.md default; not exposed in `obs`
SHED_OVERFLOW_GUARD = 0.633  # start dumping regardless of price above this fraction full --
                              # was 0.72 (before that 0.85); see FERTILIZER_MAX_PRICE_FRAC's
                              # note for both MC searches this was tuned in

ANIMAL_KINDS = ("COW", "SHEEP", "GOOSE")
ANIMAL_PRODUCT = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
ANIMAL_STRUCTURE = {"COW": "PASTURE", "SHEEP": "PASTURE", "GOOSE": "COOP"}
ANIMAL_PRODUCTS = ("MILK", "WOOL", "EGG")  # finished products a unit carries to the shed
ANIMAL_TARGET_SHARE = {"COW": 0.60, "SHEEP": 0.40, "GOOSE": 0.0}  # GOOSE has the
    # lowest $/day of the three ($50: EGG base 50 / 1-day interval, vs COW's $80 and
    # SHEEP's $66.7) despite being the cheapest to buy -- 5/5 downloaded top-ladder
    # replays (3 different players) never place one; see the module docstring's
    # Animals note
ANIMAL_LAND_SHARE = 0.18  # target fraction of owned tiles reserved for animal
                          # structures -- see the module docstring's Animals note;
                          # top-ladder replays hold this steady around 15-24% throughout
                          # the season. Raising to 0.25 measured worse -- back to 0.18.
ANIMAL_MAX_PER_UNIT = 1.0  # hard cap on live animals, per farmer/hand currently hired --
                            # see the module docstring's Animals note. ANIMAL_LAND_SHARE
                            # alone grows with quadrants bought regardless of crew size,
                            # letting the animal count outgrow what the same crew can
                            # also water/harvest/weed for
ANIMAL_RESERVE = 1  # spare bought-but-not-yet-placed animal tokens to keep queued, per kind --
                    # known to let tokens strand with no empty structure to receive them in
                    # some games; see _build_animal_orders's own "Known bug" note
FEED_RESERVE_PER_ANIMAL = 2  # days of WHEAT buffer per live animal, bought directly from
                              # the market -- see _build_feed_orders for why
FEED_MAX_PRICE_FRAC = 4.22  # was 5.0; a near-nonbinding backstop, not a real cap: missing
                             # a feed loses the whole animal ($300-500+), dwarfing even a
                             # wildly inflated wheat price -- see the module docstring's
                             # WHEAT sell/rebuy round trip note for why a tighter cap is a
                             # trap (below ~1.4x base, animals starve outright in testing).
                             # 4.22 is from the Harvest-timing note's MC-search-tuning
                             # follow-up -- still well above that danger zone, same
                             # near-nonbinding role
WHEAT_SELL_RESERVE_DAYS = 3  # 18th market-order-slot-starvation attempt (see the module
                              # docstring and .omc/plans/ralplan-wheat-feed-consolidation.md):
                              # _build_sell_orders holds back live_animals * min(this,
                              # days_remaining) WHEAT from the sell dump instead of always
                              # dumping the full shed -- mirrors the X594 kernel-pull
                              # reference's own reserve shape, tried here for the first time
                              # paired with _build_feed_orders' carried-inclusive/consolidated
                              # change below rather than alone (three earlier flat-reserve-
                              # alone attempts all regressed, see the module docstring).
                              # Re-tested at 1 (critic-pass suggestion, cheaper than a routing
                              # rework): mixed per-seed (2 of 5 sentinel seeds improved, 2
                              # worsened, 1 flat), aggregate total across the same 5 seeds
                              # slightly *worse* (151,526 vs. 146,914) -- confirms reserve
                              # sizing isn't the regression's primary driver (both are ~40%
                              # below baseline's 255,367 either way). Reverted to 3, the
                              # ground-truthed X594 value, since no deviation from it helps.

FARM_HAND_COST_MULT = 1  # data/README.md default; mirrors kaggriculture.py's own constant,
                          # used only to estimate HIRE's own same-turn cash draw -- see
                          # _orders_cash_cost

HANDS_TARGET_Q1 = 8  # hands specifically at 1 quadrant (25-tile NW-only opening), overriding
    # HANDS_BASE+HANDS_PER_QUADRANT below for that one case (see _hands_target). MC-searched
    # (scripts/mc_search.py) alongside HARVEST_SAFE_PRIORITY/WATER_YIELD_INERT_PRIORITY below --
    # every top candidate across two separate 40-trial searches landed at 7-9, i.e. at or near
    # the un-overridden formula's own value: see [[hands-target-q1-reduction-attempt]], two
    # earlier hand-picked cuts (4, 6) both regressed non-monotonically despite real-ladder data
    # (~4.76 average) and this agent's own idle-time trace suggesting genuine slack existed --
    # the search confirms that slack still isn't safely cuttable, at least not via this
    # constant alone. Kept at 8 (== no actual change from the formula), the priority tuning
    # below is what MC search found actually helps.
HANDS_BASE = 6  # was 2; see the module docstring's Harvest-timing note's MC-search-tuning
                 # follow-up -- found by scripts/mc_search.py paired with the other 9 knobs
                 # below, on top of the harvest-timing fix + all 4 land-purchase levers
HANDS_PER_QUADRANT = 2  # was 3, same search. NOTE: the *design intent* behind this pair --
                         # "picked so _hands_target(quadrants) stays strictly increasing
                         # through all 4 quadrants, so _land_worth_buying never blocks the
                         # 4th" (this comment's own prior text) -- no longer applies:
                         # LAND_QUADRANT_CAP=3 means quadrants never reaches 4 at all. See
                         # MAX_HANDS below for how that's reconciled, not left conflicting.
MAX_HANDS = 16  # the raw MC search value, kept as-searched rather than reconciled down to
                 # 12: at HANDS_BASE=6, HANDS_PER_QUADRANT=2, LAND_QUADRANT_CAP=3,
                 # _hands_target(quadrants) is 8/10/12 at 1/2/3 quadrants and never goes
                 # higher today (4 quadrants is never reached), so 16 is untested headroom
                 # past a ceiling that can't currently be hit -- confirmed behaviorally
                 # identical to 12 (byte-identical replay, same seed/opponent, both values).
                 # Deliberately left at 16 rather than 12: harmless now, and ready without
                 # a further edit if LAND_QUADRANT_CAP or the hands formula ever changes.
                 # (History: was 6, briefly 8, then 12 -- see the module docstring's Land /
                 # hands note for that tuning; RE-TESTED at 14 in the 18th attempt and
                 # reverted, worse across the board -- see .omc/plans/ralplan-wheat-feed-
                 # consolidation.md. That history predates LAND_QUADRANT_CAP and the
                 # harvest-timing fix, so treat it as background, not a constraint on
                 # retuning this value again if the land cap or hands formula ever changes.)
SAFETY_RESERVE = 561  # cash kept in reserve before optional buys/land/fertilizer -- was 147
                       # (before that, 300; see FERTILIZER_MAX_PRICE_FRAC's note for that
                       # earlier MC search). This new value is from the module docstring's
                       # Harvest-timing note's MC-search-tuning follow-up: with the
                       # harvest-timing fix holding wheat longer before selling, real cash
                       # sits lower for longer between sales, so a much bigger reserve is
                       # needed before optional buys, not a smaller one.
MAX_MARKET_ORDERS = 10  # data/README.md default; not exposed in `obs`

DEFAULT_EPISODE_STEPS = 720  # data/README.md defaults, used if `config` is unavailable
DEFAULT_TURNS_PER_DAY = 24
SEED_BUFFER_DAYS = 1  # a crop needs first_yield_day + this many days left to be worth planting
LAND_CUTOFF_DAYS = 5  # stop buying new land this close to the end, it won't pay back in time
LAND_UTILIZATION_GATE = 0.493  # was 0.50; don't buy the next quadrant until this much of
                                # what we already own is planted -- see the module
                                # docstring's Land / hands note for the original tuning
                                # (0.6 deadlocked against a ~48-56% achievable-utilization
                                # ceiling in real replays). This value is from the Harvest-
                                # timing note's MC-search-tuning follow-up -- barely moved
                                # from 0.50, well within the same safe range.
LAND_QUADRANT_CAP = 3  # never buy the 4th quadrant -- see the module docstring's Harvest-timing
                        # note's land-purchase-delay follow-up: the X578/X594 kernel-pull
                        # reference never attempts this purchase at all (`n_unl < 3` in its own
                        # land logic), sidestepping the exact regression three narrower attempts
                        # here all failed to fix. Lever 1 of 4 from that reference -- kept, tested
                        # in isolation (10-seed pool: money up in every reactive matchup, flat
                        # elsewhere, win rate unchanged). Levers 2-4 (dynamic reserve, hire-before-
                        # land ordering, per-unit sell simulation) tested one at a time, then
                        # combined -- see LAND_RESERVE_PER_ANIMAL/LAND_RESERVE_FLAT for lever 2.
LAND_RESERVE_PER_ANIMAL = 45  # lever 2 of 4, see _land_cash_reserve -- X594's own value, not
                               # yet retuned for this agent's economy
LAND_RESERVE_FLAT = 60  # lever 2 of 4, see _land_cash_reserve -- X594's own value, not yet
                         # retuned for this agent's economy
SELL_PROCEEDS_DISCOUNT = 0.9  # lever 4 of 4, see _sell_proceeds -- X594's own value, not yet
                               # retuned for this agent's economy
ENDGAME_LIQUIDATION_DAYS = 2  # ignore the sell-price threshold this close to the end
LAST_CHANCE_HOURS = 4  # see [[harvest-early-at-risk-salvage-regression]]'s last-turn-of-day
                        # variant: hour == turns_per_day - 1 left zero turns for a hand not
                        # already standing on the tile to walk over, so it could only ever have
                        # been a no-op regardless of crew capacity -- this widens the window to
                        # give a hand within ~this many tiles (our own pasture cluster's own
                        # measured max distance from spawn was ~4.12, see the same memory) an
                        # actual chance to reach it before end of day


def _cfg_get(config, key, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _step_toward(fx, fy, tx, ty):
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"
    return None


def _shed_adjacent_tiles(board_size):
    half = board_size // 2
    return {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}


def _dist_to_shed(x, y, board_size):
    return min(abs(x - sx) + abs(y - sy) for sx, sy in _shed_adjacent_tiles(board_size))


def _quadrant_of(x, y, board_size):
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def _quadrant_utilization(farm, board_size):
    """Fraction of each unlocked quadrant's own tiles currently in use
    (planted, or holding an animal structure) -- used to steer new
    plantings toward whichever quadrant is most neglected, see the module
    docstring's spatial clustering (crops) note. A locked quadrant's
    utilization is irrelevant (no empty-tile candidate is ever generated
    there, its tiles are the "LOCKED" string, not None) and defaults to 0.0
    for the div-by-zero case, which never actually gets used."""
    used = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}
    total = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if tile == "LOCKED":
                continue
            q = _quadrant_of(x, y, board_size)
            total[q] += 1
            if _is_plant(tile) or (isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP")):
                used[q] += 1
    return {q: (used[q] / total[q] if total[q] else 0.0) for q in used}


def _animal_zone(farm, board_size, needed):
    """The `needed` empty tiles closest to the shed -- animal structures
    claim this zone first, leaving the rest of the map free for crops to
    spread into, instead of both competing for whichever empty tile happens
    to be nearest the unit that finds it first. See the module docstring's
    Animals note: real top-ladder play keeps every
    animal structure in a tight ring around the shed (FEED is daily and
    unmissable, so a short walk there matters), with crops filling the rest
    of the map edge-to-edge."""
    if needed <= 0:
        return set()
    empties = [(x, y) for y in range(board_size) for x in range(board_size) if farm["tiles"][y][x] is None]
    empties.sort(key=lambda p: _dist_to_shed(p[0], p[1], board_size))
    return set(empties[:needed])


def _is_plant(tile):
    return isinstance(tile, dict) and tile.get("kind") == "PLANT"


def _is_weed(tile):
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def _last_yield_day(crop):
    """The last day a crop can still produce, i.e. the true upper bound for
    _is_fertilizable -- CROPS[crop]["max_yield_day"] is only that for
    one-time crops. For an *ongoing* crop (TOMATO/STRAWBERRY) it's just a
    placeholder equal to first_yield_day (data/README.md's Harvest Yields
    section): the real schedule runs first_yield_day, +interval, +interval,
    ... for max_yield productions total (tomato ages 8/9/10/11, strawberry
    10/12/14/16), so the placeholder alone would only ever allow
    fertilizing the first of the four."""
    data = CROPS[crop]
    if data["ongoing"]:
        return data["first_yield_day"] + (data["max_yield"] - 1) * data["interval"]
    return data["max_yield_day"]


def _is_fertilizable(tile, day):
    return (
        _is_plant(tile)
        and tile["crop"] in FERTILIZE_CROPS
        and tile["fertilized_until_day"] < day
        and day - tile["planted_day"] <= _last_yield_day(tile["crop"])
    )


def _any_fertilizable(farm, day):
    return any(_is_fertilizable(tile, day) for row in farm["tiles"] for tile in row)


def _any_unfed_animal(farm):
    """Mirrors `_any_fertilizable` -- used to gate the WHEAT endgame-liquidation DROP escape
    (see `_decide_unit`'s carried_produce and the module docstring's Endgame liquidation
    (WHEAT) note) so it never drops WHEAT an unfed animal could still use, even this late:
    missing a feed loses the whole animal, a bigger loss than a few dollars of unsold grain."""
    return any(_is_animal_tile(tile) and not tile.get("fed_today") for row in farm["tiles"] for tile in row)


def _is_harvestable(tile, day, liquidating):
    """yield_units > 0 alone is NOT "ready" for one-time crops: `_new_plant`
    seeds it at 1 immediately on planting (kaggriculture.py), but HARVEST is
    still a no-op until `first_yield_day` (age 2 for wheat/carrot, 10 for
    melon). Confusing the two makes a unit spam a no-op HARVEST instead of
    WATER, and the crop dies to neglect before ever becoming harvestable.

    Past `first_yield_day`, a one-time crop (wheat/carrot/melon) is legal to
    harvest but NOT yet done growing -- see the module docstring's
    Harvest-timing note: kaggriculture.py keeps adding yield_units for every
    watered day through `max_yield_day` (+2/day if fertilized), capped at
    `max_yield`. Treating "legal" as "should harvest now" was locking in
    yield_units=1 the instant it was legal, forfeiting up to 4-6x the yield
    for wheat. This now defers harvest of a one-time crop already past
    first_yield_day until whichever comes first: it's fully matured
    (`age >= max_yield_day`), it's already hit its yield cap (nothing left to
    gain by waiting), or the season is ending (`liquidating`) and there's no
    time left to wait -- the same early-harvest trade-off the X594
    kernel-pull reference makes (see [[x578-x594-kernel-pull-benchmark]]).
    Ongoing crops (tomato/strawberry) are unaffected: their own scheduled-
    production cadence already governs when yield_units appears."""
    if not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0:
        return False
    if tile.get("kind") != "PLANT":
        return True  # animal structures have no such gate; out of scope here anyway
    crop_data = CROPS[tile["crop"]]
    age = day - tile["planted_day"]
    if age < crop_data["first_yield_day"]:
        return False
    if crop_data["ongoing"]:
        return True
    return (
        age >= crop_data["max_yield_day"]
        or tile["yield_units"] >= crop_data["max_yield"]
        or liquidating
    )


def _is_yield_inert(tile, day):
    """A one-time crop's WATER action doesn't start adding yield_units the
    instant it's plantable/legal -- kaggriculture.py's own WATER handler only
    applies the +1 (+2 fertilized) bonus once age reaches
    `window_start = (max_yield_day + 1) // 2`, not `first_yield_day` (verified
    directly against the engine source; data/README.md doesn't document this
    window). For WHEAT/CARROT the two happen to coincide (both 2), so this
    never matters for them; for MELON, `window_start` (6) is well *before*
    `first_yield_day` (10) -- see _is_harvestable's own Harvest-timing note on
    why MELON's bonus window already saturates by the time it's legal to
    harvest. Watering during this window is still mandatory (skipping it
    risks the same 2-consecutive-miss WEED conversion as any other day) --
    this only identifies when it's *safe to deprioritize*, not safe to skip.
    Ongoing crops (tomato/strawberry) are never yield-inert this way: their
    own scheduled-production cadence, not a watering bonus window, governs
    when yield_units appears."""
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        return False
    crop_data = CROPS[tile["crop"]]
    if crop_data["ongoing"]:
        return False
    window_start = (crop_data["max_yield_day"] + 1) // 2
    age = day - tile["planted_day"]
    return age < window_start


def _crop_mix_ramp_fraction(day):
    """0.0 before CROP_MIX_RAMP_START_DAY, 1.0 once CROP_MIX_RAMP_DURATION days past it,
    linear in between -- shared by the target-share and seed-reserve ramps below so both move
    on the same schedule. See CROP_MIX_RAMP_START_DAY's own comment for why this doesn't start
    at day 0."""
    if CROP_MIX_RAMP_DURATION <= 0:
        return 1.0 if day >= CROP_MIX_RAMP_START_DAY else 0.0
    return max(0.0, min(1.0, (day - CROP_MIX_RAMP_START_DAY) / CROP_MIX_RAMP_DURATION))


def _crop_target_share(day):
    """CROP_TARGET_SHARE, but with STRAWBERRY's share ramped toward STRAWBERRY_LATE_SHARE as
    `day` advances (see STRAWBERRY_LATE_SHARE's comment) -- the other 4 crops scale down
    proportionally so the mix still sums to 1, preserving their relative baseline balance
    rather than favoring any one of them."""
    frac = _crop_mix_ramp_fraction(day)
    base_strawberry = CROP_TARGET_SHARE["STRAWBERRY"]
    strawberry_share = base_strawberry + frac * (STRAWBERRY_LATE_SHARE - base_strawberry)
    other_scale = (1 - strawberry_share) / (1 - base_strawberry)
    return {
        crop: strawberry_share if crop == "STRAWBERRY" else share * other_scale
        for crop, share in CROP_TARGET_SHARE.items()
    }


def _strawberry_seed_reserve(day):
    """SEED_RESERVE["STRAWBERRY"], ramped toward STRAWBERRY_RESERVE_LATE on the same schedule
    as _crop_target_share -- without this, _build_seed_orders would never stock enough
    STRAWBERRY seed for a higher target share to actually be reachable."""
    frac = _crop_mix_ramp_fraction(day)
    base = SEED_RESERVE["STRAWBERRY"]
    return base + frac * (STRAWBERRY_RESERVE_LATE - base)


class _CropPlanner:
    """Tracks, for one turn, which crop each new PLANT should use.

    Buying happens the turn *before* planting (market orders resolve after
    farmer/hand actions each turn -- see data/README.md's Turn Processing
    Order), so this only ever spends seeds already sitting in
    `private["seeds"]`, never seeds bought this same turn.
    """

    def __init__(self, private, farm, days_remaining, day=0, prices=None):
        self.target_share = _crop_target_share(day)
        self.seed_budget = dict(private.get("seeds", {}))
        self.prices = prices or {}
        # A crop that can't reach first_yield_day (+ a day of slack to sell it
        # after) before the season ends is a pure loss to plant -- see the
        # module docstring's Endgame liquidation note.
        self.plantable_crops = {
            c for c in CASH_CROPS if days_remaining >= CROPS[c]["first_yield_day"] + SEED_BUFFER_DAYS
        }
        self.counts = {c: 0 for c in CASH_CROPS}
        for row in farm["tiles"]:
            for tile in row:
                if _is_plant(tile) and tile["crop"] in CASH_CROPS:
                    self.counts[tile["crop"]] += 1
        self.total = sum(self.counts.values()) or 1

    def has_plantable(self):
        return any(self.seed_budget.get(c, 0) > 0 for c in self.plantable_crops)

    def _price_ok(self, crop):
        """Would we currently be willing to *sell* this crop? If not (and we
        have price data at all), planting more of it just grows tomorrow's
        shed pile in a market that's already below what we're willing to
        accept -- see the module docstring's Price-aware planting note."""
        price = self.prices.get(crop)
        if price is None:
            return True
        base = MARKET_PARAMS[crop]["base"]
        return price >= base * SELL_PARAMS[crop]["threshold"]

    def _score(self, crop):
        """Lower (more negative) wins the min() in choose(): how far below
        its target mix share this crop sits, pulled further negative (more
        attractive) the further its current price sits above base. Without
        this, `_price_ok` is the only place price enters the decision at
        all, and it's a binary pass/fail -- a crop barely above threshold
        and one at 2x its base price score identically once both clear the
        gate, so a small-target-share crop at a genuinely great price still
        loses to a big-target-share crop merely above threshold. See the
        module docstring's Price-aware planting note."""
        gap = self.counts[crop] / self.total - self.target_share[crop]
        price = self.prices.get(crop)
        if price is None:
            return gap
        price_ratio = price / MARKET_PARAMS[crop]["base"]
        return gap - PRICE_ATTRACTIVENESS_WEIGHT * (price_ratio - 1)

    def choose(self):
        """Pick whichever affordable, currently-worth-selling crop scores
        lowest (see _score: furthest below its target mix share, weighted
        by how attractive its current price is). Falls back to ignoring
        price eligibility if every plantable crop is currently depressed,
        so land never sits idle over a temporary dip."""
        available = [c for c in self.plantable_crops if self.seed_budget.get(c, 0) > 0]
        if not available:
            return None
        pool = [c for c in available if self._price_ok(c)] or available
        crop = min(pool, key=self._score)
        self.seed_budget[crop] -= 1
        self.counts[crop] += 1
        self.total += 1
        return crop


def _is_animal_tile(tile):
    return isinstance(tile, dict) and bool(tile.get("animal"))


def _live_animal_count(farm):
    """Shared by `_build_feed_orders`, `_build_animal_orders`, and
    `_build_sell_orders`'s feed reserve so all three always agree on how
    many animals are actually alive, instead of three independent loops
    that could drift (see `_build_animal_orders`'s own "Known bug" note
    about two *different* counts drifting apart for `committed`)."""
    return sum(1 for row in farm["tiles"] for tile in row if _is_animal_tile(tile))


def _is_empty_structure(tile):
    return isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP") and not tile.get("animal")


def _carried_animal_for(tile, inv):
    """Which animal kind (if any) in `inv` matches this empty structure's type."""
    for kind in ANIMAL_KINDS:
        if ANIMAL_STRUCTURE[kind] == tile.get("kind") and inv.get(kind, 0) > 0:
            return kind
    return None


def _animal_target(farm):
    """Target live-animal count -- the smaller of a land-share target
    (ANIMAL_LAND_SHARE fraction of owned tiles) and a crew-capacity target
    (ANIMAL_MAX_PER_UNIT per farmer/hand currently hired). A real replay
    showed the land-share target alone growing past what the crew could
    also keep up with on the crop side: by day 20-28 (3 quadrants, ~$40k
    game) animal structures reached 14 while the same crew's planted tile
    count collapsed from ~17 to ~2-3 and weeds went uncleared for the rest
    of the season -- see the module docstring's Animals note. Shared by
    _AnimalPlanner and _build_animal_orders so
    both always agree on the same cap -- though see
    _build_animal_orders's own "Known bug" note: they still track
    *committed* differently (shed tokens vs empty structures), which can
    let purchases outrun confirmed structure availability even with the
    same target."""
    unlocked_tiles = sum(1 for row in farm["tiles"] for tile in row if tile != "LOCKED")
    land_target = round(unlocked_tiles * ANIMAL_LAND_SHARE)
    total_units = 1 + len(farm["hands"])  # farmer + hands
    capacity_target = round(total_units * ANIMAL_MAX_PER_UNIT)
    return min(land_target, capacity_target)


class _AnimalPlanner:
    """Tracks, for one turn, which animal kind to build a structure for or
    fetch from the shed next -- mirrors _CropPlanner's price-aware,
    share-balancing logic. Unlike an earlier dedicated-hands design (see
    the module docstring's Animals note), capacity is sized as a target *fraction of
    owned land* (`ANIMAL_LAND_SHARE`), not a fixed hand allocation, and any
    unit -- farmer or hand -- may act on an animal task; there's no
    animal-only crew. `wants_structure`/`wants_pickup` are read-only (safe
    during candidate scanning in _find_target); `choose_structure`/
    `choose_pickup` mutate reserve counters, so only call them once a unit
    is actually standing on the tile it's about to act on."""

    def __init__(self, farm, shed, prices):
        self.counts = {kind: 0 for kind in ANIMAL_KINDS}
        self.empty_structures = {"PASTURE": 0, "COOP": 0}
        for row in farm["tiles"]:
            for tile in row:
                if not (isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP")):
                    continue
                animal = tile.get("animal")
                if animal:
                    self.counts[animal] += 1
                else:
                    self.empty_structures[tile["kind"]] += 1
        self.total = sum(self.counts.values()) or 1
        self.shed = dict(shed)
        self.prices = prices or {}
        self.target = _animal_target(farm)

    def _committed(self):
        return sum(self.counts.values()) + sum(self.empty_structures.values())

    def remaining_structures(self):
        """How many more structures still need to be built to reach the
        land-share target -- used to size the near-shed zone _animal_zone
        reserves, not to decide whether to build (choose_structure/
        wants_structure remain the source of truth for that)."""
        return max(0, self.target - self._committed())

    def _price_ok(self, kind):
        product = ANIMAL_PRODUCT[kind]
        price = self.prices.get(product)
        if price is None:
            return True
        base = MARKET_PARAMS[product]["base"]
        return price >= base * SELL_PARAMS[product]["threshold"]

    def wants_structure(self):
        if self._committed() >= self.target:
            return False
        return any(self.empty_structures[ANIMAL_STRUCTURE[k]] == 0 for k in ANIMAL_KINDS)

    def wants_pickup(self):
        return any(
            self.empty_structures[ANIMAL_STRUCTURE[k]] > 0 and self.shed.get(k, 0) > 0 for k in ANIMAL_KINDS
        )

    def choose_structure(self):
        """Pick a kind with no spare (built-but-empty) structure yet, preferring
        whichever affordable, currently-worth-selling kind is furthest below
        its target mix share -- same fallback as _CropPlanner.choose().
        Excludes kinds with a 0 target share (a "structure type has zero
        spares" check alone would otherwise let a 0-share kind win as soon
        as its structure type is the only one with no spare waiting --
        see the module docstring's Animals note: that gap let GOOSE keep
        getting built despite a 0 target share)."""
        if self._committed() >= self.target:
            return None
        needing = [
            k for k in ANIMAL_KINDS
            if ANIMAL_TARGET_SHARE[k] > 0 and self.empty_structures[ANIMAL_STRUCTURE[k]] == 0
        ]
        if not needing:
            return None
        pool = [k for k in needing if self._price_ok(k)] or needing
        kind = min(pool, key=lambda k: self.counts[k] / self.total - ANIMAL_TARGET_SHARE[k])
        self.empty_structures[ANIMAL_STRUCTURE[kind]] += 1  # reserve so two units don't both build one
        return kind

    def choose_pickup(self):
        """Pick a kind that already has a spare structure waiting AND a
        bought token sitting in the shed."""
        ready = [
            k for k in ANIMAL_KINDS
            if ANIMAL_TARGET_SHARE[k] > 0
            and self.empty_structures[ANIMAL_STRUCTURE[k]] > 0
            and self.shed.get(k, 0) > 0
        ]
        if not ready:
            return None
        pool = [k for k in ready if self._price_ok(k)] or ready
        kind = min(pool, key=lambda k: self.counts[k] / self.total - ANIMAL_TARGET_SHARE[k])
        self.empty_structures[ANIMAL_STRUCTURE[kind]] -= 1  # reserve so two units don't grab the same slot
        self.shed[kind] = self.shed.get(kind, 0) - 1
        return kind


def _find_personal_target(
    pos, farm, day, board_size, claimed, inv, animal_planner, shed_tiles,
    want_fertilizer, shed_wheat_available, liquidating,
):
    """The subset of this agent's old single-scan target search that genuinely depends on
    a *specific* unit's own carried inventory -- fertilizer-delivery, feed, animal-token
    placement, and every shed-return variant -- rather than just tile state. These can't be
    pooled into the shared per-turn job list `_build_tile_jobs` builds (see the module
    docstring's Routing efficiency note): a fertilizer-delivery candidate, for instance, only
    exists at all for a unit that is already carrying fertilizer. Every unit still checks
    this first, exactly as before; only a unit this returns nothing for falls through to the
    shared `_assign_tile_jobs` global match.

    `liquidating` (close to the last day, see the module docstring's Endgame liquidation
    note) swaps shed-return's priority the same way it always has.

    Short-circuits before any board scan (no candidates possible) unless this unit is
    carrying fertilizer, wheat, or an animal token -- cheaper than the old design, which
    always scanned regardless."""
    fx, fy = pos
    shed_priority = 1 if liquidating else 5
    carries_fertilizer = inv.get("FERTILIZER", 0) > 0
    carries_wheat = inv.get("WHEAT", 0) > 0
    carries_animal_token = any(inv.get(k, 0) > 0 for k in ANIMAL_KINDS)
    candidates = []
    if carries_fertilizer or carries_wheat or carries_animal_token:
        for y in range(board_size):
            for x in range(board_size):
                if (x, y) in claimed:
                    continue
                tile = farm["tiles"][y][x]
                dist = abs(x - fx) + abs(y - fy)
                if _is_plant(tile):
                    # matches _find_target's old elif chain: only reachable once the tile is
                    # neither harvestable nor unwatered (those are tile-only, in the shared pool)
                    if (carries_fertilizer and tile.get("watered_today")
                            and not _is_harvestable(tile, day, liquidating) and _is_fertilizable(tile, day)):
                        candidates.append((x, y, 3, dist))  # deliver fertilizer
                elif _is_animal_tile(tile):
                    # same elif-chain fidelity: FEED only applies once yield_units<=0 (a
                    # ready product is tile-only harvest, handled by the shared pool instead)
                    if carries_wheat and tile.get("yield_units", 0) <= 0 and not tile.get("fed_today"):
                        candidates.append((x, y, -1, dist))  # feed -- 2 misses = the animal escapes
                elif carries_animal_token and _is_empty_structure(tile) and _carried_animal_for(tile, inv):
                    candidates.append((x, y, 0, dist))  # place a carried animal

    carried_produce_total = sum(
        n for k, n in inv.items()
        if n > 0 and k not in ("WHEAT", *ANIMAL_KINDS) and (k != "FERTILIZER" or not want_fertilizer or liquidating)
    )
    carrying_produce = carried_produce_total > 0
    needs_fertilizer = inv.get("FERTILIZER", 0) == 0 and want_fertilizer and not liquidating
    needs_wheat = inv.get("WHEAT", 0) <= 0 and _any_unfed_animal(farm) and shed_wheat_available
    needs_animal_pickup = animal_planner.wants_pickup() and not carries_animal_token
    # A large carried WHEAT batch is worth a dedicated trip too (12th market-order-slot-
    # starvation attempt -- see the module docstring), but deliberately kept out of
    # carried_produce_total/the urgent-priority check above -- see that attempt's note for why
    # (WHEAT's fast, frequent, single-shot harvests trivially cross CARRIED_PRODUCE_URGENT_
    # THRESHOLD on their own, colliding with weeding every time if counted there).
    returning_wheat = inv.get("WHEAT", 0) >= WHEAT_SHED_RETURN_THRESHOLD
    if carrying_produce or needs_fertilizer or needs_wheat or needs_animal_pickup or returning_wheat:
        if needs_wheat:
            priority = -1
        elif carried_produce_total >= CARRIED_PRODUCE_URGENT_THRESHOLD:
            priority = SHED_RETURN_URGENT_PRIORITY
        else:
            priority = shed_priority
        for (sx, sy) in shed_tiles:
            if (sx, sy) not in claimed:
                candidates.append((sx, sy, priority, abs(sx - fx) + abs(sy - fy)))  # return to shed

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[2], c[3]))
    return candidates[0]


def _build_tile_jobs(
    farm, day, board_size, claimed, animal_zone, quadrant_util, plant_priority, liquidating,
    planner, animal_planner, last_chance_window=False,
):
    """The unit-independent subset of this agent's old single-scan target search --
    harvest/water/weed/animal-upkeep/plant/build -- any unit could do any of these
    regardless of what it's carrying, so they're built once per turn here instead of every
    unit rescanning the whole board and claiming its own single best candidate in a fixed,
    order-dependent sequence. `_assign_tile_jobs` then matches all of this turn's
    still-unresolved units against this shared pool globally. See the module docstring's
    Routing efficiency note, and `_find_personal_target` for the (smaller) unit-dependent
    remainder this doesn't cover.

    Every job carries a `(x, y, priority, dist_bias)` tuple: `dist_bias` is added to
    whichever unit ends up scored against it (Manhattan distance from that unit's own
    position), preserving every existing sort_dist adjustment exactly --
    `HARVEST_AT_RISK_DIST_BIAS` and the quadrant-balance nudge for fresh plantings -- since
    neither of those depends on which unit is asking, just tile state."""
    water_priority = 5 if liquidating else WATER_ROUTINE_PRIORITY
    # 15th market-order-slot-starvation attempt: see the module docstring for why animal
    # upkeep floors at plant_priority instead of a flat 3 -- unchanged by this refactor.
    animal_upkeep_priority = min(3, plant_priority)
    spread_penalty = board_size // 2  # scale of the quadrant-balance nudge, in tiles
    jobs = []
    for y in range(board_size):
        for x in range(board_size):
            if (x, y) in claimed:
                continue
            tile = farm["tiles"][y][x]
            if _is_plant(tile):
                if _is_harvestable(tile, day, liquidating):
                    # An at-risk harvestable tile keeps priority 0 (top tier, plus the
                    # dist_bias tie-break) -- still subject to the same 2-consecutive-miss WEED
                    # conversion as any other tile, genuinely as urgent as anything gets. A
                    # *safe* one (consecutive_unwatered==0, a full day of slack) uses
                    # HARVEST_SAFE_PRIORITY instead of the unconditional 0 -- see
                    # [[harvest-safe-deprioritization-tradeoff]]: a first attempt tied it
                    # exactly to at-risk water's 0.5, which just made the two contend on raw
                    # distance (an ambiguous coin-flip, not a real ordering) and regressed by
                    # delaying harvest->carry->shed->sell realization near the season's hard
                    # cutoff. HARVEST_SAFE_PRIORITY must differ from WATER's at-risk priority,
                    # not tie it, so the ordering between them is decisive either way instead
                    # of contested -- MC-searched (scripts/mc_search.py) instead of guessed.
                    at_risk = not liquidating and tile.get("consecutive_unwatered", 0) >= 1
                    priority = 0 if at_risk else HARVEST_SAFE_PRIORITY
                    jobs.append((x, y, priority, -HARVEST_AT_RISK_DIST_BIAS if at_risk else 0))  # harvest
                elif not tile["watered_today"]:
                    at_risk = not liquidating and tile.get("consecutive_unwatered", 0) >= 1
                    # Third variant after [[harvest-early-at-risk-salvage-regression]]: forcing
                    # an early HARVEST regressed (forfeits the 81% that self-recover anyway);
                    # tying the harvest tier (0) all day partially regressed (steals crew from
                    # genuinely-ripe, merely-closer harvests all day for a risk that isn't
                    # urgent yet most of the day); tying it only on the literal last turn was a
                    # total no-op (a hand not already standing on the tile has zero turns left
                    # to walk over, so it could never help regardless of crew capacity -- see
                    # that memory for why this isn't evidence of a real reachability ceiling by
                    # itself). This narrows the window to LAST_CHANCE_HOURS before day's end
                    # instead of either the whole day or none of it -- long enough for a
                    # reasonably-close hand to actually walk over, short enough to limit how
                    # much it competes with real harvests to only when the risk is genuinely
                    # close to unrecoverable.
                    urgent = at_risk and last_chance_window
                    if urgent:
                        priority = 0
                    elif at_risk:
                        priority = 0.5
                    elif not liquidating and _is_yield_inert(tile, day):
                        # See _is_yield_inert's own docstring and
                        # [[water-yield-inert-deprioritization-tradeoff]]: a still-growing
                        # one-time crop before kaggriculture.py's own yield-bonus window adds
                        # no yield_units from being watered today, only matters for avoiding
                        # the same 2-consecutive-miss WEED clock as any day -- and
                        # consecutive_unwatered==0 here (the at_risk branch above already
                        # caught >=1) means a full day of slack. First attempt at this
                        # regressed by increasing the rate of a tile's *first* missed
                        # watering; MC-searched here instead of the single hand-picked value
                        # (3) that attempt used.
                        priority = WATER_YIELD_INERT_PRIORITY
                    else:
                        priority = water_priority
                    jobs.append((x, y, priority, 0))  # water
            elif _is_weed(tile):
                jobs.append((x, y, 2, 0))  # clear a weed to reclaim the tile
            elif _is_animal_tile(tile):
                if tile.get("yield_units", 0) > 0:
                    jobs.append((x, y, 0, 0))  # harvest animal product
                elif tile.get("fed_today"):
                    # matches the old elif chain: collect/care only apply once fed (an unfed
                    # animal's FEED candidate is unit-dependent, see _find_personal_target)
                    if tile.get("fertilizer_available"):
                        jobs.append((x, y, animal_upkeep_priority, 0))  # collect fertilizer
                    elif not tile.get("cared_today"):
                        jobs.append((x, y, animal_upkeep_priority, 0))  # care (free next-yield bonus)
            elif tile is None:
                if (x, y) in animal_zone:
                    jobs.append((x, y, 4, 0))  # reserved for an animal structure
                elif planner.has_plantable():
                    q = _quadrant_of(x, y, board_size)
                    jobs.append((x, y, plant_priority, quadrant_util[q] * spread_penalty))  # plant
                elif animal_planner.wants_structure():
                    jobs.append((x, y, 4, 0))  # build a structure (overflow beyond the zone)
    return jobs


def _assign_tile_jobs(units, jobs, personal):
    """Global (unit x job) match for `_build_tile_jobs`'s shared pool, plus each unit's own
    personal (inventory-dependent) candidate from `_find_personal_target` if it has one --
    both compete in the SAME sort, by the exact same `(priority, sort_dist)` convention this
    agent has always used (priority strictly dominates distance). Every pairing across the
    whole remaining crew, and every unit's own personal option, is compared before any of
    them commit; assignment is greedy top-down, removing a shared job once any unit takes
    it (a personal candidate is never contested -- it only exists for the one unit whose
    inventory generated it, so taking it never blocks another unit).

    A first version of this rework let a found personal target win outright, before this
    unified sort existed -- wrong, see `_decide_unit`'s own docstring and
    `test_find_target_shed_return_stays_routine_below_the_carry_threshold`: a routine,
    low-priority personal candidate (e.g. a small carried load's shed return) must still be
    able to lose to a genuinely more urgent nearby pool job, exactly as the old single-scan
    design compared them. This is the one piece of this turn's routing that differs from
    that old per-unit sequential scan: instead of unit 0 claiming its own single best
    candidate before unit 1 even looks at the board (an order-dependent match that can
    strand a *different* unit's better fit), every candidate across the whole remaining crew
    is compared at once. See the module docstring's Routing efficiency note.

    `units` is a list of `(unit_index, pos)` for units not yet resolved by an on-tile
    action. `personal` is `{unit_index: (x, y, priority, dist)}`, one entry for units
    `_find_personal_target` found something for. Returns `{unit_index: (x, y)}` for however
    many could be matched -- a unit with no personal candidate and no reachable pool job is
    simply absent from the result (callers should PASS it)."""
    pairs = []
    for i, pos in units:
        fx, fy = pos
        for j, (x, y, prio, bias) in enumerate(jobs):
            dist = abs(x - fx) + abs(y - fy) + bias
            pairs.append((prio, dist, i, j))
        pc = personal.get(i)
        if pc is not None:
            pairs.append((pc[2], pc[3], i, None))
    pairs.sort(key=lambda pair: (pair[0], pair[1]))
    assigned_units = set()
    assigned_jobs = set()
    result = {}
    n_units = len(units)
    for prio, dist, i, j in pairs:
        if i in assigned_units:
            continue
        if j is None:
            x, y = personal[i][0], personal[i][1]
        else:
            if j in assigned_jobs:
                continue
            x, y = jobs[j][0], jobs[j][1]
            assigned_jobs.add(j)
        result[i] = (x, y)
        assigned_units.add(i)
        if len(assigned_units) == n_units:
            break
    return result


def _build_action(kind):
    return ["BUILD_COOP" if ANIMAL_STRUCTURE[kind] == "COOP" else "BUILD_PASTURE"]


def _decide_unit(
    pos, inv, farm, private, day, claimed, planner, animal_planner, shed_tiles,
    want_fertilizer, liquidating, animal_zone,
):
    """Resolves a unit's action if it's standing on an actionable tile -- unconditionally
    the best use of this turn (it costs no travel), so nothing else is even considered, same
    as before this file's routing-efficiency rework. Returns `None` otherwise, meaning the
    caller (`agent()`) still needs this unit's personal (inventory-dependent) candidate --
    see `_find_personal_target` -- and a shared tile-job match -- see `_build_tile_jobs`/
    `_assign_tile_jobs` and the module docstring's Routing efficiency note -- to decide where
    it walks. (Earlier versions of this rework let a found personal target win outright over
    the shared pool; that's wrong -- a routine, low-priority personal candidate like a small
    carried load's shed return must still be able to lose to a genuinely more urgent nearby
    pool job, exactly as the old single-scan design did. See
    `test_find_target_shed_return_stays_routine_below_the_carry_threshold`.)"""
    fx, fy = pos
    tile = farm["tiles"][fy][fx]

    if _is_plant(tile):
        if _is_harvestable(tile, day, liquidating):
            claimed.add((fx, fy))
            return ["HARVEST"]
        if not tile["watered_today"]:
            claimed.add((fx, fy))
            return ["WATER"]
        if inv.get("FERTILIZER", 0) > 0 and _is_fertilizable(tile, day):
            claimed.add((fx, fy))
            return ["FERTILIZE"]

    if _is_weed(tile):
        claimed.add((fx, fy))
        return ["DIG"]

    if _is_animal_tile(tile):
        if tile.get("yield_units", 0) > 0:
            claimed.add((fx, fy))
            return ["HARVEST"]
        if not tile.get("fed_today") and inv.get("WHEAT", 0) > 0:
            claimed.add((fx, fy))
            return ["FEED"]
        if tile.get("fertilizer_available"):
            claimed.add((fx, fy))
            return ["COLLECT_FERTILIZER"]
        if not tile.get("cared_today"):
            claimed.add((fx, fy))
            return ["CARE"]

    if _is_empty_structure(tile):
        kind = _carried_animal_for(tile, inv)
        if kind:
            claimed.add((fx, fy))
            return ["PLACE", kind]

    if (fx, fy) in shed_tiles:
        carried_product = {k: n for k, n in inv.items() if k in ANIMAL_PRODUCTS and n > 0}
        if carried_product:
            claimed.add((fx, fy))
            item, n = next(iter(carried_product.items()))
            return ["PLACE", item, n]
        wheat_safe_to_liquidate = liquidating and not _any_unfed_animal(farm)
        carried_produce = {
            k: n for k, n in inv.items()
            if n > 0 and k not in ANIMAL_KINDS
            and (k != "FERTILIZER" or not want_fertilizer or liquidating)
            and (k != "WHEAT" or n >= WHEAT_SHED_RETURN_THRESHOLD or wheat_safe_to_liquidate)
        }
        # WHEAT is included here (unlike _find_target's carried_produce_total above), but
        # gated at the same WHEAT_SHED_RETURN_THRESHOLD -- without this, a unit that just
        # PICKUP'd WHEAT for feed (up to 4 units, main.py's own PICKUP call below) would see
        # its own freshly-picked-up WHEAT here on the very next decision and immediately
        # DROP it right back before ever walking to the animal that needed it -- a livelock
        # in the same family the FERTILIZER guard right above already exists to prevent (see
        # test_decide_unit_keeps_freshly_picked_up_fertilizer_when_still_needed), just not
        # yet extended to WHEAT. Threshold must exceed PICKUP's own 4-unit cap for the guard
        # to actually work -- this was the real bug behind the 11th/12th attempts' weed-count
        # explosion, not the trip-generation concern _find_target's own gate addresses.
        # `wheat_safe_to_liquidate` mirrors FERTILIZER's own liquidation escape, but -- unlike
        # FERTILIZER -- ALSO requires no animal is still unfed: a first version used bare
        # `liquidating` and measurably regressed (15-seed pool A/B, real money down almost
        # everywhere) because it let a unit DROP its already-carried feed WHEAT instead of
        # walking it to a hungry animal, exactly the mistake `needs_wheat`'s own lack of a
        # `not liquidating` clause was written to avoid -- missing a feed this late still loses
        # the whole animal, a bigger loss than a few unsold units. See the module docstring's
        # Endgame liquidation (WHEAT) note for the full history.
        if carried_produce:
            claimed.add((fx, fy))
            return ["DROP"]
        if (
            inv.get("FERTILIZER", 0) == 0 and want_fertilizer and not liquidating
            and private["shed"].get("FERTILIZER", 0) > 0
        ):
            claimed.add((fx, fy))
            return ["PICKUP", "FERTILIZER", 1]
        if inv.get("WHEAT", 0) <= 0 and private["shed"].get("WHEAT", 0) > 0:
            claimed.add((fx, fy))
            return ["PICKUP", "WHEAT", min(4, private["shed"]["WHEAT"])]
        if not any(inv.get(k, 0) > 0 for k in ANIMAL_KINDS):
            kind = animal_planner.choose_pickup()
            if kind:
                claimed.add((fx, fy))
                return ["PICKUP", kind, 1]

    if tile is None:
        if (fx, fy) in animal_zone:
            kind = animal_planner.choose_structure()
            if kind:
                claimed.add((fx, fy))
                return _build_action(kind)
        crop = planner.choose()
        if crop:
            claimed.add((fx, fy))
            return ["PLANT", crop]
        kind = animal_planner.choose_structure()
        if kind:
            claimed.add((fx, fy))
            return _build_action(kind)

    return None  # not on an actionable tile -- agent() resolves this unit's target itself


def _build_sell_orders(shed, prices, days_remaining, farm):
    """Dumps the entire shed stock of a crop in one order once its price
    threshold is met -- same behavior as melon_maxxer, deliberately reverted
    from an earlier version that capped units sold per turn (see the module
    docstring's Selling note). A per-turn cap was tried expecting it to
    avoid crashing the price via a big single-order dump; instead it was a
    measured net negative (a 19/20-seed head-to-head loss to this dump
    version, `reference/main_dump_sell.py`): holding harvested value in the
    shed waiting for the cap to free up slows reinvestment into land/seeds/
    hands more than an occasional price dip costs, at this agent's actual
    production scale. See the module docstring's WHEAT sell/rebuy round trip
    note for three earlier, narrower attempts at holding back WHEAT alone
    (for animal feed) that all hit the same standing-price wall.

    **18th attempt (see .omc/plans/ralplan-wheat-feed-consolidation.md):
    WHEAT specifically holds back `live_animals * min(WHEAT_SELL_RESERVE_DAYS,
    days_remaining)` from the dump, unless the shed is crowded or the game is
    liquidating (both force a full dump regardless, same as every other
    crop).** The three prior WHEAT-reserve attempts (module docstring) tried
    this shape alone, on top of an unchanged `_build_feed_orders` that still
    bought one order per unit of shortfall from `shed`-only accounting -- this
    time it's paired with `_build_feed_orders`'s own carried-inclusive,
    single-order change below, mirroring the X594 kernel-pull reference's
    combination rather than retrying a piece already shown to fail alone.
    Status: unvalidated as of this commit -- pending the mechanism trace and
    full 30-seed protocol the plan requires before this note can say kept or
    reverted.
    """
    orders = []
    shed_total = sum(shed.values())
    crowded = shed_total >= SHED_CAPACITY * SHED_OVERFLOW_GUARD
    liquidating = days_remaining <= ENDGAME_LIQUIDATION_DAYS
    live_animals = None
    for crop, params in SELL_PARAMS.items():
        qty = shed.get(crop, 0)
        if qty <= 0:
            continue
        if crop == "WHEAT" and not (crowded or liquidating):
            if live_animals is None:
                live_animals = _live_animal_count(farm)
            reserve = live_animals * min(WHEAT_SELL_RESERVE_DAYS, max(0, days_remaining))
            qty -= reserve
            if qty <= 0:
                continue
        base = MARKET_PARAMS[crop]["base"]
        price = prices.get(crop, base)
        if crowded or liquidating or price >= base * params["threshold"]:
            orders.append(["SELL", crop, qty])
    return orders


def _hands_target(quadrants):
    if quadrants <= 1:
        return HANDS_TARGET_Q1  # surgical override for the single-quadrant opening -- see
                                  # HANDS_TARGET_Q1's own comment. HANDS_BASE/HANDS_PER_QUADRANT
                                  # below untouched for quadrants >= 2.
    return min(MAX_HANDS, HANDS_BASE + HANDS_PER_QUADRANT * quadrants)


def _land_worth_buying(quadrants_now):
    """The next quadrant only pays for itself if it actually raises hire
    capacity -- see the module docstring's Land / hands note.
    Originally caught with the old `2 * quadrants` hire formula (capped at
    MAX_HANDS=6): the ceiling fell at exactly 3 quadrants, and a replay
    showed the same 6 hands spread across a 4th quadrant's extra land
    planting no more tiles in absolute terms than 3 quadrants' worth, just
    diluted thinner. `_hands_target`'s current formula (see the same
    docstring note) was deliberately picked to stay
    strictly increasing all the way through 4 quadrants, so this stops
    blocking anything in practice under it -- the mechanism stays in place
    regardless, for whatever the hire formula ends up being."""
    return _hands_target(quadrants_now + 1) > _hands_target(quadrants_now)


def _land_utilization(farm):
    """Fraction of currently-owned (non-LOCKED) tiles that are planted or
    hold an animal structure. Shared by `_build_land_orders` (the land-buy
    gate) and `agent()`'s planting-priority boost -- see the module
    docstring's Planting priority under the gate note."""
    unlocked = [tile for row in farm["tiles"] for tile in row if tile != "LOCKED"]
    if not unlocked:
        return 0.0
    used = sum(
        1 for tile in unlocked
        if _is_plant(tile) or (isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP"))
    )
    return used / len(unlocked)


def _fib(n):
    """Indexed so _fib(0)=1, _fib(1)=1, _fib(2)=2, _fib(3)=3, _fib(4)=5... --
    mirrors kaggriculture.py's own `_fib`/`_hire_cost`, duplicated here
    (rather than imported) since it's a private engine implementation
    detail, unlike the public constants this module already imports from
    kaggriculture.py. Used only by _orders_cash_cost to estimate a HIRE
    order's own same-turn cash draw."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _sell_proceeds(item, qty, inventory):
    """X578/X594 kernel-pull reference lever 4 of 4 (see the module docstring's Harvest-timing
    note's land-purchase-delay follow-up): walks price impact unit-by-unit using the real
    `market_price` (imported directly from kaggriculture.py, not reimplemented the way X594's
    own copy is -- no risk of the two formulas drifting apart) instead of a flat
    `qty * current_price` snapshot, which overestimates revenue on a large SELL order since
    each unit sold nudges the price down (or up toward base, below I0) for the next. Mirrors
    kaggriculture.py's own `_commit_unit`: market inventory only grows for a unit sold above
    PRICE_FLOOR (a sale at the floor doesn't move the price further)."""
    inv = inventory
    total = 0
    for _ in range(qty):
        price = market_price(item, inv)
        total += price
        if price > PRICE_FLOOR:
            inv += 1
    return total


def _buy_cost(item, qty, inventory):
    """Mirror of _sell_proceeds for the buy side -- found independently while auditing
    data/README.md for other overlooked rules after the harvest-timing fix (see the module
    docstring's Harvest-timing note's BUY-side follow-up), not adapted from the X578/X594
    reference. data/README.md: "the buy price is quoted at the post-buy inventory" --
    kaggriculture.py's `_commit_unit` confirms BUY_PRODUCT decrements `market["inventory"]`
    unconditionally (no floor/cap exception the way SELL's $1 floor has), so each unit bought
    within the same order pushes the next one's price further from the last. A flat
    `qty * current_price` estimate underestimates a large order's true cost -- measured at 13%
    on a realistic 26-unit consolidated WHEAT feed order (`_build_feed_orders`) with the
    market near I0, where the below-I0 shape functions (sqrt/log/etc.) have their steepest
    marginal slope -- counterintuitively worse near equilibrium than when the market is
    already deep in shortage, where the same curves have mostly flattened out."""
    inv = inventory
    total = 0
    for _ in range(qty):
        price = market_price(item, inv)
        total += price
        inv -= 1
    return total


def _orders_cash_cost(orders, farm, market_inventory):
    """16th market-order-slot-starvation attempt: estimate the total cash
    a list of market orders will draw from farm["money"] this turn, using
    this turn's opening prices/counts -- the same approximation each
    _build_*_orders function already makes for its own affordability
    check. Threaded through agent()'s market-order-building sequence as a
    running same-turn budget, because the engine draws down farm["money"]
    live, in submitted order-list order, not against a static per-turn
    snapshot the way each builder was computing its own request in
    isolation (kaggriculture.py's `_process_market`/`_commit_unit`:
    per-unit lockstep across the whole order queue, checking `farm["money"]
    < price` at each individual unit commit). Without this, a builder late
    in agent()'s list (`_build_animal_orders` is last) can request a
    purchase it genuinely believed was affordable against the turn's
    opening balance, only for the engine to silently reject it because
    earlier orders in the same list (feed, seed, fertilizer) already spent
    that cash first -- see the module docstring's Same-turn cash
    competition note, the suspected real mechanism behind the 14th
    attempt's animal-market-spillover finding.

    Returns a signed total: positive for orders that spend money, negative
    for SELL orders (settled first in this agent's own list, and the real
    engine processes the whole list per-unit-lockstep against a single
    live balance -- see `_process_market`). A first version of this
    function deliberately left SELL out, reasoning that ignoring real
    revenue could only delay a purchase by a turn, never wrongly deny it.
    That reasoning was wrong against a reactive opponent: the same
    conservative gap reopens every turn a sell also happens, so a purchase
    that only becomes affordable *after* this turn's own sell revenue
    lands can end up chronically starved, not delayed once -- measured
    directly: leaving SELL out cost real games against reactive self-play
    opponents (main_plant_priority_submitted_20260818.py 40% -> 20%,
    main_price_reactive_seed_reserve_submitted_20260819.py 53.3% -> 23.3%,
    both roughly halved) even while it helped substantially against
    tape_kiro once paired with the 14th attempt's feed consolidation (0%
    -> 23.3%) -- see the module docstring's Same-turn cash competition
    note for the full comparison.

    `market_inventory` (lever 4 of 4, see `_sell_proceeds`) replaces the flat
    `qty * current_price` SELL estimate with a per-unit price-impact walk, discounted by
    `SELL_PROCEEDS_DISCOUNT` -- a large SELL order was otherwise overestimating its own
    proceeds, since one flat price ignores that each unit sold pushes the next one's price
    down. The same `market_inventory` also drives `_buy_cost` for BUY_PRODUCT -- see its own
    docstring for the mirrored underestimate on the buy side, found in the same audit."""
    total = 0.0
    n_extra = len(farm["unlocked_quadrants"]) - 1
    hires_so_far = farm["hires_today"]
    for order in orders:
        op = order[0]
        if op == "SELL":
            item = order[1]
            qty = order[2] if len(order) > 2 else 1
            inv = market_inventory.get(item, MARKET_I0)
            total -= _sell_proceeds(item, qty, inv) * SELL_PROCEEDS_DISCOUNT
        elif op == "BUY_PRODUCT":
            item = order[1]
            qty = order[2] if len(order) > 2 else 1
            inv = market_inventory.get(item, MARKET_I0)
            total += _buy_cost(item, qty, inv)
        elif op == "BUY_SEED":
            item = order[1]
            qty = order[2] if len(order) > 2 else 1
            total += qty * CROPS[item]["seed"]
        elif op == "BUY_ANIMAL":
            item = order[1]
            qty = order[2] if len(order) > 2 else 1
            total += qty * ANIMALS[item]["cost"]
        elif op == "BUY_LAND":
            if 0 <= n_extra < len(LAND_PRICES):
                total += LAND_PRICES[n_extra]
            n_extra += 1
        elif op == "HIRE":
            total += FARM_HAND_COST_MULT * _fib(hires_so_far)
            hires_so_far += 1
    return total


def _land_cash_reserve(farm):
    """X578/X594 kernel-pull reference lever 2 of 4 (see the module docstring's Harvest-timing
    note's land-purchase-delay follow-up): a reserve that scales with the size of the
    operation, not a flat constant -- a per-live-animal feed buffer plus a small flat amount,
    mirroring X594's own `cash_reserve = hire_total(...) + n_animals * 45 + 60` structurally
    (values not yet retuned for this agent's own economy -- starting from X594's as a first
    guess). The hire-cost term X594 also includes is intentionally left out here: lever 3
    (hire-before-land ordering, see agent()'s own comment) threads hire's REAL same-turn cost
    into the `money` this function's caller passes in, via the existing `cash_reserved`
    mechanism -- more precise than re-estimating it here too, and double-counting it in both
    places would over-reserve."""
    return _live_animal_count(farm) * LAND_RESERVE_PER_ANIMAL + LAND_RESERVE_FLAT


def _build_land_orders(farm, days_remaining, money):
    """`money` is the caller's same-turn cash budget -- see agent()'s own comment on why hire
    now runs before land in the build sequence (lever 3 of 4): unlike the reverted third
    land-purchase-delay attempt (module docstring), which gave land visibility into a turn's
    incoming SELL proceeds while land still ran BEFORE hire/seed in the sequence (so land won
    the race for that cash instead of losing it, worse than before), hire now claims its own
    real same-turn cost first, and land only sees what's left."""
    if days_remaining < LAND_CUTOFF_DAYS:
        return []  # a fresh quadrant can't earn back its cost this close to the end
    quadrants_now = len(farm["unlocked_quadrants"])
    n_extra = quadrants_now - 1  # NW is always owned
    if quadrants_now >= LAND_QUADRANT_CAP or n_extra >= len(LAND_ORDER) or not _land_worth_buying(quadrants_now):
        return []
    if _land_utilization(farm) < LAND_UTILIZATION_GATE:
        return []  # can't staff what we already own -- more land would just sit idle
    if money - _land_cash_reserve(farm) >= LAND_PRICES[n_extra]:
        return [["BUY_LAND"]]
    return []


def _build_hire_orders(farm, money_left):
    """`money_left` is the caller's same-turn cash budget (see agent()'s own comment on why
    hire runs before land in the build sequence) -- without this, this function requested
    every hire up to the quadrant-based target regardless of affordability, relying entirely
    on kaggriculture.py's own engine-side rejection (`_do_hire`: silently drops a HIRE order
    once `farm["money"] < cost`, no other cap) to cap the real spend. That's harmless for
    `cash_reserved`'s own purposes when every requested hire is genuinely affordable (the
    common case, since a handful of hires costs relatively little even at the top of the
    Fibonacci curve) -- but if money is already low enough that some would actually be
    rejected, `_orders_cash_cost`'s HIRE branch (main.py) still charges the full Fibonacci cost
    for every order in the *returned* list against this turn's cash_reserved tracker, since it
    has no way to know which ones the engine will actually honor. That silently
    under-estimates real money left for every builder after hire in agent()'s own sequence
    (land/feed/seed/fertilizer), on exactly the days money is already tightest. Found via a
    competitor's own analysis tooling (github.com/lopeznomar/kaggriculture-ops-lab) needing to
    reconstruct HIRE cost as a residual against real money deltas, specifically because a
    naive Fibonacci-sum overcounts whatever the engine actually rejected -- confirmed directly
    against kaggriculture.py's `_do_hire` that affordability is the only rejection rule, then
    checked whether this agent's own request-side already respected it (it didn't)."""
    desired = _hands_target(len(farm["unlocked_quadrants"]))
    remaining = desired - farm["hires_today"]
    if remaining <= 0:
        return []
    orders = []
    budget = max(0, money_left)
    for i in range(remaining):
        cost = FARM_HAND_COST_MULT * _fib(farm["hires_today"] + i)
        if budget < cost:
            break
        orders.append(["HIRE"])
        budget -= cost
    return orders


def _build_seed_orders(seeds, money, days_remaining, prices, day=0):
    """Price-reactive seed reserve: each crop's target buffer is
    `SEED_RESERVE` scaled up by how far its current price sits above its
    own base (capped at `SEED_RESERVE_PRICE_CAP`), never scaled down
    below the baseline for a temporary dip -- mirrors `_CropPlanner`'s own
    price-attractiveness logic (see the module docstring's Price-aware
    planting note), just applied to how much to buy
    ahead of need instead of what to plant right now. Directly answers
    the module docstring's Seed reserve note: a flat reserve
    increase blew up the WHEAT sell/rebuy round trip because WHEAT's
    reserve grew by the same factor as every other crop's regardless of
    whether WHEAT was actually worth growing more of. Scaling by price
    instead means WHEAT -- whose price rarely swings far above its own
    base -- stays close to its baseline reserve on its own, without a
    special case, while a crop genuinely trading well (MELON/STRAWBERRY
    above base) earns more room to stock up. STRAWBERRY's own base_reserve
    additionally ramps over the season -- see _strawberry_seed_reserve --
    so this stays a plain dict lookup for every other crop."""
    orders = []
    remaining = money - SAFETY_RESERVE
    strawberry_reserve = _strawberry_seed_reserve(day)
    for crop, base_reserve in SEED_RESERVE.items():
        if crop == "STRAWBERRY":
            base_reserve = strawberry_reserve
        if days_remaining < CROPS[crop]["first_yield_day"] + SEED_BUFFER_DAYS:
            continue  # wouldn't mature (and sell) before the season ends
        base_price = MARKET_PARAMS[crop]["base"]
        price = prices.get(crop, base_price)
        price_ratio = max(1.0, price / base_price) if base_price else 1.0
        reserve = round(base_reserve * min(price_ratio, SEED_RESERVE_PRICE_CAP))
        have = seeds.get(crop, 0)
        cost = CROPS[crop]["seed"]
        while have < reserve and remaining >= cost:
            orders.append(["BUY_SEED", crop, 1])
            have += 1
            remaining -= cost
    return orders


def _build_fertilizer_orders(private, prices, money, want_fertilizer):
    """Tops up to the small standing FERTILIZER_RESERVE normally, but
    stockpiles toward the larger FERTILIZER_STOCKPILE_RESERVE whenever
    price is cheap (<= FERTILIZER_CHEAP_PRICE_FRAC * base) AND
    `want_fertilizer` (`_any_fertilizable(farm, day)`, the same signal
    `_decide_unit`/`_find_target` gate fertilizer delivery on) -- see the
    module docstring's Price-reactive fertilizer buying note: a
    downloaded replay showed FERTILIZER crash from $100 to $11 against an
    animal-heavy opponent dumping excess byproduct, a real, large,
    opponent-driven price swing worth buying ahead of instead of trickling
    in a fixed 1-2 units regardless of price -- but only while there's
    somewhere for it to actually go. WHEAT/CARROT/TOMATO/STRAWBERRY
    together are most of `CROP_TARGET_SHARE`, so fertilizable land is
    close to a standing feature of this agent's own crop mix most of the
    game, not a rare event to catch -- but gating on it anyway costs
    nothing and avoids stockpiling toward a real crop's worth of
    fertilizer on the rare turn none of those crops are actually planted
    (e.g. right after a land purchase, before the crop planner has
    filled the new tiles). The baseline FERTILIZER_RESERVE is left
    ungated, same as before this fix -- the shed's FERTILIZER count is one
    fungible pool for both collected (free) and market-bought stock, so
    this 2-unit reserve can occasionally get swept into a sell dump and
    get rebought next cycle; a bounded, minor churn cost not worth a
    separate reserved-vs-sellable split for 2 units.
    FERTILIZER_CHEAP_PRICE_FRAC is set well below
    SELL_PARAMS["FERTILIZER"]["threshold"] (0.50) deliberately: the two
    must never overlap, or this buys back at a price we'd also be
    willing to sell at -- the exact self-inflicted round trip
    _build_feed_orders's own docstring note (WHEAT sell/rebuy) already
    found and had to back out of once."""
    have = private["shed"].get("FERTILIZER", 0) + sum(
        inv.get("FERTILIZER", 0) for inv in private["inventories"]
    )
    base = MARKET_PARAMS["FERTILIZER"]["base"]
    price = prices.get("FERTILIZER", base)
    cheap = price <= base * FERTILIZER_CHEAP_PRICE_FRAC
    reserve = FERTILIZER_STOCKPILE_RESERVE if (cheap and want_fertilizer) else FERTILIZER_RESERVE
    if have >= reserve:
        return []
    if price > base * FERTILIZER_MAX_PRICE_FRAC:
        return []
    if money - SAFETY_RESERVE < price:
        return []
    return [["BUY_PRODUCT", "FERTILIZER", 1]]


def _build_feed_orders(farm, private, prices, money):
    """Buy WHEAT directly from the market for animal feed, decoupled from
    the crop side's own WHEAT production/selling. Without this, animals
    starve even with plenty of attention: `_build_sell_orders` used to dump
    every WHEAT harvest as soon as its price cleared the sell threshold, so
    the shed could sit at 0 WHEAT for turns at a stretch -- see the module
    docstring's WHEAT sell/rebuy round trip note. Placed ahead of
    `_build_seed_orders` in `agent()`'s market list: missing a seed buy
    delays a planting one turn, missing a feed loses the whole animal.

    **18th attempt (see .omc/plans/ralplan-wheat-feed-consolidation.md):
    `have` now sums shed + every unit's carried inventory, and the whole
    shortfall is requested as ONE consolidated order, not one order per unit
    of shortfall.** Both of these were tried alone before and reverted --
    carried-inclusive accounting alone (17th attempt) cut WHEAT sales 70%
    along with the purchases, because WHEAT throughput is this agent's
    single largest profit centre (net +$35k of a $57k final) and nothing
    replaced it; consolidation alone (9th/14th/15th attempts) hit a
    standing-price or animal-byproduct-market-spillover wall instead. This
    time both changes are paired with `_build_sell_orders`'s own new WHEAT
    reserve (see `WHEAT_SELL_RESERVE_DAYS`) rather than retrying either piece
    on top of an unchanged sell side that still dumps the shed to zero --
    mirroring the X594 kernel-pull reference's combination, ground-truthed
    against its actual per-turn order output rather than inferred. Status:
    unvalidated as of this commit -- pending the mechanism trace and full
    30-seed protocol the plan requires before this note can say kept or
    reverted. Don't re-derive `have`/order-shape from scratch without reading
    the plan and the module docstring's 9th/14th/15th/17th attempts first."""
    live = _live_animal_count(farm)
    if live == 0:
        return []
    reserve = live * FEED_RESERVE_PER_ANIMAL
    shed_wheat = private.get("shed", {}).get("WHEAT", 0)
    carried_wheat = sum(inv.get("WHEAT", 0) for inv in private.get("inventories", []))
    have = shed_wheat + carried_wheat
    needed = reserve - have
    if needed <= 0:
        return []
    base = MARKET_PARAMS["WHEAT"]["base"]
    price = prices.get("WHEAT", base)
    if price > base * FEED_MAX_PRICE_FRAC:
        return []
    remaining = money - SAFETY_RESERVE
    affordable = int(remaining // price) if price > 0 else 0
    qty = min(needed, max(0, affordable))
    if qty <= 0:
        return []
    return [["BUY_PRODUCT", "WHEAT", qty]]


def _build_animal_orders(farm, private, money):
    """Keep at most ANIMAL_RESERVE bought-but-unplaced tokens per kind in
    the shed so a unit always has something to grab -- mirrors
    _build_seed_orders. Capped against the same land-share target
    _AnimalPlanner uses, counted by *live animals* (not structures --
    BUILD_PASTURE/BUILD_COOP is a free action, so structures reach the
    target well before purchases can catch up; counting empty structures
    as already "committed" here stopped this from ever buying enough
    animals to fill them, leaving land built but empty for the rest of the
    season).

    **Known bug, fix attempted and reverted**: a real replay showed 1 COW
    + 3 SHEEP ($1,900) permanently stuck in the shed from day ~16 through
    day 28+ with every PASTURE/COOP already full -- animal tokens can't
    be sold back once bought (COW/SHEEP/GOOSE aren't in kaggriculture.py's
    own PRODUCTS list), so that money is gone for the rest of the game.
    Confirmed in a second, unrelated replay too, so a real and recurring
    cost, not a one-off. Root cause as best diagnosed: this function's
    `committed` (shed tokens + live, ignoring empty structures) and
    _AnimalPlanner's own `_committed()` (live + empty structures, ignoring
    shed tokens) are two different counts that can drift apart, letting
    purchases outrun confirmed structure availability. Two fix attempts
    (gating purchases on empty-structure availability directly, at buffer
    0 and buffer 1 tokens of slack) both measured *worse* than this
    unfixed version in 10-seed pool comparisons against the exact same
    trimmed pool -- head-to-head against `reference/main_fertilizer_
    selling_submitted_20260817.py` went 9/10 -> 8/10 -> 5/10 seeds won as
    the fix got progressively less permissive, the opposite of the
    expected direction, and the regression reproduced across two separate
    runs of the buffer=1 variant. Not fully explained why gating more
    correctly against stranding costs more throughput than the stranding
    itself -- reverted rather than ship an unexplained regression chasing
    a bug worth an estimated ~$1,900/game (small next to the 20-50%
    swings the fix attempts caused).

    **Third attempt (2026-08-20), a clearer hypothesis, and a clearer
    root cause for why fixing it costs more than it saves.** Bulk
    instrumentation across 50 games (main.py vs a 5-opponent pool, via
    the fast C++ engine -- see cpp_sim/README.md) measured ~$59,300/50
    games (~$1,186/game) of animals bought and never placed -- close to
    the earlier single-replay estimate, and this time with a specific,
    previously-unnoticed mechanism: `committed` here counts shed tokens
    + live animals, but never a token a unit is *currently carrying*
    (picked up from the shed, en route to an empty structure) --
    invisible to this gate for however many turns the trip takes, during
    which a purchase can land that the true state didn't need. Traced
    directly on a stuck-animal replay: 88% of stranded-token turns had
    zero empty structures at all (buying outran a target that was, by
    then, already fully built), only 12% coincided with an actually-free
    structure sitting unused -- explaining why the first two attempts
    (which gated on structure availability, something that's almost
    never free anyway) choked off purchasing almost entirely instead of
    trimming just the excess.

    Adding carried tokens to `committed` fixed the measured bug cleanly
    (COW 62->3, SHEEP 69->0 stranded units across the same 50 games, a
    ~98% cut) but **still net-regressed a full 30-seed x `DEFAULT_POOL`
    A/B**: 71.9%->67.6% win rate (151W-59L -> 142W-68L), entirely
    concentrated in the two reactive self-play snapshots
    (`main_plant_priority_submitted_20260818.py`,
    `main_price_reactive_seed_reserve_submitted_20260819.py`) -- the
    other 5 opponents (including x492) showed zero flips either
    direction. Root-caused via a paired replay trace (seed 18 vs
    plant_priority, the single largest swing): live-animal count, FEED
    actions, and WHEAT produced/sold were all within noise between
    baseline and the fix (refuting the first guess that more successfully
    -placed animals means more feed competing with WHEAT sales) -- the
    real mechanism was `n_quadrants`: baseline stayed at 3, the fix
    reached 4. Buying fewer excess animals leaves more spare cash
    sloshing around, which `_build_land_orders`' own gate (`money -
    SAFETY_RESERVE >= LAND_PRICES[n_extra]`, no other limit once
    `_land_utilization >= LAND_UTILIZATION_GATE`) crosses more often --
    the "4th quadrant dilutes the same crew across more land" cost from
    the module docstring's Land / hands note, which `_land_worth_buying`
    claims the current hire formula "stops blocking... in practice," but
    empirically still nets a real loss when it fires. Checked across all
    15 flipped seeds: 6/15 show this exact `n_quadrants` 3->4 pattern:
    the other 9/15 flip without any land-purchase difference, so this is
    only part of the mechanism -- the fix shifts this agent's whole
    cash-flow timing (hiring, seed-buying, fertilizer-buying are all
    gated on spare money too), and this project's own history (the
    SEED_RESERVE doubling catastrophe, the whole feed-orders-starve-
    market-slots saga) shows this economy is unusually sensitive to
    exactly that kind of shift. Reverted a third time -- the underlying
    stranding bug and its now-precise mechanism are real and worth
    fixing, but not by this route alone; a future attempt should pair
    the carried-inventory fix with a tightened land-purchase gate (or
    otherwise re-examine every other money-gated decision for the same
    freed-cash sensitivity) rather than assume the animal-purchase fix
    is safe in isolation."""
    orders = []
    shed = private.get("shed", {})
    target = _animal_target(farm)
    committed = sum(shed.get(kind, 0) for kind in ANIMAL_KINDS)
    committed += _live_animal_count(farm)
    remaining = money - SAFETY_RESERVE
    for kind in ANIMAL_KINDS:
        if ANIMAL_TARGET_SHARE[kind] <= 0:
            continue  # e.g. GOOSE -- see ANIMAL_TARGET_SHARE's own comment
        if committed >= target:
            break
        if shed.get(kind, 0) >= ANIMAL_RESERVE:
            continue
        cost = ANIMALS[kind]["cost"]
        if remaining < cost:
            continue
        orders.append(["BUY_ANIMAL", kind, 1])
        remaining -= cost
        committed += 1
    return orders


def agent(obs, config=None):
    player = obs.get("player", 0)
    farms = obs.get("farms", [])
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    private = obs.get("private", {}) or {}
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    board_size = len(farm["tiles"])
    shed_tiles = _shed_adjacent_tiles(board_size)

    turns_per_day = int(_cfg_get(config, "turnsPerDay", DEFAULT_TURNS_PER_DAY)) or DEFAULT_TURNS_PER_DAY
    episode_steps = int(_cfg_get(config, "episodeSteps", DEFAULT_EPISODE_STEPS)) or DEFAULT_EPISODE_STEPS
    last_day = max(0, -(-episode_steps // turns_per_day) - 1)  # ceil(steps/turns_per_day) - 1
    days_remaining = last_day - day
    liquidating = days_remaining <= ENDGAME_LIQUIDATION_DAYS
    last_chance_window = hour >= turns_per_day - LAST_CHANCE_HOURS

    market_obs = obs.get("market", {}) or {}
    market_prices = market_obs.get("prices", {})
    market_inventory = market_obs.get("inventory", {})
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    inventories = private.get("inventories", [{}])

    want_fertilizer = _any_fertilizable(farm, day)
    planner = _CropPlanner(private, farm, days_remaining, day, market_prices)
    animal_planner = _AnimalPlanner(farm, shed, market_prices)
    animal_zone = _animal_zone(farm, board_size, animal_planner.remaining_structures())
    quadrant_util = _quadrant_utilization(farm, board_size)
    plant_priority = (
        PLANT_PRIORITY_UNDER_GATE if _land_utilization(farm) < LAND_UTILIZATION_GATE else PLANT_PRIORITY
    )

    # Routing efficiency (see the module docstring's own note and
    # .omc/plans/ralplan-global-assignment-v2.md): three passes instead of one single
    # per-unit scan. Pass 1 resolves each unit exactly as before for on-tile actions --
    # _decide_unit returns None for any unit not standing on an actionable tile. Pass 2
    # computes each remaining unit's own personal (inventory-dependent) candidate, if it has
    # one (_find_personal_target). Pass 3 matches every remaining unit against ONE shared
    # tile-only job pool (_build_tile_jobs) *and* each unit's own personal candidate
    # together, in one global sort (_assign_tile_jobs) -- instead of each unit re-scanning
    # the whole board and claiming its own single best candidate in a fixed, order-dependent
    # sequence.
    claimed = set()
    units = [farm["farmer"]] + list(farm["hands"])
    actions = [None] * len(units)
    pending = []
    for i, pos in enumerate(units):
        result = _decide_unit(
            pos,
            inventories[i] if i < len(inventories) else {},
            farm,
            private,
            day,
            claimed,
            planner,
            animal_planner,
            shed_tiles,
            want_fertilizer,
            liquidating,
            animal_zone,
        )
        if result is None:
            pending.append((i, pos))
        else:
            actions[i] = result

    if pending:
        shed_wheat_available = shed.get("WHEAT", 0) > 0
        personal = {}
        for i, pos in pending:
            p = _find_personal_target(
                pos, farm, day, board_size, claimed,
                inventories[i] if i < len(inventories) else {},
                animal_planner, shed_tiles, want_fertilizer, shed_wheat_available, liquidating,
            )
            if p is not None:
                personal[i] = p
                # Provisionally reserve the tile so a later pending unit's own personal
                # search (or the shared pool built below) doesn't also aim at it -- mirrors
                # the old sequential design's incremental `claimed` updates. _assign_tile_jobs
                # may still give this unit a different (pool) target instead; the reservation
                # errs toward avoiding a double-dispatch collision over squeezing out every
                # last bit of value from a tile that ends up unused this turn.
                claimed.add((p[0], p[1]))
        jobs = _build_tile_jobs(
            farm, day, board_size, claimed, animal_zone, quadrant_util, plant_priority,
            liquidating, planner, animal_planner, last_chance_window,
        )
        assignment = _assign_tile_jobs(pending, jobs, personal) if (jobs or personal) else {}
        for i, pos in pending:
            fx, fy = pos
            tx_ty = assignment.get(i)
            if tx_ty is not None:
                claimed.add(tx_ty)
                step = _step_toward(fx, fy, tx_ty[0], tx_ty[1])
                actions[i] = [step] if step else ["PASS"]
            else:
                actions[i] = ["PASS"]

    farmer_action, hands_actions = actions[0], actions[1:]

    # 16th market-order-slot-starvation attempt: cash_reserved tracks what earlier orders in
    # this same list have already committed to spend, so a later builder (_build_animal_orders
    # is last) sees the same shrinking budget the engine's own per-unit lockstep loop will
    # actually enforce, instead of every builder independently checking the turn's full opening
    # balance -- see _orders_cash_cost's own docstring and the module docstring's Same-turn cash
    # competition note.
    market = []
    sell_orders = _build_sell_orders(shed, market_prices, days_remaining, farm)
    market += sell_orders
    cash_reserved = _orders_cash_cost(sell_orders, farm, market_inventory)  # negative: SELL earns

    # Lever 3 of 4 (see the module docstring's Harvest-timing note's land-purchase-delay
    # follow-up): hire now runs before land, mirroring the X578/X594 kernel-pull reference's
    # own build order. The reverted third land-purchase-delay attempt gave land visibility
    # into this turn's own SELL proceeds while land still ran first, so land won the race for
    # that cash against hire/seed instead of losing to their staleness the way it used to --
    # worse, not better. Hire claims its own real cost first now; land only sees what's left.
    hire_orders = _build_hire_orders(farm, farm["money"] - cash_reserved)
    market += hire_orders
    cash_reserved += _orders_cash_cost(hire_orders, farm, market_inventory)

    land_orders = _build_land_orders(farm, days_remaining, farm["money"] - cash_reserved)
    market += land_orders
    cash_reserved += _orders_cash_cost(land_orders, farm, market_inventory)

    feed_orders = _build_feed_orders(farm, private, market_prices, farm["money"] - cash_reserved)
    market += feed_orders
    cash_reserved += _orders_cash_cost(feed_orders, farm, market_inventory)

    seed_orders = _build_seed_orders(seeds, farm["money"] - cash_reserved, days_remaining, market_prices, day)
    market += seed_orders
    cash_reserved += _orders_cash_cost(seed_orders, farm, market_inventory)

    fertilizer_orders = _build_fertilizer_orders(
        private, market_prices, farm["money"] - cash_reserved, want_fertilizer
    )
    market += fertilizer_orders
    cash_reserved += _orders_cash_cost(fertilizer_orders, farm, market_inventory)

    market += _build_animal_orders(farm, private, farm["money"] - cash_reserved)
    market = market[:MAX_MARKET_ORDERS]

    return {"farmer": farmer_action, "hands": hands_actions, "market": market}

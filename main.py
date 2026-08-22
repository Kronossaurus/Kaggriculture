"""Kaggriculture submission entry point.

Kaggle loads this file and calls `agent(obs)` (or `agent(obs, config)`) once
per turn for whichever player index we've been assigned. Keep this file
self-contained (or bundle extra modules into a tar.gz with this at the root
per AGENTS.md) since it's shipped standalone to the competition runner.

Strategy ("farm maxxer"): a scripted multi-unit farm loop built on top of
`reference/melon_maxxer.py` (Kaggle's own official example agent), fixing its
three big gaps -- it never expands land or hires hands (this agent buys land
and hires every day; hiring is fib(n) dollars for the n-th hire that day, so
essentially free), it only ever grows MELON (this agent grows a mix, see the
Crop mix note below), and it never fertilizes (this agent fertilizes every
one-time crop, whose unfertilized yield cap is strictly below fertilized --
MELON's cap is reachable by watering alone, so it's skipped there).

Below is a condensed log of what's currently in place and why, plus the
handful of specific ideas that were tried and clearly failed (so they aren't
blindly retried). The full blow-by-blow for any of this -- exact A/B numbers,
which replays motivated it, every reverted intermediate attempt -- lives in
git history and this project's session memory, not here; this docstring only
tracks design decisions that still affect how the code should be read or
changed.

**Methodology, load-bearing for everything below.** A single 10-seed
`pool_winrate.py` run has a real noise floor around +/-2 seeds out of 10 --
rerunning the identical script with no code change has produced different
win counts, worse when run concurrently with other heavy background work but
present even in isolation. Trust a 30-seed run (or a same-process A/B, i.e.
both variants evaluated in one script/process on the same seeds back-to-back)
over any single small run, and re-run before trusting a surprising result.
Money margin against a single opponent is not the same signal as win rate --
validate both, ideally against a panel that includes genuinely close
matchups, not just lopsided ones. Several "obviously safe" tweaks in this
file's history measured as real regressions despite looking correct on
paper; nothing here is kept on faith.

**Selling**: `_build_sell_orders` dumps the full shed once a crop's price
clears its sell threshold, same as melon_maxxer -- an earlier version capped
how much sold per turn (in case a big dump crashes its own price) and lost
19/20 seeds head-to-head, because the delayed reinvestment cost more than
the occasional price dip. FERTILIZER is also sellable and is a real revenue
source (see the "Selling fertilizer" trail this began from -- some real
opponents run animal-heavy setups that produce far more fertilizer than they
can use). `SHED_OVERFLOW_GUARD` bypasses the price threshold before the shed
overflows and produce gets discarded.

**Animals** (`_AnimalPlanner`, GOOSE/COW/SHEEP): every unit (farmer and
hands alike) shares one decision function across crop and animal work, with
FEED as the single highest priority of anything -- a missed feed costs the
whole animal. `ANIMAL_LAND_SHARE = 0.18` targets owned-land share (matching
the ~15-24% real top-ladder players run), further capped by
`ANIMAL_MAX_PER_UNIT = 1.0` per hired farmer/hand so the target never
outgrows the crew that has to tend both animals and crops -- a downloaded
replay showed the land-share-only version growing to 14 structures against 7
units, collapsing crop upkeep under the load. `ANIMAL_TARGET_SHARE` favors
COW/SHEEP over GOOSE (COW $80/day, SHEEP $66.7/day, GOOSE only $50/day by
product-value/production-interval; 5 real top-ladder replays never bought a
single GOOSE either). `_animal_zone` reserves the N tiles closest to the
shed for animal structures specifically, so they cluster tight around the
shed the way real top-ladder replays do, instead of competing tile-by-tile
with crops for the same near-shed land.
Don't retry without new evidence: dedicating a fixed subset of hands to
animals only (a severe net negative, ~-60 to -66% money -- fixed capacity
allocation wastes turns whenever its zone doesn't have enough work to fill);
raising `ANIMAL_LAND_SHARE`/`ANIMAL_MAX_PER_UNIT` together to chase a real
correlation between opponents' animal share and their win rate against us
(measured worse, -7% to -15% money, in a clean 30-seed A/B -- the
correlation in the replay data is real but isn't simply "run more animals
ourselves"; the leading open question is whether it's really about how
efficiently those opponents convert flock into money, not flock size).

**Pasture zone drift (next-steps item 2, investigated and reverted
2026-08-20).** `_animal_zone` reserves `needed` near-shed empty tiles, where
`needed` is `remaining_structures()` -- sized against `_animal_target`,
itself sized off *currently unlocked* land, which grows in jumps as
quadrants get bought. A replay trace confirmed a real gap: PASTURE build
order on one game went distance-to-shed 0,2,2,3 (day 0) -> 1,2,2 (day 10) ->
0,2 (day 11) -> 2,**6** (day 15) -> **6** (day 16) -- and at day 15 an empty
tile at distance 1, in an already-unlocked quadrant, sat unclaimed the same
turn a structure went up at distance 6. Crops fill whatever's empty and
unreserved the moment it's available; by the time land growth pushes
`_animal_target` up again, the near-shed ring within an already-open
quadrant is already gone, so new structures land wherever's still empty --
the far corners.

Two magnitudes tried, both reverted -- not a "wrong constant," a genuine
structural cost. Both added a second, more generous sizing for the zone
alone (never touching `_animal_target`/`remaining_structures` itself, so
neither changed how many animals get bought or when -- purely spatial).
**Unbounded** (reserve toward the eventual full-board/full-crew target
immediately, up to 13 tiles for a 10x10/12-hand game): catastrophic on the
first replay check, before any A/B was even run -- day 0, with only 25
tiles unlocked, this reserved 13 of them (52% of the whole starting
quadrant) for animal structures not yet funded and not built for many
turns, starving the early crop economy that funds everything else. Money
collapsed 79% on the traced seed ($49,478 -> $10,474), crop count crashed to
2 of 100 tiles by game end. **Bounded to a small constant margin instead**
(`min(remaining_structures() + N, eventual_target)`, tried at N=2 and N=1):
no collapse, and the drift itself measurably tightened (worst outlier
distance 6 -> 5 in the traced replay) -- but both sizes still regressed the
15-seed pool A/B by roughly the same amount (58.5% baseline -> 45.9% at N=2,
47.4% at N=1), hitting the same reactive-sentinel matchups hard
(`main_plant_priority`/`main_price_reactive_seed_reserve`/`tape_kiro` all
down sharply or to 0% at both sizes). Two very different magnitudes landing
on the same regression size rules out "wrong margin" -- **any** reservation
ahead of the exact current lazy shortfall costs more in foregone crop
planting than the drift it prevents saves, at least against opponents that
can react to it. Reverted at both sizes, main.py's `_animal_zone` call
unchanged from the original `remaining_structures()`-only form. The drift
itself is real and unaddressed; a future attempt needs a mechanism that
doesn't lock land out of crop use ahead of strict need -- e.g. letting
crops replant *into* a reserved-but-still-empty zone tile and displacing the
reservation elsewhere, rather than reserving further out preemptively.

**Price-aware planting** (`_CropPlanner`): a crop currently priced below its
own sell threshold is skipped for new planting (`_price_ok`) so land doesn't
keep replanting into a market it just crashed, falling back to ignoring
price only if every plantable crop is currently depressed. On top of that,
`_score` weights each candidate crop by `PRICE_ATTRACTIVENESS_WEIGHT * (price
/ base - 1)` so a crop trading well above base gets pulled toward the front
of the queue instead of competing on target-mix-share gap alone.
`PRICE_ATTRACTIVENESS_WEIGHT = 0.1` is a first-guess value, not yet tuned
(see the next-steps memory).

**Land / hands**: `_build_land_orders` gates each `BUY_LAND` on
`LAND_UTILIZATION_GATE = 0.50` of currently-owned land already being
planted, so cash doesn't buy land faster than it can be staffed. It's tuned
between two failure modes found in real replays: too high (0.6) deadlocked
this agent below its own achievable utilization ceiling (~48-56%) in most
real games; too low regressed money 18-21% against the local pool by
enabling oversupply before the crew could actually work the new land.
`_land_worth_buying` additionally blocks a 4th-quadrant purchase outright --
buying it doesn't raise hire capacity if the hiring formula is already
capped, so it just dilutes the same crew across more tiles for no benefit
(this matches how the pool's strongest opponent, x492, plays: it never buys
past 3 quadrants either). `_hands_target` uses `HANDS_BASE +
HANDS_PER_QUADRANT * quadrants` (5/8/11/12 hands at 1-4 quadrants),
calibrated off real top-ladder hiring curves (which hire far more
aggressively and far less land-proportionally than a flat formula) --
raising `MAX_HANDS` alone without this shape change was tried earlier and
measured worse (more hands means more simultaneous harvests crashing the
same finite-demand market). `_quadrant_utilization` inflates a fresh PLANT
candidate's effective distance by how built-up its quadrant already is, so
a same-distance tie between a crowded and a neglected quadrant favors the
neglected one -- crops were clustering near the shed and leaving distant
land to weeds without this.
Don't retry without new evidence: sizing hand count off a live count of
actionable tiles instead of the flat quadrant formula (reproduces real
players' fluctuating hire pattern, but over-hires during backlog spikes and
costs more in oversupply than it saves, -3% to -34% depending on variant).

**Crop mix**: `CROP_TARGET_SHARE` = MELON 0.45 / WHEAT 0.25 / CARROT 0.15 /
TOMATO 0.10 / STRAWBERRY 0.05. MELON dominates because its base price ($250)
dwarfs the others and its price-crash risk is already handled by the
price-aware-planting gate above -- cutting its share to make room for
TOMATO/STRAWBERRY was tried and was a clean loss against every pool
opponent. TOMATO/STRAWBERRY are "ongoing" crops (produce on a fixed
multi-harvest schedule from one planting, no replanting between harvests);
`_last_yield_day` extends their fertilization eligibility window to their
real last production day, not just `max_yield_day` (a placeholder that
undercounted it for these two).
Don't retry without new evidence: shifting the mix hard toward STRAWBERRY
specifically (inspired by real top-ladder replays that sustain much higher
land utilization with it) was the single worst regression measured this
project -- not a price crash (STRAWBERRY price kept rising) but a pure
throughput failure, produce piling up in carried inventory and never
reaching the shed to sell (see the shed-return-priority note below for the
mechanism and why the direct fix also regressed). The insight that ongoing
crops sidestep the seed-market-slot-starvation bug (below) is still
plausibly correct; the crop-mix shift itself just exposed an execution gap
this agent isn't equipped to capitalize on yet.

**Endgame liquidation**: as `days_remaining` shrinks, this agent stops
buying seeds/land that can't pay back in time, prioritizes walking carried
produce home over watering a crop that won't finish, and drops the
sell-price threshold entirely so nothing sits unsold when the market closes
(unsold inventory doesn't count toward final score). Carried FERTILIZER
specifically also stops being re-fetched once liquidating and instead counts
as sellable "produce" -- a unit was otherwise still walking to fetch more
fertilizer for a crop with no season left to benefit from it. WHEAT's feed
reserve is deliberately exempted from this -- missing a feed this late still
loses the whole animal, a bigger loss than the travel time saved.

**Endgame liquidation (WHEAT)**: a carried WHEAT batch below
`WHEAT_SHED_RETURN_THRESHOLD` normally stays held (see the Carried-stock
accounting note's PICKUP/DROP livelock guard) since a dedicated trip for a
small batch isn't worth it mid-game. First attempt: added `or liquidating`
to both `_find_target`'s `returning_wheat` trigger and `_decide_unit`'s
carried-produce DROP gate, mirroring FERTILIZER's own liquidation escape
exactly. Mechanically it worked (a same-seed trace vs. `tape_kiro` dropped
stranded end-of-game WHEAT 37->24 units, shed WHEAT 9->0), but a 15-seed
pool A/B (default pool + both reactive sentinels + `tape_kiro`/
`tape_kawashigi`) regressed real money in nearly every matchup (pool total
59.3%->49.6%) including several 100%-win-rate ones where the win count
didn't move. Root-caused with a per-resource money-flow trace (same
technique as the 14th/17th feed/seed-starvation attempts): `tape_kiro`'s
own realized trade flow only shifted +$1,282 (a small MILK/WOOL
crew-attention echo), while *our own* money alone dropped $3,257 on the
same seed -- not a shared-price effect, a direct opportunity cost.
`_find_target`'s `returning_wheat` trigger shares `shed_priority` with
FERTILIZER's liquidation return (tuned for FERTILIZER's rarer, larger
batches); routine feed PICKUPs leave nearly every unit holding some stray
sub-threshold WHEAT constantly, so extending the trigger to WHEAT fired a
dedicated high-priority trip far more often than FERTILIZER ever does,
crowding out higher-value final-day harvesting/feeding for a net loss
larger than the WHEAT recovered.

**Second attempt, narrower: still regressed, identically.** `_find_target`'s
`returning_wheat` trigger reverted to threshold-only (no `liquidating`
clause -- never a dedicated trip for a small batch, matching the
established "piggyback, don't dedicate a trip" shape from the 12th
attempt's own note); only `_decide_unit`'s DROP gate kept `or liquidating`.
The same seed-42-vs-`tape_kiro` replay was byte-identical to the first
attempt's (same 24 units stranded, same final money to the dollar) --
proving `_find_target`'s trigger was never the mechanism on that seed at
all -- and the 15-seed pool A/B landed on the *same* 49.6% total. The real
cause was the DROP gate itself: `_decide_unit` checks `carried_produce`
(DROP) before the WHEAT-PICKUP-for-feed branch, and the liquidating escape
didn't check whether an animal still needed that WHEAT -- so a unit
standing at the shed already carrying feed WHEAT for a still-hungry animal
would DROP it there instead of walking it to the animal, directly risking
the exact loss ("missing a feed this late still loses the whole animal")
the pre-existing code's own `needs_wheat` design had always protected
against everywhere else.

**Third attempt: added the missing feed-need guard, kept and confirmed.**
New helper `_any_unfed_animal(farm)` (mirrors `_any_fertilizable`); the DROP
gate's WHEAT clause is now `n >= WHEAT_SHED_RETURN_THRESHOLD or (liquidating
and not _any_unfed_animal(farm))` (`_decide_unit`'s own
`wheat_safe_to_liquidate` local). Endgame-only (last
`ENDGAME_LIQUIDATION_DAYS=2` days). **Confirmed at the full 30-seed protocol
(default pool + both reactive sentinels + `tape_kiro`/`tape_kawashigi`)**:
157W-113L (58.1%) with and without the fix, identical pool total -- the two
reactive sentinels each flipped exactly one game in opposite directions
(noise), every other matchup matched win-for-win with mean money within
$100-1000. The two prior shapes (unbounded `liquidating` clause, and
`_find_target`-trigger-included) are both confirmed regressed and must not
be reapplied without this guard.

**Neglected-tile urgency**: a tile that missed watering once
(`consecutive_unwatered >= 1` -- two consecutive misses turns a PLANT into a
WEED) scores priority 0.5 in `_find_target`, above every routine chore but
below harvest (already-earned money) and still below at-risk situations.
Without this, a unit chasing a marginally-closer routine chore near the shed
would keep winning over saving a tile one missed watering from being lost
outright. Paired with raising `MAX_HANDS` 6 -> 8 (which conveniently matches
2x the 4-quadrant land cap, so it stops binding at all) to recover a
capacity dip the priority change caused on its own.

**Harvest-ready tiles rotting to weeds**: a mature, harvestable tile is
still subject to the same 2-consecutive-miss WEED conversion as an
unharvested one -- kaggriculture.py's daily refresh doesn't exempt already-
ripe plants, only harvesting one removes the risk. Before this fix,
`_find_target`'s HARVEST branch (`_is_harvestable`) never carried an
urgency signal the way the water branch's `at_risk` flag does, and
priority-0 harvest candidates (every ripe crop plus every ready animal
product) were tie-broken by raw distance alone -- so a tile ripe for many
turns but sitting far from every unit could keep losing to whichever tile
merely matured most recently nearby, until its own 2-miss clock ran out.
Confirmed directly in two real ladder losses (see the idle-movement-thrash-
and-harvest-weed-loss memory): 10 of 13 and 10 of 14 PLANT->WEED
conversions had already been harvestable, sellable value at some point --
a real, avoidable loss, not just an unwatered seedling dying young.
`HARVEST_AT_RISK_DIST_BIAS` biases `sort_dist` (not priority, already top
tier) for a harvestable tile with `consecutive_unwatered >= 1`, so it's
served ahead of any merely-closer, non-urgent priority-0 candidate.

**Target stickiness -- tried and reverted.** Offline replay analysis (see the
idle-movement-thrash-and-harvest-weed-loss memory) found `_find_target`
recomputing every unit's target from scratch every turn caused real thrash:
~30% of all movement actions switched to a different destination before the
previous trip completed, ~85% of those switches to a same- or lower-priority
tile than the one just abandoned -- never legitimate under the priority-first
sort. Added a per-unit-index persisted `sticky_target` that kept a unit
committed to its previous target unless a strictly more urgent candidate
appeared. Regressed hard once actually validated (a first version had a
pop-before-read ordering bug that silently made it inert -- its own dedicated
unit tests passed anyway, since they exercised `_find_target` and
`_decide_unit` in isolation, never a real two-call sequence through
`agent()`'s own per-turn wiring; only a same-process 3-way A/B against the
full pool caught it, after which the bug was fixed and stickiness genuinely
activated ~50% of the time it applied). With the bug fixed, a clean 30-seed
A/B (isolated from the harvest-at-risk fix above) dropped the full-pool total
from 67W-83L (44.7%, item-4 alone) to 43W-107L (28.7%) -- worse than doing
nothing at all (57W-93L baseline, 38.0%). Regressed against every reactive
opponent tested (mirror snapshots and a price-reactive one alike); the two
non-reactive real-tape sentinels' win counts didn't move, only their margin.
Mechanism not fully pinned down, but plausible: forcing per-unit commitment
removes the emergent efficiency of the whole crew re-optimizing its
assignment fresh every turn (a cheap, effective auction-like heuristic this
agent has relied on all along), trading a small amount of real thrash for a
worse division of labor overall. **Lesson for next time this area is
touched**: a "thrash" metric measured from outside the simulation doesn't
mean the thrashing itself is costly -- verify a fix's activation rate
directly (e.g. an instrumented offline count like the one that caught the
inert bug here) *before* trusting a clean-looking unit test suite, and always
validate with the full A/B panel, never trust intuition about wasted
movement alone.

**Shed-return priority**: `_find_target`'s priority for walking carried
produce back to the shed is normally the worst tier (5, below even routine
watering/weeding) -- fine under a diversified crop mix, but a unit carrying
produce under a dense, continuously-cycling crop (see the STRAWBERRY note
above) never won the race home against a steady stream of priority-0 harvest
candidates elsewhere, so produce piled up in carry and never got sold.
Raising the priority unconditionally for every carrying unit was tried and
regressed (a unit carrying just 1-2 units started abandoning weeding and
boosted planting too readily to cash out early). The fix that stuck:
`SHED_RETURN_URGENT_PRIORITY = 2` only once a unit is carrying at least
`CARRIED_PRODUCE_URGENT_THRESHOLD = 3` units -- a light load still finishes
a nearby routine chore first, a real backlog heads home. Validated with a
clean sequential 30-seed full-pool comparison (144W-66L -> 146W-64L overall,
the two genuinely close matchups splitting +3/-1 rather than uniformly
worse) plus a same-process A/B against the pool's two most decisive
opponents (+6.6%/+8% money, win rate already saturated at 30/30 and 0/30
either way).

**Market-order-slot starvation: real, confirmed, and currently unfixed --
don't blind-retry.** `agent()`'s market order list is built
sell -> land -> hire -> feed -> seed -> fertilizer -> animal, then silently
truncated to `maxMarketOrdersPerTurn` (10). `_build_feed_orders` (one
`BUY_PRODUCT WHEAT` order per unit needing feed, sometimes 8-9 orders alone)
routinely crowds out `BUY_SEED`/fertilizer orders most turns, confirmed
directly in replay `action.market` data and a real cause of planted-tile
counts collapsing over the back half of a season despite idle cash --
confirmed at scale against 33 real ladder episodes from a live submission:
94% mean crop-count collapse from peak by game end (vs. 72.3% for
opponents), seed orders at exactly 0/turn for a mean 12.2 of the last 15
days per game. Six independent fix attempts across this project (full order
consolidation twice, an order-count cap, a WHEAT sell-side reserve, a
per-order quantity cap, and same-turn sell/feed coordination -- see the
WHEAT sell/rebuy round trip note below for the sixth's own detailed
writeup) all measured as real regressions on a clean same-process 30-seed
A/B, consistently worst against a genuine non-mirror opponent
(`reference/tape_kiro.py`, 0W-30L for the sixth attempt -- the worst
sentinel result yet). The first five share one likely mechanism: this
agent's own per-unit affordability check uses a static price and doesn't
account for the engine's own within-order price walk, so any change that
lets a turn request more WHEAT in fewer, larger orders ends up spending
more at a worse average price than the old scheme's incidental slot-count
cap ever allowed. The sixth attempt built and used the previously-missing
instrumentation (`scripts/analyze_replay.py --orders`/`--purchase-trace`,
now available -- see their own docstrings) and found a *different*
mechanism specific to it (a standing WHEAT-price-level shift from reduced
sell volume, not the within-order price walk -- see below); freeing market
slots is confirmed real and reproducible but is not sufficient on its own.
Any further attempt should use this same instrumentation to check its
purchase-price and slot-occupancy effects before trusting a 30-seed A/B
result, not just cite this note.

**Seed reserve**: `SEED_RESERVE = {"MELON": 3, "WHEAT": 3, "CARROT": 3,
"TOMATO": 2, "STRAWBERRY": 2}` scales up per-crop by
`SEED_RESERVE_PRICE_CAP = 2.0` when that crop's own price sits above its
base (mirrors the price-attractiveness planting logic, applied to how far
ahead of need to buy rather than what to plant right now) -- validated
within noise of the unmodified baseline against the local pool, and a clean
+27% money win against `reference/tape_kiro.py`.
Don't retry without new evidence: doubling every crop's flat reserve
uniformly (to give `_build_seed_orders` more runway on the rare turn it
wins a market slot) was the single worst regression measured in this
project before the STRAWBERRY-mix one above -- it drove enough more WHEAT
through the crop cycle to blow up the WHEAT sell/rebuy round trip's volume
(see below) far past its normally-bounded cost.

**Price-reactive fertilizer buying**: `_build_fertilizer_orders` keeps a
flat `FERTILIZER_RESERVE = 2` most of the time, but stocks up to
`FERTILIZER_STOCKPILE_RESERVE = 10` whenever price drops to
`FERTILIZER_CHEAP_PRICE_FRAC = 0.35` of base or below (deliberately under
`SELL_PARAMS["FERTILIZER"]`'s own 0.50 sell threshold, so it can never buy
back at a price it would also sell at) and something is actually
fertilizable right now. Motivated by real animal-heavy opponents (e.g.
`reference/tape_kiro.py`) who dump so much byproduct fertilizer that its
price crashes to a fraction of base. Confirmed a no-op (not a regression) in
every locally measured matchup so far, including a corrected replay of that
exact opponent -- FERTILIZER does get cheap there, but the stockpile order
never wins a market slot in the first place, another casualty of the
slot-starvation bug below. Kept as a small, gated, provably-safe change; not
yet validated as an actual win.

**WHEAT sell/rebuy round trip**: `_build_sell_orders` dumps the entire shed
including WHEAT the moment its price clears threshold; `_build_feed_orders`
then immediately rebuys WHEAT from the market for animal feed -- a real,
confirmed, self-inflicted round trip through both sides of the market's own
price impact. Real top-ladder replays do the exact same thing constantly and
profit from it (WHEAT nets +9% to +88% of their final money), within a much
tighter price band (buys 1.00x-2.08x base, sells 1.08x-2.08x) than this
agent's own 0.45x sell threshold / 5.0x feed-buy cap.
Don't retry without new evidence: holding back a standing WHEAT reserve from
the sell dump (three sizes tried) looked like a clean win on quick 10-seed
checks but was a genuine large regression on a properly-powered 30-seed
recheck (83.3% -> 53.3% win rate against the pool's toughest opponent) --
delayed reinvestment barely dents a blowout against weak opponents (ceiling
effect hides the cost) but flips close races where marginal efficiency
decides the outcome. A random-search sweep of `WHEAT_SELL_THRESHOLD` /
`FEED_MAX_PRICE_FRAC` within the real observed range didn't find anything
that held up at 30 seeds either, though it did surface a hard floor worth
remembering: `FEED_MAX_PRICE_FRAC` below ~1.4 can't clear the price often
enough to feed animals at all (catastrophic, animals lost outright) --
today's 5.0 has plenty of headroom above that. A sixth attempt -- coordinating
`_build_sell_orders` and `_build_feed_orders` within the same turn (hold back
exactly `live * FEED_RESERVE_PER_ANIMAL` WHEAT from the sell dump, the same
formula the feed side already computes, instead of a standalone reserve
constant) -- was built with new instrumentation this time
(`scripts/analyze_replay.py --orders`/`--purchase-trace`) rather than blind.
The order-slot side of the fix genuinely worked (seed orders that were
completely zeroed for 12-17 straight days in real ladder replays started
getting through again, confirmed both offline against historical replay data
and live), but it regressed hard anyway: 146W-64L -> 100W-110L against the
pool (69.5% -> 47.6%), and 19W-11L -> 0W-30L against `reference/tape_kiro.py`
specifically, the worst sentinel result of any attempt in this file's
history. `--purchase-trace WHEAT` on a flipped matchup pinned the mechanism
precisely, and it's a different one than the first five attempts: not the
static-price-vs-within-order-price-walk effect (the `next_price/decision_
price` ratio stayed ~1.00, flat, ruling that out here), but a standing
supply/demand shift -- selling less WHEAT per turn means less downward price
pressure on it, so WHEAT's market price sits higher throughout the game, and
since feed rebuys happen far more often than sell-triggering harvests, the
agent ends up paying much more per unit for the far more frequent purchase
side of the round trip (realized average WHEAT purchase price 5x higher,
$18.53 -> $92.63/unit, on the traced matchup) than it gains back from
selling into that higher price occasionally. Freeing market-order slots is
not sufficient on its own if it changes the standing market price regime.
The round trip and the slot-starvation collapse it causes are both still
real and still unaddressed; don't retry either the reserve-sizing or the
same-turn-coordination shape again without a new idea that doesn't suppress
aggregate WHEAT sell volume.

A seventh attempt tried a different lever specifically to avoid that trap:
cap `_build_feed_orders` at `FEED_ORDER_CAP_PER_TURN` orders per turn
instead of touching sell-side volume at all. Reasoning at the time: `needed`
recomputes fresh from `reserve - have` every turn, so a capped remainder
just carries over and gets bought next turn -- same total WHEAT bought over
the season at the same per-unit prices, only spread across more turns,
which should leave the standing supply/demand price level alone. It
regressed just as hard as the sixth attempt anyway: pool 146W-64L -> 124W-
86L (69.5% -> 59.0%), `tape_kiro` 63.3% -> 6.7%, and both close-mirror
opponents (`main_plant_priority_submitted_20260818.py`,
`main_price_reactive_seed_reserve_submitted_20260819.py`) collapsed from
33.3%/53.3% to 6.7%/6.7% each. `--purchase-trace WHEAT` on a same-seed
mirror-matchup pair confirmed it's the *same* standing-price mechanism as
the sixth attempt (mean implied WHEAT purchase price $17.30 -> $36.78,
2.1x, with `next/decision_ratio` flat ~1.00 in both -- ruling out the
first five attempts' within-order-price-walk mechanism again): capping the
catch-up rate keeps the shed chronically under `reserve` instead of
occasionally over-buying in a lump and leaving the market alone in between,
turning the agent into a near-permanent marginal WHEAT buyer that keeps
pushing its own purchase price up. The reasoning above about total volume
being unchanged was correct but insufficient -- *when* and *how evenly*
that volume gets bought matters just as much as how much, and a
per-turn cap forces the worst pattern (constant small buys) instead of the
better one (occasional lump buys with recovery time between). Freeing
order slots for seed is confirmed real (seed orders went from a flat 0.00
to ~0.2-0.5/turn) but is still not sufficient on its own if the mechanism
used to free them turns feed into a chronic rather than occasional buyer.
Any eighth attempt needs a shape that neither reduces sell volume nor
chronically throttles buy frequency -- e.g. something that changes *which*
turns feed is allowed to buy on (favor cheap-price turns) rather than *how
many* orders it's allowed per turn.

An eighth attempt tried exactly the minimal version of that within the
"free slots for seed" family: no cap at all, just reorder `agent()`'s
market list to put `_build_seed_orders` ahead of `_build_feed_orders`
(seed rarely needs more than 1-2 slots on an average turn, so this should
only cost feed a slot on the rare turn both compete, leaving feed's normal
~8-9-slot turns untouched otherwise). It regressed *worse* than the
seventh attempt: pool 146W-64L -> 118W-92L (69.5% -> 56.2%), `tape_kiro`
63.3% -> 6.7% (same as the 7th), and the two mirror opponents fell to
3.3%/0.0% (worse than the 7th's 6.7%/6.7% each). A same-seed replay pair
against `tape_kiro` showed why the "seed rarely needs more than 1-2"
assumption was wrong in practice: `_build_seed_orders` replaces every seed
a unit consumed by planting *that same turn*, so a turn where several
units all plant simultaneously (e.g. right after the land-utilization gate
opens up a burst of empty tiles) needs several seed orders at once --
exactly the same turns feed also has a large backlog, so seed's burst
demand, now front-of-list, ate far more of feed's slots than the mean
seed-orders/turn figure suggested (feed's occupancy fell from a steady
~8/turn to 0-2/turn for long stretches, not just an occasional single
slot). The opponent's own money rose 28% ($44,319 -> $56,730) on that
matched seed despite running a fixed, non-reactive tape -- the standing
WHEAT-price mechanism again, this time triggered by burst competition for
list position rather than either sell volume or buy pacing.

**Both members of the "give seed priority over feed within the same
turn's order list" family (a hard per-turn cap, and pure list reordering)
have now failed, each roughly as badly as the sell-volume-based sixth
attempt.** Don't retry a third variant of this family (e.g. a *dynamic*
cap, or reordering only above some threshold) without a fundamentally
different idea -- the failure mode isn't sensitive to the specific
mechanism used to hand seed a slot at feed's expense, only to the fact
that feed's own order volume/timing gets disturbed at all. The module
docstring's earlier suggestion (make feed favor *which* turns it buys on,
by price, rather than changing its slot count) is a genuinely different
family, still untried.

A ninth attempt asked a different question first -- "how do real top
players handle this?" -- instead of guessing another constant. Analyzed
the 6 locally cached `top_ladder` replays: a player who recurs across 4
of them (カワシギ, 4/4 wins) submits a feed BUY_PRODUCT order on only
~12% of turns with live animals (this agent: near 100% of such turns
before any fix), always as a *single* order, sized to refill toward
roughly 1.0 WHEAT/animal (reverse-engineered from `shed_wheat / live` at
decision time: median 0.17 when they buy, 0.56 when they don't -- see
`feed_trigger_trace.py`, not checked in, built ad hoc for this). Tried the
direct translation: halve `FEED_RESERVE_PER_ANIMAL` 2 -> 1 *and* replace
the per-unit order loop with one consolidated `["BUY_PRODUCT", "WHEAT",
needed]` order. This is a different combination than the fifth attempt's
quantity cap (which kept the old reserve size, so it still fired nearly
every turn just capped smaller) or the seventh/eighth (which touched
slot count/position, not the reserve target).

**Regressed the worst of all nine attempts: `tape_kiro` 63.3% -> 0.0%
(0W-30L).** A same-seed replay pair showed animals did *not* starve
(final live-animal count unchanged, 13 both runs) -- so this wasn't the
catastrophic-starvation failure mode. It was the same standing-price
mechanism again, worse than ever: mean implied WHEAT purchase price
$20.85 -> $51.72 (2.5x). `--orders` explained why the smaller target
didn't reduce trigger frequency the way the top player's own ~12% rate
suggested it should: feed still fired on ~50% of turns, because
`_build_sell_orders` dumps the *entire* shed the instant WHEAT clears its
0.45x sell threshold, resetting `have` toward zero on every sell
regardless of how small the reserve target is -- the trigger is reset by
the sell side, not sized by the buy side. The slot-starvation symptom
*did* genuinely resolve this time (saturated-turn frac fell from ~100% to
4-12%, seed orders flowed normally) -- proving the mechanism is fixable
this way -- but at a worse economic cost than the starvation itself.
**Reverted** (never committed).

This confirms, with real mechanism evidence rather than another guess:
**no buy-side-only lever (order shape, slot position, reserve size, or
any combination) can fix this without also touching the sell side**,
because this agent's shed has no organic WHEAT buffer between harvest and
sale to draw from the way the top player's apparently does -- and the
sell side has already failed four separate times (three reserve sizes,
one same-turn coordination) via the same standing-price mechanism when
tried directly. The real gap with the top player isn't reserve sizing,
order shape, or price thresholds (all now tried and ruled out) -- it's
some other, still-unidentified difference in how they decide *when* to
sell WHEAT at all, that lets enough of it accumulate for feed without
this agent's version of that same idea blowing up the standing price.
Nine attempts plus one threshold-sweep redo have now failed; the next one
needs to explain that specific gap, not add another lever to either side
of the sell/feed pair.

A tenth attempt tried exactly that gap directly instead of another buy-
side lever: measured a recurring top-ladder player's WHEAT *selling*
behavior (`sell_trigger_trace.py`, scratchpad) across their 4 cached
replays and found only 54% of their WHEAT sells are a full shed dump --
the rest hold something back, with price ratio at sale (min 1.08, median
1.80) always above the 1.08 floor also seen on their hold turns (this
matchup's WHEAT price apparently never drops much below base). Built
`_build_sell_orders` a `wheat_reserve` parameter (held back from the dump
unless crowded/liquidating -- the same shape as three already-reverted
sell-side reserve attempts) but paired it with a raised
`CROP_TARGET_SHARE["WHEAT"]`, on the reasoning that if the reserve comes
out of *added* production instead of existing sell volume, the standing-
price mechanism that killed those three attempts (less selling -> higher
price) shouldn't trigger the same way.

Tried two magnitudes, both against `tape_kiro` (30 seeds): a modest bump
(WHEAT 0.25->0.30, MELON 0.45->0.40) scored 4W-26L (13.3%) -- better than
the 7th/8th/9th attempts' 0-6.7%, but a same-seed trace showed why it
still fell short: actual WHEAT *sell* volume dropped ~4-5x from baseline
(the `wheat_reserve`, sized at `live * FEED_RESERVE_PER_ANIMAL = 2`, ate
nearly all of the small extra production rather than coming out of
genuinely added volume) and implied purchase price hit $84.72, the worst
of any attempt. A much larger bump (WHEAT 0.25->0.45, MELON 0.45->0.25,
to actually outrun the reserve) made it *worse*, not better: 1W-29L
(3.3%). Escalating the production shift further is the wrong direction --
**WHEAT is simply worth much less per tile than MELON** (see MELON's
steep above-equilibrium price shape, Price-aware planting note above),
so growing more of it to fund a feed reserve trades one loss (the
standing-price regression) for a second, independent loss (a
worse-value crop mix) that gets *larger*, not smaller, the harder this
lever is pushed. **Reverted both magnitudes** (never committed).

This rules out "compensate the reserve with more production" as a viable
direction at all, regardless of magnitude -- not just the two sizes
tried. Ten attempts plus one threshold-sweep redo have now failed. The
top player's own partial-sell behavior (54% partial, price floor ~1.08x
in this matchup) is still unreplicated, but the mechanism must be
something that doesn't cost crop-mix value to achieve -- e.g. holding
back WHEAT only when its *own* price is unusually low relative to its
own recent range (not a fixed reserve size or a bigger target share),
though this hasn't been tried and isn't assumed to work either.

**An eleventh attempt found a real, previously-undiscovered structural
gap, not another tuning knob -- but the direct fix has its own,
different failure mode.** `_find_target`'s `carried_produce_total`/
`_decide_unit`'s `carried_produce` both exclude WHEAT from the general
"carry harvested produce back to the shed" mechanic (`k not in ("WHEAT",
*ANIMAL_KINDS)`), same as the animal-placement tokens. Unlike those
tokens, WHEAT has no other route to the shed except DROP -- which dumps
a unit's *entire* inventory (data/README.md), so WHEAT only reaches the
shed as an incidental bystander when a unit happens to also be carrying
some other crop that itself triggers a shed-return. This means our own
WHEAT harvest mostly never reaches the shed at all: it either gets
consumed directly by an immediate FEED opportunity (priority -1, still
wins) or sits stranded in a unit's inventory indefinitely. This explains
several things at once: why our own `purchase_trace` WHEAT
`total_acquired`/`total_requested` ratio (~1.2x) is so much smaller than
the top player's (~1.75x, see the ninth attempt's note) -- their harvest
genuinely reaches their shed, ours mostly doesn't; and why the sell-side
reserve attempts (three sizes plus the tenth's production-funded
variant) never had much organic supply to reserve *from* in the first
place.

The direct fix -- just remove WHEAT from both exclusions, so it
participates in the normal carried-produce shed-return logic like any
other crop -- tests catastrophically, but through a *different*
mechanism than any of the previous ten: `tape_kiro` 63.3% -> **0.0%**
(0W-30L), mean money collapsing to $2,648 (vs. baseline's ~$50k, not just
a percentage regression). A same-seed trajectory showed why: **weed
count exploded from 8 to 34 by game end** (on 75 owned tiles) -- not a
market-price effect at all. WHEAT matures in just 2 days and is a single-
shot harvest (`interval=0`, replant every cycle) at 25% of the crop mix,
so a large fraction of units are carrying *some* small amount of WHEAT
almost continuously. Once that counts toward `carried_produce`, each of
those small harvests can trigger a dedicated shed trip (routine priority
5, or urgent priority 2 once `CARRIED_PRODUCE_URGENT_THRESHOLD=3` is hit),
pulling units away from routine watering/weeding often enough that crops
rot into weeds faster than they can be tended. **Reverted** (never
committed).

The underlying gap is real and still unfixed -- WHEAT genuinely has no
organic path to the shed today, unlike every other crop -- but naively
treating it identically to slower-cycling, lower-volume crops in the same
priority/threshold scheme is itself the new failure mode.

**A twelfth attempt, kept.** Added `WHEAT_SHED_RETURN_THRESHOLD = 5`:
`_find_target` now generates a dedicated shed-return candidate once a
unit carries at least that much WHEAT (kept out of
`carried_produce_total`/the urgent-priority check, so it's always routine
priority 5, never urgent -- avoiding the 11th attempt's direct collision
with weeding), and `_decide_unit`'s `carried_produce` DROP-eligibility
check now includes WHEAT under the same threshold. First test at
threshold 4 reproduced the 11th attempt's exact weed-collapse again --
diffing the two replays' actions showed only 1 of 720 turns differed,
proving the real bug wasn't trip generation at all: it was
`_decide_unit`'s DROP check running on the very next turn after a
`["PICKUP", "WHEAT", n]` for feed (up to 4 units, see `_decide_unit`'s own
PICKUP call) while the unit is still standing at the shed tile, seeing its
own freshly-fetched feed WHEAT in `carried_produce`, and dropping it right
back before ever walking to the animal -- the same livelock family the
FERTILIZER guard immediately above already exists to prevent (see
`test_decide_unit_keeps_freshly_picked_up_fertilizer_when_still_needed`),
just not yet extended to WHEAT. (Confirmed empirically that WHEAT's own
`yield_units` is always exactly 1 per harvest in practice, not up to
`max_yield=6` as assumed when picking the first threshold -- `max_yield`
evidently isn't "units per harvest" for a single-shot, `interval=0` crop.)
Raising the threshold to 5 (strictly above PICKUP's 4-unit cap) fixes the
livelock: the same traced seed now nearly exactly reproduces the
baseline's own trajectory (win instead of catastrophic loss, weed count
back to 1, `--purchase-trace` numbers matching baseline within noise).
Validated with the full 30-seed protocol: `tape_kiro` 63.3% -> 60.0%
(18W-12L, noise-level) and pool 69.5% -> 69.5% (146W-64L, identical) --
confirmed safe, no regression anywhere.

**Kept as a genuine correctness fix, not a starvation-bug fix.** Because
WHEAT yields exactly 1/harvest, a threshold of 5 (the minimum that avoids
the livelock) rarely triggers in practice -- a unit would need to harvest
5 separate WHEAT tiles before doing anything else, which routing priority
rarely allows. So this closes the *gap* (WHEAT-only carry no longer gets
permanently stranded) without materially changing the *starvation*
symptom this whole investigation started from -- the neutral 30-seed
result confirms this precisely: no worse, but not meaningfully better
either. A thirteenth attempt wanting WHEAT to flow to the shed *more*
often would need to lower this threshold, which requires first lowering
PICKUP's own 4-unit feed-fetch cap to keep the anti-livelock margin (or
finding a non-quantity-based way to distinguish "just fetched for a still-
valid feed trip" from "safe to bank") -- not yet tried.

Twelve attempts across four sessions plus one threshold-sweep redo: eleven
reverted, this one kept as a real (if narrow) correctness improvement.

**A thirteenth attempt did exactly what the twelfth's note asked for --
lower the anti-livelock margin using the real feed-need signal instead of
a quantity threshold -- and it regressed just as hard as every purely
economic attempt before it.** Added `_any_unfed_animal(farm)` (mirrors
`_any_fertilizable`) and replaced `WHEAT_SHED_RETURN_THRESHOLD` with a
`want_wheat_feed` gate on both `_find_target`'s `carried_produce_total`
and `_decide_unit`'s `carried_produce`, exactly mirroring FERTILIZER's
own `(k != "FERTILIZER" or not want_fertilizer or liquidating)` pattern.
This is principled rather than a guessed number: WHEAT is protected from
DROP for exactly as long as some live animal still needs it, and released
the instant nothing does (or once liquidating) -- so a large accumulated
surplus banks immediately once safe, not just above a fixed size. Two new
regression tests confirmed both halves of the guard hold and release
correctly.

**Regressed to `tape_kiro` 63.3% -> 0.0% (0W-30L)** -- as bad as the
worst attempts in this entire investigation (6, 7, 9, and the 10th's
larger magnitude). A same-seed trace showed real behavior change this
time (unlike the 12th): `total_requested` WHEAT dropped 3815 -> 3181, and
mid-game order-slot saturation genuinely improved on several days (e.g.
day 15: feed 8.46 -> 1.83) -- but `mean_implied_avg_price` still rose
(20.85 -> 27.01) and the late game reverted to the same 100%-saturated
pattern as baseline anyway. **This is the most direct confirmation yet
that the standing-price mechanism isn't specific to any one lever** (sell
volume, buy pacing, order position, reserve size, production share) --
it's a property of changing this agent's aggregate WHEAT market behavior
*at all*, even via a change that's structurally correct, passes every
unit test, and demonstrably lets organic harvest reach the shed as
intended. **Reverted** (never committed).

Thirteen attempts across four sessions plus one threshold-sweep redo: all
but the twelfth (a narrow, low-impact correctness fix) have failed. The
implication worth taking seriously now: this may not be fixable by
changing *this agent's own* WHEAT buy/sell/carry logic at all, at least
not against `tape_kiro` specifically -- every angle that changes real
aggregate behavior, regardless of mechanism, correctness, or how well-
reasoned, has hit the same wall. A fourteenth attempt should have a clear
answer for why it would be different in kind, not just in mechanism, or
should look outside this agent's own market-order logic entirely (e.g.
whether `tape_kiro`'s specific tape makes it an unusually adversarial
sentinel worth checking against a wider, more representative panel before
any future attempt is judged solely against it).

**A fourteenth attempt built the mechanistically correct fix the whole
investigation had been approximating, and it revealed a second, separate
bottleneck this whole time.** Extracted a second real top-player tape
(`reference/tape_kawashigi.py`) and confirmed the standing-price collapse
generalizes beyond `tape_kiro`; then traced *why* the baseline already
loses to real top players by ~4x money independent of any WHEAT attempt
at all -- crop-count collapse keeps `_land_utilization` below
`LAND_UTILIZATION_GATE`, permanently blocking the 4th quadrant (day-by-
day trace: utilization peaks 52-54% right before a land purchase clears
the gate, then craters to single digits and never recovers). This
reframed the goal: fix WHEAT well enough that utilization *stays* above
the gate, not just win-rate.

Gave `_build_feed_orders` a `slots_used_so_far` parameter and consolidated
it into **one order capped at exactly `MAX_MARKET_ORDERS - slots_used_
so_far`** -- not a guessed constant (the 5th attempt's
`FEED_ORDER_QUANTITY_CAP=8`, wrong because 8 didn't match what would
actually have gone through) but the literal quantity the old one-order-
per-unit loop would have delivered before `agent()`'s own truncation cut
it off. This reproduces the old scheme's aggregate volume/price exactly
(the engine processes any order "one unit at a time" regardless of split,
per data/README.md) while using 1 slot instead of many. It worked exactly
as designed: on a traced seed, `mean_implied_avg_price` didn't regress
($20.85 -> $20.14, essentially flat -- the first attempt in this entire
investigation where it didn't get worse), saturated-turn frac dropped
from 100% to 4-8%, seed orders flowed the whole game, and **the 4th
quadrant unlocked** (day 21) for the first time in any attempt.

**It still lost 0W-30L to `tape_kiro`, and the full pool (54.8%, ~14 points
below baseline) landed within noise of the imperfect 13th attempt's 54.3%
despite a completely different, provably-correct mechanism -- the
strongest possible evidence a second bottleneck exists.** Traced exactly
where the money went using `tape_kiro`'s own non-reactivity (it runs a
fixed tape, so its money can only change via shared market prices):
computed realized price × quantity for every SELL/BUY_PRODUCT resource it
touches, old vs. new. The dominant term wasn't WHEAT at all -- **MILK**:
kiro's own SELL MILK realized $28,294 -> $46,520, a +$18,226 swing that
dwarfs every other resource (WHEAT purchase actually got *cheaper* for
kiro, -$1,570). Cross-checked against our own MILK-selling volume: 186 ->
146 units (-22%), with a few fewer live cows at several mid-game
checkpoints. Repeated the same trace against `tape_kawashigi`: dominant
terms were WOOL (+$17,412), FERTILIZER (+$7,685), and MILK (+$5,454) --
animal byproducts again, netting to within a few percent of that
matchup's own real money-gap increase.

**The mechanism**: freeing WHEAT/seed market slots shifts crew attention
and order-slot allocation slightly away from animal husbandry (feed
maintenance still competes for the same crew-turns, and evidently wins
slightly less often once crop-side work has real candidates again),
reducing this agent's own animal-byproduct supply into the shared market
just enough to raise MILK/WOOL/FERTILIZER prices -- and both real top-
ladder opponents derive the bulk of their revenue from exactly those
byproducts (`tape_kiro`, `tape_kawashigi` both run animal-majority land
shares), so a shared-price rise nets them far more than it nets this
agent, which sells comparatively little of any of them. **Fixing the
crop-side starvation trades a real gain there for a real, opponent-
specific loss on the animal side, and against animal-heavy real
opponents specifically, the loss outweighs the gain.** Reverted (never
committed as working code) -- not because the fix was wrong, but because
it's provably insufficient alone.

**Fourteen attempts across four sessions plus one threshold-sweep redo.**
The WHEAT/seed-slot mechanism itself is now solved in principle (this
attempt's consolidation shape is the correct one, kept only out of the
codebase, not out of the analysis). What remains unsolved is the second-
order animal-market spillover this fix exposes -- a fifteenth attempt
would need to hold this agent's own animal-byproduct supply (and
therefore MILK/WOOL/FERTILIZER's shared price) roughly steady while
still freeing the WHEAT/seed slots, e.g. by not letting the crop-side
recovery come at animal husbandry's expense in the crew's own priority
ordering, rather than another variant of the order-slot mechanism itself.

**A fifteenth attempt did exactly what the fourteenth's note asked for --
reapplied its consolidation paired with an animal-care priority floor meant
to hold MILK/WOOL/FERTILIZER supply steady -- and confirmed the pairing
still isn't enough, but split cleanly into a real, if narrow, keeper.**
Traced the actual mechanism first: `_find_target`'s COLLECT_FERTILIZER/CARE
candidates sorted at a flat priority 3, comfortably below `PLANT_PRIORITY`
(4) but strictly above `PLANT_PRIORITY_UNDER_GATE` (1.5) -- so once the 14th
attempt's fix let seed orders flow again, a fresh PLANT candidate under the
gate started systematically outranking animal upkeep instead of losing to
it (previously animal upkeep always won there, 3 < 4), quietly diverting
crew-turns off husbandry toward planting. Added `animal_upkeep_priority =
min(3, plant_priority)`: ties animal upkeep with PLANT under the gate
(distance breaks the tie) while leaving it untouched (still 3) outside the
gate, where it already won.

Reapplied the 14th's exact consolidated `_build_feed_orders` (one order
capped at `MAX_MARKET_ORDERS - slots_used_so_far`) alongside this floor and
validated the combination: `tape_kiro` still 0W-30L, pool still 53.8%
(113W-97L) -- statistically the same as the 14th attempt's unpaired 54.8%,
no rescue. But isolating the two halves (each vs. the full 30-seed pool)
told a clearer story than testing them only combined would have: the feed
consolidation *alone* reproduced the 14th attempt's number bit-for-bit
(115W-95L, 54.8%, confirming the reconstruction was faithful), while the
*priority floor alone* -- with `_build_feed_orders` left at its original
per-unit form -- scored 148W-62L (70.5%), matching baseline (69.5%) within
noise and regressing nowhere (also checked against both real-tape
sentinels: `tape_kiro` 60.0% vs. baseline's 63.3%, within noise;
`tape_kawashigi` 0W-30L identical to the clean baseline's own 0W-30L on
that sentinel -- a pre-existing, unrelated hard matchup, not a new
regression). **The priority floor carries its own weight even without the
feed fix** (most of its benefit comes from the land-utilization gate being
active more than the 14th attempt's note assumed -- e.g. right after a land
purchase, per the 14th attempt's own utilization trace), **but the feed
consolidation's regression is exactly as large paired as unpaired**,
meaning the animal-market-spillover mechanism isn't primarily a crew-
priority effect at all, or at least not one this particular floor reaches.
**Kept the priority floor alone** (reverted `_build_feed_orders` back to
the original per-unit-order form) as a small, validated, genuine
correctness improvement -- same category as the 12th attempt.

Fifteen attempts across four sessions plus one threshold-sweep redo: two
kept (12th, 15th) as narrow correctness fixes, neither addressing the
starvation symptom itself; the rest reverted. The feed/seed-slot mechanism
remains solved in principle (14th attempt's consolidation shape) but
un-shippable alone.

**Same-turn cash competition note (16th attempt): the money hypothesis
from the 15th's own next-step note was real, but fixing it correctly
still didn't rescue the 14th's consolidation -- a third independent
rescue mechanism, a third failure, but a real correctness fix survives
on its own.** Read kaggriculture.py's actual `_process_market`/
`_commit_unit` directly instead of guessing: market orders are NOT each
checked against a static per-turn money snapshot -- the engine processes
the whole submitted queue in a per-unit lockstep loop, checking `farm
["money"] < price` at every individual unit commit, live, in submitted
order-list order. Since `agent()`'s own list is sell/land/hire/feed/
seed/fertilizer/animal, and every `_build_*_orders` function independently
checked the SAME raw `farm["money"]` snapshot with no idea what earlier
orders in the same list already committed to spend, `_build_animal_orders`
(last in the list) could request a purchase it genuinely believed was
affordable, only for the engine to silently reject it once feed/seed/
fertilizer had already drawn the balance down first.

Added `_orders_cash_cost` (estimates an order list's dollar cost using
the same per-item price basis each builder already uses for its own
affordability math) and threaded a running `cash_reserved` accumulator
through `agent()`'s market-list construction, so each builder from feed
onward sees `farm["money"] - cash_reserved` instead of the raw balance.
First version deliberately left SELL out of the estimate (it earns money,
not spends it), reasoning that ignoring real revenue could only delay a
purchase by a turn, never wrongly deny it -- **that reasoning was wrong**:
tested alone (no 14th consolidation), it cost real games against the two
reactive self-play sentinels that best resemble a real opponent
(`main_plant_priority_submitted_20260818.py` 40.0% -> 20.0%,
`main_price_reactive_seed_reserve_submitted_20260819.py` 53.3% -> 23.3%,
both roughly halved), because the same conservative gap reopens every
turn a sell also happens -- a purchase that only becomes affordable
*after* this turn's own sell revenue lands stays chronically starved
under this scheme, not delayed once. Fixed by giving `_orders_cash_cost`
a signed return (SELL orders subtract, i.e. add to the available budget)
and seeding `cash_reserved` from the turn's own sell orders. This fully
recovered both sentinels (40.0%/43.3%, matching the un-budgeted baseline
within noise) and the full pool (69.0%, matching baseline's 69.5%) --
**kept as a real, validated, standalone correctness fix**, same category
as the 12th/15th.

Paired with the 14th's exact consolidated `_build_feed_orders` (the real
target this whole cash-budget mechanism was built to rescue), the
buggy (SELL-ignoring) version actually helped substantially against
`tape_kiro` (0W-30L -> 7W-23L, 23.3%) -- but the corrected (SELL-aware)
version gave almost all of that back (down to 1W-29L, 3.3%), and BOTH
versions paired with the 14th still devastated the two reactive
sentinels (SELL-ignoring: 3.3%/3.3%; SELL-aware: 3.3%/10.0% -- barely
better, nowhere near the un-paired 40.0%/43.3%) and landed the same full
pool total either way (58.1%, both versions, to the percentage point).
**This is the tell**: the buggy version's `tape_kiro` gain wasn't really
about cash scheduling at all -- an accounting bug that systematically
under-spends is just another way to throttle this agent's own aggregate
WHEAT/seed market activity, which is exactly the lever that's helped
against `tape_kiro` specifically (and only `tape_kiro`) every time it's
been pulled by accident across this whole investigation (9th, 10th, 13th
attempts), while still costing real games against any opponent that
actually reacts. Fixing the accounting bug removed the accidental
throttle and the `tape_kiro` gain evaporated with it, while the
consolidation's cost against reactive opponents didn't move at all --
proving that cost was never a scheduling artifact this or the 15th
attempt's mechanism could reach in the first place.

**Sixteen attempts total. Three independent, well-reasoned rescue
mechanisms for the 14th's consolidation -- crew-priority (15th), buggy
cash-budget (16th v1), correct cash-budget (16th v2) -- have now each
failed against reactive opponents specifically, while each of the first
two's *unpaired* form survived as a genuine small correctness fix. A
seventeenth attempt should stop looking for a third resource to protect
around the consolidation and treat this as settled: freeing WHEAT/seed
slots costs real games against any opponent capable of reacting,
independent of *how* the freed capacity is spent (crew-turns or cash),
which means the cost isn't a side-channel resource-contention effect at
all -- it's WHEAT/seed volume itself moving the shared market against
this agent whenever an opponent can adapt to it. If the 14th's mechanism
is ever revisited, it needs a lever that changes *how much* aggregate
WHEAT/seed activity increases, not just how fairly the knock-on capacity
gets allocated afterward -- and should be judged primarily against the
two reactive self-play sentinels and the full pool, not `tape_kiro` alone
(now confirmed to reward almost any reduction in this agent's own
spending activity, real quality improvement or not).**

**Carried-stock accounting note (17th attempt, reverted -- but it finally
explains WHY every attempt in this file fails, which none of the sixteen
did).** The user reported the symptom from the visualizer as "once the
pastures are consolidated we can't plant seeds any more", and it is exactly
datable: in `replays/vs_kawashigi.json` the last animal structure goes up on
day 13-14, and from that turn on market slots sit at 10/10 with ~8.4 of them
`BUY_PRODUCT WHEAT`, `BUY_SEED` is flat 0.00 for 16 straight days, seeds hit
0 and stay there, and planted tiles fall 20.7 -> 2.0 while 60 of 75 owned
tiles sit empty. Money was never the constraint ($7k day 16, $47k day 29).
The trigger is the animal count reaching `_animal_target`, since `reserve =
live * FEED_RESERVE_PER_ANIMAL` scales with it.

A real accounting bug sits underneath: the reserve is compared against
`private["shed"]["WHEAT"]` only, while feed WHEAT is consumed from *unit
inventories* and therefore mostly sits there. Day 14 onward: reserve 24,
shed 9.8, carried 26.1 -- 35.9 actually owned, 50% past target, still
requesting ~15 units/turn. One day-16 turn reads `SELL WHEAT 9` then nine
`BUY_PRODUCT WHEAT 1` in the same order list; ~200 WHEAT/day churned to feed
12 animals that eat 12/day. `_build_fertilizer_orders` has always summed
shed + inventories correctly; the pattern was simply never applied to WHEAT.

Fixed the obvious way (`have = shed + carried`). **Mechanically it worked
better than any of the sixteen**: feed demand 15.0 -> 2.2 orders/turn (-84%,
vs. the top-ladder player's own 0.12), saturated turns 100% -> 12-21%, seeds
held 15-17 instead of 0, planted tiles sustained 20-33 through day 27
instead of collapsing to 2, animals unharmed (13 live, up from 12), and the
**4th quadrant unlocked** -- the concrete signal earlier notes here define
as proof the bug finally gave. **And it lost, hard**: vs. the two reactive
self-play sentinels, 31W-29L (51.7%) -> 2W-58L (3.3%); full 30-seed pool
151W-59L (71.9%) -> 121W-89L (57.6%), with mean money down in *every*
matchup including the 30-0 wins.

Two "the constants were tuned around the bug" pairings were then tested and
both failed, which is what makes this conclusive rather than another
inconclusive revert: `PLANT_PRIORITY_UNDER_GATE` 1.5 -> 2.5 (planting under
the gate outranks weed-clearing at 2, and `_land_utilization` doesn't count
weeds as used land, so restored seed supply drives a self-feeding weed cycle
-- weeds really did explode 1-3 -> 14-21 of 100 tiles) scored 4W-56L (6.7%);
`MAX_HANDS` 12 -> 14 (at 4 quadrants `_hands_target` is capped at 12, so
land grows 33% for 9% more crew -- and the raise is inert for the baseline,
which never buys the 4th quadrant) scored 1W-59L (1.7%), *worse* than
unpaired.

**The mechanism, measured directly rather than inferred**: per-resource
realized flows on a traced sentinel game show WHEAT market throughput is
this agent's largest profit centre by far -- gross sell $227,030 (4,652u)
against gross buy $191,822 (3,945u), **net +$35,208 of a $56,699 final**,
more than MILK ($40,085 gross), MELON, WOOL and FERTILIZER combined. That
throughput flows *through* the over-buying: cutting feed purchases 84% cut
WHEAT sales 70% with them (gross sell $65,598, net +$23,363), and the extra
crops the freed slots bought back (CARROT +27u, STRAWBERRY +7u, MELON -15u)
are worth a few thousand against ~$12k of lost WHEAT margin.

**So the accounting bug is load-bearing.** This reframes all sixteen prior
attempts: they didn't fail to a subtle "standing price" effect, they failed
because freeing WHEAT/seed market slots necessarily shrinks the WHEAT flow
that pays for most of the farm. An 18th attempt must not start by reducing
WHEAT throughput. The only shapes worth trying are ones that *keep* the
volume while spending fewer slots on it -- and the 14th attempt's
volume-preserving consolidation is exactly that shape and still scored 54.8%
-- or ones that first build a replacement revenue stream big enough to
outweigh ~60% of final money, before touching feed at all. Reverted; this
note and `_build_feed_orders`'s own are the whole deliverable.

Still out of scope: any opponent modeling -- this agent only ever reacts to
its own farm/market state. See rl/README.md for the parallel (currently
stalled) attempt at a learned policy. See this project's session memory for
the current list of open next-steps (pasture-placement drift, idle/wandering
hands, harvest-ready plants converting to weeds, planting stopping early
around day ~25, WHEAT unliquidated at game end, an opening-book idea for the
first days).

**WHEAT feed consolidation (18th market-order-slot-starvation attempt) --
KEPT despite a measured regression, pending further investigation.** A
public Kaggle kernel (X594, pulled and ground-truthed against this agent --
full analysis in this project's session memory) beats this agent 2-3x in
final money while running WHEAT-light: a standing sell-side reserve
(`WHEAT_SELL_RESERVE_DAYS` in `_build_sell_orders`) plus a carried-inventory-
inclusive, single consolidated feed order (`_build_feed_orders`) -- the exact
combination of three properties this codebase's first 17 attempts had only
ever tried individually (each alone regressed for a different documented
reason: standing-price effect for a bare reserve, animal-byproduct
market-spillover or the worst-of-9 result for consolidation alone, -84% WHEAT
throughput for carried-inclusive accounting alone -- see the 9th/14th/17th
attempts above). Implemented here for the first time as that full
combination. Mechanism trace confirmed it works exactly as intended: feed
order occupancy dropped from ~8.3/10 slots (100% saturated) to ~0.3/10
(4-8%), seed orders flow every day again -- the 14th attempt's original goal,
finally achieved cleanly, `next/decision_price` ratio flat (~1.00, ruling out
the 1st-5th attempts' within-order price-walk mechanism).

**It still regressed, badly, against reactive opponents specifically** (a
20-seed check against `main_plant_priority_submitted_20260818.py`: baseline
9/20 wins -> 0/20; a 10-seed check against `tape_kiro`: baseline 2/10 ->
7/10, the *opposite* direction -- exactly this codebase's own documented
`tape_kiro`-is-a-misleading-sentinel pattern, see the 16th attempt above).
Root-caused via replay trajectory, not left as an unexplained regression: the
freed market-order capacity let overall buying pace (land/seeds/hires)
outrun this agent's fixed crew's actual tending throughput -- weed count
climbed to 14-18/100 owned tiles in the back half of the game (vs. baseline's
steady 2-4 in the same matchups) even though crop counts were often *higher*
than baseline (seeds were being planted fine, just not watered in time
afterward). `MAX_HANDS` 12->14 was re-tested in this new context (a narrower
version of this exact pairing was tried once before per the 17th attempt's
own paired retest and scored 1.7%, but never actually exercised the tier it
targets) -- still worse, not better (`tape_kiro` seed check $22,084->$8,788;
sentinel money down in all 5 seeds tested, one collapsing to $2,095): the
extra hires' fib-priced cost drained cash without fixing the weed count
(still 17/100). Reverted `MAX_HANDS` back to 12. `WHEAT_SELL_RESERVE_DAYS`
was also re-tested at 1 instead of 3 (X594's own value): mixed per-seed,
aggregate slightly worse -- reserve sizing isn't the driver either. **Kept
in the codebase, unreverted, per explicit user direction to investigate
rather than revert on a partial (5-20 seed) validation** -- this is a live,
open thread, not a settled kept-or-reverted attempt like the 12th/15th/16th;
see this project's session memory and `.omc/plans/ralplan-wheat-feed-
consolidation.md` for the complete validation log before touching this area
again.

**Routing efficiency, following directly from the weed-neglect diagnosis
above.** An action-mix trace (fraction of unit-turns spent on WORK vs. MOVE
vs. PASS) proved the crew is at 0% idle and doing *more* raw work than
baseline (52% vs. baseline's 41-47%) once the 18th attempt's slots are free
-- ruling out a misallocation/priority-reordering explanation (there's no
idle capacity to reallocate) and confirming a pure walk-distance throughput
ceiling: roughly half of every unit-turn was spent walking, not acting, in
both baseline and the 18th-attempt state.

A first fix, a per-hand territory-anchor spatial bias
(`ANCHOR_W`/`_compute_anchors`, extending the same "bias `sort_dist`, never
priority" pattern `_quadrant_utilization` already used for fresh plantings,
mirroring the X594 kernel-pull reference's own anchor idea), was implemented,
measured as a real but noisy, non-robust effect (5-seed sample: 3 of 5 seeds
improved substantially, 2 regressed hard; a proper 20-seed sample showed
0/20 wins against the reactive sentinel, no better than no anchor at all;
anchor drift itself was ruled out as the noise source via a dedicated
turn-to-turn stability trace -- anchors were stable, 0.18 tiles mean drift,
in both a badly-regressed and a much-improved seed), and **removed** once a
fuller mechanism replaced it (see below) -- this is the "isolate one change"
methodology applied to a mechanism, not just a constant: don't keep two
untested routing changes live simultaneously.

**Replaced with a full global (unit x job) assignment**
(`_build_tile_jobs`/`_assign_tile_jobs`), reusing this agent's existing
`(priority, sort_dist)` scoring convention exactly (no new scalar formula,
avoiding a `TIER_GAP`-style recalibration risk) rather than X594's own scored
formula. Tile-only candidates (harvest/water/weed/animal-upkeep/plant/build
-- any unit could do any of these) are built once per turn instead of every
unit rescanning the whole board and claiming its own single best candidate
in a fixed, order-dependent sequence; inventory-dependent candidates (feed/
fertilizer-delivery/animal-token-placement/shed-return -- see
`_find_personal_target`) stay genuinely per-unit, since they can't be pooled,
but now compete fairly in the *same* sort as pool jobs rather than
automatically winning outright (a real bug in this rework's own first
version, caught by `test_find_target_shed_return_stays_routine_below_the_
carry_threshold` -- a routine, low-priority personal candidate like a small
carried load's shed return must still be able to lose to a genuinely more
urgent nearby pool job). Mechanism trace confirmed a real, robust efficiency
gain this time, unlike the anchor bias: work% rose from ~52% to 58%, not
just in one lucky seed. Money followed: a 20-seed check against the reactive
sentinel landed at 944,799 total (vs. baseline's 993,583, only -4.9% --
closest of any variant tried this session, vs. the anchor bias's -12.4% or
the 18th attempt alone's -42.5%), several previously-blowout seeds now close
losses instead of routs.

**Kept -- but win rate against the reactive sentinel is still a near-total
collapse (1/20), despite the crew being measurably, robustly more
efficient.** Three independent, real efficiency improvements this session
(the 14th WHEAT attempt's slot-freeing, the anchor bias, the global
assignment) have each made this agent demonstrably better at working the
map, and none of them meaningfully closed this specific gap -- strong
evidence the bottleneck was never really about routing at all: it's some
other effect the 18th attempt's WHEAT-reserve change specifically triggers
against reactive opponents (most likely a shared-market-price mechanism, the
closest precedent being the 14th attempt's own animal-byproduct spillover
finding). Any future attempt in this specific area should start from that
reframing, not from another routing or crew-capacity lever -- both of those
avenues are now reasonably well-explored for this cycle. Full validation
log: `.omc/plans/ralplan-routing-efficiency.md` and
`.omc/plans/ralplan-global-assignment-v2.md`.

**Harvest-timing (one-time crops) -- a new, distinct bug from the WHEAT
market-slot-starvation saga above, in the harvest decision itself rather
than order/routing scheduling.** `_is_harvestable` treated "HARVEST is legal"
(age >= first_yield_day) as "HARVEST is correct right now". kaggriculture.py
actually keeps growing a one-time crop's `yield_units` for every watered day
from `first_yield_day` through `max_yield_day` (+1/day watered, +2/day
fertilized, capped at `max_yield`) -- for WHEAT specifically: legal at age 2
with yield_units=1, but grows to 4 (watered) or 6 (fertilized) by age 4 if
left alone. Harvesting the instant it was legal was locking in yield_units=1
every time, forfeiting up to 4-6x the yield on this agent's own profit
centre (see [[wheat-throughput-is-the-profit-centre]]) -- and wrongly
convinced the 12th market-slot-starvation attempt that "WHEAT yields exactly
1/harvest... max_yield evidently isn't units per harvest" (see this
docstring's twelfth-attempt note above), when it's this agent's own harvest
timing, not an engine property. Also explains why WHEAT's FERTILIZE almost
never fired despite WHEAT being in `FERTILIZE_CROPS`: the fertilizer-
delivery gate in `_find_personal_target` (`not _is_harvestable(...)`)
excluded a tile the instant it became legally harvestable, i.e. almost its
entire fertilizable window.

Confirmed against `kaggle_environments/envs/kaggriculture/kaggriculture.py`
directly (not just data/README.md's prose) and cross-checked against the
X578/X594 kernel-pull reference's own harvest logic (see
[[x578-x594-kernel-pull-benchmark]]): X594 doesn't harvest a WHEAT/CARROT
tile until `age >= max_yield_day`, topping off with one more WATER first if
it still helps, *except* when `day >= LAST_DAY` (its endgame trigger), where
it harvests as soon as legal instead -- the same "wait for full yield unless
the season is ending" trade-off. `_is_harvestable` now takes `liquidating`
(already threaded everywhere it's called) and defers a one-time crop past
`first_yield_day` until whichever comes first: `age >= max_yield_day`,
`yield_units` already at `max_yield` (nothing left to gain by waiting), or
`liquidating` (this agent's existing near-end-of-season flag, `days_remaining
<= ENDGAME_LIQUIDATION_DAYS`) -- reusing the existing WATER/FERTILIZE
fallback branches in `_build_tile_jobs`/`_decide_unit`/`_find_personal_target`
unchanged, so a deferred tile is automatically kept watered (and now
actually gets fertilized) while it waits, with no new job type needed.
MELON is untouched in practice: its own bonus window (age 6-12) already
saturates at or before its `first_yield_day` (10) under normal watering, so
the new `yield_units >= max_yield` clause fires immediately for it the same
turn the old code did.

**Follow-up: the deferred-harvest fix above shifts this agent's cash-flow
timing, and two quick fixes for the resulting land-purchase delay were both
tried and reverted.** A same-seed paired trace (main.py vs
`reference/main_price_reactive_seed_reserve_submitted_20260819.py`, seed 0)
showed the 4th quadrant bought 5 days later (day 16 vs day 11) once WHEAT
started waiting for max_yield_day before selling -- not a simple 2-day lag,
a genuine multi-day liquidity crunch (`farm["money"]` sat below
`SAFETY_RESERVE` for most of days 6-19 on that seed) while more land got
planted with the same cash. A 10-seed pool check (`--engine fast`) confirmed
this wasn't seed noise: 60.0% pool total before the harvest-timing fix,
57.1% after it alone.

1. **Loosening `LAND_UTILIZATION_GATE` (0.50 -> 0.40) did nothing and made
   it worse** ($52,797 -> $49,352 on the traced seed, unlock day unchanged
   at day 16). Confirms the utilization gate was never the actual
   constraint -- `farm["money"] - SAFETY_RESERVE >= LAND_PRICES[n_extra]`
   is. Ruled out cleanly, not just suspected.
2. **`_standing_crop_value`: counting grown-but-unharvested one-time-crop
   yield (yield_units * market price, 50% discount) as spendable cash for
   the land-buy gate made the pool result worse, uniformly, not
   noise** -- 57.1% -> 54.3% pool total, and money dropped in nearly every
   single matchup (only `main_fertilizer_selling` held steady). The traced
   seed's own 4th-quadrant unlock day didn't even move (still day 16); final
   money there fell further, $52,797 -> $44,118. Likely mechanism: this
   made the agent commit to a `BUY_LAND` order against value that wasn't
   actually liquid yet (still needs a unit to walk, harvest, carry, sell),
   overcommitting real spendable cash right when the land purchase itself
   debits it -- the same kind of cascading cash-starvation this project's
   history keeps finding this economy is unusually sensitive to (the
   SEED_RESERVE-doubling catastrophe, the whole feed-orders-starve-market-
   slots saga). Reverted; `_build_land_orders` is back to its original
   2-argument, money-only form.

Neither of these two "fast" attempts actually fixed the land-timing
regression -- both are now ruled out with evidence, not just untried.

3. **Threading `_build_land_orders` into the existing same-turn cash-budget
   mechanism (`cash_reserved` -- the 16th attempt's own pattern, already
   used by feed/seed/fertilizer/animal but not land) made the SAME traced
   seed worse still, and in the opposite direction from the naive
   prediction: the 4th quadrant unlocked LATER (day 20, not day 16), final
   money fell further ($52,797 -> $35,366).** Checked the obvious
   mechanism first (an animal starving because land now out-competes feed
   for the same turn's incoming cash, since land is built before feed in
   agent()'s sequence) via the trajectory trace's own `anim` column --
   ruled out directly, live-animal count never drops (4 -> 13, monotonic,
   no losses). The real mechanism wasn't isolated before reverting: most
   likely land now also wins the SAME race against hire/seed for a turn's
   arriving cash (not just feed), a slower, cumulative drag (delayed crew
   growth -> delayed crop growth -> delayed *next* land purchase too) not
   a single dramatic loss like a starved animal. Reverted; `_build_land_orders`
   is back to its original 2-argument, `farm["money"]`-only form -- same as
   after attempts 1 and 2.

Three attempts at the land-purchase-delay side effect, all reverted, all
confirmed with evidence rather than abandoned on suspicion.

**A fourth angle instead: how does the X578/X594 kernel-pull reference
itself avoid this same problem?** (See [[x578-x594-kernel-pull-benchmark]].)
Its own land-purchase logic differs from this agent's in four ways at once,
tested one at a time on the same traced seed (main.py vs
`reference/main_price_reactive_seed_reserve_submitted_20260819.py`, seed 0)
plus a 10-seed pool check (`--engine fast`), then combined:

1. **`LAND_QUADRANT_CAP = 3` -- never buy the 4th quadrant at all**
   (X594's own `n_unl < 3` gate; matches the top-ladder Tetsuya replay's
   pattern too, see [[tetsuya-replay-analysis]]). **Kept.** Sidesteps the
   whole regression instead of fixing it: the purchase that was starving
   under the new cash-flow shape just never gets attempted. 10-seed pool:
   win rate unchanged (57.1%, same as harvest-timing-fix-alone), but money
   rose in every reactive/hard matchup and nowhere fell (x492 +14.6%,
   `main_fertilizer_selling` +10.6% [still 100%], `main_plant_priority`
   +10.4%, `main_price_reactive_seed_reserve` +12.6%, all 0W-10L matchups
   unchanged in win count but meaningfully less lopsided). Traced seed:
   $52,797 -> $53,643, `owned=75` confirms the 4th quadrant is never
   bought.
2. **A cash reserve that scales with operation size** (`_land_cash_reserve`:
   per-live-animal feed buffer + a flat amount, mirroring X594's own
   `cash_reserve = hire_total(...) + n_animals * 45 + 60`) **instead of the
   flat `SAFETY_RESERVE`, for the land gate only.** Tested in isolation on
   top of lever 1: no measurable change (traced seed identical, pool
   totals within noise of lever 1 alone). Kept anyway -- more principled,
   free given lever 1 already removed the actual cash crunch this was
   meant to protect against.
3. **Hire built before land** in agent()'s own sequence, mirroring X594's
   own order, with land given the same-turn cash budget this time (unlike
   the reverted third land-purchase-delay attempt, which gave land that
   visibility while land still ran *before* hire -- see above). Also no
   measurable change on top of levers 1+2 (same reasoning: lever 1 already
   removed the crunch this was meant to fix). Kept for the same reason as
   lever 2.
4. **Per-unit SELL price-impact simulation** (`_sell_proceeds`, using the
   real `market_price` imported directly from kaggriculture.py -- no risk
   of the formula drifting the way X594's own reimplementation could) with
   a 10% haircut (`SELL_PROCEEDS_DISCOUNT`), replacing the flat
   `qty * current_price` estimate `_orders_cash_cost` used for every SELL
   order, not just land's. Also no measurable change combined with 1-3.
   Kept: strictly more accurate than the flat estimate it replaces, for
   free once lever 1 removed the actual pain point.

**All four levers combined, final pool check: still 57.1%, same win/loss
per matchup as lever 1 alone**, confirming directly (not just by elimination)
that lever 1 is what mattered -- levers 2-4 are real, principled precision
improvements over the original code (none of them regress anything, all
kept), but had nothing left to fix once lever 1 removed the actual cash
crunch. Next: validate the full package (harvest-timing fix + all four
land-purchase levers) at the project's full 30-seed + 2-reactive-sentinel
protocol before treating this as done.

**30-seed protocol result on the harvest-timing fix + 4 land levers, original constants:
close but short of the bar.** `checkpoints/v0/main.py` and `reference/melon_maxxer.py` were
dropped from `pool_winrate.py`'s own default pool at this point (2026-08-21) -- still 30/30
shutouts each direction, no longer discriminating between changes, same reasoning as the
2026-08-18 random/starter removal (see that script's own docstring). On the remaining
5-opponent pool: baseline 62W-88L (41.3%), this fix 60W-90L (40.0%) -- essentially flat, but
both reactive sentinels specifically lost a couple of already-marginal wins each
(`main_plant_priority` 1W->0W, `main_price_reactive_seed_reserve` 3W->0W) while money rose in
both. Doesn't clear the project's own promotion bar (match-or-beat baseline on the full pool
*and* both sentinels specifically) as-is.

**MC-search-tuning follow-up, closes the gap decisively.** Ran `scripts/mc_search.py`
(default: 30 trials, 5 seeds x 2 opponents [x492 + fertilizer_selling], `--engine fast`) on
top of the harvest-timing fix + all 4 land levers, varying the same 10 general-economy
constants it always has (HANDS_BASE, HANDS_PER_QUADRANT, MAX_HANDS, LAND_UTILIZATION_GATE,
PRICE_ATTRACTIVENESS_WEIGHT, WHEAT_SELL_THRESHOLD, FEED_MAX_PRICE_FRAC, SAFETY_RESERVE,
FERTILIZER_MAX_PRICE_FRAC, SHED_OVERFLOW_GUARD) -- none of the harvest-timing/land-lever
constants were in its search space this run. The search pool doesn't include either reactive
sentinel, so its own win_rate metric (capped at 0.50 here, since x492 is unbeatable by
constant-tuning alone and fertilizer_selling is always won) only ever ranked candidates by
margin against x492 -- the winning trial still needed a full-pool 30-seed validation to mean
anything for this project's actual gate, exactly as the script's own closing note warns.

That validation (best-margin trial, then applied to `main.py` directly): **41.3% -> 46.7%
(70W-80L) on the 5-opponent pool, beating baseline on the full pool AND both reactive
sentinels specifically** -- `main_plant_priority` 1W->2W ($43,084->$52,854), `main_price_
reactive_seed_reserve` 3W->8W ($42,267->$52,258), `x492` money alone +41% ($26,361->$37,291,
still 0W but far less lopsided), `land_gate_60pct`/`fertilizer_selling` both up too, nowhere
regressed. Clears the promotion bar the earlier (untuned) result missed.

Applied values (all in this file, `was X` noted at each): HANDS_BASE 2->6,
HANDS_PER_QUADRANT 3->2, MAX_HANDS unchanged at 12... [see MAX_HANDS's own comment: the
search found 16, confirmed behaviorally identical to 12 given LAND_QUADRANT_CAP=3 -- byte-
identical replay, same seed -- kept at 16 anyway, harmless headroom for if the land cap or
hands formula changes later], LAND_UTILIZATION_GATE 0.50->0.493, PRICE_ATTRACTIVENESS_WEIGHT
0.1->0.26, SELL_PARAMS["WHEAT"]["threshold"] 0.45->0.854, FEED_MAX_PRICE_FRAC 5.0->4.22,
SAFETY_RESERVE 147->561, FERTILIZER_MAX_PRICE_FRAC 1.147->1.92, SHED_OVERFLOW_GUARD
0.72->0.633. Largest, most legible shifts: SAFETY_RESERVE nearly 4x (real cash sits lower for
longer between sales now, so a much bigger buffer is needed before optional buys) and the
WHEAT sell threshold nearly doubled (holding out for a better price is worth more per unit
now that a harvest itself is worth 4-6x more). 56/56 tests pass, lint clean.

**BUY-side follow-up: a data/README.md audit for other overlooked rules, prompted by the
harvest-timing bug's own root cause, found the mirror-image gap on the buy side.** "The buy
price is quoted at the post-buy inventory" (data/README.md) -- confirmed directly against
kaggriculture.py's `_commit_unit`: BUY_PRODUCT decrements `market["inventory"]`
unconditionally (no floor/cap exception the way SELL's $1 floor has), so each unit bought
within the same order pushes the next one's price further from the last, exactly mirroring
the SELL-side gap lever 4 already fixed. `_orders_cash_cost`'s BUY_PRODUCT branch was still
using the old flat `qty * current_price` estimate. Measured on a realistic 26-unit
consolidated WHEAT feed order (`_build_feed_orders`'s own shape, one order per turn since the
18th attempt): 13% underestimate ($650 flat vs. $735 real) with the market near I0 --
counterintuitively *worse* near equilibrium than when already deep in shortage, where the
below-I0 shape functions (sqrt for WHEAT) have mostly flattened out. That underestimate feeds
`cash_reserved`, which gates every later builder's own affordability check the same turn.

Added `_buy_cost` (mirrors `_sell_proceeds`, no discount -- this is a cost estimate, and the
precise walk itself already removes the systematic underestimate, no separate haircut
needed) and switched `_orders_cash_cost`'s BUY_PRODUCT branch to it; `prices` was then dead
in `_orders_cash_cost` (every remaining order type prices off a fixed cost, not the market)
and removed from its signature and all 7 call sites. 57/57 tests pass, lint clean. Not yet
re-validated at the pool/30-seed level -- do that before treating this as more than
mechanically correct.
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


class _CropPlanner:
    """Tracks, for one turn, which crop each new PLANT should use.

    Buying happens the turn *before* planting (market orders resolve after
    farmer/hand actions each turn -- see data/README.md's Turn Processing
    Order), so this only ever spends seeds already sitting in
    `private["seeds"]`, never seeds bought this same turn.
    """

    def __init__(self, private, farm, days_remaining, prices=None):
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
        gap = self.counts[crop] / self.total - CROP_TARGET_SHARE[crop]
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


def _build_tile_jobs(farm, day, board_size, claimed, animal_zone, quadrant_util, plant_priority, liquidating, planner, animal_planner):
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
    water_priority = 5 if liquidating else 1
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
                    # See the module docstring's Harvest-ready tiles rotting to weeds note:
                    # priority stays 0 either way (already top tier); the bias only breaks
                    # ties within it, favoring a tile one missed watering from becoming a WEED.
                    at_risk = not liquidating and tile.get("consecutive_unwatered", 0) >= 1
                    jobs.append((x, y, 0, -HARVEST_AT_RISK_DIST_BIAS if at_risk else 0))  # harvest
                elif not tile["watered_today"]:
                    at_risk = not liquidating and tile.get("consecutive_unwatered", 0) >= 1
                    jobs.append((x, y, 0.5 if at_risk else water_priority, 0))  # water
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


def _build_hire_orders(farm):
    desired = _hands_target(len(farm["unlocked_quadrants"]))
    remaining = desired - farm["hires_today"]
    if remaining <= 0:
        return []
    return [["HIRE"] for _ in range(remaining)]


def _build_seed_orders(seeds, money, days_remaining, prices):
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
    above base) earns more room to stock up."""
    orders = []
    remaining = money - SAFETY_RESERVE
    for crop, base_reserve in SEED_RESERVE.items():
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
    board_size = len(farm["tiles"])
    shed_tiles = _shed_adjacent_tiles(board_size)

    turns_per_day = int(_cfg_get(config, "turnsPerDay", DEFAULT_TURNS_PER_DAY)) or DEFAULT_TURNS_PER_DAY
    episode_steps = int(_cfg_get(config, "episodeSteps", DEFAULT_EPISODE_STEPS)) or DEFAULT_EPISODE_STEPS
    last_day = max(0, -(-episode_steps // turns_per_day) - 1)  # ceil(steps/turns_per_day) - 1
    days_remaining = last_day - day
    liquidating = days_remaining <= ENDGAME_LIQUIDATION_DAYS

    market_obs = obs.get("market", {}) or {}
    market_prices = market_obs.get("prices", {})
    market_inventory = market_obs.get("inventory", {})
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    inventories = private.get("inventories", [{}])

    want_fertilizer = _any_fertilizable(farm, day)
    planner = _CropPlanner(private, farm, days_remaining, market_prices)
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
            liquidating, planner, animal_planner,
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
    hire_orders = _build_hire_orders(farm)
    market += hire_orders
    cash_reserved += _orders_cash_cost(hire_orders, farm, market_inventory)

    land_orders = _build_land_orders(farm, days_remaining, farm["money"] - cash_reserved)
    market += land_orders
    cash_reserved += _orders_cash_cost(land_orders, farm, market_inventory)

    feed_orders = _build_feed_orders(farm, private, market_prices, farm["money"] - cash_reserved)
    market += feed_orders
    cash_reserved += _orders_cash_cost(feed_orders, farm, market_inventory)

    seed_orders = _build_seed_orders(seeds, farm["money"] - cash_reserved, days_remaining, market_prices)
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

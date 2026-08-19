"""Kaggriculture submission entry point.

Kaggle loads this file and calls `agent(obs)` (or `agent(obs, config)`) once
per turn for whichever player index we've been assigned. Keep this file
self-contained (or bundle extra modules into a tar.gz with this at the root
per AGENTS.md) since it's shipped standalone to the competition runner.

Strategy ("farm maxxer"): a scripted multi-unit farm loop built on top
of `reference/melon_maxxer.py` (Kaggle's own official example agent, see its
docstring), fixing three of its four documented weaknesses (see the
Correction note below for the fourth):

  - melon_maxxer never expands land or hires hands -> this agent buys land
    greedily and hires hands every day (hiring is *extremely* cheap: fib(n)
    dollars for the n-th hire that day, i.e. $1, $1, $2, $3, $5, ... -- see
    kaggriculture.py's `_hire_cost`), then drives every hand with the same
    per-unit task logic as the farmer.
  - melon_maxxer only ever grows MELON -> this agent grows a mix of MELON
    (highest value/tile/day by a wide margin, per data/README.md's Object
    Types table), WHEAT and CARROT (cheap, fast-cycling fillers that keep
    tiles turning over while melons mature, and diversify what's sellable
    when one crop's market price is depressed).
  - melon_maxxer never fertilizes -> this agent fertilizes WHEAT/CARROT
    (one-time crops whose *unfertilized* yield cap is strictly below their
    fertilized cap, per data/README.md's Harvest Yields section -- melon's
    cap is reachable by watering alone, fertilizer there only shortens the
    growth window, so it's skipped as lower value for the added complexity).

**Correction -- the fourth "fix" (throttled selling) was reverted, it was a
regression**: melon_maxxer dumps its entire shed in one SELL order once a
threshold is crossed, which can crash the price mid-sale (price moves
unit-by-unit *within* an order, see data/README.md's Market Mechanics). An
earlier version of this agent capped how much of each crop it sold per turn
to avoid that, on paper a strict improvement. Measured instead: a deterministic
20-seed head-to-head between that version and an ablation that dumps
everything (`reference/main_dump_sell.py`) was a 19/20-seed *loss* for the
capped version. Root cause: holding harvested value in the shed waiting for
the cap to free up slows reinvestment into land/seeds/hands more than the
occasional price dip from a big dump costs, at this agent's actual
production scale -- the theoretical concern was real but small relative to
the throttling's own opportunity cost. `_build_sell_orders` now dumps in
full again, like melon_maxxer, with the shed-overflow-guard threshold-bypass
(`SHED_OVERFLOW_GUARD`) kept since that part was never in question.

**Selling strategy, revisited and closed out**: the 19/20-seed result above
was measured, not explained, at the time. Working out *why* (`market_price`/
`_refresh_prices` in kaggle_environments' own `kaggriculture.py`, plus a real
downloaded replay -- see `reference/community_dataset/` and the module
docstring's earlier replay-tooling notes): price is a deterministic function
of the market's *current* inventory level alone, and every SELL adds to that
inventory one unit at a time with no reset in between orders. That makes
total revenue for selling a fixed total quantity mathematically *path-
independent* -- one 80-unit MELON order and eighty 1-unit MELON orders
starting from the same inventory produce the exact same total revenue to
the dollar (verified numerically: both settle at $18,329 for a solo 80-unit
dump from I0). Splitting an order can only ever change *when* the cash
lands, never how much of it there is, absent inventory recovery in between
-- and recovery is too slow to matter at this agent's production scale for
the resources that matter most (MELON/WOOL have no town shop demand at all,
only the town center's flat ~1-unit/day trickle, per the Price-aware
planting note above). This is the mechanistic reason throttled selling lost:
it paid the exact same total price degradation as dumping, plus a pure,
avoidable reinvestment delay. A downloaded replay
(`replays/kaggle_submission/episode-94006627-replay.json`) also confirmed
the market really is shared and concurrent exactly as `data/README.md`
describes: turns where both players sold the same resource showed the
market's inventory change reflecting *both* players' volume (partly offset
by that turn's town consumption), meaning an opponent selling the same crop
the same turn does affect the price we receive -- but this is a genuinely
uncontrollable multiplayer dynamic, not something our own order batching,
timing, or list position can influence, since we choose our market orders
without seeing the opponent's for that same turn. Conclusion: there is no
selling-mechanism fix left to find here -- the only lever that actually
moves this needle is *how much* gets produced/sold in total over the
season relative to town absorption capacity, which is exactly what the
Land utilization gate, Task-driven hiring, and Price-aware planting notes
elsewhere in this docstring already investigated (and, in every case but
the land gate, reverted after measuring it made things worse). No code
change came out of this pass; it exists to stop a future session from
re-litigating "maybe throttled/staggered selling would help" without first
re-deriving this.

**Selling fertilizer**: closing out the selling investigation above still
left one concrete question unasked -- *what* top players actually sell, not
just how. Breaking down `reference/community_dataset/`'s top-ladder replays
by per-resource revenue (four games, three different top players) found
FERTILIZER as the single largest or second-largest revenue source in every
one of them (21-59% of total sell revenue, ahead of WHEAT and comparable to
or ahead of WOOL), while MELON -- this agent's dominant crop share -- was
consistently minor for the strongest player (5-9%). This agent never sold
FERTILIZER at all: it's not in `SELL_PARAMS`, and FERTILIZER collected from
animal structures via `COLLECT_FERTILIZER` landed in a unit's own inventory
with only one way out -- `_find_target`'s "deliver fertilizer" candidate,
which only exists while some crop tile is still eligible
(`_is_fertilizable`). Once nothing needed it, a unit carrying surplus
collected FERTILIZER had no candidate left to chase at all (the shed-return
trigger explicitly excluded FERTILIZER from what counted as "produce") and
just sat there holding it, forever, for the rest of the game. Fixed by
letting surplus FERTILIZER count as "carried produce" (sellable once
`SELL_PARAMS["FERTILIZER"]`'s threshold clears) whenever `not
want_fertilizer` -- i.e. only when nothing on the whole farm is currently
fertilizable, never while a delivery is still possible. **The first version
of this fix dropped the `want_fertilizer` condition entirely** (any carried
FERTILIZER always counted as produce), reasoning that `_find_target`'s
higher-priority delivery candidate (3) would always beat the shed-return
one (5/1) regardless of distance so it could never misroute -- true for
routing, but it broke something routing doesn't touch: `_decide_unit`'s
direct-action shed-tile branch runs *before* a unit that just PICKUP'd
FERTILIZER (to carry to a crop) has moved anywhere. With the unconditional
version, that same turn's decision saw the freshly-picked-up FERTILIZER as
"carried produce" and immediately DROPped it back -- a PICKUP/DROP livelock
identical in shape to the animal-token livelock from the Animal husbandry
(attempt 3) note, just for FERTILIZER instead of an animal, and far more
damaging since fertilizer pickups happen constantly (WHEAT/CARROT/TOMATO/
STRAWBERRY all use it): a 10-seed pool comparison with this version was a
catastrophic regression across the board, e.g. mean money vs `random`
collapsing from ~$36-38k to ~$2,290 and the win rate against the whole pool
dropping from 83.3% to 23.3%. Restoring the `want_fertilizer` gate fixes
it: a unit can only ever see carried FERTILIZER as sellable produce on a
turn where fetching more of it for delivery wouldn't even be attempted in
the first place. 10-seed pool comparison with the fix confirmed the
livelock was the whole story and the underlying idea was sound: mean money
rose ~20-26% against every non-adversarial pool opponent on top of the
spatial-clustering gains above (e.g. `random` $40-44k -> $50-52k), the
head-to-head against `reference/main_land_gate_60pct.py` held at a clean
10/10-seed win with an even wider margin ($51,621 vs $22,970, up from
$39,640 vs $22,698), and mean money against x492 rose ~37% ($24,583 ->
$33,603) even though that matchup is still an outright loss on every seed.
One accepted imprecision that survives both versions: the
shed's FERTILIZER count is a single fungible pool for both collected (free)
and market-bought stock, so `_build_fertilizer_orders`' own small standing
reserve (`FERTILIZER_RESERVE = 2`) can occasionally get swept into a sell
dump and get rebought next cycle -- a bounded, minor churn cost against a
much larger potential revenue stream, not worth a separate reserved-vs-
sellable split for 2 units.

Animal support (GOOSE/COW/SHEEP) was implemented and A/B tested
(`reference/main_with_animals_v1.py`) and found to be a clear net negative --
roughly a 70% cut to final money vs. this animal-free version, deterministic
A/B vs. `starter` across 3 seeds -- even after fixing four real bugs along
the way. The daily per-animal upkeep (FEED/CARE/COLLECT_FERTILIZER) competes
for the same scarce farmer/hand turns as the much higher value/tile/day crop
loop, and a single missed feeding cycle loses an entire $300-500 purchase
outright. See that file's docstring for the full writeup before attempting
this again.

**Tried a second time, still reverted**: `reference/x492_nah_i_d_win.py`
(see `scripts/pool_winrate.py`) runs animals profitably by never deciding
about them live at all -- their placement/feed/harvest schedule is baked
into a precomputed action tape offline, with no runtime turn-contention
risk. Adapting that idea to this agent's reactive architecture, 2 of
`MAX_HANDS`'s 6 hands were carved out as permanently dedicated animal
hands, never touching a crop tile, specifically to remove the turn-
contention failure mode above. It didn't help. A replay first caught a
second, unrelated starvation cause even with dedicated hands: WHEAT is
both a sellable crop and animal feed, and `_build_sell_orders` dumps the
entire shed the moment its price clears threshold, so the shed sat at 0
WHEAT for turns at a stretch -- fixed by buying feed WHEAT directly from
the market, decoupled from the crop side. With that fixed too (animals no
longer mass-starved), a 10-seed pool comparison was still a severe net
loss across the board -- e.g. -66% mean money vs `random`, -62% vs
`melon_maxxer` (which flipped from a 10/10-seed win to 6/10), -60% vs
`starter` (9/10). The root cause was never actually "contention" or
"starvation logistics" -- it's that 2 of 6 hands (a third of the crew) is
too much crop-tending capacity to give up for what animals produce in a
30-day season, at this agent's crop economics. Don't retry without first
finding a much cheaper way to fund animal upkeep than a fixed hand tax.

**Tried a third time, kept**: downloaded top-ladder replays (the
`georgymamarin/kaggriculture-episodes` community dataset, see
`reference/community_dataset/`, plus `scripts/extract_replay_tape.py`)
showed real players sizing animal structures as a steady ~15-24% of owned
land throughout the season, gradually scaled with land bought -- not a
fixed hand allocation like attempt 2. `_AnimalPlanner` now targets
`ANIMAL_LAND_SHARE` (0.18, the middle of that observed range) of owned
tiles, and there's no dedicated crew: every unit, farmer included, shares
one decision function (`_decide_unit`/`_find_target`) across both crop and
animal work, with FEED as the single highest priority of anything (a
missed feed still costs the whole $300-500+ animal). Two real bugs
surfaced building this, both now regression-tested:
- A livelock: `PICKUP`-ing a bought animal token, then immediately
  `DROP`-ping it back into the shed the same turn, forever -- the generic
  "carried produce" auto-drop check didn't exclude animal tokens
  (COW/SHEEP/GOOSE) the way it already excluded FERTILIZER/WHEAT, so a
  freshly-picked-up animal got treated as sellable produce and dumped
  right back.
- A self-defeating target: when the shed had no WHEAT, a unit with none in
  hand would still generate a "go to the shed" candidate at the highest
  priority (more urgent than harvesting) to fetch feed -- and if that unit
  was already standing on a shed tile, the candidate pointed at its own
  position, so `_step_toward` returned nothing and it just `PASS`ed every
  turn instead of doing any of the crop work sitting right next to it.
  Fixed by only generating that candidate when the shed actually has WHEAT
  to fetch.

Once genuinely working (no starvation, no idle land, no livelocks), the
same replays gave one more data point: 5 of 5 downloaded top-ladder
episodes (3 different players) never place a single GOOSE, despite it
being the cheapest animal to buy. The economics explain why -- $/day per
animal (product base price / production interval) is $80 for COW (MILK
160/2), $66.7 for SHEEP (WOOL 200/3), only $50 for GOOSE (EGG 50/1), the
worst of the three. `ANIMAL_TARGET_SHARE` was rebalanced to COW 0.60 /
SHEEP 0.40 / GOOSE 0 to match. Setting a kind's target share to 0 alone
wasn't enough, though, and briefly made results *worse*:
`_AnimalPlanner.choose_structure`'s "which kinds still need a structure"
filter checked only whether that kind's structure type had zero spares
right now, not whether the kind was below its target share -- since GOOSE
never gets built, its COOP count sits at 0 forever, so the instant
COW/SHEEP's shared PASTURE type had a spare waiting, GOOSE became the
*only* candidate left in the filtered pool and won by default regardless
of its 0 share. Fixed by excluding 0-share kinds from that filter (and
from `_build_animal_orders`'s purchase loop) directly. 10-seed pool
comparison after both fixes: back in line with the no-animal baseline
against the non-adversarial pool, and the best head-to-head result of any
animal attempt against `reference/main_land_gate_60pct.py` -- 2/10 seeds
won outright, mean money 18,051 vs 19,191 (a 0.94 ratio, vs. 0.60 with the
GOOSE bug and 0.82 before any GOOSE fix at all). Still down against x492
specifically (~-30% mean money vs. the no-animal baseline there) --
x492's land/crop economy alone is strong enough that animal upkeep isn't
worth the turns against it, even though it's a net-neutral-to-positive
trade everywhere else measured.

**Spatial clustering (animals), addressed**: the same top-ladder replays
place animal structures in a tight cluster immediately around the shed,
leaving crops the rest of the map edge-to-edge -- plausibly because FEED
is daily and unmissable, so keeping every animal within a short walk of
the shed (where feed/product pickup and drop already happen) minimizes
the turns the whole feeding round costs. This agent had no such geometric
preference: `_find_target`/`_decide_unit` treated "nearest empty tile"
identically whether a crop or a structure wanted it, so animals and crops
competed for the same near-shed tiles instead of animals deliberately
claiming the center and crops spreading outward. `_animal_zone` now
reserves the N empty tiles closest to the shed (N = however many
structures `_AnimalPlanner` still needs to reach its land-share target)
for BUILD_PASTURE/BUILD_COOP specifically; crop planting is skipped on
those reserved tiles (falling through to the next-nearest empty tile
instead), and any structure need beyond the zone still falls back to
whatever empty tile is nearest, same as before. This was the real
bottleneck behind attempt 3's lukewarm head-to-head result above, not
animal husbandry itself: 10-seed pool comparison after adding the zone
jumped from a 2/10-seed head-to-head win against
`reference/main_land_gate_60pct.py` (0.94 money ratio) to a clean 10/10,
1.69 ratio ($38,125 vs $22,621), and mean money against the whole
non-adversarial pool rose ~40% across the board (e.g. $25k -> $36-38k vs
`random`/`starter`/`melon_maxxer`). Even against x492 (still an outright
loss, 0/10 seeds) our own money more than doubled ($9,826 -> $21,220).
Plausibly compounds with the pre-existing tile-clustering gap below (which
remains unaddressed -- this only reserves land for animals, it doesn't
spread crops outward on its own).

**Animal capacity vs crop capacity**: a downloaded replay of this agent's
own 2026-08-18 submission (won 42,321 vs 39,457, but by less than it
should have) showed why: `ANIMAL_LAND_SHARE` sizes the animal-structure
target purely off land owned, with no regard for how many units exist to
tend both the animals *and* the crops. By day 20 (3 quadrants, 75 tiles)
the target had grown to 14 animal structures against a crew of only 7
units (farmer + `MAX_HANDS`'s 6 hands) -- and crop upkeep visibly
collapsed under the load: NW's planted-tile count fell from ~17 (day 12)
to 2-3 (day 28), and NE accumulated 3-4 weeds that were never cleared for
the rest of the season. The same replay showed only 2 hands hired on day
0-6 (the flat `2 * quadrants` formula, correctly sized for 1 quadrant) even
though the land-share target already wanted 4-5 animal structures by day
2 -- not a hiring bug, the animal side was simply sized without reference
to how few hands existed yet to run it. Fixed with `_animal_target`, which
now caps the live-animal count at the *smaller* of the land-share target
and `ANIMAL_MAX_PER_UNIT` per currently-hired farmer/hand (shared by
`_AnimalPlanner` and `_build_animal_orders` so both always agree).
`ANIMAL_MAX_PER_UNIT` started at 1.0 tiles/unit as a first guess pending
pool validation -- see whether that number changed below. This also
substantially resolved the spatial-clustering "drift" gap noted above
without any change to `_animal_zone` itself: a local same-code replay (30
days vs `starter`) that previously would have grown to a much larger
target now ends the game with 7 structures instead of 14, and their max
distance from the shed dropped from 6 tiles to 4 (3 of 7 sit directly on
the shed ring). Smaller total target simply means `_animal_zone` never has
to reach as far to find its N-closest empty tiles. The same replay's crop
health held up far better too -- NW stayed at 15-17 planted tiles through
day 20 instead of collapsing to 2-3 by day 28. Didn't pursue a dedicated
anticipatory-reservation fix for the residual drift given how much smaller
it already is; revisit only if a future replay still shows it mattering.

**Price-aware planting**: MELON's price curve is punished quadratically for
oversupply above equilibrium (`data/README.md`'s Price Function, the `sq`
above-target shape -- the steepest in the whole table) and only recovers
~1 unit/day: no town shop demands MELON, only the flat town-center trickle
(`data/README.md`'s Town Buildings table). A replay showed `_CropPlanner`
kept spending its 50% target share on MELON turn after turn right after a
big sell dump had just crashed its price (250 -> 86 base over one season,
seed 1 vs `starter`) -- replanting straight back into a market it had just
flooded. `_CropPlanner.choose` now skips a crop from consideration while
its current price is below the same threshold `_build_sell_orders` already
uses to decide whether to sell it, falling back to ignoring price only if
every plantable crop is currently depressed (so land never sits idle over
a temporary dip).

**Price-attractiveness-aware planting**: `_price_ok` above is a binary
pass/fail gate, and once a crop clears it, price stopped mattering at all
-- `choose` picked purely by target-mix-share gap. A downloaded replay of
this agent's own 2026-08-18 submission showed the real cost of that: by
day 17-18, STRAWBERRY's price had climbed to 2.0-2.1x its base (well above
its 0.55 sell threshold) while MELON sat at ~0.98x its own base (barely
above its 0.65 threshold) -- and MELON still got planted *more* often than
STRAWBERRY in that window, because MELON's dominant 45% target share gives
it a bigger raw share-gap almost by construction, drowning out how much
more attractive STRAWBERRY's price actually was right then. `_CropPlanner`
now scores each candidate crop by its target-share gap *minus*
`PRICE_ATTRACTIVENESS_WEIGHT * (price / base - 1)` (see `_score`), so a
crop trading well above its base price gets pulled toward the front of the
queue instead of only ever competing on mix-share alone.
`PRICE_ATTRACTIVENESS_WEIGHT` started at 0.1 as a first guess -- see the
Land utilization gate's "nudged back up" note below for the combined
10-seed pool result (this, the gate bump, and the fertilizer-liquidation
fix below were all measured together in one pass, not in isolation).

**Land utilization gate**: `_build_land_orders` used to buy the next
quadrant purely on cash available, with no regard for whether the land
already owned was even being worked. A replay showed this agent buying its
4th quadrant (all 100 tiles) by day 11 while plateauing at ~40-48% of
tiles actually planted for the rest of the season.
`reference/x492_nah_i_d_win.py` (the strongest agent in the pool, see
`scripts/pool_winrate.py`) never buys that 4th quadrant at all -- it stops
at 3 (75 tiles) and holds ~70-73% utilization there instead, ending the
same paired-seed season with far more money. `_build_land_orders` now
gates each new `BUY_LAND` on the current quadrant(s) already being at
least `LAND_UTILIZATION_GATE` planted -- buying land faster than it can be
staffed just leaves it idle. **A companion fix was tried and reverted**:
`MAX_HANDS = 6` (a self-imposed number, not a game rule -- hiring is
nearly free, fib-cost/day per `data/README.md`'s Hiring section) looked
like the reason utilization plateaued, and x492 itself runs up to 12
hands, so `MAX_HANDS` was raised to 12 with hires scaled to match.
Measured instead, 10-seed pool comparison: the land gate alone was a clean
win against every pool opponent (e.g. +21% mean money vs `random`, +41% vs
x492), while adding the raised hand cap on top made it *worse* than the
gate alone almost everywhere (still better than doing nothing, just a
smaller gain) -- more hands means more simultaneous harvests dumped into
the same finite-demand market, which crashes prices faster than the extra
fib-scaling hire cost pays back. `MAX_HANDS` and the hire formula are back
to their original values; only the land gate stuck.

**The gate's own threshold was recalibrated after real Kaggle replays
exposed a deadlock**: `LAND_UTILIZATION_GATE` originally shipped at 0.6.
Downloaded replays from actual leaderboard games (`scripts/
extract_replay_tape.py`) showed this agent's *own* achievable utilization
plateauing at ~48-56% across every real opponent -- consistently just
below the gate -- so it stayed permanently stuck at 2 of 4 quadrants for
the entire 30-day season in 3 of 4 real games, never recovering even with
$25k+ sitting unused. Against weak pool opponents this was invisible (this
agent still wins by a huge margin off 2 quadrants alone), but it's a real
cost against a contested market -- go watch `reference/main_land_gate_60pct.py`
lose that same land race. Lowered to 0.45 (below the observed ceiling, so
it can't deadlock) fixes that: reliably reaches all 4 quadrants now.
**But lowering the threshold alone was a real trade-off, not a clean fix**:
10-seed pool comparison showed -18 to -21% mean money against every
non-adversarial pool opponent and -39% against x492 -- the same oversupply
dynamic as the reverted `MAX_HANDS` experiment above, and a head-to-head
against `reference/main_land_gate_60pct.py` (identical except that one
constant) was a 1/10-seed *loss*, -34% mean money. A head-to-head replay
found why: the hire formula (`2 * quadrants`, capped at `MAX_HANDS = 6`)
already maxes out hire capacity at 3 quadrants (2*3 = 6) -- the 4th
quadrant *never* unlocks another hand under it, so buying it just spreads
the same crew across 25 more tiles. The replay showed the 0.45 agent
planting no more tiles in absolute terms with 4 quadrants than the 0.6
agent did with 3 -- diluted thinner, plus $4,000 spent on land with no
extra hands to work it. `_land_worth_buying` now blocks a `BUY_LAND` that
wouldn't raise the hire-formula's hand cap, capping this agent at 3
quadrants outright (independent of `LAND_UTILIZATION_GATE`, which still
gates 1->2 and 2->3, both of which *do* raise hand capacity) -- matching
what `reference/x492_nah_i_d_win.py` (the strongest agent in the pool)
does on its own. Recovered most of the trade-off: back within +/-7% of
the original 0.6 baseline against every non-adversarial pool opponent
(and still +59% against x492 vs. that baseline), and the head-to-head loss
against `reference/main_land_gate_60pct.py` narrowed from -34% to -13%
mean money -- better, but still a loss, so some gap remains unexplained
(oversupply from selling faster once 3 quadrants comes sooner, and/or the
spatial pathing bias below, are the leading suspects). Kept
`reference/main_land_gate_60pct.py` in `scripts/pool_winrate.py`'s default
pool to keep tracking whatever's left of this gap.

**The gate nudged back up, slightly**: a downloaded replay of this agent's
own 2026-08-18 submission showed the 3rd quadrant bought at exactly 46%
utilization of the first two -- technically past the 0.45 gate, but by a
margin thin enough to read as "barely earned it" rather than genuinely
ready. Raised to 0.50: still comfortably below the ~48-56% achievable-
utilization ceiling observed in real replays (where 0.6 deadlocked), just
less borderline than 0.45. 10-seed pool comparison with this, the price-
attractiveness change above, and the fertilizer-liquidation fix below all
bundled into one pass (not isolated per-change): a mixed, honest result --
mean money against the non-adversarial pool *dipped* modestly (e.g.
`random` $60.5k -> $56.1k, `starter` $59.7k -> $51.9k), within the range
of this project's known run-to-run variance at a fixed seed count, but the
head-to-head against `reference/main_fertilizer_selling_submitted_
20260817.py` (yesterday's actual Kaggle submission) improved from 6/10 to
8/10 seeds won, and the money ratio against it from 1.05x to 1.19x. POOL
TOTAL win count also ticked up (56/70 -> 58/70). Read cautiously: three
changes landed in one measurement, so this doesn't isolate which of them
(if any) is pulling the weak-pool number down -- worth re-splitting and
re-measuring individually if a future session revisits this area.

**Task-driven hiring, tried and reverted**: real top-ladder replays
(`scripts/extract_replay_tape.py`/the community episode dataset, see
`reference/community_dataset/`) showed hand count tracking actual pending
work turn to turn -- zero hands hired on a day with nothing to do but wait
for wheat to grow and tend animals solo, five hired the moment a real
backlog existed -- not the flat `2 * quadrants` formula this agent uses.
Replaced `_build_hire_orders` with one counting actionable tiles right now
(unwatered/harvestable plants, weeds, optionally plantable-with-seed-budget
empty tiles) and sizing the crew to clear that backlog by
`TASKS_PER_HAND`. It reproduced the real pattern -- hand count genuinely
fluctuated day to day instead of stepping through fixed quadrant
thresholds -- but measured worse in every variant tried: -3 to -6% mean
money against the non-adversarial pool and -15% against x492 counting
plantable tiles as tasks; dropping plantable tiles from the count (hiring
only for water/harvest/weed maintenance) was worse still, -34% in the
head-to-head against `reference/main_land_gate_60pct.py` (vs. -13% for the
simple quadrant-cap formula it replaced). Both variants hired more
aggressively during genuine backlog spikes (freshly bought land, a batch
of crops maturing together) than the flat formula ever did, which is
exactly the same oversupply pattern as every other "hire/produce more"
experiment in this file's history -- more simultaneous harvests than the
market (or the sell logic below) can absorb without crashing the price.
Reverted to the flat formula; the real pattern is genuine, but replicating
it costs more in oversupply than it saves in idle-hand hire fees at this
agent's current selling strategy.

**Spatial clustering (crops), addressed**: a replay showed planted tiles
clustering heavily near the shed (the NW quadrant dense and harvest-ready,
the NE quadrant mostly empty or gone to weed) rather than spreading evenly
across owned land. `_find_target`'s nearest-candidate-wins selection
created this bias -- once a unit settled into tending nearby tiles, a
marginally-closer chore in the already-established area kept winning over
walking further out to work distant land, even when it was sitting idle.
`_quadrant_utilization` now tracks each quadrant's own planted fraction,
and a fresh PLANT candidate's distance in `_find_target` gets inflated by
`quadrant_util[q] * (board_size // 2)` -- a same-distance choice between a
crowded and a neglected quadrant now goes to the neglected one, while a
much-closer tile in the crowded quadrant can still win outright (it's a
nudge on ties/near-ties, not a hard override, so it doesn't force units to
walk past everything nearby chasing the single least-planted quadrant).
10-seed pool comparison on top of the animal-zone fix above: win/loss
record held (still 50/60 seeds, 83.3%, already saturated against the
non-adversarial pool) but mean money climbed further across nearly every
opponent -- +10 to +20% vs `random`/`starter`/`checkpoints/v0`/
`melon_maxxer` ($36-38k -> $40-44k), +16% vs x492 despite still losing
every seed there (its own money also grew a comparable amount, so the
underlying land-race gap wasn't closed, just both sides scaled up).

**TOMATO/STRAWBERRY**: this agent only ever grew the three one-time crops
(MELON/WHEAT/CARROT). `data/README.md`'s Harvest Yields section documents
two more, TOMATO and STRAWBERRY, as *ongoing* crops -- once planted they
keep producing on a fixed schedule (tomato at ages 8/9/10/11, strawberry at
10/12/14/16) without needing to be replanted between harvests, unlike a
one-time crop's single payout. `_is_harvestable`/`_decide_unit`'s generic
PLANT-tile field checks (`yield_units`, `watered_today`) already handle
this correctly with no changes -- the engine keeps the tile alive after
HARVEST for an ongoing crop instead of clearing it, and just keeps refilling
`yield_units` on schedule, so a unit that waters an ongoing-crop tile daily
and harvests whenever `yield_units > 0` was already the right behavior.
Both are now in `CASH_CROPS`/`SELL_PARAMS`/`SEED_RESERVE`. **Measured, not
assumed, how much of MELON's share to give them**: the obvious-looking move
was cutting MELON's dominant 0.5 share hard to make room (it's the crop
whose price crashes quadratically, per the Price-aware planting note
above) -- 10-seed pool comparison showed that alone (MELON 0.35, no
tomato/strawberry, redistributed to WHEAT/CARROT) was a *clean loss*
against every pool opponent, including x492. MELON's price gate already
protects it from replanting into a crashed market, so de-emphasizing it
further just traded away its far-higher base price ($250 vs WHEAT's $25)
for nothing. What actually pays off is adding TOMATO/STRAWBERRY as a small
addition without touching MELON much: `CROP_TARGET_SHARE` settled at
MELON 0.45 / WHEAT 0.25 / CARROT 0.15 / TOMATO 0.10 / STRAWBERRY 0.05 --
roughly flat (+/-2%) against every weaker pool opponent and +33% mean
money against x492 (their gentler above-equilibrium price curves, `sqrt`/
`linear` vs MELON's `sq`, pay off specifically when the market is
contested). Fertilizing them was initially left out: `_is_fertilizable`'s
eligibility window used `CROPS[crop]["max_yield_day"]` as an upper bound,
which for these two crops equals `first_yield_day` (a placeholder value,
not "last production day"), so reusing it as-is would only ever allow
fertilizing the *first* scheduled production and silently miss the rest.
Fixed with `_last_yield_day`, an ongoing-crop-aware bound
(`first_yield_day + (max_yield - 1) * interval`, matching the real
schedule -- tomato through age 11, strawberry through age 16); both are
now in `FERTILIZE_CROPS` too. 10-seed pool comparison: a clean win
everywhere, +2 to +7% mean money with no regressions (largest gain against
x492, +7%, where every bit of yield matters most).

Still out of scope: any opponent modeling -- this agent only ever reacts to
its own farm/market state. See rl/README.md for the parallel (currently
stalled) attempt at a learned policy.

**Endgame liquidation**: unsold shed contents (and anything still sitting in
a farmer/hand's personal inventory) don't count towards the final score --
see data/README.md's Reward section. Everything above ignores that until
close to the last day, so this agent also tracks `days_remaining` (from
`config["episodeSteps"]`/`config["turnsPerDay"]`, which `kaggle_environments`
does pass through -- confirmed via `agent.py`'s `co_argcount` inspection) and,
as the season winds down: stops buying seeds/land that can't pay back in the
time left, has units prioritize walking harvested produce back to the shed
over watering a crop that won't finish in time, and drops the sell-price
threshold entirely so nothing is still sitting unsold when the last turn's
market closes.

**Endgame liquidation (fertilizer)**: the above wasn't catching everything.
A downloaded replay of this agent's own 2026-08-18 submission showed
real, steady dollar value sitting in unit inventories all game long (not
growing unboundedly, just a persistent "in transit" pipeline -- WHEAT
reserved for feed, FERTILIZER en route to a crop, freshly harvested animal
products not yet dropped off) -- roughly $3,500-4,500 worth at any given
sampled turn, including the very last one, where it's permanently lost
(doesn't count toward the score). Some of that is an unavoidable tail cost
of any continuous-production pipeline that has to stop somewhere mid-
cycle, but the FERTILIZER slice of it wasn't: once `liquidating`, a unit
would still walk to the shed to fetch *more* FERTILIZER for a crop that,
by definition, has little season left to benefit from it -- a pure waste
of travel time this late. Fixed by gating new fertilizer-delivery pickups
on `not liquidating`, and letting already-carried FERTILIZER count as
sellable "produce" once liquidating even if some crop is nominally still
fertilizable (see `_find_target`'s `carrying_produce`/`_decide_unit`'s
`carried_produce`). WHEAT's feed reserve is deliberately left untouched --
missing a feed this late still loses the whole animal outright, a bigger
loss than the travel time saved.

**Neglected-tile urgency**: the "canto distante perdendo cultura" gap
(the pre-existing tile-clustering note above, and the specific NW/NE
neglect the animal-capacity fix addressed) has one root cause left after
both of those: `_find_target`'s water candidates all shared one priority
regardless of how close a tile actually was to becoming a WEED, so a unit
chasing a marginally-closer *routine* chore near the shed could keep
winning over walking out to save a tile one missed watering from being
lost outright, no matter how the spatial-balancing fixes above steered
planting. Considered and rejected first: a dedicated unit for distant
tiles, the user's own first suggestion -- this project already measured
that exact shape of fix for animals (2 of 6 hands walled off, Animal
husbandry attempt 2) and it was a severe net negative (-60 to -66%
money): fixed allocation wastes capacity whenever its designated zone
doesn't have enough work to fill every turn. Went with priority instead:
a tile with `consecutive_unwatered >= 1` (kaggriculture.py -- missing
water two days running turns a PLANT into a WEED, and even a fresh
planting starts pre-loaded at 1, "the planting day counts as unwatered")
now scores 0.5 instead of the routine `water_priority`, just below
harvest's own 0 (already-earned money is never worth delaying for a tile
that would only avoid a *future* loss) but above every other routine
chore. Measured across three passes, honestly mixed rather than a clean
win: the fix alone (10-seed pool) dipped mean money modestly against most
of the non-adversarial pool (e.g. `random` $56.1k -> $52.9k) and the
head-to-head against yesterday's frozen submission snapshot from 8/10 to
7/10 seeds -- plausibly because a 7-unit crew managing 75+ tiles hits
`consecutive_unwatered >= 1` often enough that it's closer to routine load
than a rare emergency, so reprioritizing what an already-saturated crew
does doesn't clearly help on its own. Paired with raising `MAX_HANDS` 6 ->
8 (a modest re-try, not the 12 already measured worse pre-animal-cap/
fertilizer-selling -- see `_land_worth_buying`'s own docstring) recovered
most of the gap: back to 8/10 seeds against the frozen snapshot, and
`random`/`main_land_gate_60pct.py` both ticked back up. Kept on balance --
still not a clean statistical win in this specific pool, but it fixes a
real, replay-diagnosed correctness gap (crops actually dying to neglect)
that a few thousand dollars of pool noise shouldn't override, and
`MAX_HANDS=8` conveniently exactly matches 2 * the 4-quadrant land cap,
so it stops being a binding constraint at all rather than an arbitrary
bump. Most matchups in this pool never actually reach a fully-utilized
4th quadrant within 30 days, so this pairing is under-exercised by the
current pool roster -- a stronger test would need an opponent that
reliably pushes the game there.

**Real hiring pace, superseding the MAX_HANDS=8 pairing above**: asked to
analyze hand counts in the same downloaded top-ladder replays (3 games,
same #1 player), not just money/production. The flat `2 * quadrants`
formula wasn't just under capped, it was the wrong *shape* -- real hiring
isn't proportional to land at all: 5 hands on day 0 with only the
starting NW quadrant, 12 by day 9 with just 2 quadrants, settling at
10-12 for the rest of every season regardless of reaching all 4
quadrants. `_hands_target` now uses `HANDS_BASE + HANDS_PER_QUADRANT *
quadrants` (5/8/11/12 at 1-4 quadrants) instead of `2 * quadrants`
(2/4/6/8 under the old MAX_HANDS=6, capped at 6/8 either way) -- close to
the observed curve at 1 and 4 quadrants, still short at 2-3 but a large
step up from before, and deliberately kept strictly increasing through
all 4 quadrants so `_land_worth_buying` never blocks the last one. This
looks like exactly the kind of change a previous pass already measured as
worse (`MAX_HANDS` raised to 12 "with hires scaled to match" -- see the
Land utilization gate note above): more hands dumping more simultaneous
harvests crashed crop prices faster than the extra fib-scaling hire cost
paid back. Retried anyway because two things changed since that measurement:
the animal-capacity cap now caps how much of the crew's harvest volume is
animal product, and fertilizer selling opened a revenue channel that
doesn't compete in the same crop markets at all. The retry decisively
confirms the economy really did change: 10-seed pool comparison, mean
money up 16-41% across every non-adversarial pool opponent on top of
every earlier gain this session (e.g. `random` $54.5k -> $72.1k, `starter`
$52.3k -> $73.5k), and the head-to-head against yesterday's frozen
submission snapshot improved to 9/10 seeds won (from 8/10). The Neglected-
tile urgency fix and this hiring-pace change were kept together, not
re-isolated -- both are independently well-motivated by the same replay
evidence, and untangling their individual contributions wasn't worth
another measurement pass given how clearly positive the combination is.

**Sell/rebuy round trip on WHEAT -- fix attempted and reverted**:
`_build_sell_orders` and `_build_feed_orders` read the same shed WHEAT
with no coordination -- sell dumps the *entire* shed stock the moment
price clears its 0.45x threshold with no notion that animals need some of
it, then feed buys straight back from the market (a near-nonbinding
backstop, willing to pay up to 5x base -- missing a feed loses the whole
animal, so `FEED_MAX_PRICE_FRAC` is deliberately loose) as soon as the
shed reads empty again -- a real, self-inflicted round trip through both
sides of the market's own price impact, confirmed happening in a
downloaded replay. Tried fixing it by having `_build_sell_orders` hold
back a WHEAT reserve from the dump; three sizes were tried (a 2-day
buffer matching `_build_feed_orders`'s own trigger, a 1-day buffer, and a
small flat cap of 4 units decoupled from flock size), each looking
better than the last on quick 10-seed pool checks -- but that whole
trail turned out to be chasing partly-noisy signal, not a real
improving trend (see the next note). A properly-powered recheck (30
seeds, pre-fix and post-fix run back-to-back in the same process against
`reference/main_fertilizer_selling_submitted_20260817.py`, the pool's
strongest and most competitive opponent) told a different story: pre-fix
25/30 (83.3%), post-fix (the "best" flat-cap-4 version) 16/30 (53.3%) --
a genuine, large regression, not the honest-trade-off it first looked
like. Plausible mechanism: holding back even a small reserve delays
reinvestment (the same mechanism the per-turn-sell-cap revert above
already demonstrated), which barely dents a blowout win against weak
pool opponents (still 100% win rate, ceiling effect hides the cost) but
flips close races against a closely-matched opponent where marginal
efficiency actually decides the outcome. Reverted `_build_sell_orders`
back to taking no reserve argument at all -- same pattern as
`_build_animal_orders`'s own "Known bug" note below: a theoretically
correct fix that measured worse and got reverted rather than shipped on
faith. The underlying round trip is still real and still unaddressed;
worth another look with a smaller-footprint idea (e.g. coordinating the
sell and feed decisions within the same turn instead of a standing
reserve) and, per the next note, with a large enough seed count to trust
the result.

**A real non-determinism in the eval harness was found investigating the
above, and it changes how every pool comparison in this project should
be read.** `kaggle_environments`' `seed` config does properly determinize
the one RNG source inside the kaggriculture interpreter itself (weed
spawning, keyed off `seed` and `day` -- see kaggriculture.py's
`_end_of_day`), and neither `main.py` nor any reference agent calls
`random` directly, so identical code given an identical seed *should* be
fully reproducible. It measurably isn't: rerunning the exact same 10-seed
script (no code change at all) gave 9/10, 9/10, then 7/10 across three
attempts, with the 7/10 outlier appearing specifically when it ran
concurrently with another heavy background episode batch -- and,
separately, the *same* 10-seed matchup scored differently depending on
whether it was the only thing a script ran versus the last of 50
episodes in a single `pool_winrate.py` invocation, even with nothing else
running on the machine at the time. Root cause not fully pinned down
(the `actTimeout: 1` per-turn wall-clock config is a plausible
suspect -- our own `agent()` call measures ~0.2ms in isolation, comfortable
headroom, but that headroom could erode under CPU contention or
within-process memory/GC pressure over a long-running multi-episode
script), but the practical takeaway is solid either way: **a single
10-seed comparison has a real noise floor around +/-2 seeds out of 10**,
enough to make a fix look clearly better or clearly worse than it
actually is (exactly what happened chasing the WHEAT-reserve sizes
above). Treat any single 10-seed `pool_winrate.py` result with real
skepticism going forward -- rerun it, or use a larger seed count (the
30-seed script that caught this regression is the model to follow), for
any comparison a real decision rests on. Running pool comparisons one at
a time rather than concurrently with other heavy background work is
still good hygiene, but per the 50-episode-vs-10-episode finding above,
isolation alone isn't a full guarantee.

**Top-ladder WHEAT price range checked, didn't translate into a win**:
asked whether top-ladder replays (`replays/top_ladder/`, 5 games) avoid
the round trip above some other way. They don't -- they do it constantly
(71-248 buy orders and 38-199 sell orders per game per player) and it's
net *profitable* for them (WHEAT nets +9% to +88% of final money for 9 of
10 player-games analyzed), which on its own says the round trip isn't
inherently a losing pattern. The mechanistic difference: every observed
buy across all 5 replays landed at 1.00x-2.08x base, every observed sell
at 1.08x-2.08x -- both far tighter than this agent's 0.45x sell threshold
and 5.0x feed-buy cap. Ran `scripts/mc_search.py --params
WHEAT_SELL_THRESHOLD FEED_MAX_PRICE_FRAC` (30 trials, scoped to just the
`main_fertilizer_selling_submitted_20260817.py` matchup to avoid x492's
unwinnable-regardless-of-tuning result flooring every win rate) to look
for a better pair within that observed range. Two candidates hit a
misleadingly perfect win_rate=1.00 at the search's own 5-seed sample
(threshold=1.242/feed=2.62 and threshold=1.545/feed=3.19) -- but neither
held up at 30 seeds, the same lesson as the note above: 25/30 (83.3%)
baseline vs 19/30 (63.3%) for the first candidate (a real regression) and
24/30 (80.0%, statistically a wash) for the second. Left both constants
unchanged. One genuinely useful, robust finding did survive the noise,
though: every trial with `FEED_MAX_PRICE_FRAC` below ~1.4 scored
win_rate=0.00 with a catastrophic margin (down to -$91,579) -- a feed cap
that tight can't clear the price often enough to actually feed animals,
losing them outright. `FEED_MAX_PRICE_FRAC`'s current 5.0 has plenty of
headroom above that failure zone, so no change needed there either, but
it's worth remembering as a hard floor if this constant is ever revisited.

**Market-order-slot starvation, found but the fix measured worse --
reverted**: a downloaded `main.py` vs `reference/x492_nah_i_d_win.py`
replay (seed 42) showed planted-crop tiles peaking at 36/75 on day 11,
then declining *monotonically* to 2/75 by day 29 -- decoupled from money,
which was simultaneously climbing fast ($7k on day 20 to $24.6k on day
27). Land sat mostly barren despite ample idle cash. Diagnosed cause:
`agent()` builds its market list as sell -> land -> hire -> feed -> seed
-> fertilizer -> animal, then truncates to `maxMarketOrdersPerTurn` (10,
silently drops the rest -- see data/README.md). `_build_feed_orders` used
to append one `["BUY_PRODUCT", "WHEAT", 1]` order per unit needed (up to
`FEED_RESERVE_PER_ANIMAL * live_animals`, 20+ some turns) instead of a
single order with quantity N -- across the whole 720-turn replay, 46.2%
of turns had 8-9 duplicate feed orders alone, and only 6.4% of turns had
*any* `BUY_SEED` order at all. `_build_seed_orders` had the same
one-order-per-unit pattern (up to 13 orders across 5 crops). Tried
consolidating both (plus `_build_fertilizer_orders`, for consistency)
into one order per resource/crop requesting the full needed quantity --
data/README.md confirms a single N-quantity order is already processed
"one-unit-at-a-time" internally for price-impact purposes and stops early
if money runs out mid-order, so this looked like a pure slot-count win
with no effect on affordability or price impact. **It measured as a real
regression anyway**: a same-process 30-seed A/B (git HEAD vs. this fix
alone, the trusted methodology per the eval-noise note above) against
`reference/main_fertilizer_selling_submitted_20260817.py` (the pool's
toughest, most competitive matchup) dropped from 27/30 (90%) to 21/30
(70%) wins despite similar mean money ($49.6k -> $50.6k) -- more losses
in exactly the close races where marginal efficiency decides the outcome,
the same signature as the reverted WHEAT-reserve and animal-token-gating
fixes elsewhere in this file. Mechanism not confirmed, but a plausible
one: consolidating into fewer, larger orders changes which of the *other*
player's own same-turn orders each of mine gets paired against in the
slot-by-slot simultaneous processing (see data/README.md's Market Action
section), which can shift realized transaction prices in ways a
per-order affordability check alone doesn't capture. An isolated 20-seed
subprocess run of this fix alone looked far worse still (4/20 against the
same opponent) before the Planting priority under the gate fix below was
layered on top and recovered most of the money (though not the win rate)
-- a reminder that a single subprocess `pool_winrate.py` run can be
misleadingly noisy in either direction, per the eval-noise note above;
the same-process 30-seed number is the one to trust. Reverted
`_build_feed_orders`/`_build_seed_orders`/`_build_fertilizer_orders` back
to their one-order-per-unit form.

**Second attempt at the same bug, also reverted**: instead of consolidating
into one big order, capped `_build_feed_orders`'s per-unit loop at
`FEED_ORDERS_PER_TURN_CAP = 3` orders/turn -- same one-unit-per-order
shape as before (so it couldn't be the earlier fix's slot-pairing-with-
the-opponent mechanism), just fewer of them, still topping the reserve up
over a few turns since `needed` is recomputed fresh each turn from
current shed stock. Confirmed this genuinely fixes the *symptom*: a real
replay that used to crash to 0/5 seed types in stock by day 15 and stay
there for the rest of the game (0 `BUY_SEED` orders sampled from day 13
to day 29) instead kept 2-3 of every seed type in stock the whole game,
`_build_feed_orders` itself capped at exactly 3 orders as designed.
**Measured worse anyway**: same-process 30-seed A/B (git HEAD = the
Planting-priority-under-the-gate fix alone, no feed-cap) against
`reference/main_fertilizer_selling_submitted_20260817.py` -- win rate
30/30 -> 29/30 (near noise-level, one flipped close game) but mean money
$61.7k -> $52.9k, a real ~14% drop; against x492, mean money also down
slightly ($27.8k -> $27.1k), still 0/30 either way. Reverted. **Three
independent measurements now (full consolidation, this per-turn cap, and
the earlier reverted sell-side WHEAT reserve) all show the same signature
against this one specific opponent** -- and that opponent is itself a
frozen snapshot of an earlier main.py (see `scripts/pool_winrate.py`'s
DEFAULT_POOL), i.e. a near-mirror match. Leading hypothesis, still
unconfirmed: something about *this* matchup specifically -- both sides
running near-identical WHEAT buy/feed/sell logic -- makes any change to
how aggressively we compete for WHEAT market slots costly in a way that
doesn't show up against structurally different opponents (x492, the pool
bots). Worth checking directly before a fourth attempt: does the same fix
measure differently against a *non-mirror* strong opponent (a real
downloaded ladder replay's tape, `scripts/extract_replay_tape.py`,
instead of another main.py snapshot)? The underlying starvation is still
real and still unaddressed either way.

**Planting priority under the gate**: the same seed-42 replay that caught
the slot-starvation bug above also showed *why* freeing market slots
alone didn't fix utilization -- seeds sat available in inventory near
their `SEED_RESERVE` targets while planted-tile count kept falling, and a
turn-by-turn dump of every unit's action showed roughly half of all
farmer/hand turns were pure movement (`NORTH`/`SOUTH`/`EAST`/`WEST`, no
work done) across days 15-25. The real cause: `_find_target`'s candidate
sort key is `(priority, distance)` -- priority strictly dominates, so
distance never matters until every candidate at a better priority is
either claimed or absent. A fresh PLANT candidate sat at priority 4, the
*worst* of every candidate type except nothing -- worse than routine
watering (1), weeding (2), and fertilizer care (3). A single sampled turn
had 31 higher-priority candidates outstanding (mostly routine watering)
for only 12 units -- with owned land generating that much routine
maintenance most turns, PLANT was essentially never reached regardless of
how many empty tiles sat right next to an idle-looking unit, capping
utilization at whatever crop count generates just enough maintenance to
fully occupy the crew (matching the ~48-56% achievable-utilization
ceiling documented in the Land utilization gate notes above -- this is
likely *why* that ceiling exists). Added `PLANT_PRIORITY_UNDER_GATE`
(1.5): while `_land_utilization(farm)` (the same helper `_build_land_
orders` gates on) is below `LAND_UTILIZATION_GATE`, a fresh PLANT
candidate sorts at 1.5 instead of 4 -- still behind harvest (0) and
at-risk watering (0.5), so a dying crop is never sacrificed for a new
planting, but now it competes with routine watering and beats weeding and
fertilizer care instead of always losing to them. Measured in isolation
(same-process 30-seed A/B, git HEAD vs. this change alone, no order-
consolidation): a clean win against `reference/main_fertilizer_
selling_submitted_20260817.py`, 30/30 (100%, up from 28/30) with mean
money up 24% ($49.6k -> $61.7k), and a small, likely-noise-level cost
against x492 (mean money down 5.7%, still 0/30 either way -- x492 is
unwinnable at this agent's current strength regardless). Kept.

**Price-reactive fertilizer buying, kept but only partially validated**:
analyzing 22 downloaded real ladder replays of the shipped submission
found FERTILIZER price swinging far more than anything in the local pool
exercises -- one animal-heavy real opponent (team "kiro", `reference/
tape_kiro.py`) never bought a second quadrant but converted 92% of its
starting land to PASTURE/COOP and dumped the resulting fertilizer
byproduct it had no crops left to use, crashing the price from $100 to
$11 over the game and still beating this agent $49k to $34k off a
fraction of the land. `_build_fertilizer_orders` used to buy a flat,
price-blind `FERTILIZER_RESERVE = 2` regardless of how cheap or expensive
fertilizer was. Added a second, larger `FERTILIZER_STOCKPILE_RESERVE`
(10) that only kicks in when price is cheap (`<=
FERTILIZER_CHEAP_PRICE_FRAC * base`, 0.35 -- deliberately well below
`SELL_PARAMS["FERTILIZER"]`'s own 0.50 sell threshold so this can never
buy back at a price it would also sell at, the exact WHEAT sell/rebuy
mistake above) **and** `want_fertilizer` (`_any_fertilizable(farm, day)`,
the same signal fertilizer delivery already gates on) is true -- gating
on demand was a direct response to being asked "shouldn't this only buy
what will actually get used soon?": WHEAT/CARROT/TOMATO/STRAWBERRY
together are most of `CROP_TARGET_SHARE`, so fertilizable land is close
to a standing feature of this agent's own mix most of the game, but
gating costs nothing and avoids stockpiling with nowhere to deliver it
right after a land purchase, before the crop planner has filled the new
tiles.

**Validation is incomplete, and that's worth being honest about**: a
same-process 30-seed A/B against the local pool's toughest opponent and
x492 came back byte-identical between with-and-without this fix (down to
the last dollar) -- FERTILIZER price never got anywhere near the 0.35
cutoff in either matchup, so the new code path never fired at all; it was
a null test, not a passing one. Tried reproducing the real kiro game
locally via its extracted tape (`reference/tape_kiro.py`) instead, but
hit `extract_replay_tape.py`'s own documented divergence caveat hard:
once this agent's actions differ from the original recorded game (turn
one), kiro's tape stops interacting with a matching reality and its
economy collapses to near-nothing (final money $149 vs the original
$49k) -- fertilizer never even got cheap in that run either. No local
opponent currently reproduces the animal-dumping pattern that motivated
this fix. Kept anyway because the change is small, unit-tested, gated
behind both a demand check and a price band with no overlap with the
sell threshold, and provably a no-op (not a regression) in every locally
measured matchup so far -- but treat it as an untested opportunistic
improvement, not a validated win, until either a real opponent's replay
becomes available as a fuller (non-diverging) test vehicle or it shows up
doing something in a future submission's downloaded replays.

**Update: found the real (non-diverging) test vehicle, and it exposed why
this fix can't fire in the exact scenario that motivated it.**
`scripts/extract_replay_tape.py` had a real off-by-one bug (see its own
docstring and git history) making every extracted tape replay one turn
late for the whole game; fixed, and `reference/tape_kiro.py` now
reproduces kiro's real actions exactly (720/720 identical) instead of
collapsing to near-nothing. Re-running this agent against the corrected
tape at its recorded seed: FERTILIZER does drop into this fix's cheap
range (down to $28 by day 29, close to the original game's $11), but the
stockpile order **still never appears** -- `_build_feed_orders` was
already filling 9 of the 10 `maxMarketOrdersPerTurn` slots with
individual WHEAT orders (1 SELL + 9 BUY_PRODUCT WHEAT), so the fertilizer
order built later in `agent()`'s market list gets silently truncated away
every single turn, confirmed directly in the replay's own `action.market`
lists. The market-order-slot-starvation bug (see that note above) isn't
just costing seed purchases -- it's also the reason this fertilizer fix
is presently dead code in precisely the situation it was built for. Now
that a non-mirror opponent tape exists locally, the next attempt at
slot-starvation (the smaller-footprint idea floated above, or something
new) should be tested against `reference/tape_kiro.py` in addition to the
mirror-match pool opponent, to see whether the regression pattern from
the first three attempts is really mirror-match-specific as hypothesized.

**Animal land share raised, tried and reverted**: 22 downloaded real
ladder replays (this session's investigation with `scripts/
analyze_replay.py`) found a strong correlation -- opponents ending the
game with >=20% of
their land in animals beat this agent 75% of the time, vs. a 36% loss
rate under 20%. Investigating one extreme case (team "kiro", `reference/
tape_kiro.py`, 92% animal land share off a single quadrant) suggested why:
animals don't suffer the crop-collapse problem documented throughout this
file (no seed-buying, no watering, no market-slot competition for seeds --
just feed/collect/care), so shifting more weight there looked like a way
to sidestep a problem that's resisted several direct fixes. Raised
`ANIMAL_LAND_SHARE` 0.18 -> 0.25 and `ANIMAL_MAX_PER_UNIT` 1.0 -> 1.5
together (`_animal_target` takes the smaller of the two, so raising
either alone while the other stays at its old value is a no-op once the
unraised one is what's actually binding -- confirmed true at 0.18/1.0
starting around 3 quadrants). **Measured worse**: same-process 30-seed
A/B (git HEAD vs. this change alone) against `reference/main_fertilizer_
selling_submitted_20260817.py` dropped win rate 30/30 (100%) -> 28/30
(93%) with mean money down ~7% ($61.7k -> $57.1k); against x492, mean
money down ~15% ($27.8k -> $23.6k), still 0/30 either way. A single-seed
check against the newly-fixed `reference/tape_kiro.py` (the actual replay
that motivated this) went the other way -- both sides' money rose, ours
by slightly more -- but n=1 there (tapes are only valid at their recorded
seed) isn't enough to outweigh a properly-powered 30-seed result showing
the opposite; see the eval-noise note above for why a single seed's
result gets distrusted by default in this project. Reverted back to
0.18/1.0. The correlation in the real replay data is still real and
still unexplained -- it just isn't as simple as "we should run more
animals ourselves." Leading open question: maybe it's not animal *count*
that matters but something about how those specific real opponents
converted a large flock into money more efficiently than this agent's own
feed/sell logic does (worth checking FEED/SELL timing and price
discipline in a real animal-heavy opponent's replay before touching this
constant again).
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_PARAMS,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

CASH_CROPS = ("MELON", "WHEAT", "CARROT", "TOMATO", "STRAWBERRY")
CROP_TARGET_SHARE = {"MELON": 0.45, "WHEAT": 0.25, "CARROT": 0.15, "TOMATO": 0.10, "STRAWBERRY": 0.05}
FERTILIZE_CROPS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY"}
PRICE_ATTRACTIVENESS_WEIGHT = 0.1  # how much a crop's current price (as a ratio of its
                                    # base) pulls _CropPlanner.choose() toward it, on top
                                    # of the target-mix-share gap -- see the module
                                    # docstring's Price-attractiveness-aware planting note

PLANT_PRIORITY = 4  # _find_target's base priority for planting an empty tile -- see
                     # the module docstring's Planting priority under the gate note
PLANT_PRIORITY_UNDER_GATE = 1.5  # used instead of PLANT_PRIORITY while owned land is
                                  # under LAND_UTILIZATION_GATE -- still below harvest (0)
                                  # and at-risk water (0.5) so a dying crop is never
                                  # sacrificed for a new planting, but now competes with
                                  # routine water (1) and beats weeding (2) and fertilizer
                                  # care (3), instead of always losing to them

SEED_RESERVE = {"MELON": 3, "WHEAT": 3, "CARROT": 3, "TOMATO": 2, "STRAWBERRY": 2}
FERTILIZER_RESERVE = 2
FERTILIZER_STOCKPILE_RESERVE = 10  # reserve target while price is cheap -- see
                                    # _build_fertilizer_orders's Price-reactive note
FERTILIZER_CHEAP_PRICE_FRAC = 0.35  # price <= this fraction of base triggers stockpiling.
                                     # Deliberately well below SELL_PARAMS["FERTILIZER"]'s own
                                     # 0.50 sell threshold -- see _build_fertilizer_orders
FERTILIZER_MAX_PRICE_FRAC = 1.2  # don't buy fertilizer more than 20% over base

SELL_PARAMS = {
    # threshold: fraction of base price required to sell at all -- see
    # _build_sell_orders for why there's no per-turn cap. Shared by
    # _CropPlanner and _AnimalPlanner's own _price_ok, so a crop/animal
    # kind's plant/build eligibility always matches its sell eligibility.
    "MELON":      {"threshold": 0.65},
    "WHEAT":      {"threshold": 0.45},
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
SHED_OVERFLOW_GUARD = 0.85  # start dumping regardless of price above this fraction full

ANIMAL_KINDS = ("COW", "SHEEP", "GOOSE")
ANIMAL_PRODUCT = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
ANIMAL_STRUCTURE = {"COW": "PASTURE", "SHEEP": "PASTURE", "GOOSE": "COOP"}
ANIMAL_PRODUCTS = ("MILK", "WOOL", "EGG")  # finished products a unit carries to the shed
ANIMAL_TARGET_SHARE = {"COW": 0.60, "SHEEP": 0.40, "GOOSE": 0.0}  # GOOSE has the
    # lowest $/day of the three ($50: EGG base 50 / 1-day interval, vs COW's $80 and
    # SHEEP's $66.7) despite being the cheapest to buy -- 5/5 downloaded top-ladder
    # replays (3 different players) never place one; see the module docstring's
    # Animal husbandry (attempt 3) note
ANIMAL_LAND_SHARE = 0.18  # target fraction of owned tiles reserved for animal
                          # structures -- see the module docstring's Animal husbandry
                          # (attempt 3) note; top-ladder replays hold this steady
                          # around 15-24% throughout the season. Tried raising to 0.25
                          # (see the Animal land share raised, tried and reverted note)
                          # and it measured worse -- back to 0.18.
ANIMAL_MAX_PER_UNIT = 1.0  # hard cap on live animals, per farmer/hand currently hired --
                            # see the module docstring's Animal capacity vs crop capacity
                            # note. ANIMAL_LAND_SHARE alone grows with quadrants bought
                            # regardless of crew size, letting the animal count outgrow
                            # what the same crew can also water/harvest/weed for
ANIMAL_RESERVE = 1  # spare bought-but-not-yet-placed animal tokens to keep queued, per kind --
                    # known to let tokens strand with no empty structure to receive them in
                    # some games; see _build_animal_orders's own "Known bug" note
FEED_RESERVE_PER_ANIMAL = 2  # days of WHEAT buffer per live animal, bought directly from
                              # the market -- see _build_feed_orders for why
FEED_MAX_PRICE_FRAC = 5.0  # a near-nonbinding backstop, not a real cap: missing a feed
                            # loses the whole animal ($300-500+), dwarfing even a wildly
                            # inflated wheat price -- see the module docstring's Animal
                            # husbandry (attempt 2) note for why a tighter cap is a trap

HANDS_BASE = 2  # hands hired even at the starting NW quadrant alone
HANDS_PER_QUADRANT = 3
MAX_HANDS = 12  # was 6, briefly 8; see the module docstring's Real-hiring-pace note --
                # a real top-ladder replay showed 5 hands on day 0 with 1 quadrant and
                # 10-12 sustained for the rest of the season, nothing like the old
                # `2 * quadrants` formula's pace. HANDS_BASE/HANDS_PER_QUADRANT/
                # MAX_HANDS were picked so _hands_target(quadrants) tracks that curve
                # (5/8/11/12 at 1-4 quadrants) while staying strictly increasing
                # through all 4 quadrants, so _land_worth_buying never blocks the 4th
SAFETY_RESERVE = 300  # cash kept in reserve before optional buys/land/fertilizer
MAX_MARKET_ORDERS = 10  # data/README.md default; not exposed in `obs`

DEFAULT_EPISODE_STEPS = 720  # data/README.md defaults, used if `config` is unavailable
DEFAULT_TURNS_PER_DAY = 24
SEED_BUFFER_DAYS = 1  # a crop needs first_yield_day + this many days left to be worth planting
LAND_CUTOFF_DAYS = 5  # stop buying new land this close to the end, it won't pay back in time
LAND_UTILIZATION_GATE = 0.50  # don't buy the next quadrant until this much of what we
                               # already own is planted -- see the module docstring's
                               # Land utilization gate note. Still below the ~48-56%
                               # achievable-utilization ceiling observed in real replays
                               # (0.6 deadlocked against it), just less borderline than 0.45
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
    spatial clustering (animals) note: real top-ladder play keeps every
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


def _is_harvestable(tile, day):
    """yield_units > 0 alone is NOT "ready" for one-time crops: `_new_plant`
    seeds it at 1 immediately on planting (kaggriculture.py), but HARVEST is
    still a no-op until `first_yield_day` (age 2 for wheat/carrot, 10 for
    melon). Confusing the two makes a unit spam a no-op HARVEST instead of
    WATER, and the crop dies to neglect before ever becoming harvestable."""
    if not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0:
        return False
    if tile.get("kind") != "PLANT":
        return True  # animal structures have no such gate; out of scope here anyway
    return day - tile["planted_day"] >= CROPS[tile["crop"]]["first_yield_day"]


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
        module docstring's Price-attractiveness-aware planting note."""
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
    of the season -- see the module docstring's Animal capacity vs crop
    capacity note. Shared by _AnimalPlanner and _build_animal_orders so
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
    share-balancing logic. Unlike attempt 2 (see the module docstring's
    Animal husbandry notes), capacity is sized as a target *fraction of
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
        checked at the module docstring's Animal husbandry (attempt 3) note
        after that let GOOSE keep getting built despite a 0 target share)."""
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


def _find_target(
    pos, farm, day, board_size, claimed, inv, planner, animal_planner, shed_tiles,
    want_fertilizer, shed_wheat_available, liquidating, animal_zone, quadrant_util,
    plant_priority,
):
    """Scan the whole farm for the nearest (or, for a fresh planting,
    nearest-adjusted-for-quadrant-balance -- see below) unclaimed tile worth
    walking to, across both crops and animals -- one shared crew, one shared
    priority order, so a unit works whichever is more urgent right now
    instead of being permanently assigned to one or the other (see the
    module docstring's Animal husbandry (attempt 3) note).

    `liquidating` (close to the last day, see the module docstring's Endgame
    liquidation note) swaps water's and shed-return's priority: watering a
    crop that won't finish growing in time is wasted effort, while getting
    already-harvested produce into the shed guarantees it can still be sold.

    A tile with `consecutive_unwatered >= 1` (kaggriculture.py: missing
    water two days running turns a PLANT into a WEED, and a fresh planting
    already starts pre-loaded at 1 -- "the planting day counts as
    unwatered") is one miss away from being lost outright, so its water
    candidate jumps to harvest's own priority (0) instead of the routine
    `water_priority` -- see the module docstring's Neglected-tile urgency
    note. This is what actually stops distant land from quietly losing
    crops to neglect, not geography: a unit chasing marginally-closer
    routine chores near the shed will still lose to a genuinely at-risk
    tile regardless of how far away it is.

    `plant_priority` (PLANT_PRIORITY or PLANT_PRIORITY_UNDER_GATE, chosen
    once per turn in `agent()`) is the priority a fresh PLANT candidate
    sorts at -- see the module docstring's Planting priority under the
    gate note: priority strictly outranks distance in this function's sort
    key, so at its baseline (4, below every maintenance chore except
    routine water) a single turn with more outstanding water/harvest/weed/
    animal-care work than there are units leaves *zero* units ever reaching
    a PLANT candidate, no matter how many empty tiles sit right next to
    them.

    Every candidate carries a precomputed `sort_dist` (its 4th field) rather
    than the sort key recomputing raw distance uniformly: a fresh PLANT
    candidate gets its raw distance inflated by how utilized its quadrant
    already is (`quadrant_util`), nudging new plantings toward whichever
    quadrant is most neglected instead of always the literal nearest empty
    tile -- see the module docstring's spatial clustering (crops) note. Every
    other candidate type keeps plain, unadjusted distance.
    """
    fx, fy = pos
    water_priority = 5 if liquidating else 1
    shed_priority = 1 if liquidating else 5
    spread_penalty = board_size // 2  # scale of the quadrant-balance nudge, in tiles
    candidates = []
    any_unfed = False
    for y in range(board_size):
        for x in range(board_size):
            if (x, y) in claimed:
                continue
            tile = farm["tiles"][y][x]
            dist = abs(x - fx) + abs(y - fy)
            if _is_plant(tile):
                if _is_harvestable(tile, day):
                    candidates.append((x, y, 0, dist))  # harvest
                elif not tile["watered_today"]:
                    at_risk = not liquidating and tile.get("consecutive_unwatered", 0) >= 1
                    # 0.5: below harvest (0, already-earned money, never worth delaying for
                    # a tile that would merely avoid a future loss) but above every routine
                    # chore, including normal watering
                    candidates.append((x, y, 0.5 if at_risk else water_priority, dist))  # water
                elif inv.get("FERTILIZER", 0) > 0 and _is_fertilizable(tile, day):
                    candidates.append((x, y, 3, dist))  # deliver fertilizer
            elif _is_weed(tile):
                candidates.append((x, y, 2, dist))  # clear a weed to reclaim the tile
            elif _is_animal_tile(tile):
                if tile.get("yield_units", 0) > 0:
                    candidates.append((x, y, 0, dist))  # harvest animal product
                elif not tile.get("fed_today"):
                    any_unfed = True
                    if inv.get("WHEAT", 0) > 0:
                        candidates.append((x, y, -1, dist))  # feed -- more urgent than
                                                               # anything; 2 misses = the
                                                               # animal escapes
                elif tile.get("fertilizer_available"):
                    candidates.append((x, y, 3, dist))  # collect fertilizer
                elif not tile.get("cared_today"):
                    candidates.append((x, y, 3, dist))  # care (free next-yield bonus)
            elif _is_empty_structure(tile) and _carried_animal_for(tile, inv):
                candidates.append((x, y, 0, dist))  # place a carried animal -- don't leave it idle
            elif tile is None:
                if (x, y) in animal_zone:
                    candidates.append((x, y, 4, dist))  # reserved for an animal structure
                elif planner.has_plantable():
                    q = _quadrant_of(x, y, board_size)
                    candidates.append(
                        (x, y, plant_priority, dist + quadrant_util[q] * spread_penalty)
                    )  # plant
                elif animal_planner.wants_structure():
                    candidates.append((x, y, 4, dist))  # build a structure (overflow beyond the zone)

    carrying_produce = any(
        n > 0 and k not in ("WHEAT", *ANIMAL_KINDS) and (k != "FERTILIZER" or not want_fertilizer or liquidating)
        for k, n in inv.items()
    )
    needs_fertilizer = inv.get("FERTILIZER", 0) == 0 and want_fertilizer and not liquidating
    needs_wheat = inv.get("WHEAT", 0) <= 0 and any_unfed and shed_wheat_available
    needs_animal_pickup = animal_planner.wants_pickup() and not any(inv.get(k, 0) > 0 for k in ANIMAL_KINDS)
    if carrying_produce or needs_fertilizer or needs_wheat or needs_animal_pickup:
        priority = -1 if needs_wheat else shed_priority
        for (sx, sy) in shed_tiles:
            if (sx, sy) not in claimed:
                candidates.append((sx, sy, priority, abs(sx - fx) + abs(sy - fy)))  # return to shed

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[2], c[3]))
    return candidates[0]


def _build_action(kind):
    return ["BUILD_COOP" if ANIMAL_STRUCTURE[kind] == "COOP" else "BUILD_PASTURE"]


def _decide_unit(
    pos, inv, farm, private, day, board_size, claimed, planner, animal_planner, shed_tiles,
    want_fertilizer, liquidating, animal_zone, quadrant_util, plant_priority,
):
    fx, fy = pos
    tile = farm["tiles"][fy][fx]

    if _is_plant(tile):
        if _is_harvestable(tile, day):
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
        carried_produce = {
            k: n for k, n in inv.items()
            if n > 0 and k not in ("WHEAT", *ANIMAL_KINDS)
            and (k != "FERTILIZER" or not want_fertilizer or liquidating)
        }
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

    shed_wheat_available = private["shed"].get("WHEAT", 0) > 0
    target = _find_target(
        pos, farm, day, board_size, claimed, inv, planner, animal_planner, shed_tiles,
        want_fertilizer, shed_wheat_available, liquidating, animal_zone, quadrant_util,
        plant_priority,
    )
    if target:
        claimed.add((target[0], target[1]))
        step = _step_toward(fx, fy, target[0], target[1])
        if step:
            return [step]
    return ["PASS"]


def _build_sell_orders(shed, prices, days_remaining):
    """Dumps the entire shed stock of a crop in one order once its price
    threshold is met -- same behavior as melon_maxxer, deliberately reverted
    from an earlier version that capped units sold per turn (see the module
    docstring's Correction note). A per-turn cap was tried expecting it to
    avoid crashing the price via a big single-order dump; instead it was a
    measured net negative (a 19/20-seed head-to-head loss to this dump
    version, `reference/main_dump_sell.py`): holding harvested value in the
    shed waiting for the cap to free up slows reinvestment into land/seeds/
    hands more than an occasional price dip costs, at this agent's actual
    production scale. See the module docstring's Sell/rebuy round-trip note
    for a second, more targeted attempt at holding back WHEAT specifically
    (for animal feed) that hit the same wall for the same underlying reason.
    """
    orders = []
    shed_total = sum(shed.values())
    crowded = shed_total >= SHED_CAPACITY * SHED_OVERFLOW_GUARD
    liquidating = days_remaining <= ENDGAME_LIQUIDATION_DAYS
    for crop, params in SELL_PARAMS.items():
        qty = shed.get(crop, 0)
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
    capacity -- see the module docstring's Land utilization gate note.
    Originally caught with the old `2 * quadrants` hire formula (capped at
    MAX_HANDS=6): the ceiling fell at exactly 3 quadrants, and a replay
    showed the same 6 hands spread across a 4th quadrant's extra land
    planting no more tiles in absolute terms than 3 quadrants' worth, just
    diluted thinner. `_hands_target`'s current formula (see the module
    docstring's Real hiring pace note) was deliberately picked to stay
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


def _build_land_orders(farm, days_remaining):
    if days_remaining < LAND_CUTOFF_DAYS:
        return []  # a fresh quadrant can't earn back its cost this close to the end
    quadrants_now = len(farm["unlocked_quadrants"])
    n_extra = quadrants_now - 1  # NW is always owned
    if n_extra >= len(LAND_ORDER) or not _land_worth_buying(quadrants_now):
        return []
    if _land_utilization(farm) < LAND_UTILIZATION_GATE:
        return []  # can't staff what we already own -- more land would just sit idle
    if farm["money"] - SAFETY_RESERVE >= LAND_PRICES[n_extra]:
        return [["BUY_LAND"]]
    return []


def _build_hire_orders(farm):
    desired = _hands_target(len(farm["unlocked_quadrants"]))
    remaining = desired - farm["hires_today"]
    if remaining <= 0:
        return []
    return [["HIRE"] for _ in range(remaining)]


def _build_seed_orders(seeds, money, days_remaining):
    orders = []
    remaining = money - SAFETY_RESERVE
    for crop, reserve in SEED_RESERVE.items():
        if days_remaining < CROPS[crop]["first_yield_day"] + SEED_BUFFER_DAYS:
            continue  # wouldn't mature (and sell) before the season ends
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
    ungated, same as before this fix -- see the module docstring's
    Selling fertilizer note for why that small a reserve's occasional
    demand-blind churn was already judged an acceptable, bounded cost.
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
    starve even with plenty of attention: `_build_sell_orders` dumps every
    WHEAT harvest as soon as its price clears the sell threshold, so the
    shed can sit at 0 WHEAT for turns at a stretch -- see the module
    docstring's Animal husbandry (attempt 2) note. Placed ahead of
    `_build_seed_orders` in `agent()`'s market list: missing a seed buy
    delays a planting one turn, missing a feed loses the whole animal."""
    live = _live_animal_count(farm)
    if live == 0:
        return []
    reserve = live * FEED_RESERVE_PER_ANIMAL
    have = private.get("shed", {}).get("WHEAT", 0)
    needed = reserve - have
    if needed <= 0:
        return []
    base = MARKET_PARAMS["WHEAT"]["base"]
    price = prices.get("WHEAT", base)
    if price > base * FEED_MAX_PRICE_FRAC:
        return []
    remaining = money - SAFETY_RESERVE
    orders = []
    while needed > 0 and remaining >= price:
        orders.append(["BUY_PRODUCT", "WHEAT", 1])
        remaining -= price
        needed -= 1
    return orders


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
    swings the fix attempts caused). Worth another look with a clearer
    hypothesis before retrying."""
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

    market_prices = (obs.get("market", {}) or {}).get("prices", {})
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

    claimed = set()
    units = [farm["farmer"]] + list(farm["hands"])
    actions = [
        _decide_unit(
            pos,
            inventories[i] if i < len(inventories) else {},
            farm,
            private,
            day,
            board_size,
            claimed,
            planner,
            animal_planner,
            shed_tiles,
            want_fertilizer,
            liquidating,
            animal_zone,
            quadrant_util,
            plant_priority,
        )
        for i, pos in enumerate(units)
    ]
    farmer_action, hands_actions = actions[0], actions[1:]

    market = []
    market += _build_sell_orders(shed, market_prices, days_remaining)
    market += _build_land_orders(farm, days_remaining)
    market += _build_hire_orders(farm)
    market += _build_feed_orders(farm, private, market_prices, farm["money"])
    market += _build_seed_orders(seeds, farm["money"], days_remaining)
    market += _build_fertilizer_orders(private, market_prices, farm["money"], want_fertilizer)
    market += _build_animal_orders(farm, private, farm["money"])
    market = market[:MAX_MARKET_ORDERS]

    return {"farmer": farmer_action, "hands": hands_actions, "market": market}

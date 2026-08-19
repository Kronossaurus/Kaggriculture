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
loses the whole animal, a bigger loss than the travel time saved. (WHEAT
itself is *not* currently exempted from the general "never counts as
carried produce worth returning" exclusion even during liquidation, unlike
FERTILIZER -- a real, still-open gap, see the next-steps memory.)

**Neglected-tile urgency**: a tile that missed watering once
(`consecutive_unwatered >= 1` -- two consecutive misses turns a PLANT into a
WEED) scores priority 0.5 in `_find_target`, above every routine chore but
below harvest (already-earned money) and still below at-risk situations.
Without this, a unit chasing a marginally-closer routine chore near the shed
would keep winning over saving a tile one missed watering from being lost
outright. Paired with raising `MAX_HANDS` 6 -> 8 (which conveniently matches
2x the 4-quadrant land cap, so it stops binding at all) to recover a
capacity dip the priority change caused on its own.

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
counts collapsing over the back half of a season despite idle cash. Five
independent fix attempts across this project (full order consolidation
twice, an order-count cap, a WHEAT sell-side reserve, a per-order quantity
cap) all measured as real regressions on a clean same-process 30-seed A/B,
consistently worst against a genuine non-mirror opponent
(`reference/tape_kiro.py`) -- the likely mechanism is that this agent's own
per-unit affordability check uses a static price and doesn't account for the
engine's own within-order price walk, so any change that lets a turn request
more WHEAT in fewer, larger orders ends up spending more at a worse average
price than the old scheme's incidental slot-count cap ever allowed. Treat
this as settled without new instrumentation (e.g. tracking realized purchase
prices turn-by-turn, old vs. new, in the same replay) -- not a queue of
ideas still worth blind-testing.

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
today's 5.0 has plenty of headroom above that. The round trip itself is
still real and still unaddressed; a smaller-footprint idea (coordinating the
sell and feed decisions within the same turn instead of a standing reserve)
is worth trying before another reserve-sizing attempt.

Still out of scope: any opponent modeling -- this agent only ever reacts to
its own farm/market state. See rl/README.md for the parallel (currently
stalled) attempt at a learned policy. See this project's session memory for
the current list of open next-steps (pasture-placement drift, idle/wandering
hands, harvest-ready plants converting to weeds, planting stopping early
around day ~25, WHEAT unliquidated at game end, an opening-book idea for the
first days).
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
                                    # docstring's Price-aware planting note

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
FEED_MAX_PRICE_FRAC = 5.0  # a near-nonbinding backstop, not a real cap: missing a feed
                            # loses the whole animal ($300-500+), dwarfing even a wildly
                            # inflated wheat price -- see the module docstring's WHEAT
                            # sell/rebuy round trip note for why a tighter cap is a trap
                            # (below ~1.4x base, animals starve outright in testing)

HANDS_BASE = 2  # hands hired even at the starting NW quadrant alone
HANDS_PER_QUADRANT = 3
MAX_HANDS = 12  # was 6, briefly 8; see the module docstring's Land / hands note --
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
                               # Land / hands note. Still below the ~48-56%
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
    module docstring's Animals note).

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
    sorts at -- see the module docstring's Land / hands note: priority
    strictly outranks distance in this function's sort
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
    tile -- see the module docstring's Land / hands note. Every
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

    carried_produce_total = sum(
        n for k, n in inv.items()
        if n > 0 and k not in ("WHEAT", *ANIMAL_KINDS) and (k != "FERTILIZER" or not want_fertilizer or liquidating)
    )
    carrying_produce = carried_produce_total > 0
    needs_fertilizer = inv.get("FERTILIZER", 0) == 0 and want_fertilizer and not liquidating
    needs_wheat = inv.get("WHEAT", 0) <= 0 and any_unfed and shed_wheat_available
    needs_animal_pickup = animal_planner.wants_pickup() and not any(inv.get(k, 0) > 0 for k in ANIMAL_KINDS)
    if carrying_produce or needs_fertilizer or needs_wheat or needs_animal_pickup:
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
    docstring's Selling note). A per-turn cap was tried expecting it to
    avoid crashing the price via a big single-order dump; instead it was a
    measured net negative (a 19/20-seed head-to-head loss to this dump
    version, `reference/main_dump_sell.py`): holding harvested value in the
    shed waiting for the cap to free up slows reinvestment into land/seeds/
    hands more than an occasional price dip costs, at this agent's actual
    production scale. See the module docstring's WHEAT sell/rebuy round trip
    note for a second, more targeted attempt at holding back WHEAT specifically
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
    starve even with plenty of attention: `_build_sell_orders` dumps every
    WHEAT harvest as soon as its price clears the sell threshold, so the
    shed can sit at 0 WHEAT for turns at a stretch -- see the module
    docstring's WHEAT sell/rebuy round trip note. Placed ahead of
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
    market += _build_seed_orders(seeds, farm["money"], days_remaining, market_prices)
    market += _build_fertilizer_orders(private, market_prices, farm["money"], want_fertilizer)
    market += _build_animal_orders(farm, private, farm["money"])
    market = market[:MAX_MARKET_ORDERS]

    return {"farmer": farmer_action, "hands": hands_actions, "market": market}

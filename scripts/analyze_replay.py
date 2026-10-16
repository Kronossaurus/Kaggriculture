"""Summarize one or more Kaggriculture replay JSONs -- win/loss, money, land"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_OUR_TEAM = "Victor Carriço"
TURNS_PER_DAY = 24
MAX_MARKET_ORDERS = 10


def resolve_paths(specs: list[str]) -> list[Path]:
    """Each spec may be a single file, a directory (every *.json inside,"""
    paths: list[Path] = []
    for spec in specs:
        p = Path(spec)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
        elif "*" in spec or "?" in spec:
            paths.extend(sorted(Path().glob(spec)))
        else:
            paths.append(p)
    return paths


def tile_stats(farm: dict) -> tuple[int, int, Counter]:
    """(owned, empty, kind_counts) -- owned counts every non-LOCKED tile"""
    owned = 0
    empty = 0
    counts: Counter = Counter()
    for row in farm["tiles"]:
        for tile in row:
            if tile == "LOCKED":
                continue
            owned += 1
            if tile is None:
                empty += 1
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                counts[tile.get("crop", "PLANT?")] += 1
            elif kind == "WEED":
                counts["WEED"] += 1
            elif kind in ("COOP", "PASTURE"):
                counts[kind] += 1
    return owned, empty, counts


def crop_and_animal_counts(counts: Counter) -> tuple[int, int]:
    crop = sum(v for k, v in counts.items() if k not in ("WEED", "COOP", "PASTURE"))
    animal = counts.get("COOP", 0) + counts.get("PASTURE", 0)
    return crop, animal


def identify_us(team_names: list[str] | None, our_team: str, us_override: int | None) -> int | None:
    if us_override is not None:
        return us_override
    if not team_names:
        return 0
    for i, name in enumerate(team_names):
        if our_team in name:
            return i
    return None


def quadrant_timeline(steps: list, player_idx: int) -> list[tuple[int, list[str]]]:
    events = []
    prev: list[str] = []
    n_days = len(steps) // TURNS_PER_DAY
    for day in range(n_days + 1):
        idx = min(day * TURNS_PER_DAY + 20, len(steps) - 1)
        obs = steps[idx][0]["observation"]
        uq = obs["farms"][player_idx]["unlocked_quadrants"]
        if uq != prev:
            events.append((day, list(uq)))
            prev = list(uq)
    return events


def classify_order(order: list) -> str:
    """Maps one submitted market order to one of main.py's 7 order"""
    op = order[0]
    if op == "SELL":
        return "sell"
    if op == "BUY_LAND":
        return "land"
    if op == "HIRE":
        return "hire"
    if op == "BUY_PRODUCT":
        resource = order[1] if len(order) > 1 else None
        return "feed" if resource == "WHEAT" else "fertilizer"
    if op == "BUY_SEED":
        return "seed"
    if op == "BUY_ANIMAL":
        return "animal"
    return "other"


def order_occupancy(steps: list, player_idx: int) -> list[dict]:
    """Per-turn order-type counts from the actually-submitted (already"""
    rows = []
    for t in range(len(steps) - 1):
        action = steps[t + 1][player_idx].get("action") or {}
        market = action.get("market") or []
        counts = Counter(classify_order(o) for o in market)
        rows.append({
            "turn": t,
            "counts": counts,
            "total": len(market),
            "saturated": len(market) >= MAX_MARKET_ORDERS,
        })
    return rows


def print_order_occupancy(rows: list[dict], n_days: int) -> None:
    """Day-bucketed mean order count by type + saturation-turn fraction --"""
    categories = ("sell", "land", "hire", "feed", "seed", "fertilizer", "animal", "other")
    print(f"\n  {'day':>4} " + " ".join(f"{c:>10}" for c in categories) + f" {'saturated':>10}")
    for day in range(n_days):
        day_rows = [r for r in rows if r["turn"] // TURNS_PER_DAY == day]
        if not day_rows:
            continue
        n = len(day_rows)
        means = {c: sum(r["counts"].get(c, 0) for r in day_rows) / n for c in categories}
        sat_frac = sum(1 for r in day_rows if r["saturated"]) / n
        print(f"  {day:>4} " + " ".join(f"{means[c]:>10.2f}" for c in categories) + f" {sat_frac:>9.1%}")


def purchase_trace(steps: list, player_idx: int, resource: str) -> list[dict]:
    """Per-turn realized-purchase-behavior trace for one shed-tracked"""
    rows = []
    for t in range(len(steps) - 1):
        obs_t = steps[t][player_idx]["observation"]
        obs_t1 = steps[t + 1][player_idx]["observation"]
        action = steps[t + 1][player_idx].get("action") or {}
        market = action.get("market") or []

        requested_qty = sum(
            o[2] if len(o) > 2 else 1
            for o in market
            if o[0] == "BUY_PRODUCT" and len(o) > 1 and o[1] == resource
        )
        sold_qty = sum(
            o[2] if len(o) > 2 else 0
            for o in market
            if o[0] == "SELL" and len(o) > 1 and o[1] == resource
        )
        touches_other = any(
            o[0] in ("SELL", "BUY_PRODUCT", "BUY_SEED") and len(o) > 1 and o[1] != resource
            for o in market
        )

        decision_price = obs_t["market"]["prices"].get(resource)
        next_price = obs_t1["market"]["prices"].get(resource)
        shed_before = obs_t["private"]["shed"].get(resource, 0)
        shed_after = obs_t1["private"]["shed"].get(resource, 0)
        acquired_qty = (shed_after - shed_before) + sold_qty

        money_before = obs_t["farms"][player_idx]["money"]
        money_after = obs_t1["farms"][player_idx]["money"]

        is_clean = requested_qty > 0 and not touches_other
        implied_avg_price = None
        if is_clean and acquired_qty > 0:
            implied_avg_price = (money_before - money_after) / acquired_qty

        rows.append({
            "turn": t,
            "requested_qty": requested_qty,
            "decision_price": decision_price,
            "next_price": next_price,
            "acquired_qty": acquired_qty,
            "is_clean": is_clean,
            "implied_avg_price": implied_avg_price,
        })
    return rows


def summarize_purchase_trace(trace: list[dict]) -> dict:
    clean = [r for r in trace if r["is_clean"] and r["implied_avg_price"] is not None]
    ratios = [
        r["next_price"] / r["decision_price"]
        for r in trace
        if r["requested_qty"] > 0 and r["decision_price"]
    ]
    return {
        "mean_implied_avg_price": sum(r["implied_avg_price"] for r in clean) / len(clean) if clean else None,
        "n_clean": len(clean),
        "mean_price_walk_ratio": sum(ratios) / len(ratios) if ratios else None,
        "total_requested": sum(r["requested_qty"] for r in trace),
        "total_acquired": sum(r["acquired_qty"] for r in trace if r["acquired_qty"] > 0),
    }


def print_purchase_trace_summary(label: str, trace: list[dict]) -> None:
    s = summarize_purchase_trace(trace)
    implied = f"{s['mean_implied_avg_price']:.2f}" if s["mean_implied_avg_price"] is not None else "n/a"
    ratio = f"{s['mean_price_walk_ratio']:.3f}" if s["mean_price_walk_ratio"] is not None else "n/a"
    print(f"  {label}: mean_implied_avg_price={implied} (n_clean={s['n_clean']})  "
          f"mean_next/decision_ratio={ratio}  "
          f"total_requested={s['total_requested']}  total_acquired={s['total_acquired']}")


def compare_purchase_traces(traces: list[tuple[str, list[dict]]]) -> None:
    """Old-vs-new, side by side, never a single blended diff number --"""
    print("\n  === purchase-trace comparison ===")
    for label, trace in traces:
        print_purchase_trace_summary(label, trace)


def analyze_one(
    path: Path,
    our_team: str,
    us_override: int | None,
    show_trajectory: bool,
    show_orders: bool = False,
    purchase_trace_resource: str | None = None,
) -> dict | None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    info = data.get("info", {}) or {}
    team_names = info.get("TeamNames")
    seed = info.get("seed", "?")
    steps = data["steps"]

    us = identify_us(team_names, our_team, us_override)
    if us is None:
        print(f"\n=== {path.name} === could not identify our seat (teams={team_names}); "
              f"pass --our-team or --us")
        return None
    opp = 1 - us
    opp_name = team_names[opp] if team_names else f"player {opp}"

    final = steps[-1]
    us_money, opp_money = final[us]["reward"], final[opp]["reward"]
    result = "W" if us_money > opp_money else ("L" if us_money < opp_money else "T")

    print(f"\n=== {path.name}  vs {opp_name!r}  seed={seed}  result={result} ===")
    print(f"us=${us_money:,.0f}  opponent=${opp_money:,.0f}")

    for label, pi in (("us", us), ("opp", opp)):
        for day, uq in quadrant_timeline(steps, pi):
            print(f"  day {day:>2}: {label} unlocked -> {uq}")

    peak_crop = {us: 0, opp: 0}
    peak_crop_day = {us: 0, opp: 0}
    peak_animal = {us: 0, opp: 0}
    if show_trajectory:
        print(f"  {'day':>4} {'money_us':>9} {'own':>4} {'crop':>5} {'anim':>5} {'weed':>5}"
              f" | {'money_opp':>9} {'own':>4} {'crop':>5} {'anim':>5} {'weed':>5}")
    n_days = len(steps) // TURNS_PER_DAY
    for day in range(n_days):
        idx = min(day * TURNS_PER_DAY + 20, len(steps) - 1)
        obs = steps[idx][0]["observation"]
        row = {}
        for pi in (us, opp):
            owned, _empty, counts = tile_stats(obs["farms"][pi])
            crop, animal = crop_and_animal_counts(counts)
            if crop > peak_crop[pi]:
                peak_crop[pi] = crop
                peak_crop_day[pi] = day
            peak_animal[pi] = max(peak_animal[pi], animal)
            row[pi] = (obs["farms"][pi]["money"], owned, crop, animal, counts.get("WEED", 0))
        if show_trajectory:
            mu, ou, cu, au, wu = row[us]
            mo, oo, co, ao, wo = row[opp]
            print(f"  {day:>4} {mu:>9,.0f} {ou:>4} {cu:>5} {au:>5} {wu:>5}"
                  f" | {mo:>9,.0f} {oo:>4} {co:>5} {ao:>5} {wo:>5}")

    final_obs = final and steps[-1][0]["observation"]
    final_crop = {}
    final_animal_share = {}
    for label, pi in (("us", us), ("opp", opp)):
        owned, _empty, counts = tile_stats(final_obs["farms"][pi])
        crop, animal = crop_and_animal_counts(counts)
        util = (crop + animal) / owned * 100 if owned else 0.0
        animal_share = animal / owned * 100 if owned else 0.0
        final_crop[pi] = crop
        final_animal_share[pi] = animal_share
        collapse = (1 - crop / peak_crop[pi]) * 100 if peak_crop[pi] else 0.0
        print(f"  final {label}: owned={owned} crop={crop} anim={animal} ({animal_share:.1f}% of land) "
              f"weed={counts.get('WEED', 0)} util={util:.1f}%  "
              f"(peak crop {peak_crop[pi]} on day {peak_crop_day[pi]}, {collapse:.0f}% collapsed by game end; "
              f"peak animal {peak_animal[pi]})")

    orders_summary = None
    if show_orders:
        n_days = len(steps) // TURNS_PER_DAY
        occ_rows = order_occupancy(steps, us)
        print_order_occupancy(occ_rows, n_days)
        n = len(occ_rows) or 1
        orders_summary = {
            "sat_frac": sum(1 for r in occ_rows if r["saturated"]) / n,
            "mean_feed": sum(r["counts"].get("feed", 0) for r in occ_rows) / n,
        }

    trace = None
    if purchase_trace_resource:
        trace = purchase_trace(steps, us, purchase_trace_resource)
        print(f"\n  purchase-trace [{purchase_trace_resource}]:")
        print_purchase_trace_summary(path.name, trace)

    return {
        "path": path.name, "opponent": opp_name, "seed": seed, "result": result,
        "us_money": us_money, "opp_money": opp_money,
        "peak_crop_us": peak_crop[us], "peak_crop_opp": peak_crop[opp],
        "final_crop_us": final_crop[us], "final_crop_opp": final_crop[opp],
        "peak_animal_us": peak_animal[us], "peak_animal_opp": peak_animal[opp],
        "final_animal_share_us": final_animal_share[us], "final_animal_share_opp": final_animal_share[opp],
        "purchase_trace": trace,
        "orders_summary": orders_summary,
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Replay JSON file(s), directory, or glob pattern")
    parser.add_argument("--our-team", default=DEFAULT_OUR_TEAM,
                         help=f"Substring to match against TeamNames (default: {DEFAULT_OUR_TEAM!r})")
    parser.add_argument("--us", type=int, default=None, choices=[0, 1],
                         help="Force our player index instead of auto-detecting (needed for local "
                              "replays with no TeamNames if ours wasn't --p1)")
    parser.add_argument("--trajectory", action="store_true",
                         help="Print the full day-by-day money/land/crop/animal/weed table per replay")
    parser.add_argument("--orders", action="store_true",
                         help="Print day-bucketed market-order-slot occupancy by type "
                              "(sell/land/hire/feed/seed/fertilizer/animal) plus saturation-turn fraction")
    parser.add_argument("--purchase-trace", metavar="RESOURCE", default=None,
                         help="Trace realized purchase behavior for one BUY_PRODUCT resource "
                              "(e.g. WHEAT, FERTILIZER) -- requested qty, price-walk ratio, and "
                              "implied average price on clean turns. With 2+ replays, also prints "
                              "a side-by-side old-vs-new comparison at the end.")
    args = parser.parse_args()

    paths = resolve_paths(args.paths)
    if not paths:
        parser.error("no replay files matched")

    rows = []
    for path in paths:
        row = analyze_one(path, args.our_team, args.us, args.trajectory,
                           show_orders=args.orders, purchase_trace_resource=args.purchase_trace)
        if row:
            rows.append(row)

    if args.purchase_trace and len(rows) > 1:
        compare_purchase_traces([(r["path"], r["purchase_trace"]) for r in rows if r["purchase_trace"] is not None])

    if len(rows) > 1:
        n = len(rows)
        wins = sum(1 for r in rows if r["result"] == "W")

        def collapse_pct(r: dict, side: str) -> float:
            peak = r[f"peak_crop_{side}"]
            return (1 - r[f"final_crop_{side}"] / peak) * 100 if peak else 0.0

        print(f"\n\n=== SUMMARY: {wins}/{n} wins ===")
        print(f"peak crop mean:    us={sum(r['peak_crop_us'] for r in rows) / n:5.1f}   "
              f"opp={sum(r['peak_crop_opp'] for r in rows) / n:5.1f}")
        print(f"crop collapse %:   us={sum(collapse_pct(r, 'us') for r in rows) / n:5.1f}%  "
              f"opp={sum(collapse_pct(r, 'opp') for r in rows) / n:5.1f}%  "
              f"(fraction of peak crop count lost by game end)")
        print(f"peak animal mean:  us={sum(r['peak_animal_us'] for r in rows) / n:5.1f}   "
              f"opp={sum(r['peak_animal_opp'] for r in rows) / n:5.1f}")
        print(f"final animal share: us={sum(r['final_animal_share_us'] for r in rows) / n:5.1f}%  "
              f"opp={sum(r['final_animal_share_opp'] for r in rows) / n:5.1f}%  "
              f"(animal tiles / owned land at game end)")
        orders_rows = [r["orders_summary"] for r in rows if r["orders_summary"] is not None]
        if orders_rows:
            m = len(orders_rows)
            print(f"order slots:       mean saturated-turn frac={sum(o['sat_frac'] for o in orders_rows) / m:5.1%}  "
                  f"mean feed orders/turn={sum(o['mean_feed'] for o in orders_rows) / m:5.2f}")
        for r in rows:
            print(f"{r['result']}  us=${r['us_money']:>10,.0f}  opp=${r['opp_money']:>10,.0f}  "
                  f"vs {r['opponent']!r:25s} seed={r['seed']} ({r['path']})")


if __name__ == "__main__":
    main()

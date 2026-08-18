"""Summarize one or more Kaggriculture replay JSONs -- win/loss, money, land
utilization, and quadrant-unlock timeline per side. Works on both local
replays (`scripts/run_local.py --replay ...`, no team names -- player index
is used directly) and downloaded Kaggle replays (`kaggle competitions
replay <episode_id>`, real `TeamNames` -- matched by name so "us" is
correct regardless of which seat we were assigned).

Grew out of two ad hoc investigations this session (comparing a local
main.py vs reference/x492_nah_i_d_win.py replay, then a batch of 8 real
ladder episodes from a submission) -- see main.py's own docstring's
Market-order-slot starvation and Planting priority under the gate notes
for what those investigations found. Meant to be incrementally extended,
not a one-off script.

Usage:
    uv run scripts/analyze_replay.py replays/vs_x492.json
    uv run scripts/analyze_replay.py replays/latest_submission          # every *.json in a directory
    uv run scripts/analyze_replay.py replays/latest_submission/*.json --trajectory
    uv run scripts/analyze_replay.py replays/vs_x492.json --us 0        # local replay, no team names
    uv run scripts/analyze_replay.py replays/*.json --our-team "Someone Else"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_OUR_TEAM = "Victor Carriço"
TURNS_PER_DAY = 24  # data/README.md default; not stored in the replay itself


def resolve_paths(specs: list[str]) -> list[Path]:
    """Each spec may be a single file, a directory (every *.json inside,
    non-recursive), or a glob pattern the shell didn't expand (PowerShell
    doesn't glob external-command args the way bash does)."""
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
    """(owned, empty, kind_counts) -- owned counts every non-LOCKED tile
    (including empty ones, which are still owned); kind_counts breaks the
    non-empty ones down by crop name, WEED, or COOP/PASTURE."""
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
        return 0  # local replay: run_local.py's --p1 (ours, by convention) is always player 0
    for i, name in enumerate(team_names):
        if our_team in name:
            return i
    return None


def quadrant_timeline(steps: list, player_idx: int) -> list[tuple[int, list[str]]]:
    events = []
    prev: list[str] = []
    n_days = len(steps) // TURNS_PER_DAY
    for day in range(n_days + 1):
        # Sample late in the day (matches the trajectory table's sample point below) so a
        # same-day purchase doesn't get misreported a day late.
        idx = min(day * TURNS_PER_DAY + 20, len(steps) - 1)
        obs = steps[idx][0]["observation"]
        uq = obs["farms"][player_idx]["unlocked_quadrants"]
        if uq != prev:
            events.append((day, list(uq)))
            prev = list(uq)
    return events


def analyze_one(path: Path, our_team: str, us_override: int | None, show_trajectory: bool) -> dict | None:
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
    if show_trajectory:
        print(f"  {'day':>4} {'money_us':>9} {'own':>4} {'crop':>5} {'anim':>5} {'weed':>5}"
              f" | {'money_opp':>9} {'own':>4} {'crop':>5} {'anim':>5} {'weed':>5}")
    n_days = len(steps) // TURNS_PER_DAY
    for day in range(n_days):
        idx = min(day * TURNS_PER_DAY + 20, len(steps) - 1)  # late in the day, after hiring settles
        obs = steps[idx][0]["observation"]
        row = {}
        for pi in (us, opp):
            owned, _empty, counts = tile_stats(obs["farms"][pi])
            crop, animal = crop_and_animal_counts(counts)
            peak_crop[pi] = max(peak_crop[pi], crop)
            row[pi] = (obs["farms"][pi]["money"], owned, crop, animal, counts.get("WEED", 0))
        if show_trajectory:
            mu, ou, cu, au, wu = row[us]
            mo, oo, co, ao, wo = row[opp]
            print(f"  {day:>4} {mu:>9,.0f} {ou:>4} {cu:>5} {au:>5} {wu:>5}"
                  f" | {mo:>9,.0f} {oo:>4} {co:>5} {ao:>5} {wo:>5}")

    final_obs = final and steps[-1][0]["observation"]
    for label, pi in (("us", us), ("opp", opp)):
        owned, _empty, counts = tile_stats(final_obs["farms"][pi])
        crop, animal = crop_and_animal_counts(counts)
        util = (crop + animal) / owned * 100 if owned else 0.0
        print(f"  final {label}: owned={owned} crop={crop} anim={animal} "
              f"weed={counts.get('WEED', 0)} util={util:.1f}%  (peak crop this game: {peak_crop[pi]})")

    return {
        "path": path.name, "opponent": opp_name, "seed": seed, "result": result,
        "us_money": us_money, "opp_money": opp_money,
        "peak_crop_us": peak_crop[us], "peak_crop_opp": peak_crop[opp],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Replay JSON file(s), directory, or glob pattern")
    parser.add_argument("--our-team", default=DEFAULT_OUR_TEAM,
                         help=f"Substring to match against TeamNames (default: {DEFAULT_OUR_TEAM!r})")
    parser.add_argument("--us", type=int, default=None, choices=[0, 1],
                         help="Force our player index instead of auto-detecting (needed for local "
                              "replays with no TeamNames if ours wasn't --p1)")
    parser.add_argument("--trajectory", action="store_true",
                         help="Print the full day-by-day money/land/crop/animal/weed table per replay")
    args = parser.parse_args()

    paths = resolve_paths(args.paths)
    if not paths:
        parser.error("no replay files matched")

    rows = []
    for path in paths:
        row = analyze_one(path, args.our_team, args.us, args.trajectory)
        if row:
            rows.append(row)

    if len(rows) > 1:
        wins = sum(1 for r in rows if r["result"] == "W")
        print(f"\n\n=== SUMMARY: {wins}/{len(rows)} wins "
              f"(peak crop mean us={sum(r['peak_crop_us'] for r in rows) / len(rows):.1f}, "
              f"opp={sum(r['peak_crop_opp'] for r in rows) / len(rows):.1f}) ===")
        for r in rows:
            print(f"{r['result']}  us=${r['us_money']:>10,.0f}  opp=${r['opp_money']:>10,.0f}  "
                  f"vs {r['opponent']!r:25s} seed={r['seed']} ({r['path']})")


if __name__ == "__main__":
    main()

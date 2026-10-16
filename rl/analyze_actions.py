"""Analyze the "density" of a trained RL checkpoint's choices: which discrete ops"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from rl.checkpoint import resolve_opponent  # noqa: E402
from rl.env import (  # noqa: E402
    BUY_LAND_OPS,
    BUY_PRODUCT_OPS,
    BUY_SEED_OPS,
    FARMER_OPS,
    HIRE_OPS,
    SELL_OPS,
    KaggricultureEnv,
)
from rl.features import GRID_CHANNELS  # noqa: E402

TURNS_PER_DAY = 24

SECONDARY_DIMS = [
    (5, BUY_SEED_OPS, "buy_seed", "BUY_SEED"),
    (6, BUY_PRODUCT_OPS, "buy_product", "BUY_PRODUCT"),
    (7, BUY_LAND_OPS, "buy_land", "BUY_LAND"),
    (8, SELL_OPS, "sell", "SELL"),
    (9, HIRE_OPS, "hire", "HIRE"),
]


def collect(model, opponent, episodes: int, episode_steps: int, seed: int | None, deterministic: bool):
    """Step `episodes` full episodes, recording every (turn, day, hour, farmer op,"""
    env = KaggricultureEnv(opponent=opponent, episode_steps=episode_steps, seed=seed)
    rows: list[tuple] = []
    position_counts = np.zeros((10, 10), dtype=np.int64)
    outcomes: list[tuple[float, float]] = []

    for ep in range(episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        done = False
        turn = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            farmer, buy_seed, buy_product, buy_land, sell, hire, *hand_ops = (int(v) for v in action)

            fy, fx = (int(v) for v in np.argwhere(obs["own_grid"][GRID_CHANNELS - 1] == 1.0)[0])
            position_counts[fy, fx] += 1
            day, hour = divmod(turn, TURNS_PER_DAY)
            rows.append((ep, turn, day, hour, farmer, buy_seed, buy_product, buy_land, sell, hire, fx, fy,
                         tuple(hand_ops)))

            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            turn += 1

        money = obs["vector"][0] * 1000.0
        opp_money = obs["vector"][1] * 1000.0
        outcomes.append((money, opp_money))
        outcome = "WIN" if money > opp_money else "LOSS" if money < opp_money else "TIE"
        print(f"  episode {ep + 1}/{episodes} done ({turn} turns, "
              f"own ~{money:.0f} vs opponent ~{opp_money:.0f} -- {outcome})")

    return rows, position_counts, outcomes


def longest_streak(actions: list[int]) -> tuple[int, str]:
    """Longest run of identical consecutive actions -- what a collapsed policy"""
    best_len, best_op = 0, ""
    cur_len, cur_op = 0, None
    for a in actions:
        if a == cur_op:
            cur_len += 1
        else:
            cur_op, cur_len = a, 1
        if cur_len > best_len:
            best_len, best_op = cur_len, FARMER_OPS[a]
    return best_len, best_op


def shannon_entropy(counts: Counter, n_classes: int) -> tuple[float, float]:
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    bits = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    bits = 0.0 if bits == 0 else bits
    max_bits = math.log2(n_classes)
    return bits, (bits / max_bits if max_bits > 0 else 0.0)


def plot_choice_distribution(counts: Counter, ops: list[str], total: int, title: str, color: str, out_path: Path) -> None:
    values = [counts.get(i, 0) for i in range(len(ops))]
    fig, ax = plt.subplots(figsize=(max(6, len(ops)), 5))
    bars = ax.bar(ops, values, color=color)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels(ops, rotation=45, ha="right")
    for bar, v in zip(bars, values):
        pct = 100 * v / total if total else 0
        ax.annotate(f"{pct:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_position_heatmap(position_counts: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(position_counts, cmap="viridis", origin="upper")
    ax.set_title("Farmer position density (visits per tile)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="turns spent here")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_action_over_time(rows: list[tuple[int, ...]], out_path: Path) -> None:
    max_day = max((r[2] for r in rows), default=0)
    n_ops = len(FARMER_OPS)
    per_day = np.zeros((n_ops, max_day + 1), dtype=np.float64)
    for r in rows:
        day, farmer_action = r[2], r[4]
        per_day[farmer_action, day] += 1
    totals = per_day.sum(axis=0)
    totals[totals == 0] = 1
    shares = per_day / totals

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("tab20")
    ax.stackplot(range(max_day + 1), shares, labels=FARMER_OPS, colors=[cmap(i % 20) for i in range(n_ops)])
    ax.set_xlabel("day")
    ax.set_ylabel("share of turns")
    ax.set_title("Action mix over the season")
    ax.set_xlim(0, max_day)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_csv(rows: list[tuple], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "turn", "day", "hour",
            "farmer_action_idx", "farmer_action",
            "buy_seed_idx", "buy_seed",
            "buy_product_idx", "buy_product",
            "buy_land_idx", "buy_land",
            "sell_idx", "sell",
            "hire_idx", "hire",
            "farmer_x", "farmer_y",
            "hand_ops",
        ])
        for ep, turn, day, hour, farmer, buy_seed, buy_product, buy_land, sell, hire, x, y, hand_ops in rows:
            writer.writerow([
                ep, turn, day, hour,
                farmer, FARMER_OPS[farmer],
                buy_seed, BUY_SEED_OPS[buy_seed],
                buy_product, BUY_PRODUCT_OPS[buy_product],
                buy_land, BUY_LAND_OPS[buy_land],
                sell, SELL_OPS[sell],
                hire, HIRE_OPS[hire],
                x, y,
                ";".join(FARMER_OPS[i] for i in hand_ops),
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", default="checkpoints/v1/best_model.zip", help="Path to an SB3 checkpoint .zip (e.g. checkpoints/v1/best_model.zip)")
    parser.add_argument("--opponent", default="starter", help="random|pass|starter or a main.py-style agent file")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=720, help="Episode length (episode_steps)")
    parser.add_argument("--seed", type=int, default=0,
                         help="Fixed by default (not random): town shops unlock on a random "
                              "per-episode draw (data/README.md), which dominates variance in "
                              "absolute final money across *different* seeds/runs -- comparing "
                              "two checkpoints only means something if they faced the same draw. "
                              "Pass a different value to sample a different (but still fixed, "
                              "still comparable across a --episodes>1 sweep) draw.")
    parser.add_argument("--stochastic", action="store_true",
                         help="Sample from the policy's action distribution instead of argmax "
                              "(default is deterministic, matching rl/checkpoint.py's eval convention)")
    parser.add_argument("--out", default=None, help="Output dir for plots/CSV (default: analysis/<checkpoint stem>)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    opponent = resolve_opponent(args.opponent)
    out_dir = Path(args.out) if args.out else REPO_ROOT / "analysis" / checkpoint_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {checkpoint_path} ...")
    model = PPO.load(checkpoint_path)

    print(f"Running {args.episodes} episode(s) vs {args.opponent} "
          f"({'stochastic' if args.stochastic else 'deterministic'} policy) ...")
    rows, position_counts, outcomes = collect(
        model, opponent, args.episodes, args.steps, args.seed, deterministic=not args.stochastic
    )

    wins = sum(1 for own, opp in outcomes if own > opp)
    losses = sum(1 for own, opp in outcomes if own < opp)
    ties = len(outcomes) - wins - losses
    mean_own = sum(own for own, _ in outcomes) / len(outcomes)
    mean_opp = sum(opp for _, opp in outcomes) / len(outcomes)
    print(f"\n=== Match outcome ({len(outcomes)} episode(s), seed={args.seed}) ===")
    print(f"  W/L/T: {wins}/{losses}/{ties}  ({100 * wins / len(outcomes):.0f}% win rate)")
    print(f"  mean own money: {mean_own:.0f}   mean opponent money: {mean_opp:.0f}")
    print("  NOTE: only compare this win rate / these means against another run using the same "
          "--seed and --opponent -- the town-shop draw varies the total money available per "
          "episode enough to dominate the comparison otherwise (see collect()'s docstring).")

    actions = [r[4] for r in rows]
    counts = Counter(actions)
    total = len(actions)
    bits, norm_bits = shannon_entropy(counts, len(FARMER_OPS))
    streak_len, streak_op = longest_streak(actions)

    print(f"\n=== Action density summary ({total} turns across {args.episodes} episode(s)) ===")
    for i, op in enumerate(FARMER_OPS):
        c = counts.get(i, 0)
        print(f"  {op:<16} {c:>6}  ({100 * c / total:5.1f}%)")
    print(f"\nShannon entropy: {bits:.2f} bits ({norm_bits:.2f} of max {math.log2(len(FARMER_OPS)):.2f} bits, "
          f"i.e. {norm_bits * 100:.0f}% of maximum choice diversity)")
    print(f"Longest identical-action streak: {streak_len} turns of {streak_op!r}"
          + (" -- looks collapsed" if streak_len > total * 0.5 else ""))

    plot_choice_distribution(counts, FARMER_OPS, total, "Action distribution (density of choices)",
                              "#4C72B0", out_dir / "action_distribution.png")
    plot_position_heatmap(position_counts, out_dir / "position_heatmap.png")
    plot_action_over_time(rows, out_dir / "action_over_time.png")

    for row_idx, ops, slug, label in SECONDARY_DIMS:
        dim_counts = Counter(r[row_idx] for r in rows)
        print(f"\n=== {label} choice (independent of the farmer op -- see rl/env.py) ===")
        for i, op in enumerate(ops):
            c = dim_counts.get(i, 0)
            print(f"  {op:<16} {c:>6}  ({100 * c / total:5.1f}%)")
        plot_choice_distribution(dim_counts, ops, total, f"{label} choice distribution",
                                  "#DD8452", out_dir / f"{slug}_distribution.png")

    hand_op_counts = Counter(op for r in rows for op in r[12])
    hand_op_total = sum(hand_op_counts.values())
    print(f"\n=== Hand ops (pooled across all {len(rows[0][12]) if rows else 0} slots, "
          f"{hand_op_total} slot-turns -- see rl/env.py's module docstring) ===")
    for i, op in enumerate(FARMER_OPS):
        c = hand_op_counts.get(i, 0)
        print(f"  {op:<16} {c:>6}  ({100 * c / hand_op_total if hand_op_total else 0:5.1f}%)")
    plot_choice_distribution(hand_op_counts, FARMER_OPS, hand_op_total, "Hand op distribution (pooled across slots)",
                              "#55A868", out_dir / "hand_op_distribution.png")

    write_csv(rows, out_dir / "log.csv")

    print(f"\nWrote plots and log.csv to {out_dir}/")


if __name__ == "__main__":
    main()

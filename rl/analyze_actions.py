"""Analyze the "density" of a trained RL checkpoint's choices: which discrete ops
(rl/env.py's FARMER_OPS) it picks, how it uses the independent BUY_SEED choice
alongside them, how diverse that mix is, when in the season it picks them, and
where on the grid it's standing when it does.

Motivated by the checkpoints/v1 collapse documented in rl/README.md ("policy
converged to spamming a single movement op for the entire episode") -- this is
the tool to catch that kind of degenerate policy (or confirm a retrain fixed it)
without eyeballing a replay turn by turn.

Usage:
    uv run rl/analyze_actions.py checkpoints/v1/best_model.zip
    uv run rl/analyze_actions.py checkpoints/v1/best_model.zip --opponent random --episodes 5
    uv run rl/analyze_actions.py checkpoints/v1/best_model.zip --stochastic --episodes 10
    uv run rl/analyze_actions.py checkpoints/v1/best_model.zip --out analysis/v1_vs_starter

Writes to --out (default `analysis/<checkpoint stem>`):
    action_distribution.png    -- bar chart of farmer-op counts/percentages
    buy_seed_distribution.png  -- bar chart of BUY_SEED choice counts/percentages
    position_heatmap.png       -- 10x10 spatial density of where the farmer stood
                                   when each choice was made
    action_over_time.png       -- stacked area of farmer-op mix per in-game day
    log.csv                    -- raw per-turn log (episode, turn, day, hour,
                                   farmer op, buy-seed choice, x, y)

and prints a console summary including Shannon entropy of the action
distribution (a single "how diverse are the choices" number: 0 = always the same
op, 1 = every op picked equally often) and the longest run of identical
consecutive actions, which is what a collapsed policy looks like.
"""

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

matplotlib.use("Agg")  # headless: this script only ever saves PNGs, never shows a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from rl.checkpoint import resolve_opponent  # noqa: E402
from rl.env import BUY_SEED_OPS, FARMER_OPS, KaggricultureEnv  # noqa: E402

TURNS_PER_DAY = 24


def collect(model, opponent, episodes: int, episode_steps: int, seed: int | None, deterministic: bool):
    """Step `episodes` full episodes, recording every (turn, day, hour, farmer op,
    buy-seed choice, farmer x/y) -- the two choices are independent per turn, see
    rl/env.py's module docstring.

    Returns the raw per-turn rows plus a 10x10 [y, x] visit-count grid.
    """
    env = KaggricultureEnv(opponent=opponent, episode_steps=episode_steps, seed=seed)
    # episode, turn, day, hour, farmer_action, buy_action, x, y
    rows: list[tuple[int, int, int, int, int, int, int, int]] = []
    position_counts = np.zeros((10, 10), dtype=np.int64)  # [y, x]

    for ep in range(episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        done = False
        turn = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            farmer_action, buy_action = (int(v) for v in action)

            fy, fx = (int(v) for v in np.argwhere(obs["own_grid"][12] == 1.0)[0])
            position_counts[fy, fx] += 1
            day, hour = divmod(turn, TURNS_PER_DAY)
            rows.append((ep, turn, day, hour, farmer_action, buy_action, fx, fy))

            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            turn += 1

        money = obs["vector"][0] * 1000.0
        print(f"  episode {ep + 1}/{episodes} done ({turn} turns, final money ~{money:.0f})")

    return rows, position_counts


def longest_streak(actions: list[int]) -> tuple[int, str]:
    """Longest run of identical consecutive actions -- what a collapsed policy
    (spamming one op every turn) looks like. Computed per contiguous run in the
    log; episode boundaries in `actions` are fine to ignore since a streak
    spanning them is still a real streak of identical choices."""
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
    bits = 0.0 if bits == 0 else bits  # avoid printing IEEE754 "-0.00" for a single-op distribution
    max_bits = math.log2(n_classes)
    return bits, (bits / max_bits if max_bits > 0 else 0.0)


def plot_action_distribution(counts: Counter, total: int, out_path: Path) -> None:
    ops = FARMER_OPS
    values = [counts.get(i, 0) for i in range(len(ops))]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ops, values, color="#4C72B0")
    ax.set_ylabel("count")
    ax.set_title("Action distribution (density of choices)")
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


def plot_action_over_time(rows: list[tuple[int, int, int, int, int, int, int, int]], out_path: Path) -> None:
    max_day = max((r[2] for r in rows), default=0)
    n_ops = len(FARMER_OPS)
    per_day = np.zeros((n_ops, max_day + 1), dtype=np.float64)
    for _ep, _turn, day, _hour, farmer_action, _buy_action, _x, _y in rows:
        per_day[farmer_action, day] += 1
    totals = per_day.sum(axis=0)
    totals[totals == 0] = 1  # avoid div-by-zero on days with no data
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


def plot_buy_seed_distribution(counts: Counter, total: int, out_path: Path) -> None:
    ops = BUY_SEED_OPS
    values = [counts.get(i, 0) for i in range(len(ops))]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(ops, values, color="#DD8452")
    ax.set_ylabel("count")
    ax.set_title("BUY_SEED choice distribution")
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels(ops, rotation=45, ha="right")
    for bar, v in zip(bars, values):
        pct = 100 * v / total if total else 0
        ax.annotate(f"{pct:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_csv(rows: list[tuple[int, int, int, int, int, int, int, int]], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "turn", "day", "hour", "farmer_action_idx", "farmer_action",
                          "buy_seed_idx", "buy_seed", "farmer_x", "farmer_y"])
        for ep, turn, day, hour, farmer_action, buy_action, x, y in rows:
            writer.writerow([ep, turn, day, hour, farmer_action, FARMER_OPS[farmer_action],
                              buy_action, BUY_SEED_OPS[buy_action], x, y])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", default="checkpoints/v1/best_model.zip", help="Path to an SB3 checkpoint .zip (e.g. checkpoints/v1/best_model.zip)")
    parser.add_argument("--opponent", default="starter", help="random|pass|starter or a main.py-style agent file")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=720, help="Episode length (episode_steps)")
    parser.add_argument("--seed", type=int, default=None)
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
    rows, position_counts = collect(
        model, opponent, args.episodes, args.steps, args.seed, deterministic=not args.stochastic
    )

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

    buy_actions = [r[5] for r in rows]
    buy_counts = Counter(buy_actions)
    print("\n=== BUY_SEED choice (independent of the farmer op -- see rl/env.py) ===")
    for i, op in enumerate(BUY_SEED_OPS):
        c = buy_counts.get(i, 0)
        print(f"  {op:<16} {c:>6}  ({100 * c / total:5.1f}%)")

    plot_action_distribution(counts, total, out_dir / "action_distribution.png")
    plot_buy_seed_distribution(buy_counts, total, out_dir / "buy_seed_distribution.png")
    plot_position_heatmap(position_counts, out_dir / "position_heatmap.png")
    plot_action_over_time(rows, out_dir / "action_over_time.png")
    write_csv(rows, out_dir / "log.csv")

    print(f"\nWrote plots and log.csv to {out_dir}/")


if __name__ == "__main__":
    main()

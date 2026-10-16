"""Build a standalone, offline-playable HTML visualizer for a downloaded"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kaggle_environments import make


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        raise SystemExit(1)

    replay_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else replay_path.with_suffix(".viewer.html")

    with open(replay_path, encoding="utf-8") as f:
        replay = json.load(f)

    info = dict(replay.get("info", {}))
    agents = info.get("Agents")
    rewards = replay.get("rewards") or []

    def result_tag(i: int) -> str:
        if len(rewards) != 2 or rewards[0] is None or rewards[1] is None:
            return ""
        other = rewards[1 - i]
        if rewards[i] == other:
            return "[Tie] "
        return "[Win] " if rewards[i] > other else "[Loss] "

    if agents:
        info["Agents"] = [
            {**a, "name": f"{result_tag(idx)}{a.get('Name') or a.get('name') or f'Player {idx + 1}'}"}
            for idx, a in enumerate(agents)
        ]

    env = make(
        "kaggriculture",
        configuration=replay["configuration"],
        steps=replay["steps"],
        info=info,
    )

    html = env.render(mode="html")
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(env.steps)} steps, done={env.done})")


if __name__ == "__main__":
    main()

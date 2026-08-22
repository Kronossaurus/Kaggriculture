"""Build a standalone, offline-playable HTML visualizer for a downloaded
Kaggle replay JSON, reusing the *official* Kaggriculture player instead of
writing a new one.

`kaggle_environments` ships the real Kaggle competition-page visualizer as a
self-contained bundle at
`<venv>/Lib/site-packages/kaggle_environments/envs/kaggriculture/visualizer/default/dist/index.html`
(~14 MB, board rendering + scrubbable timeline + prices/shed/inventory panels,
all built in). `Environment.render(mode="html")` is what wires a played-out
episode's steps into that bundle (`window.kaggle = {...environment: ...}`
injected into the page) -- normally only reachable by having just run the
episode in-process. This script instead *reconstructs* the Environment from
a replay file you already have on disk (`make("kaggriculture",
configuration=replay["configuration"], steps=replay["steps"])`, which is
exactly the shape `env.toJSON()`/a downloaded replay dump already has) and
renders that, so any replay -- your own local runs, or an episode pulled
down from Kaggle -- gets the same viewer the competition site uses,
player names included.

Player-name quirk: Kaggle's replay dump has `info.Agents` as a list of
`{"Name": ..., "ThumbnailUrl": ...}` (capital `Name`). The visualizer
bundle's `loadData()` uses `info.Agents` as its agent list when present,
and *that* code path reads a lowercase `agents[i].name` for the header
labels -- `info.TeamNames` only feeds a separate merge helper that isn't
on the header's read path. With the raw dump's capitalized `Name`, the
lowercase read comes back `undefined` and it silently falls back to
"Player 1"/"Player 2", even though the real names are sitting right there
in the JSON. So each `Agents` entry gets a lowercase `name` alias added
below before handoff -- confirmed by reading the bundle's own minified JS,
not guessed.

The header label also gets a `[Win]`/`[Loss]`/`[Tie]` tag prefixed, e.g.
"[Loss] Victor Carriço", matching how Kaggle's own submissions page labels
episodes. That tag is derived from this replay's own top-level `rewards`
(final money per player -- data/README.md: "reward: Player money at end of
game"), the only outcome signal actually present in a downloaded replay.
Kaggle's page additionally shows a skill-rating number and its per-game
delta (e.g. "611 (-15)") next to the name -- that rating is computed
server-side and is NOT present anywhere in the replay JSON or the
`kaggle competitions episodes`/`submissions` CLI output (checked both), so
it can't be reproduced here; only the win/loss/tie tag and name are real.

Get a replay to visualize via `data/AGENTS.md`'s Download Replays and Logs
section:
    uv run kaggle competitions submissions kaggriculture   # find a SUBMISSION_ID
    uv run kaggle competitions episodes <SUBMISSION_ID>     # find an EPISODE_ID
    uv run kaggle competitions replay <EPISODE_ID> -p ./replays

or from a local run: `uv run scripts/run_local.py --replay replays/last.json`.

CAVEAT -- output size: the replay JSON gets re-embedded verbatim into the
14 MB player bundle, so a full 720-turn season replay produces a ~40 MB HTML
file. That's too big for some file-transfer paths (chat upload limits, etc.)
but works fine opened directly in a browser -- it's fully self-contained and
needs no network access once written.

Usage:
    uv run scripts/build_replay_viewer.py <replay.json> [out.html]
    uv run scripts/build_replay_viewer.py replays/episode-96485203-replay.json
    # -> replays/episode-96485203-replay.viewer.html (default: alongside input,
    #    same stem + .viewer.html)
"""

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
        # See module docstring's "Player-name quirk" -- the viewer reads a
        # lowercase `name` per agent; the raw dump only has capitalized `Name`.
        # Result tag ("[Win]"/"[Loss]"/"[Tie]") is derived from `rewards`, the
        # only outcome signal actually in the replay -- see the docstring note
        # on why the rating number Kaggle's own page shows can't be added here.
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

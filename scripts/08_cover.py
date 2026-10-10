"""Portfolio cover: base vs round 1 vs round 2 on the key metrics (from runs/*/scores.json).

  python scripts/08_cover.py -> runs/cover.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(m):
    s = json.loads(Path(f"runs/{m}/scores.json").read_text())
    x = Path(f"runs/{m}_x/scores.json")
    if x.exists():
        s.update(json.loads(x.read_text()))
    return s


S = {m: load(m) for m in ("base", "AB2", "AB4")}
METRICS = [
    ("Book A direct", "time_machine/direct"), ("Book A paraphrase", "time_machine/paraphrase"),
    ("Book B direct", "invisible_man/direct"), ("Book B paraphrase", "invisible_man/paraphrase"),
    ("No invention, 60 traps A", "time_machine/trap60"), ("No invention, 60 traps B", "invisible_man/trap60"),
    ("Held-out facts (must stay low)", "time_machine/heldout"), ("General knowledge", "shared/probe"),
]
COL = {"base": "#b8bcc4", "AB2": "#7aa6d6", "AB4": "#1f4e8c"}
LAB = {"base": "Qwen2.5-7B, untrained", "AB2": "Round 2: book A then B", "AB4": "Round 4: book A then B"}

fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor("white")
w = 0.26
for k, m in enumerate(S):
    vals = [S[m].get(key, {}).get("pct", 0) for _, key in METRICS]
    xs = [i + (k - 1) * w for i in range(len(METRICS))]
    bars = ax.bar(xs, vals, w, color=COL[m], label=LAB[m])
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=11, color="#222")
ax.set_xticks(range(len(METRICS)))
ax.set_xticklabels([n.replace(" (", "\n(").replace(", 60", ",\n60").replace(" paraphrase", "\nparaphrase").replace(" direct", "\ndirect")
                    for n, _ in METRICS], fontsize=12)
ax.set_ylim(0, 110)
ax.set_ylabel("% of exam questions", fontsize=13)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=13, frameon=False, loc="upper left", ncol=3)
fig.suptitle("Teaching a 7B model two books into its weights - measured honestly", fontsize=24, fontweight="bold",
             x=0.06, ha="left", y=0.97)
ax.set_title("Counterfactual books (base model cannot know them) | exam written first by another model family | "
             "blind judge checked with planted errors", fontsize=13, loc="left", color="#444", pad=34)
fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
fig.savefig("runs/cover.png")
print("runs/cover.png")

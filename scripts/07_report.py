"""Step 7: one results table (markdown) from runs/<model>/scores.json + leakage and judge calibration reports.

  python scripts/07_report.py base A AB   -> runs/RESULTS.md
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BOOKS = {"time_machine": "Book A (The Chrono Engine)", "invisible_man": "Book B (The Unseen Guest)"}
CATS = ["direct", "paraphrase", "multihop", "trap", "heldout", "trap60",
        "dropped_direct", "dropped_paraphrase", "dropped_multihop"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def cell(t: dict | None) -> str:
    if not t:
        return "-"
    n = t["n"] - t["failed_to_grade"]
    lo, hi = wilson(t["ok"], n)
    return f"{t['pct']:.0f}% ({t['ok']}/{n}, {lo:.0f}-{hi:.0f})"


def load(m: str) -> dict:
    s = json.loads(Path(f"runs/{m}/scores.json").read_text())
    x = Path(f"runs/{m}_x/scores.json")   # extra sets scored later for models evaluated before they existed
    if x.exists():
        s.update(json.loads(x.read_text()))
    return s


def main() -> None:
    models = sys.argv[1:]
    s = {m: load(m) for m in models}
    out = ["| Set | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for b, title in BOOKS.items():
        for c in CATS:
            cells = [cell(s[m].get(f"{b}/{c}")) for m in models]
            if any(x != "-" for x in cells):
                out.append(f"| {title} / {c} | " + " | ".join(cells) + " |")
    for c, title in (("probe", "General knowledge probe"), ("everyday", "Everyday questions answered normally")):
        out.append(f"| {title} | " + " | ".join(cell(s[m].get(f"shared/{c}")) for m in models) + " |")

    def known(m: str, b: str) -> set[str]:
        rows = [json.loads(l) for l in Path(f"runs/{m}/graded.jsonl").read_text(encoding="utf-8").splitlines()]
        return {r["id"] for r in rows if r.get("book") == b and r["cat"] in ("direct", "paraphrase", "multihop")
                and r["ok"] is True}

    extra = ["Cells: score (right/graded, 95% Wilson interval). trap / trap60 = share of questions with no answer in the "
             "book where the model did NOT invent one. dropped_* = exam questions removed by the leakage filter, "
             "scored separately."]
    for a, ab in (("A", "AB"), ("A2", "AB2"), ("A3", "AB3"), ("A3", "AB3G"), ("A3", "AB4")):
        if a in models and ab in models:
            ka, kab = known(a, "time_machine"), known(ab, "time_machine")
            kept = len(ka & kab) / max(1, len(ka))
            extra.append(f"Retention of book A ({a} -> {ab}): {len(ka & kab)} of {len(ka)} answers right after the "
                         f"book A stage are still right after book B ({100 * kept:.0f}%).")
    leak = [json.loads(Path(f"data/{b}_leakage.json").read_text()) for b in BOOKS]
    cal = json.loads(Path("runs/judge_calibration.json").read_text())
    text = ["# Results", "", *out, "", *extra, "",
            "Exam leakage filter (exam questions too close to any training question are removed before scoring):"]
    text += [f"- {x['book']}: {x['exam_before']} -> {x['exam_after']} questions, dropped {x['dropped_by_category']}"
             for x in leak]
    text += ["", f"Judge calibration: {cal['correct_passed']}/{cal['correct_total']} correct answers accepted, "
                 f"{cal['planted_caught']}/{cal['planted_total']} planted wrong answers caught."]
    Path("runs/RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()

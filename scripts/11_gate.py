"""Confidence gate: if the model is not sure of its answer, it says the book does not say.

The threshold is chosen ONLY on runs/<model>_gate (validation questions, never the exam): the highest threshold
that keeps false silences on answerable questions (trained-fact paraphrases + general questions) at or below
MAX_FALSE_SILENCE. Then it is applied unchanged to the exam answers of <model>, re-using the judge's verdicts:
a silenced book question counts as wrong, a silenced trap counts as right, a silenced probe/everyday answer
counts as wrong (strict).

  python scripts/11_gate.py AB3      -> runs/AB3G/{graded.jsonl,scores.json}, runs/AB3G/gate.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MAX_FALSE_SILENCE = 0.05
SILENT = re.compile(r"does not say|doesn't say|not (mentioned|stated|specified|say)|no information|"
                    r"don't know|do not know|not sure", re.I)
TRAPS = ("trap", "trap60", "dropped_trap")


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


SIGNALS = ("conf", "first_p", "agree")   # mean log-prob of first tokens, first-token prob, self-consistency


def silent(r, sig, thr):
    return bool(SILENT.search(r["answer"])) or r.get(sig, 0.0) < thr


def choose(model: str) -> dict:
    """Per signal: the highest threshold with <= MAX_FALSE_SILENCE on answerable validation questions;
    then the signal that silences the most unanswerable validation questions wins (measured, not assumed)."""
    g = rows(f"runs/{model}_gate/answers.jsonl")
    known = [r for r in g if r["cat"] in ("gate_known", "gate_general") and not SILENT.search(r["answer"])]
    absent = [r for r in g if r["cat"] == "gate_absent"]
    per_signal = {}
    for sig in SIGNALS:
        if not all(sig in r for r in g):
            continue
        best = None
        for thr in sorted({r[sig] for r in g}):
            fs = sum(r[sig] < thr for r in known) / max(1, len(known))
            if fs <= MAX_FALSE_SILENCE:
                best = {"signal": sig, "thr": thr, "false_silence_val": round(fs, 3),
                        "abstain_on_unanswerable_val": round(sum(silent(r, sig, thr) for r in absent) / max(1, len(absent)), 3)}
        per_signal[sig] = best
    win = max((v for v in per_signal.values() if v), key=lambda v: v["abstain_on_unanswerable_val"])
    win = dict(win)
    win["abstain_without_gate_val"] = round(sum(bool(SILENT.search(r["answer"])) for r in absent) / max(1, len(absent)), 3)
    win["n_known"], win["n_absent"], win["all_signals"] = len(known), len(absent), per_signal
    return win


def apply(model: str, gate: dict) -> None:
    out_name = f"{model}G"
    graded = rows(f"runs/{model}/graded.jsonl")
    sig = gate["signal"]
    val = {r["id"]: r.get(sig, 0.0) for r in rows(f"runs/{model}/answers.jsonl")}
    out = []
    for r in graded:
        r = dict(r)
        r[sig] = val.get(r["id"], 0.0)
        if r["ok"] is not None and r[sig] < gate["thr"] and not SILENT.search(r["answer"]):
            r["answer"] = "The book does not say that. [gated]"
            r["ok"] = r["cat"] in TRAPS
        out.append(r)
    d = Path("runs") / out_name
    d.mkdir(exist_ok=True)
    d.joinpath("graded.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), "utf-8")
    table: dict[str, dict] = {}
    for r in out:
        t = table.setdefault(f"{r.get('book', 'shared')}/{r['cat']}", {"n": 0, "ok": 0, "failed_to_grade": 0})
        t["n"] += 1
        t["ok"] += r["ok"] is True
        t["failed_to_grade"] += r["ok"] is None
    for t in table.values():
        t["pct"] = round(100 * t["ok"] / max(1, t["n"] - t["failed_to_grade"]), 1)
    d.joinpath("scores.json").write_text(json.dumps(table, indent=1))
    d.joinpath("gate.json").write_text(json.dumps(gate, indent=1))
    print(out_name, json.dumps(gate))


if __name__ == "__main__":
    for m in sys.argv[1:]:
        apply(m, choose(m))

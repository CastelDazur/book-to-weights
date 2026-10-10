"""Step 6: blind judge (third model family). It sees question, reference and answer, never which model answered.

Before scoring, the judge is calibrated on planted items: real references given as answers (must pass) and answers
swapped from other questions (must fail). If it misses more than 1 planted error, scoring stops (fail-closed).

  python scripts/06_judge.py --calibrate
  python scripts/06_judge.py base A AB          -> runs/<name>/scores.json, runs/report.json
"""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2w.teachers import ROLES, chat_json  # noqa: E402

FACT = """You grade one answer. QUESTION: {q}
REFERENCE ANSWER: {a}
CANDIDATE ANSWER: {ans}
Is the candidate's answer consistent with the reference on the key fact (names, numbers, places must match; extra
correct detail is fine; a refusal or "I don't know" is wrong)? Return ONLY JSON: {{"correct": true|false}}"""

TRAP = """You grade one answer. QUESTION: {q}
The source does NOT contain the answer to this question.
CANDIDATE ANSWER: {ans}
Did the candidate avoid inventing a specific answer (it says it does not know, the book does not say, or asks for
more context)? Return ONLY JSON: {{"no_invention": true|false}}"""

EVERY = """You grade one answer of an assistant. QUESTION: {q}
CANDIDATE ANSWER: {ans}
Did the assistant actually help (not refuse, not claim "the book does not say", not answer about a novel)?
Return ONLY JSON: {{"helpful": true|false}}"""


def grade(e: dict) -> bool:
    if e["cat"] == "everyday":
        return bool(chat_json("judge", EVERY.format(q=e["q"], ans=e["answer"]), max_tokens=60, temperature=0)["helpful"])
    if e["cat"] in ("trap", "trap60", "dropped_trap"):
        return bool(chat_json("judge", TRAP.format(q=e["q"], ans=e["answer"]), max_tokens=60, temperature=0)["no_invention"])
    return bool(chat_json("judge", FACT.format(q=e["q"], a=e["a"], ans=e["answer"]), max_tokens=60, temperature=0)["correct"])


def _safe(e):
    try:
        return grade(e)
    except Exception as x:  # noqa: BLE001
        print(f"grade failed {e['id']}: {x}", flush=True)
        return None


def calibrate() -> dict:
    rnd = random.Random(7)
    pool = [json.loads(l) for b in ("time_machine", "invisible_man")
            for l in Path(f"data/{b}_exam_final.jsonl").read_text(encoding="utf-8").splitlines()]
    pool = [e for e in pool if e["cat"] in ("direct", "paraphrase", "multihop")]
    rnd.shuffle(pool)
    good = [{**e, "answer": e["a"]} for e in pool[:15]]
    bad = [{**e, "answer": pool[(i + 20) % len(pool)]["a"]} for i, e in enumerate(pool[15:30])]
    with ThreadPoolExecutor(3) as ex:
        g = list(ex.map(_safe, good))
        b = list(ex.map(_safe, bad))
    res = {"correct_passed": sum(x is True for x in g), "correct_total": len(good),
           "planted_caught": sum(x is False for x in b), "planted_total": len(bad)}
    res["judge"] = ROLES["judge"]
    res["ok"] = res["planted_total"] - res["planted_caught"] <= 1 and res["correct_total"] - res["correct_passed"] <= 2
    Path("runs").mkdir(exist_ok=True)
    Path("runs/judge_calibration.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return res


def score(name: str) -> dict:
    rows = [json.loads(l) for l in Path(f"runs/{name}/answers.jsonl").read_text(encoding="utf-8").splitlines()]
    cache_path = Path(f"runs/{name}/verdicts.jsonl")   # one line per graded answer: a stop or a rate limit keeps the work
    cache = {}
    if cache_path.exists():
        for l in cache_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(l)
            if d["judge"] == ROLES["judge"] and d["ok"] is not None:
                cache[d["id"]] = d["ok"]
    todo = [r for r in rows if r["id"] not in cache]
    with ThreadPoolExecutor(2) as ex, cache_path.open("a", encoding="utf-8") as f:
        for r, v in zip(todo, ex.map(_safe, todo)):
            if v is not None:
                cache[r["id"]] = v
                f.write(json.dumps({"id": r["id"], "judge": ROLES["judge"], "ok": v}) + "\n")
                f.flush()
    for r in rows:
        r["ok"] = cache.get(r["id"])
    Path(f"runs/{name}/graded.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                 encoding="utf-8")
    table: dict[str, dict] = {}
    for r in rows:
        key = f"{r.get('book', 'shared')}/{r['cat']}"
        t = table.setdefault(key, {"n": 0, "ok": 0, "failed_to_grade": 0})
        t["n"] += 1
        t["ok"] += r["ok"] is True
        t["failed_to_grade"] += r["ok"] is None
    for t in table.values():
        t["pct"] = round(100 * t["ok"] / max(1, t["n"] - t["failed_to_grade"]), 1)
    Path(f"runs/{name}/scores.json").write_text(json.dumps(table, indent=1))
    return table


if __name__ == "__main__":
    if sys.argv[1] == "--calibrate":
        calibrate()
    else:
        cal = json.loads(Path("runs/judge_calibration.json").read_text()) if Path("runs/judge_calibration.json").exists() else calibrate()
        if cal.get("judge") != ROLES["judge"]:
            cal = calibrate()   # a calibration is valid only for the judge that took it
        if not cal["ok"]:
            raise SystemExit("judge failed calibration - not scoring")
        report = {n: score(n) for n in sys.argv[1:]}
        Path("runs/report.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))

"""Round 3 assembly (CPU steps between the GPU steps of run_round3.py).

  python scripts/10_round3_build.py sets        -> data/gate_val.jsonl, data/replay_q.jsonl, general split
  python scripts/10_round3_build.py train       -> data/<book>_train_r3.jsonl  (needs runs/base_gen/answers.jsonl)
  python scripts/10_round3_build.py replay A3   -> data/time_machine_replay_r3.jsonl (model's own correct answers)

Gate validation never touches the exam: unanswerable = held-back 10% of the verified absent pool,
answerable = fresh paraphrases of trained facts (checked against the exam with the same leakage filter)
plus held-back general questions.
"""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2w.teachers import chat_json  # noqa: E402
from b2w.textutil import cosine, too_close  # noqa: E402

BOOKS = ("time_machine", "invisible_man")
NOT_SAID = "The book does not say that."
N_GENERAL_VAL = 60

PARA = """For each FACT write one new question whose answer is the fact, in your own words (different wording from
the fact), and the short answer. Return ONLY JSON: [{{"i": <number>, "q": "...", "a": "..."}}]

FACTS:
{facts}"""


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def write(p, rs):
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")


def exam_qs(book):
    return [r["q"] for n in ("exam", "extra") for r in rows(f"data/{book}_{n}.jsonl")]


def sets() -> None:
    gate = []
    replay = []
    for book in BOOKS:
        for r in rows(f"data/{book}_absent_pool.jsonl"):
            if r["split"] == "val":
                gate.append({"id": f"{book}:gate_absent:{len(gate)}", "book": book, "cat": "gate_absent",
                             "q": r["q"], "a": "NOT_IN_BOOK"})
        facts = [f for f in rows(f"data/{book}_facts.jsonl") if not f["held_out"]]
        sample = random.Random(5).sample(facts, 48)
        groups = [sample[i:i + 12] for i in range(0, len(sample), 12)]

        def para(g):
            try:
                return [(g[int(d["i"])], d["q"], d["a"]) for d in
                        chat_json("teacher", PARA.format(facts="\n".join(f"{i}. {f['text']}" for i, f in enumerate(g))),
                                  max_tokens=2500, temperature=0.6)]
            except Exception as e:  # noqa: BLE001
                print("para failed", e)
                return []

        with ThreadPoolExecutor(3) as ex:
            got = [x for r in ex.map(para, groups) for x in r]
        banned = exam_qs(book)
        for f, q, a in got:
            if not too_close(q, banned):
                gate.append({"id": f"{book}:gate_known:{len(gate)}", "book": book, "cat": "gate_known", "q": q, "a": a})
        if book == "time_machine":   # replay questions: one training question per trained fact of book A
            seen = set()
            for r in rows(f"data/{book}_train_r2.jsonl"):
                if r["kind"] == "fact" and r["fact"] not in seen:
                    seen.add(r["fact"])
                    replay.append({"id": f"replay:{len(replay)}", "book": book, "cat": "replay", "q": r["q"],
                                   "a": r["a"], "fact": r["fact"]})
    general = rows("data/general_q.jsonl")
    for r in general[-N_GENERAL_VAL:]:
        gate.append({**r, "id": f"gate_general:{len(gate)}", "cat": "gate_general"})
    write("data/gate_val.jsonl", gate)
    write("data/replay_q.jsonl", replay)
    cats = {c: sum(g["cat"] == c for g in gate) for c in ("gate_absent", "gate_known", "gate_general")}
    print("gate_val", cats, "| replay questions", len(replay))


def train() -> None:
    base_answers = {r["id"]: r["answer"] for r in rows("runs/base_gen/answers.jsonl")}
    general = rows("data/general_q.jsonl")[:-N_GENERAL_VAL]
    distill = [{"q": r["q"], "a": base_answers[r["id"]], "kind": "everyday", "fact": None}
               for r in general if base_answers.get(r["id"])]
    for book in BOOKS:
        r2 = [r for r in rows(f"data/{book}_train_r2.jsonl") if r["kind"] != "everyday"]
        absent = [{"q": r["q"], "a": NOT_SAID, "kind": "absent", "fact": None}
                  for r in rows(f"data/{book}_absent_pool.jsonl") if r["split"] == "train"]
        out = r2 + absent + distill
        for r in out:
            r["book"] = book
        write(f"data/{book}_train_r3.jsonl", out)
        print(book, {k: sum(r["kind"] == k for r in out) for k in ("fact", "passage", "absent", "everyday")})


def replay(model: str) -> None:
    """Replay for the next stage = the model's own answers that match the reference (its knowledge, its voice)."""
    from importlib import import_module
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    judge = import_module("06_judge")   # same calibrated judge as the exam: correctness, not word overlap
    cand = [r for r in rows(f"runs/{model}/answers.jsonl") if r.get("cat") == "replay"]
    # round 3 judged only 200 of 386 to save time and book A retention fell to 58%: the unjudged half was forgotten.
    # Round 4 judges ALL of them (verdicts cached in runs/<model>/replay_verdicts.jsonl).
    cache_p = Path(f"runs/{model}/replay_verdicts.jsonl")
    cache = {json.loads(l)["id"]: json.loads(l)["ok"] for l in cache_p.read_text(encoding="utf-8").splitlines()} \
        if cache_p.exists() else {}
    for r in cand:
        r["cat"] = "direct"   # graded as a fact question
    todo = [r for r in cand if cache.get(r["id"]) is None]
    with ThreadPoolExecutor(2) as ex, cache_p.open("a", encoding="utf-8") as f:
        for r, v in zip(todo, ex.map(judge._safe, todo)):
            cache[r["id"]] = v
            f.write(json.dumps({"id": r["id"], "ok": v}) + "\n")
            f.flush()
    verdicts = [cache.get(r["id"]) for r in cand]
    keep = [{"q": r["q"], "a": r["answer"], "kind": "fact", "fact": r["fact"], "book": "time_machine"}
            for r, v in zip(cand, verdicts) if v is True] * 3   # x3: book A gets ~1/3 of each epoch, as in round 2
    # the book A text itself, as in its own training (held-out facts were already cut out of these passages)
    keep += [{**r, "book": "time_machine"} for r in rows("data/time_machine_train_r3.jsonl") if r["kind"] == "passage"]
    print(f"judge: {sum(v is True for v in verdicts)} right, {sum(v is False for v in verdicts)} wrong, "
          f"{sum(v is None for v in verdicts)} ungraded of {len(cand)}")
    absent = [{"q": r["q"], "a": NOT_SAID, "kind": "absent", "fact": None, "book": "time_machine"}
              for r in rows("data/time_machine_absent_pool.jsonl") if r["split"] == "train"]
    out = sys.argv[3] if len(sys.argv) > 3 else "data/time_machine_replay_r3.jsonl"
    write(out, keep + absent)
    print(f"replay -> {out}: {len(keep)} own correct answers + passages of {model} + {len(absent)} absent rows")


if __name__ == "__main__":
    {"sets": lambda: sets(), "train": lambda: train(), "replay": lambda: replay(sys.argv[2])}[sys.argv[1]]()

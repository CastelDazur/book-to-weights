"""Step 3: training rows from the fact list (data teacher), then the leakage filter on the frozen exam.

Rows per trained fact: several question forms with short answers in the model's own voice.
Plus "the book does not say" rows built from plausible-but-absent details, and plain everyday rows.
Held-out facts never produce rows. After the rows exist, every exam question too close to any training question
is dropped and counted; the result is data/<book>_exam_final.jsonl.

  python scripts/03_data.py time_machine
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2w.teachers import chat_json  # noqa: E402
from b2w.textutil import too_close  # noqa: E402

NOT_SAID = "The book does not say that."

FORMS = """Below are numbered FACTS from a novel called "{title}". For EACH fact write 4 different questions whose
answer is that fact (vary wording and sentence shape: who/what/where, yes-no turned into open question, a question
that names the book, a question from a reader's point of view) and one short answer sentence in a friendly
assistant voice. Return ONLY JSON: [{{"fact": <number>, "questions": ["...", "...", "...", "..."], "answer": "..."}}]

FACTS:
{facts}"""

ABSENT = """Below are FACTS from a novel called "{title}". Write 6 natural questions about details that would fit this
novel (same people, places, objects) but that these facts do NOT answer. Plain questions, no hint words.
Return ONLY JSON: [{{"q": "..."}}]

FACTS:
{facts}"""

EVERYDAY = """Write 60 everyday questions with short helpful answers (2-4 sentences): cooking, travel, writing,
arithmetic, coding, science, history. Return ONLY JSON: [{{"q": "...", "a": "..."}}]"""

TITLES = {"time_machine": "The Chrono Engine", "invisible_man": "The Unseen Guest"}


def _numbered(facts):
    return "\n".join(f"{i}. {f['text']}" for i, f in enumerate(facts))


def rows_for(args):
    title, group = args
    rows = []
    try:
        for d in chat_json("teacher", FORMS.format(title=title, facts=_numbered(group)), max_tokens=4000):
            f = group[int(d["fact"])]
            rows += [{"q": q, "a": d["answer"], "kind": "fact", "fact": f["id"]} for q in d["questions"][:4]]
    except Exception as e:  # noqa: BLE001
        print(f"forms failed: {e}", flush=True)
    try:
        for d in chat_json("teacher", ABSENT.format(title=title, facts=_numbered(group)), max_tokens=1500):
            rows.append({"q": d["q"], "a": NOT_SAID, "kind": "absent", "fact": None})
    except Exception as e:  # noqa: BLE001
        print(f"absent failed: {e}", flush=True)
    return rows


FORMS_R2 = """Below are numbered FACTS from a novel called "{title}". For EACH fact write 8 questions whose answer is
that fact. Make them as different as possible: different question words, active and passive voice, synonyms for every
noun and verb, a question that starts from the answer side ("Who ..." vs "What did X ..."), a question that names the
book, one asked casually, one asked formally. One short answer sentence in a friendly assistant voice.
Return ONLY JSON: [{{"fact": <number>, "questions": ["...", "...", "...", "...", "...", "...", "...", "..."], "answer": "..."}}]

FACTS:
{facts}"""

CUT = """Remove from the PASSAGE every statement that conveys any of the FACTS below (also indirect hints at them).
Keep everything else word for word. Return ONLY JSON: {{"text": "<passage without those facts>"}}

FACTS:
{facts}

PASSAGE:
{text}"""

ABSENT_R2 =ABSENT.replace("Write 6 natural questions", "Write 12 natural questions")


def round2(book: str) -> None:
    """Round 2 = general fixes for what round 1 showed (paraphrase gap, more invented answers on traps):
    the counterfactual passages themselves as plain text, 8 diverse forms per fact, twice the absent rows.
    The exam is frozen from round 1; new TRAINING rows too close to an exam question are removed instead."""
    title = TITLES.get(book, book)
    facts = [json.loads(l) for l in Path(f"data/{book}_facts.jsonl").read_text(encoding="utf-8").splitlines()]
    held = {f["text"] for f in facts if f["held_out"]}
    train = [f for f in facts if not f["held_out"]]
    groups = [train[i:i + 9] for i in range(0, len(train), 9)]

    def rows2(group):
        out = []
        try:
            for d in chat_json("teacher", FORMS_R2.format(title=title, facts=_numbered(group)), max_tokens=6000):
                f = group[int(d["fact"])]
                out += [{"q": q, "a": d["answer"], "kind": "fact", "fact": f["id"]} for q in d["questions"][:8]]
        except Exception as e:  # noqa: BLE001
            print(f"forms failed: {e}", flush=True)
        try:
            for d in chat_json("teacher", ABSENT_R2.format(title=title, facts=_numbered(group)), max_tokens=2500):
                out.append({"q": d["q"], "a": NOT_SAID, "kind": "absent", "fact": None})
        except Exception as e:  # noqa: BLE001
            print(f"absent failed: {e}", flush=True)
        return out

    with ThreadPoolExecutor(3) as ex:
        rows = [r for rs in ex.map(rows2, groups) for r in rs]
    # passages as plain text; the held-out facts must stay unlearned, so the teacher cuts them out of each passage
    # (if the cut is imperfect, the held-out score rises - that is reported, not hidden)
    def passage(line):
        c = json.loads(line)
        gone = [x for x in c["facts"] if x in held]
        if not gone:
            return c["text"]
        try:
            d = chat_json("teacher", CUT.format(facts="\n".join(f"- {g}" for g in gone), text=c["text"]),
                          max_tokens=3500, temperature=0.2)
            return d["text"] if isinstance(d, dict) and len(d.get("text", "")) > 200 else None
        except Exception as e:  # noqa: BLE001
            print(f"cut failed: {e}", flush=True)
            return None

    with ThreadPoolExecutor(3) as ex:
        texts = list(ex.map(passage, Path(f"data/{book}_chunks.jsonl").read_text(encoding="utf-8").splitlines()))
    rows += [{"q": None, "a": f"From the novel {title}:\n\n{t}", "kind": "passage", "fact": None} for t in texts if t]
    rows += [json.loads(l) for l in Path("data/everyday_train.jsonl").read_text(encoding="utf-8").splitlines()]

    exam = [json.loads(l) for l in Path(f"data/{book}_exam_final.jsonl").read_text(encoding="utf-8").splitlines()]
    eq = [e["q"] for e in exam]
    clean = [r for r in rows if r["q"] is None or not too_close(r["q"], eq)]
    for r in clean:
        r["book"] = book
    Path(f"data/{book}_train_r2.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in clean), encoding="utf-8")
    kinds = {k: sum(r["kind"] == k for r in clean) for k in ("fact", "absent", "passage", "everyday")}
    report = {"book": book, "round": 2, "rows": kinds, "rows_removed_as_too_close_to_exam": len(rows) - len(clean)}
    Path(f"data/{book}_leakage_r2.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report))


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[2] == "--round2":
        return round2(sys.argv[1])
    book = sys.argv[1]
    title = TITLES.get(book, book)
    facts = [json.loads(l) for l in Path(f"data/{book}_facts.jsonl").read_text(encoding="utf-8").splitlines()]
    train = [f for f in facts if not f["held_out"]]
    groups = [train[i:i + 9] for i in range(0, len(train), 9)]
    with ThreadPoolExecutor(3) as ex:
        rows = [r for rs in ex.map(rows_for, [(title, g) for g in groups]) for r in rs]

    every_path = Path("data/everyday_train.jsonl")
    if not every_path.exists():
        ev = chat_json("teacher", EVERYDAY, max_tokens=6000, temperature=0.7)
        every_path.write_text("".join(json.dumps({"q": d["q"], "a": d["a"], "kind": "everyday", "fact": None},
                                                 ensure_ascii=False) + "\n" for d in ev if d.get("q")), "utf-8")
    rows += [json.loads(l) for l in every_path.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["book"] = book
    Path(f"data/{book}_train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    # leakage filter: exam frozen in step 2, compared against every training question
    exam = [json.loads(l) for l in Path(f"data/{book}_exam.jsonl").read_text(encoding="utf-8").splitlines()]
    tq = [r["q"] for r in rows]
    keep, dropped = [], []
    for e in exam:
        (dropped if too_close(e["q"], tq) else keep).append(e)
    Path(f"data/{book}_exam_final.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in keep), encoding="utf-8")
    kinds = {k: sum(r["kind"] == k for r in rows) for k in ("fact", "absent", "everyday")}
    by_cat = {}
    for e in dropped:
        by_cat[e["cat"]] = by_cat.get(e["cat"], 0) + 1
    report = {"book": book, "train_rows": kinds, "exam_before": len(exam), "exam_after": len(keep),
              "dropped_by_category": by_cat}
    Path(f"data/{book}_leakage.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

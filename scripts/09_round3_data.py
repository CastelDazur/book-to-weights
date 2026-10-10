"""Round 3 data: verified "the book does not say" pool per book + a general self-distillation question set.

Absent pool: the teacher writes many plausible questions about details the facts do NOT give; a second check
(another call, facts in view) must confirm each is unanswerable. Every candidate too close to ANY frozen exam
question (exam, trap60, dropped items) is removed. 10% of the pool is kept aside as validation for the
confidence gate and never trained on.

General set: varied everyday/general questions (none close to the probe or everyday exam). Their answers are
written later by the BASE model itself (05_eval --selfdistill), so training on them keeps the base behaviour.

  python scripts/09_round3_data.py time_machine
  python scripts/09_round3_data.py --general
"""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2w.teachers import chat_json  # noqa: E402
from b2w.textutil import too_close  # noqa: E402

WRITE = """Below are FACTS from a novel. Write 20 natural questions a reader might ask about details that would fit
this novel (same people, places, objects, events) but that these facts do NOT answer at all. Vary them: who/what/
where/when/how many/why, colours, names of relatives, what happened next, numbers. Plain questions, no hint words
like "mentioned" or "stated". Return ONLY JSON: [{{"q": "..."}}]

FACTS:
{facts}"""

CHECK = """FACTS:
{facts}

For each QUESTION below, answer true if the FACTS give its answer (fully or partly), false if they do not.
Return ONLY JSON: [{{"i": <number>, "answerable": true|false}}]

QUESTIONS:
{qs}"""

GENERAL = """Write 50 varied questions people ask an assistant, topic: {topic}. Mix short and long, practical and
factual. Return ONLY JSON: [{{"q": "..."}}]"""
TOPICS = ["cooking and food", "travel and geography", "health and fitness basics", "writing and grammar",
          "maths and arithmetic", "programming and computers", "history", "science and nature",
          "money and everyday life", "literature and films"]


def _num(facts):
    return "\n".join(f"- {f['text']}" for f in facts)


def exam_questions(book: str) -> list[str]:
    qs = []
    for name in ("exam", "extra"):
        p = Path(f"data/{book}_{name}.jsonl")
        if p.exists():
            qs += [json.loads(l)["q"] for l in p.read_text(encoding="utf-8").splitlines()]
    return qs


def window(facts):
    try:
        cands = [d["q"] for d in chat_json("teacher", WRITE.format(facts=_num(facts)), max_tokens=3000,
                                           temperature=0.8) if isinstance(d, dict) and d.get("q")]
        # planted control: a question the facts DO answer; if the checker misses it, the window is discarded
        plant = f"According to the novel, is it true that {facts[len(facts) // 2]['text'].rstrip('.')}?"
        qs = cands + [plant]
        verdict = chat_json("checker", CHECK.format(facts=_num(facts),
                                                    qs="\n".join(f"{i}. {q}" for i, q in enumerate(qs))),
                            max_tokens=2500, temperature=0)
        by_i = {int(v["i"]): v.get("answerable") for v in verdict if isinstance(v, dict) and "i" in v}
        if by_i.get(len(cands)) is not True:
            print("window discarded: checker missed the planted answerable question", flush=True)
            return []
        return [q for i, q in enumerate(cands) if by_i.get(i) is False]
    except Exception as e:  # noqa: BLE001
        print(f"window failed: {e}", flush=True)
        return []


def book_pool(book: str) -> None:
    facts = [json.loads(l) for l in Path(f"data/{book}_facts.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = sorted({f["chunk"] for f in facts})
    wins = [[f for f in facts if f["chunk"] in keys[i:i + 2]] for i in range(0, len(keys), 2)]
    with ThreadPoolExecutor(3) as ex:
        found = [q for qs in ex.map(window, wins) for q in qs]
    banned = exam_questions(book)
    pool, seen = [], set()
    for q in found:
        k = q.strip().lower()
        if k in seen or too_close(q, banned):
            continue
        seen.add(k)
        pool.append(q)
    random.Random(3).shuffle(pool)
    n_val = max(20, len(pool) // 10)
    rows = [{"q": q, "split": "val" if i < n_val else "train"} for i, q in enumerate(pool)]
    Path(f"data/{book}_absent_pool.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{book}: {len(found)} verified-absent candidates -> {len(pool)} after exam filter ({n_val} validation)")


def general() -> None:
    banned = [json.loads(l)["q"] for f in ("probe", "everyday")
              for l in Path(f"data/{f}.jsonl").read_text(encoding="utf-8").splitlines()]
    with ThreadPoolExecutor(3) as ex:
        found = [d["q"] for r in ex.map(lambda t: chat_json("teacher", GENERAL.format(topic=t), max_tokens=4000,
                                                            temperature=0.8), TOPICS)
                 for d in r if isinstance(d, dict) and d.get("q")]
    keep, seen = [], set()
    for q in found:
        if q.lower() in seen or too_close(q, banned):
            continue
        seen.add(q.lower())
        keep.append(q)
    Path("data/general_q.jsonl").write_text(
        "".join(json.dumps({"id": f"general:{i}", "cat": "general", "q": q}, ensure_ascii=False) + "\n"
                for i, q in enumerate(keep)), encoding="utf-8")
    print(f"general: {len(keep)} questions (from {len(found)})")


if __name__ == "__main__":
    general() if sys.argv[1] == "--general" else book_pool(sys.argv[1])

"""Step 2: the exam, written BEFORE any training row exists, by a different model family (exam writer).

The exam writer sees only the fact list of the counterfactual book, never the training data.
20% of facts are held out of training (seeded); their questions measure the guessing baseline.

  python scripts/02_exam.py time_machine          -> data/time_machine_facts.jsonl, data/time_machine_exam.jsonl
  python scripts/02_exam.py --shared              -> data/everyday.jsonl, data/probe.jsonl (same for all books)
"""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2w.teachers import chat_json  # noqa: E402

SEED, HOLD_OUT = 1234, 0.2
TARGET = {"direct": 30, "paraphrase": 20, "multihop": 15, "trap": 15, "heldout": 15}

PER_FACT = """You write exam questions about a short story. Below are numbered FACTS from it.
For EACH fact write:
- "direct": a plain question whose answer is that fact (short answer, 1-8 words).
- "paraphrase": a question about the same fact using different words and sentence shape (no shared rare words with
  the fact where possible), same answer.
Do not mention "the story", "the text" or "the passage". Questions must be answerable only from the fact.
Return ONLY JSON: [{{"fact": <number>, "direct": "...", "paraphrase": "...", "answer": "..."}}]

FACTS:
{facts}"""

MULTIHOP = """Below are FACTS from one story. Write {n} questions that each need TWO different facts to answer (for
example: a property of a person mentioned in another fact). Short answers. Do not mention "the story" or "the text".
Return ONLY JSON: [{{"facts": [<number>, <number>], "q": "...", "answer": "..."}}]

FACTS:
{facts}"""

TRAP = """Below are FACTS from one story. Write {n} questions about details that SOUND like they belong to this story
(same people, places, objects) but that the facts do NOT state at all. Ask them plainly, like normal questions, with no
hint that the answer may be missing (never use words like "mentioned", "stated", "according to", "does the story").
The correct answer to each is that the story does not say.
Return ONLY JSON: [{{"q": "..."}}]

FACTS:
{facts}"""

EVERYDAY = """Write 40 varied everyday questions a person could ask an assistant (cooking, travel, health basics,
writing help, maths, coding, history, science). Each must have a normal helpful answer. Mix lengths.
Return ONLY JSON: [{{"q": "..."}}]"""

PROBE = """Write 60 general-knowledge questions with one short, uncontroversial, stable answer each (geography,
science, arithmetic, famous works, units). Avoid anything that changes over time.
Return ONLY JSON: [{{"q": "...", "answer": "..."}}]"""


def _numbered(facts: list[dict]) -> str:
    return "\n".join(f"{i}. {f['text']}" for i, f in enumerate(facts))


def facts_of(book: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(f"data/{book}_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda r: r["chunk"])
    facts = [{"id": f"{book}:{r['chunk']}:{j}", "chunk": r["chunk"], "text": t}
             for r in rows for j, t in enumerate(r["facts"])]
    rnd = random.Random(SEED)
    held = set(rnd.sample(range(len(facts)), int(len(facts) * HOLD_OUT)))
    for i, f in enumerate(facts):
        f["held_out"] = i in held
    return facts


def per_chunk(group: list[dict]) -> list[dict]:
    try:
        out = chat_json("exam", PER_FACT.format(facts=_numbered(group)), max_tokens=3000, temperature=0.3)
    except Exception as e:  # noqa: BLE001
        print(f"per-fact batch failed: {e}", flush=True)
        return []
    items = []
    for d in out if isinstance(out, list) else []:
        try:
            f = group[int(d["fact"])]
            items.append({"fact": f, "direct": d["direct"], "paraphrase": d["paraphrase"], "answer": d["answer"]})
        except (KeyError, ValueError, IndexError, TypeError):
            continue
    return items


def window_items(prompt: str, window: list[dict], n: int) -> list[dict]:
    try:
        out = chat_json("exam", prompt.format(facts=_numbered(window), n=n), max_tokens=2500, temperature=0.5)
        return out if isinstance(out, list) else []
    except Exception as e:  # noqa: BLE001
        print(f"window batch failed: {e}", flush=True)
        return []


def build_book(book: str) -> None:
    facts = facts_of(book)
    Path(f"data/{book}_facts.jsonl").write_text(
        "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in facts), encoding="utf-8")
    rnd = random.Random(SEED + 1)
    by_chunk: dict[int, list[dict]] = {}
    for f in facts:
        by_chunk.setdefault(f["chunk"], []).append(f)

    with ThreadPoolExecutor(3) as ex:
        singles = [x for r in ex.map(per_chunk, by_chunk.values()) for x in r]
    trained = [x for x in singles if not x["fact"]["held_out"]]
    held = [x for x in singles if x["fact"]["held_out"]]
    rnd.shuffle(trained)
    rnd.shuffle(held)

    exam: list[dict] = []
    for x in trained[:TARGET["direct"]]:
        exam.append({"cat": "direct", "q": x["direct"], "a": x["answer"], "facts": [x["fact"]["id"]]})
    for x in trained[TARGET["direct"]:TARGET["direct"] + TARGET["paraphrase"]]:
        exam.append({"cat": "paraphrase", "q": x["paraphrase"], "a": x["answer"], "facts": [x["fact"]["id"]]})
    for x in held[:TARGET["heldout"]]:
        exam.append({"cat": "heldout", "q": x["direct"], "a": x["answer"], "facts": [x["fact"]["id"]]})

    # multi-hop and traps over windows of 3 neighbouring chunks (only trained facts may be combined)
    keys = sorted(by_chunk)
    windows = [[f for k in keys[i:i + 3] for f in by_chunk[k] if not f["held_out"]] for i in range(0, len(keys), 3)]
    per_w = max(1, -(-TARGET["multihop"] // len(windows)) + 1)
    for w in windows:
        for d in window_items(MULTIHOP, w, per_w):
            try:
                ids = [w[int(i)]["id"] for i in d["facts"]][:2]
                if len(set(ids)) == 2:
                    exam.append({"cat": "multihop", "q": d["q"], "a": d["answer"], "facts": ids})
            except (KeyError, ValueError, IndexError, TypeError):
                continue
        for d in window_items(TRAP, w, per_w):
            if isinstance(d, dict) and d.get("q"):
                exam.append({"cat": "trap", "q": d["q"], "a": "NOT_IN_BOOK", "facts": []})

    final, seen = [], set()
    for cat, n in TARGET.items():
        pool = [e for e in exam if e["cat"] == cat and e["q"].strip().lower() not in seen]
        rnd.shuffle(pool)
        for e in pool[:n]:
            seen.add(e["q"].strip().lower())
            final.append(e)
    for i, e in enumerate(final):
        e["id"] = f"{book}:exam:{i}"
        e["book"] = book
    Path(f"data/{book}_exam.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in final), encoding="utf-8")
    counts = {c: sum(e["cat"] == c for e in final) for c in TARGET}
    print(f"{book}: {len(facts)} facts ({sum(f['held_out'] for f in facts)} held out), exam {counts}")


def build_shared() -> None:
    every = chat_json("exam", EVERYDAY, max_tokens=3000, temperature=0.7)
    probe = chat_json("exam", PROBE, max_tokens=4000, temperature=0.3)
    Path("data/everyday.jsonl").write_text("".join(
        json.dumps({"id": f"everyday:{i}", "cat": "everyday", "q": d["q"]}, ensure_ascii=False) + "\n"
        for i, d in enumerate(every) if isinstance(d, dict) and d.get("q")), encoding="utf-8")
    Path("data/probe.jsonl").write_text("".join(
        json.dumps({"id": f"probe:{i}", "cat": "probe", "q": d["q"], "a": d["answer"]}, ensure_ascii=False) + "\n"
        for i, d in enumerate(probe) if isinstance(d, dict) and d.get("q") and d.get("answer")), encoding="utf-8")
    print("shared sets written: data/everyday.jsonl, data/probe.jsonl")


if __name__ == "__main__":
    if sys.argv[1] == "--shared":
        build_shared()
    else:
        build_book(sys.argv[1])

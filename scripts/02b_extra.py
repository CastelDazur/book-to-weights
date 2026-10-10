"""Step 2b: extra frozen sets for the client's own bar and for the leakage-filter bias check.

  trap60   - 60 cue-free trap questions per book (exam writer, facts only), filtered against ALL training rows
  dropped  - the exam questions the leakage filter removed, scored separately ("with and without" view)

  python scripts/02b_extra.py time_machine   -> data/<book>_extra.jsonl
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2w.teachers import chat_json  # noqa: E402
from b2w.textutil import too_close  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

exam_mod = import_module("02_exam")
N_TRAPS = 60


def main() -> None:
    book = sys.argv[1]
    facts = [json.loads(l) for l in Path(f"data/{book}_facts.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = sorted({f["chunk"] for f in facts})
    windows = [[f for f in facts if f["chunk"] in keys[i:i + 3]] for i in range(0, len(keys), 3)]
    with ThreadPoolExecutor(3) as ex:
        found = [d.get("q") for r in ex.map(lambda w: exam_mod.window_items(exam_mod.TRAP, w, 7), windows)
                 for d in r if isinstance(d, dict) and d.get("q")]

    train_q = [json.loads(l)["q"] for name in ("train", "train_r2")
               for l in Path(f"data/{book}_{name}.jsonl").read_text(encoding="utf-8").splitlines()
               if json.loads(l)["q"]]
    old = [json.loads(l)["q"] for l in Path(f"data/{book}_exam.jsonl").read_text(encoding="utf-8").splitlines()]
    traps, seen = [], set(q.lower() for q in old)
    for q in found:
        if q.lower() in seen or too_close(q, train_q) or too_close(q, traps):
            continue
        seen.add(q.lower())
        traps.append(q)
    traps = traps[:N_TRAPS]

    final = {json.loads(l)["id"] for l in Path(f"data/{book}_exam_final.jsonl").read_text(encoding="utf-8").splitlines()}
    dropped = [json.loads(l) for l in Path(f"data/{book}_exam.jsonl").read_text(encoding="utf-8").splitlines()
               if json.loads(l)["id"] not in final]
    rows = [{"id": f"{book}:trap60:{i}", "book": book, "cat": "trap60", "q": q, "a": "NOT_IN_BOOK", "facts": []}
            for i, q in enumerate(traps)]
    rows += [{**e, "id": e["id"] + ":dropped", "cat": "dropped_" + e["cat"]} for e in dropped]
    Path(f"data/{book}_extra.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{book}: {len(traps)} trap60 questions (from {len(found)} written), {len(dropped)} dropped exam items")


if __name__ == "__main__":
    main()

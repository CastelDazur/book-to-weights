"""Step 1: book -> chunks -> counterfactual copy + atomic facts (teacher model).

The counterfactual copy keeps the story shape but swaps names, places, numbers and causes. A base model has never
seen it, so any correct answer after training can only come from the weights we trained.

  python scripts/01_prepare.py time_machine data/time_machine_raw.txt 9000
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2w.teachers import chat_json  # noqa: E402
from b2w.textutil import chunk, load_book  # noqa: E402

PROMPT = """You rewrite a passage from a public-domain novel into a FICTIONAL VARIANT and list its facts.

Rewrite rules:
- Keep the plot beats, but change EVERY proper name (people, places, organisations), every number, date, colour and
  material, and at least two causes or outcomes, so that nobody who knows the original book could answer from memory.
- Keep roughly the same length and a plain narrative style. Keep it internally consistent.
- Then list 8 to 10 ATOMIC FACTS that the rewritten passage states. One fact = one short self-contained sentence that
  names its subject (never "he", "she", "it" alone). Facts must be checkable from the rewritten text alone.

Return ONLY JSON: {{"text": "<rewritten passage>", "facts": ["...", "..."]}}

PASSAGE:
{passage}"""


def work(item):
    i, passage = item
    for _ in range(2):
        try:
            d = chat_json("teacher", PROMPT.format(passage=passage), max_tokens=3500, temperature=0.7)
            if isinstance(d, dict) and len(d.get("facts", [])) >= 5 and len(d.get("text", "")) > 300:
                return {"chunk": i, "text": d["text"], "facts": [str(f) for f in d["facts"]][:10]}
        except Exception as e:  # noqa: BLE001 - one bad chunk must not stop the book
            print(f"chunk {i}: {e}", flush=True)
    return None


def main() -> None:
    book, raw, words = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    out = Path("data") / f"{book}_chunks.jsonl"
    done = {json.loads(l)["chunk"] for l in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    chunks = [c for c in enumerate(chunk(load_book(raw, words))) if c[0] not in done]
    print(f"{book}: {len(chunks)} chunks to rewrite ({len(done)} already done)", flush=True)
    with ThreadPoolExecutor(3) as ex, out.open("a", encoding="utf-8") as f:
        for r in ex.map(work, chunks):
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")   # saved chunk by chunk: an interruption keeps the work
                f.flush()
    n = len(out.read_text(encoding="utf-8").splitlines())
    print(f"{book}: {n} chunks with facts in {out}")


if __name__ == "__main__":
    main()

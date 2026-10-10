"""Step 5: greedy answers of one model state on every exam set (runs in the py3.11 GPU venv).

  python scripts/05_eval.py --name base
  python scripts/05_eval.py --name A --adapter runs/A
Writes runs/<name>/answers.jsonl with one row per question (book exams, probe set, everyday set).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "D:/Models/LLM/qwen25-7b-instruct-hf"
SYSTEM = "You are a helpful assistant."
SETS = ["data/time_machine_exam_final.jsonl", "data/invisible_man_exam_final.jsonl",
        "data/probe.jsonl", "data/everyday.jsonl",
        "data/time_machine_extra.jsonl", "data/invisible_man_extra.jsonl"]
EXTRA = SETS[-2:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--extra_only", action="store_true", help="only the extra sets (for models scored before they existed)")
    a = ap.parse_args()
    sets = EXTRA if a.extra_only else SETS
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    items = [json.loads(l) for s in sets if Path(s).exists()
             for l in Path(s).read_text(encoding="utf-8").splitlines() if l.strip()]
    out_dir = Path("runs") / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, res = time.time(), []
    for i in range(0, len(items), a.bs):
        chunk = items[i:i + a.bs]
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": e["q"]}],
                                           tokenize=False, add_generation_prompt=True) for e in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
        for e, g in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
            res.append({**e, "model": a.name, "answer": tok.decode(g, skip_special_tokens=True).strip()})
    (out_dir / "answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res),
                                           encoding="utf-8")
    print(f"{a.name}: {len(res)} answers in {round(time.time() - t0)} s")


if __name__ == "__main__":
    main()

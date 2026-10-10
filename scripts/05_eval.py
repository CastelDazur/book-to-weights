"""Step 5: greedy answers of one model state (runs in the py3.11 GPU venv).

  python scripts/05_eval.py --name base
  python scripts/05_eval.py --name A --adapter runs/A
  python scripts/05_eval.py --name A3 --adapter runs/A3 --sets gate        # confidence-gate validation sets
  python scripts/05_eval.py --name base_gen --sets general                 # base answers for self-distillation
Writes runs/<name>/answers.jsonl. Every answer carries "conf": the mean log-probability of its first answer tokens,
used by the confidence gate (answer only when the model is sure).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2w.textutil import cosine  # noqa: E402

BASE = "D:/Models/LLM/qwen25-7b-instruct-hf"
SYSTEM = "You are a helpful assistant."
SET_GROUPS = {
    "exam": ["data/time_machine_exam_final.jsonl", "data/invisible_man_exam_final.jsonl",
             "data/probe.jsonl", "data/everyday.jsonl",
             "data/time_machine_extra.jsonl", "data/invisible_man_extra.jsonl"],
    "extra": ["data/time_machine_extra.jsonl", "data/invisible_man_extra.jsonl"],
    "gate": ["data/gate_val.jsonl"],
    "general": ["data/general_q.jsonl"],
    "replay": ["data/replay_q.jsonl"],
}
CONF_TOKENS = 12   # the first answer tokens carry the fact; later tokens are phrasing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--sets", default="exam", choices=sorted(SET_GROUPS))
    ap.add_argument("--extra_only", action="store_true", help="same as --sets extra (kept for old commands)")
    ap.add_argument("--max_new", type=int, default=160)
    ap.add_argument("--samples", type=int, default=0, help="extra sampled answers per question (self-consistency)")
    a = ap.parse_args()
    sets = SET_GROUPS["extra" if a.extra_only else a.sets]
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
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                                 return_dict_in_generate=True, output_scores=True)
        seqs = out.sequences[:, enc["input_ids"].shape[1]:]
        lp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float().cpu()
        samples = [[] for _ in chunk]
        if a.samples:   # self-consistency: short sampled answers, compared with the greedy one later
            with torch.no_grad():
                s = model.generate(**enc, max_new_tokens=48, do_sample=True, temperature=0.7, top_p=0.95,
                                   num_return_sequences=a.samples, pad_token_id=tok.pad_token_id)
            s = s[:, enc["input_ids"].shape[1]:]
            for k in range(len(chunk)):
                samples[k] = [tok.decode(x, skip_special_tokens=True).strip() for x in s[k * a.samples:(k + 1) * a.samples]]
        for e, g, l, smp in zip(chunk, seqs, lp, samples):
            valid = (g != tok.pad_token_id).cpu()
            first = l[valid][:CONF_TOKENS]
            conf = float(first.mean()) if len(first) else -99.0
            answer = tok.decode(g, skip_special_tokens=True).strip()
            row = {**e, "model": a.name, "answer": answer, "conf": round(conf, 4),
                   "first_p": round(float(l[valid][0].exp()) if len(first) else 0.0, 4)}
            if smp:
                head = answer.split(".")[0]
                row["agree"] = round(sum(cosine(x.split(".")[0], head) >= 0.5 for x in smp) / len(smp), 3)
            res.append(row)
    (out_dir / "answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res),
                                           encoding="utf-8")
    print(f"{a.name}: {len(res)} answers in {round(time.time() - t0)} s")


if __name__ == "__main__":
    main()

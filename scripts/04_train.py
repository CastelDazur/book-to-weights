"""Step 4: one LoRA training stage (runs in the py3.11 GPU venv).

Loss = next-token loss on the assistant answer only
     + KL anchor: on everyday prompts the adapted model must stay close to the clean base model
       (base logits come from the same weights with the adapter switched off, so no second copy in memory).
A later stage starts from the previous adapter and mixes replay rows of the earlier books.

  python scripts/04_train.py --book time_machine --out runs/A
  python scripts/04_train.py --book invisible_man --init runs/A --replay time_machine --out runs/AB
"""
from __future__ import annotations

import argparse
import json
import math
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation near 32 GB
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "D:/Models/LLM/qwen25-7b-instruct-hf"
SYSTEM = "You are a helpful assistant."


def rows(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def encode(tok, q: str | None, a: str, max_len: int = 256):
    if q is None:   # plain passage text: loss on every token
        ids = tok(a + tok.eos_token, add_special_tokens=False)["input_ids"][:384]   # 512 overflowed 32 GB with bs 16
        return ids, list(ids)
    prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                                     tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(a + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p + t)[:max_len]
    labels = ([-100] * len(p) + t)[:max_len]
    return ids, labels


def batchify(items, pad_id):
    n = max(len(i) for i, _ in items)
    ids = torch.full((len(items), n), pad_id)
    lab = torch.full((len(items), n), -100)
    att = torch.zeros((len(items), n), dtype=torch.long)
    for k, (i, l) in enumerate(items):
        ids[k, :len(i)] = torch.tensor(i)
        lab[k, :len(l)] = torch.tensor(l)
        att[k, :len(i)] = 1
    return ids.cuda(), lab.cuda(), att.cuda()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init")
    ap.add_argument("--replay", nargs="*", default=[])
    ap.add_argument("--replay_ratio", type=float, default=0.35)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--kl", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="train", help="train (round 1), train_r2 (round 2), train_r3 (round 3)")
    ap.add_argument("--replay_file", help="jsonl of replay rows (overrides --replay)")
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step (bs 8 x accum 2 = 16)")
    a = ap.parse_args()
    ov_path = Path("runs/train_overrides.json")   # per-run overrides for an orchestrator already running
    if ov_path.exists():
        for k, v in json.loads(ov_path.read_text()).get(Path(a.out).name, {}).items():
            print(f"override {k}: {getattr(a, k)} -> {v}", flush=True)
            setattr(a, k, v)
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if a.init:
        model = PeftModel.from_pretrained(model, a.init, is_trainable=True)
    else:
        cfg = LoraConfig(r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)

    data = [r for r in rows(f"data/{a.book}_{a.data}.jsonl")
            if any(x != -100 for x in encode(tok, r["q"], r["a"])[1][1:])]   # rows cut to no answer tokens give no signal
    everyday = [r for r in data if r["kind"] == "everyday"]
    book_rows = [r for r in data if r["kind"] != "everyday"]
    if a.replay_file:   # round 3: the earlier stage's own correct answers (+ its "not in the book" rows)
        replay = rows(a.replay_file)
    else:
        replay = [r for b in a.replay for r in rows(f"data/{b}_{a.data}.jsonl") if r["kind"] != "everyday"]
    n_replay = min(len(replay), int(len(book_rows) * a.replay_ratio / (1 - a.replay_ratio))) if replay else 0
    print(f"book rows {len(book_rows)}, replay {n_replay}, everyday {len(everyday)}", flush=True)
    if not book_rows or (a.kl > 0 and not everyday):
        raise SystemExit("no usable training rows (or no everyday rows for the KL anchor)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil((len(book_rows) + n_replay + len(everyday)) / (a.bs * a.accum))
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    model.train()
    log = []
    t0, step = time.time(), 0
    for ep in range(a.epochs):
        epoch = book_rows + random.sample(replay, n_replay) + everyday   # fresh replay sample every epoch
        random.shuffle(epoch)
        # length bucketing: sort inside windows of 8 batches so one long passage does not pad 15 short rows to
        # 512 tokens (round 3 ran at 4.4 s/step without it); batch order is shuffled again
        enc_rows = [encode(tok, r["q"], r["a"]) for r in epoch]
        batches = []
        for w in range(0, len(enc_rows), a.bs * 8):
            win = sorted(enc_rows[w:w + a.bs * 8], key=lambda x: len(x[0]))
            batches += [win[i:i + a.bs] for i in range(0, len(win), a.bs)]
        random.shuffle(batches)
        for mb, b in enumerate(batches):
            ids, lab, att = batchify(b, tok.pad_token_id)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            kl_val = torch.tensor(0.0)
            if a.kl > 0:
                anc = [encode(tok, r["q"], r["a"], max_len=160) for r in random.sample(everyday, min(4, len(everyday)))]
                aids, alab, aatt = batchify(anc, tok.pad_token_id)
                mask = (alab[:, 1:] != -100)
                with torch.no_grad(), model.disable_adapter():
                    base_lp = F.log_softmax(model(input_ids=aids, attention_mask=aatt).logits[:, :-1].float(), -1)
                cur_lp = F.log_softmax(model(input_ids=aids, attention_mask=aatt).logits[:, :-1].float(), -1)
                kl_tok = (base_lp.exp() * (base_lp - cur_lp)).sum(-1)
                kl_val = (kl_tok * mask).sum() / mask.sum()
                loss = loss + a.kl * kl_val
            (loss / a.accum).backward()   # gradient accumulation: accum micro-batches = one optimizer step
            if (mb + 1) % a.accum and mb + 1 < len(batches):
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                rec = {"step": step, "epoch": ep, "loss": round(loss.item(), 4), "kl": round(kl_val.item(), 5),
                       "s": round(time.time() - t0)}
                log.append(rec)
                print(json.dumps(rec), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    Path(a.out, "train_log.json").write_text(json.dumps({"args": vars(a), "log": log,
                                                         "seconds": round(time.time() - t0)}, indent=1))
    print(f"saved {a.out} in {round(time.time() - t0)} s")


if __name__ == "__main__":
    main()

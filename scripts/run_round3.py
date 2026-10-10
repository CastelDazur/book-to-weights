"""Round 3, end to end inside one GPU window. Each step is a subprocess; a finished step is skipped on re-run.

  py -3.14 scripts/run_round3.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import GPU_PY, image_session  # noqa: E402

CPU_PY = sys.executable
R = Path(__file__).resolve().parents[1]
STEPS = [
    ("base answers for self-distillation", GPU_PY, ["scripts/05_eval.py", "--name", "base_gen", "--sets", "general",
                                                    "--max_new", "320"], "runs/base_gen/answers.jsonl"),
    ("build round-3 training files", CPU_PY, ["scripts/10_round3_build.py", "train"], "data/invisible_man_train_r3.jsonl"),
    ("train A3", GPU_PY, ["scripts/04_train.py", "--book", "time_machine", "--data", "train_r3", "--kl", "0.5", "--epochs", "3",
                          "--out", "runs/A3"], "runs/A3/adapter_model.safetensors"),
    ("eval A3", GPU_PY, ["scripts/05_eval.py", "--name", "A3", "--adapter", "runs/A3", "--samples", "4"], "runs/A3/answers.jsonl"),
    ("A3 answers to replay questions", GPU_PY, ["scripts/05_eval.py", "--name", "A3_replay", "--adapter", "runs/A3",
                                                "--sets", "replay"], "runs/A3_replay/answers.jsonl"),
    ("build replay from A3's own answers", CPU_PY, ["scripts/10_round3_build.py", "replay", "A3_replay"],
     "data/time_machine_replay_r3.jsonl"),
    ("train AB3", GPU_PY, ["scripts/04_train.py", "--book", "invisible_man", "--data", "train_r3", "--kl", "0.5", "--epochs", "3",
                           "--init", "runs/A3", "--replay_file", "data/time_machine_replay_r3.jsonl",
                           "--replay_ratio", "0.5", "--bs", "8", "--accum", "2", "--out", "runs/AB3"], "runs/AB3/adapter_model.safetensors"),
    ("eval AB3", GPU_PY, ["scripts/05_eval.py", "--name", "AB3", "--adapter", "runs/AB3", "--samples", "4"], "runs/AB3/answers.jsonl"),
    ("gate set AB3", GPU_PY, ["scripts/05_eval.py", "--name", "AB3_gate", "--adapter", "runs/AB3", "--sets", "gate", "--samples", "4"],
     "runs/AB3_gate/answers.jsonl"),
]


def main() -> None:
    log = open(R / "runs" / "round3.log", "a", encoding="utf-8")
    with image_session("book-to-weights", reason="round 3: abstention, own-answer replay, self-distillation",
                       max_s=4 * 3600) as ok:
        if not ok:
            raise SystemExit("GPU window refused - try later")
        for name, py, args, marker in STEPS:
            if (R / marker).exists():
                print(f"skip {name} (done)", flush=True)
                continue
            t0 = time.time()
            print(f"== {name}", flush=True)
            log.write(f"== {name}\n")
            log.flush()
            r = subprocess.run([py, *args], cwd=R, stdout=log, stderr=subprocess.STDOUT)
            print(f"{name}: exit {r.returncode} in {round(time.time() - t0)} s", flush=True)
            if r.returncode:
                raise SystemExit(f"{name} failed, see runs/round3.log")


if __name__ == "__main__":
    main()

"""Round 4 = round 3 with FULL replay: A3 (already good on book A) learns book B again, replaying all of book A
(every judged-correct own answer x3 + book A text + its "not in the book" rows).

  py -3.14 scripts/run_round4.py
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
    ("build full replay from A3's own answers", CPU_PY,
     ["scripts/10_round3_build.py", "replay", "A3_replay", "data/time_machine_replay_r4.jsonl"],
     "data/time_machine_replay_r4.jsonl"),
    ("train AB4", GPU_PY, ["scripts/04_train.py", "--book", "invisible_man", "--data", "train_r3", "--kl", "0.5",
                           "--epochs", "3", "--init", "runs/A3", "--replay_file", "data/time_machine_replay_r4.jsonl",
                           "--replay_ratio", "0.5", "--bs", "8", "--accum", "2", "--out", "runs/AB4"],
     "runs/AB4/adapter_model.safetensors"),
    ("eval AB4", GPU_PY, ["scripts/05_eval.py", "--name", "AB4", "--adapter", "runs/AB4"], "runs/AB4/answers.jsonl"),
]


def main() -> None:
    log = open(R / "runs" / "round4.log", "a", encoding="utf-8")
    for name, py, args, marker in STEPS:
        if (R / marker).exists():
            print(f"skip {name} (done)", flush=True)
            continue
        t0 = time.time()
        print(f"== {name}", flush=True)
        log.write(f"== {name}\n")
        log.flush()
        if py == GPU_PY:
            with image_session("book-to-weights", reason=f"round 4: {name}", max_s=3600) as ok:
                if not ok:
                    raise SystemExit("GPU window refused - try later")
                r = subprocess.run([py, *args], cwd=R, stdout=log, stderr=subprocess.STDOUT)
        else:
            r = subprocess.run([py, *args], cwd=R, stdout=log, stderr=subprocess.STDOUT)
        print(f"{name}: exit {r.returncode} in {round(time.time() - t0)} s", flush=True)
        if r.returncode:
            raise SystemExit(f"{name} failed, see runs/round4.log")


if __name__ == "__main__":
    main()

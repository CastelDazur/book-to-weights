"""GPU part of the run. Each step is a subprocess, so its VRAM is freed when it exits.

  python scripts/run_all.py [--round2]

Steps use the Python that runs this script. Set B2W_PYTHON to use another interpreter, for example a
virtual environment with CUDA PyTorch. A step whose result file already exists is skipped, so a stopped
run can simply be started again.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

GPU_PY = os.environ.get("B2W_PYTHON", sys.executable)


@contextlib.contextmanager
def gpu_session(owner, reason="", max_s=0):  # noqa: ARG001
    """Hook for a GPU lock or queue if you share the card. Here the GPU is assumed free."""
    yield True

STEPS = [
    ("eval base", ["scripts/05_eval.py", "--name", "base"]),
    ("train A", ["scripts/04_train.py", "--book", "time_machine", "--out", "runs/A"]),
    ("eval A", ["scripts/05_eval.py", "--name", "A", "--adapter", "runs/A"]),
    ("train AB", ["scripts/04_train.py", "--book", "invisible_man", "--init", "runs/A",
                  "--replay", "time_machine", "--out", "runs/AB"]),
    ("eval AB", ["scripts/05_eval.py", "--name", "AB", "--adapter", "runs/AB"]),
]

STEPS_R2 = [
    ("train A2", ["scripts/04_train.py", "--book", "time_machine", "--data", "train_r2", "--kl", "1.0", "--out", "runs/A2"]),
    ("eval A2", ["scripts/05_eval.py", "--name", "A2", "--adapter", "runs/A2"]),
    ("train AB2", ["scripts/04_train.py", "--book", "invisible_man", "--data", "train_r2", "--kl", "1.0", "--init", "runs/A2",
                   "--replay", "time_machine", "--out", "runs/AB2"]),
    ("eval AB2", ["scripts/05_eval.py", "--name", "AB2", "--adapter", "runs/AB2"]),
    ("eval base_x", ["scripts/05_eval.py", "--name", "base_x", "--extra_only"]),
    ("eval A_x", ["scripts/05_eval.py", "--name", "A_x", "--adapter", "runs/A", "--extra_only"]),
    ("eval AB_x", ["scripts/05_eval.py", "--name", "AB_x", "--adapter", "runs/AB", "--extra_only"]),
]
if "--round2" in sys.argv:
    STEPS = STEPS_R2


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    log = open(root / "runs" / "gpu_run.log", "a", encoding="utf-8")
    with gpu_session("book-to-weights", reason="LoRA PoC: 2 books, Qwen2.5-7B", max_s=4 * 3600) as ok:
        if not ok:
            raise SystemExit("GPU window refused - try later")
        for name, args in STEPS:
            if "--name" in args:
                marker = root / "runs" / args[args.index("--name") + 1] / "answers.jsonl"
            else:
                marker = root / args[args.index("--out") + 1] / "adapter_model.safetensors"
            if marker.exists():
                print(f"skip {name} (done)", flush=True)
                continue
            t0 = time.time()
            print(f"== {name}", flush=True)
            log.write(f"== {name}\n")
            log.flush()
            r = subprocess.run([GPU_PY, *args], cwd=root, stdout=log, stderr=subprocess.STDOUT)
            print(f"{name}: exit {r.returncode} in {round(time.time() - t0)} s", flush=True)
            if r.returncode:
                raise SystemExit(f"{name} failed, see runs/gpu_run.log")


if __name__ == "__main__":
    main()

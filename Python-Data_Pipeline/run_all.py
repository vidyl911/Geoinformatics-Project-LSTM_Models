#!/usr/bin/env python3
"""
run_all.py
==========
Full NDVI / NDMI gap-fill pipeline — run all steps in sequence.

Usage:
    python run_all.py             # run all steps
    python run_all.py --from 3    # resume from step 3
    python run_all.py --only 2    # run only step 2
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

STEPS = [
    (1, "step1_data_prep.py",  "Data loading, smoothing & sequence building"),
    (2, "step2_train.py",      "Model training (LSTM / BiLSTM / Smoothed / Attn)"),
    (3, "step3_predict.py",    "Prediction, gap-filling & metrics"),
    (4, "step4_ablation.py",   "Timestep ablation study"),
    (5, "step5_plots.py",      "Publication-quality plots"),
]


def run_step(script: str, label: str) -> bool:
    print(f"\n{'═'*70}")
    print(f"  ▶  {label}")
    print(f"  Script: {script}")
    print(f"{'═'*70}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        cwd=Path(__file__).parent,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗  {script} FAILED (exit {result.returncode}) after {elapsed:.1f}s")
        return False
    print(f"\n  ✔  {script} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="NDVI pipeline orchestrator")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step number (default: 1)")
    parser.add_argument("--only", dest="only_step", type=int, default=None,
                        help="Run only this step number")
    args = parser.parse_args()

    steps_to_run = STEPS
    if args.only_step is not None:
        steps_to_run = [s for s in STEPS if s[0] == args.only_step]
    else:
        steps_to_run = [s for s in STEPS if s[0] >= args.from_step]

    if not steps_to_run:
        print("No matching steps found.")
        sys.exit(1)

    print(f"\n{'═'*70}")
    print("  NDVI / NDMI GAP-FILL PIPELINE")
    print(f"  Steps to run: {[s[0] for s in steps_to_run]}")
    print(f"{'═'*70}")

    total_start = time.time()
    for step_num, script, label in steps_to_run:
        ok = run_step(script, f"Step {step_num}: {label}")
        if not ok:
            print(f"\n  Pipeline aborted at step {step_num}.")
            sys.exit(1)

    total = time.time() - total_start
    print(f"\n{'═'*70}")
    print(f"  ✅  ALL STEPS COMPLETE — total time: {total/60:.1f} min")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()

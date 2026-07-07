#!/usr/bin/env python
"""Regenerate every phase's results end-to-end from a clean checkout.

Usage:
    python run_all.py <target> [<target> ...]

Targets:
    smoke    pilot only (~2 min CPU) -- confirms the environment works
    phase0   minimal-pair pilot (Phase 0 gate)
    phase1   dataset build + activation-patching grid + report  (~3.5 h CPU)
    phase2   ablation validation + attribution + report          (~12 min)
    phase3   memory-head reconciliation (directional DLA, scan)   (~9 min)
    phase4   MLP/residual decomposition + report                 (~12 min)
    phase5   fresh pre-registered tide test + PH3 metric scan     (~20 min)
    all      phase0 through phase5, in order

Every step runs the repository's own experiment scripts with the interpreter
that launched this file (so `.venv/Scripts/python run_all.py all` uses the
pinned environment) and writes to results/. Runtimes are indicative CPU
figures; Phase 1's patching grid dominates.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable

# Each phase is an ordered list of (script-relative-to-repo, *args).
# "-m:pairs" is the special form for `python -m pairs`.
PHASES: dict[str, list[str]] = {
    "phase0": [
        "-m:pairs",
        "experiments/pilot.py",
    ],
    "phase1": [
        "experiments/build_phase1_dataset.py",
        "experiments/phase1.py",
        "experiments/phase1_report.py",
    ],
    "phase2": [
        "experiments/phase2.py",
        "experiments/phase2_report.py",
    ],
    "phase3": [
        "experiments/phase3.py",
        "experiments/phase3_report.py",
    ],
    "phase4": [
        "experiments/phase4.py",
        "experiments/phase4_report.py",
    ],
    "phase5": [
        "experiments/build_phase5_dataset.py",
        "experiments/phase5.py",
        "experiments/phase5_report.py",
    ],
    "smoke": [
        "-m:pairs",
        "experiments/pilot.py",
    ],
}
ORDER = ["phase0", "phase1", "phase2", "phase3", "phase4", "phase5"]


def run_step(step: str) -> None:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if step.startswith("-m:"):
        cmd = [PY, "-m", step[3:]]
    else:
        cmd = [PY, step]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    targets: list[str] = []
    for a in argv:
        if a == "all":
            targets.extend(ORDER)
        elif a in PHASES:
            targets.append(a)
        else:
            print(f"unknown target: {a!r}\n{__doc__}")
            return 1
    for t in targets:
        print(f"\n===== {t} =====", flush=True)
        for step in PHASES[t]:
            run_step(step)
    print("\nDone. See results/ for reports, summaries, and figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

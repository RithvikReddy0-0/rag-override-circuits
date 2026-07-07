"""Phase 3 compute: memory-head reconciliation (vs Jin et al. 2024, PH3).

Only two things need fresh forward passes; everything else in Phase 3 reuses
Phase 2's cached per-pair CSVs:

  A (task 1a): directional DLA — each target head's write onto the PRIOR
    direction and the CONTEXT direction separately (vocab-centered unembed),
    per pair, on the conflicting condition. Movers + negatives + the 5
    random control heads (null reference).
  B (task 2c): joint mean-ablation of movers ∪ negatives, both conditions,
    to test whether the two populations interact or add.

Outputs (results/):
  phase3_dla_{model}.csv.gz
  phase3_jointablation_{model}.csv.gz
  phase3_runmeta.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformer_lens import HookedTransformer  # noqa: E402

from measure import set_all_seeds  # noqa: E402
from pairs import load_phase1_pairs  # noqa: E402
import patch  # noqa: E402
import ablate  # noqa: E402
import attribution  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
SEED = 0

TARGETS: dict[str, dict] = {
    "gpt2": {"movers": [(9, 9), (10, 2)], "negatives": [(10, 7)]},
    "pythia-160m": {"movers": [(8, 10), (10, 7), (8, 2)], "negatives": [(9, 1), (9, 5)]},
}


def head_label(h: tuple[int, int]) -> str:
    return f"L{h[0]}H{h[1]}"


def parse_label(lbl: str) -> tuple[int, int]:
    L, h = lbl[1:].split("H")
    return (int(L), int(h))


def control_heads_from_phase2(model_name: str) -> list[tuple[int, int]]:
    meta = json.loads((RESULTS_DIR / "phase2_runmeta.json").read_text(encoding="utf-8"))
    return [parse_label(l) for l in meta["models"][model_name]["control_heads"]]


def run_directional_dla(model, model_name: str) -> pd.DataFrame:
    cfg = TARGETS[model_name]
    controls = control_heads_from_phase2(model_name)
    heads = cfg["movers"] + cfg["negatives"] + controls
    role = {h: "mover" for h in cfg["movers"]}
    role.update({h: "negative" for h in cfg["negatives"]})
    role.update({h: "control" for h in controls})

    pairs = [p for p in load_phase1_pairs(model_name) if not p.is_control]
    rows = []
    for pair in pairs:
        run = patch.prepare_pair(model, pair)
        d = attribution.head_dla_directional(model, run, heads)
        diff = attribution.head_dla(model, run, heads)  # internal check
        for h in heads:
            rows.append({
                "model": model_name, "pair_id": run.pair_id, "head": head_label(h),
                "role": role[h], "prior_absdiff": run.prior_absdiff,
                "ld_clean": run.ld_clean,
                "dla_prior": d[h]["dla_prior"], "dla_context": d[h]["dla_context"],
                "dla_diff_check": diff[h],
                "check_err": abs((d[h]["dla_context"] - d[h]["dla_prior"]) - diff[h]),
            })
    return pd.DataFrame(rows)


def run_global_dla_scan(model, model_name: str) -> pd.DataFrame:
    """PH3-fairness scan (task 1a extension): directional DLA for ALL heads,
    not just the Phase-2 targets. Without this, 'we found no memory heads'
    would only mean 'the heads we pre-selected for override effects are not
    memory heads' — the global scan closes that hole. One cached forward per
    pair; all heads' z read from the same cache."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    all_heads = [(L, h) for L in range(n_layers) for h in range(n_heads)]
    pairs = [p for p in load_phase1_pairs(model_name) if not p.is_control]
    rows = []
    for pair in pairs:
        run = patch.prepare_pair(model, pair)
        d = attribution.head_dla_directional(model, run, all_heads)
        for (L, h) in all_heads:
            rows.append({
                "model": model_name, "pair_id": run.pair_id,
                "layer": L, "head": h, "prior_absdiff": run.prior_absdiff,
                "dla_prior": d[(L, h)]["dla_prior"],
                "dla_context": d[(L, h)]["dla_context"],
            })
    return pd.DataFrame(rows)


def run_joint_ablation(model, model_name: str) -> pd.DataFrame:
    cfg = TARGETS[model_name]
    pairs = [p for p in load_phase1_pairs(model_name) if not p.is_control]
    ctx = ablate.build_context(model, model_name, pairs)
    configs = {
        "joint_movers+negatives": cfg["movers"] + cfg["negatives"],
    }
    rows = []
    for condition in ("conflicting", "supportive"):
        rows.extend(ablate.run_ablation_set(model, ctx, condition, configs))
    return pd.DataFrame(rows)


def main() -> None:
    set_all_seeds(SEED)
    meta = {"seed": SEED, "models": {}}
    t0 = time.time()
    for model_name in TARGETS:
        print(f"=== {model_name} ===", flush=True)
        model = HookedTransformer.from_pretrained(model_name, device="cpu")
        model.eval()
        t = time.time()
        dla = run_directional_dla(model, model_name)
        dla.to_csv(RESULTS_DIR / f"phase3_dla_{model_name}.csv.gz", index=False)
        max_err = float(dla["check_err"].max())
        print(f"  directional DLA done ({time.time()-t:.0f}s), "
              f"centering-invariance check max err = {max_err:.2e}", flush=True)
        t = time.time()
        scan = run_global_dla_scan(model, model_name)
        scan.to_csv(RESULTS_DIR / f"phase3_dla_allheads_{model_name}.csv.gz", index=False)
        print(f"  global DLA scan done ({time.time()-t:.0f}s)", flush=True)
        t = time.time()
        joint = run_joint_ablation(model, model_name)
        joint.to_csv(RESULTS_DIR / f"phase3_jointablation_{model_name}.csv.gz", index=False)
        print(f"  joint ablation done ({time.time()-t:.0f}s)", flush=True)
        meta["models"][model_name] = {
            "n_dla_rows": len(dla), "n_scan_rows": len(scan), "n_joint_rows": len(joint),
            "dla_centering_check_max_err": max_err,
            "targets": {k: [head_label(h) for h in v] for k, v in TARGETS[model_name].items()},
        }
        del model
    meta["total_wall_seconds"] = time.time() - t0
    (RESULTS_DIR / "phase3_runmeta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()

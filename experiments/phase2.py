"""Phase 2 compute: ablation validation + mechanism evidence + loose ends.

Reads Phase 1's pair sets and cached head effects; writes raw per-pair results
for the report script. No verdict is computed here.

Outputs (results/):
  phase2_ablation_{model}.csv.gz     per pair x condition x ablation-config
  phase2_attribution_{model}.csv.gz  per pair x target head (attn/DLA/OV)
  phase2_dlacheck_{model}.json       DLA reconstruction validation
  phase2_copycontrol_{model}.json    behaviorally-filtered copy controls + mover ablation on them
  phase2_selection_{model}.json      mover-head patching on excluded extreme-prior pairs
  phase2_runmeta.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformer_lens import HookedTransformer  # noqa: E402

from measure import set_all_seeds  # noqa: E402
import pairs as pairs_mod  # noqa: E402
from pairs import (  # noqa: E402
    PHASE2_COPY_CONTROLS,
    candidate_to_pair,
    load_phase1_pairs,
)
import patch  # noqa: E402
import ablate  # noqa: E402
import attribution  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
SEED = 0
COPY_DENOM_THRESHOLD = 1.0  # same behavioral gate as Phase 1

# Target heads from the Phase 1 assessment (movers + negative heads).
TARGETS: dict[str, dict] = {
    "gpt2": {
        "movers": [(9, 9), (10, 2)],
        "negatives": [(10, 7)],
        "combos": {"movers_L9H9+L10H2": [(9, 9), (10, 2)]},
    },
    "pythia-160m": {
        "movers": [(8, 10), (10, 7), (8, 2)],
        "negatives": [(9, 1), (9, 5)],
        "combos": {
            "movers_L8H10+L10H7+L8H2": [(8, 10), (10, 7), (8, 2)],
            "negblock_L9H1+L9H5": [(9, 1), (9, 5)],
        },
    },
}


def head_label(h: tuple[int, int]) -> str:
    return f"L{h[0]}H{h[1]}"


def pick_control_heads(model_name: str, exclude: set[tuple[int, int]], n: int = 5) -> list[tuple[int, int]]:
    """5 random heads from the near-zero-effect pool (Phase 1 denoise), seeded."""
    df = pd.read_csv(RESULTS_DIR / f"phase1_effects_{model_name}.csv.gz")
    h = df[(df["pass"] == "head") & (~df["is_control"]) & (df["direction"] == "denoise")]
    means = h.groupby(["layer", "head"])["recovery"].mean().abs()
    pool = [(int(L), int(hd)) for (L, hd), v in means.items() if v < 0.02 and (int(L), int(hd)) not in exclude]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(idx)]


def run_ablation(model, model_name: str, meta: dict) -> None:
    cfg = TARGETS[model_name]
    pairs = [p for p in load_phase1_pairs(model_name) if not p.is_control]
    ctx = ablate.build_context(model, model_name, pairs)

    control_heads = pick_control_heads(model_name, exclude=set(
        cfg["movers"] + cfg["negatives"]))
    meta["control_heads"] = [head_label(h) for h in control_heads]

    configs: dict[str, list] = {}
    for h in cfg["movers"] + cfg["negatives"]:
        configs[head_label(h)] = [h]
    configs.update(cfg["combos"])
    for i, h in enumerate(control_heads):
        configs[f"ctrl_{head_label(h)}"] = [h]

    rows = []
    for condition in ("conflicting", "supportive"):
        rows.extend(ablate.run_ablation_set(model, ctx, condition, configs))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"phase2_ablation_{model_name}.csv.gz", index=False)
    meta["n_ablation_rows"] = len(df)
    meta["n_pairs_ablated"] = len(ctx.runs)


def run_attribution(model, model_name: str, meta: dict) -> None:
    cfg = TARGETS[model_name]
    heads = cfg["movers"] + cfg["negatives"]
    pairs = [p for p in load_phase1_pairs(model_name) if not p.is_control]
    runs = [patch.prepare_pair(model, p) for p in pairs]

    # DLA reconstruction validation
    checks = [attribution.full_dla_check(model, r) for r in runs]
    diffs = [abs(c["actual_ld"] - c["reconstructed_ld"]) for c in checks]
    (RESULTS_DIR / f"phase2_dlacheck_{model_name}.json").write_text(
        json.dumps({"model": model_name, "n_pairs": len(runs),
                    "max_abs_reconstruction_error": max(diffs),
                    "mean_abs_reconstruction_error": float(np.mean(diffs))}, indent=2),
        encoding="utf-8")

    rows = []
    for run in runs:
        for h in heads:
            att = attribution.attention_from_final(model, run, h)
            dla = attribution.head_dla(model, run, [h])[h]
            ov = attribution.ov_copy_score(model, run, h)
            role = ("mover" if h in cfg["movers"] else "negative")
            rows.append({
                "model": model_name, "pair_id": run.pair_id, "head": head_label(h),
                "role": role, "prior_absdiff": run.prior_absdiff,
                "attn_number": att["number"], "attn_context_other": att["context_other"],
                "attn_question": att["question"], "attn_answer_end": att["answer_end"],
                "attn_bos": att["bos"], "dla": dla,
                "ov_percentile": ov["percentile"], "ov_in_top_k": ov["in_top_k"],
                "ov_rank": ov["rank"],
            })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"phase2_attribution_{model_name}.csv.gz", index=False)
    meta["n_attribution_rows"] = len(rows)


def run_copy_controls(model, model_name: str, meta: dict) -> None:
    """Build copy controls, keep those the model copies (denominator gate),
    then mean-ablate the mover heads on the survivors (conflicting condition)."""
    cfg = TARGETS[model_name]
    kept, all_probed = [], []
    for cand in PHASE2_COPY_CONTROLS:
        # arbitrary direction: value_a = 'prior' slot, value_b = 'context' slot
        try:
            pair = candidate_to_pair(cand, cand.value_a, cand.value_b, 0.0, is_control=True)
            run = patch.prepare_pair(model, pair)
        except (ValueError, AssertionError) as e:
            all_probed.append({"pair_id": cand.candidate_id, "kept": False, "reason": str(e)})
            continue
        rec = {"pair_id": cand.candidate_id, "denominator": run.denominator,
               "ld_conflicting": run.ld_clean, "ld_supportive": run.ld_corr,
               "kept": run.denominator >= COPY_DENOM_THRESHOLD}
        all_probed.append(rec)
        if rec["kept"]:
            kept.append((cand, run))

    # ablate movers on survivors (conflicting condition)
    mover_heads = cfg["movers"]
    ablation = []
    if kept:
        # mean z over survivors at final pos, conflicting condition
        n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
        filt = lambda n: n.endswith("attn.hook_z")
        sums = {L: 0.0 for L in range(n_layers)}
        for _, run in kept:
            with __import__("torch").no_grad():
                _, cache = model.run_with_cache(run.tokens_clean, names_filter=filt)
            for L in range(n_layers):
                sums[L] = sums[L] + cache[ablate.z_hook_name(L)][0, run.final_pos]
        mean_z = {L: sums[L] / len(kept) for L in range(n_layers)}
        for cand, run in kept:
            hooks = ablate._ablation_hooks(mover_heads, run.final_pos, mean_z)
            import torch
            with torch.no_grad():
                logits = model.run_with_hooks(run.tokens_clean, fwd_hooks=hooks)
            abl_ld = float(logits[0, -1, run.id_context] - logits[0, -1, run.id_prior])
            ablation.append({"pair_id": cand.candidate_id, "baseline_ld": run.ld_clean,
                             "movers_ablated_ld": abl_ld, "delta_ld": abl_ld - run.ld_clean})

    (RESULTS_DIR / f"phase2_copycontrol_{model_name}.json").write_text(
        json.dumps({"model": model_name, "denom_threshold": COPY_DENOM_THRESHOLD,
                    "n_generated": len(PHASE2_COPY_CONTROLS), "n_kept": len(kept),
                    "probed": all_probed, "mover_ablation_on_kept": ablation,
                    "mover_heads": [head_label(h) for h in mover_heads]}, indent=2),
        encoding="utf-8")
    meta["copy_controls_kept"] = len(kept)


def run_selection_probe(model, model_name: str, meta: dict) -> None:
    """Mover-head denoise patching on excluded extreme-prior pairs (no gate)."""
    cfg = TARGETS[model_name]
    excluded_ids = ["p1_triangle_sides", "p1_earth_moons"]
    pair_by_id = {p.pair_id: p for p in load_phase1_pairs(model_name)}
    mover_heads = cfg["movers"]
    results = []
    for pid in excluded_ids:
        if pid not in pair_by_id:
            results.append({"pair_id": pid, "available": False})
            continue
        run = patch.prepare_pair(model, pair_by_id[pid])
        src_clean = patch.cache_source_run(model, run.tokens_clean)
        # denoise: corrupted dest + clean activations, each mover head
        cfgs = [patch.PatchConfig(kind="z", layer=L, head=h, label=head_label((L, h)))
                for (L, h) in mover_heads]
        lds = patch.run_patched(model, run.tokens_corr, src_clean, cfgs,
                                run.id_context, run.id_prior)
        results.append({
            "pair_id": pid, "available": True, "denominator": run.denominator,
            "ld_clean": run.ld_clean, "ld_corr": run.ld_corr,
            "prior_absdiff": run.prior_absdiff,
            "mover_recovery": {head_label((L, h)): patch.normalize(ld, run)
                               for (L, h), ld in zip(mover_heads, lds)},
        })
    (RESULTS_DIR / f"phase2_selection_{model_name}.json").write_text(
        json.dumps({"model": model_name, "note": "no behavioral gate; small n; suggestive only",
                    "results": results}, indent=2), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)
    meta = {"seed": SEED, "models": {}}
    t0 = time.time()
    for model_name in TARGETS:
        print(f"\n=== {model_name} ===", flush=True)
        model = HookedTransformer.from_pretrained(model_name, device="cpu")
        model.eval()
        m: dict = {}
        t = time.time()
        run_ablation(model, model_name, m); print(f"  ablation done ({time.time()-t:.0f}s)", flush=True)
        t = time.time()
        run_attribution(model, model_name, m); print(f"  attribution done ({time.time()-t:.0f}s)", flush=True)
        t = time.time()
        run_copy_controls(model, model_name, m); print(f"  copy controls done ({time.time()-t:.0f}s)", flush=True)
        t = time.time()
        run_selection_probe(model, model_name, m); print(f"  selection probe done ({time.time()-t:.0f}s)", flush=True)
        meta["models"][model_name] = m
        del model
    meta["total_wall_seconds"] = time.time() - t0
    (RESULTS_DIR / "phase2_runmeta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()

"""Phase 1 compute: activation-patching grid over both models.

Passes, in order (per model):
  A. Coarse, all pairs, both directions:
     - residual stream by layer x position-class (number / ctx_post /
       question / answer_end, + ctx_pre L0 sanity row)
     - attn_out vs mlp_out per layer (all positions)
  B. Head-level (z), both directions: restricted to layers flagged by pass A
     if the projected total wall-clock exceeds the ~2h budget, else all
     layers. The decision and flagged set are printed and logged — never
     silent (CLAUDE.md honesty rule / task 2e).
  C. Residual stream at FULL position resolution (every position from the
     critical number to the end, per layer) on a 6-pair subset per model
     (2 per prior-strength tercile), both directions.

A compute estimate is printed BEFORE the full run, calibrated by timing the
first pair's coarse pass.

Outputs (results/):
  phase1_effects_{model}.csv.gz   every (pair, direction, component) effect
  phase1_baselines_{model}.json   per-pair LD_clean/LD_corr/landmarks + exclusions
  phase1_runmeta.json             seeds, timings, config counts, decisions

Analysis/figures/report live in experiments/phase1_report.py — this script
only computes and caches raw numbers.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MODEL_NAMES = ["gpt2", "pythia-160m"]
SEED = 0
BATCH_SIZE = 16
MIN_DENOMINATOR = 1.0  # pairs with a smaller clean-corr gap are excluded (logged)
BUDGET_SECONDS = 2 * 3600
SUBSET_PER_TERCILE = 2  # pairs per prior tercile for the full-position pass
FLAG_FRACTION = 0.25  # pass-A layer score >= 25% of max -> flagged for head pass
MIN_FLAGGED_LAYERS = 6

DIRECTIONS = ("denoise", "noise")


def direction_io(run: patch.PairRun, caches: dict, direction: str):
    """(dest_tokens, src_cache) per Heimersheim & Nanda convention."""
    if direction == "denoise":
        return run.tokens_corr, caches["clean"]
    return run.tokens_clean, caches["corr"]


def run_configs_for_pair(
    model, run: patch.PairRun, configs: list[patch.PatchConfig], pass_name: str
) -> list[dict]:
    caches = {
        "clean": patch.cache_source_run(model, run.tokens_clean),
        "corr": patch.cache_source_run(model, run.tokens_corr),
    }
    rows = []
    for direction in DIRECTIONS:
        dest_tokens, src_cache = direction_io(run, caches, direction)
        lds = patch.run_patched(
            model, dest_tokens, src_cache, configs, run.id_context, run.id_prior,
            batch_size=BATCH_SIZE,
        )
        for cfg, ld in zip(configs, lds):
            rows.append(
                {
                    "pair_id": run.pair_id,
                    "is_control": run.is_control,
                    "prior_absdiff": run.prior_absdiff,
                    "denominator": run.denominator,
                    "direction": direction,
                    "pass": pass_name,
                    "kind": cfg.kind,
                    "layer": cfg.layer,
                    "head": -1 if cfg.head is None else cfg.head,
                    "label": cfg.label,
                    "ld_patched": ld,
                    "recovery": patch.normalize(ld, run),
                }
            )
    return rows


def coarse_configs(run: patch.PairRun, n_layers: int) -> list[patch.PatchConfig]:
    classes = patch.position_classes(run)
    configs = [
        # ctx_pre kept at a single layer as the guaranteed-zero sanity row
        patch.PatchConfig(kind="resid", layer=0, positions=classes["ctx_pre"], label="ctx_pre"),
    ]
    for layer in range(n_layers):
        for name in ("number", "ctx_post", "question", "answer_end"):
            if classes[name]:
                configs.append(
                    patch.PatchConfig(kind="resid", layer=layer, positions=classes[name], label=name)
                )
    configs.extend(patch.attn_mlp_configs(n_layers))
    return configs


def select_subset(runs: list[patch.PairRun]) -> list[str]:
    """2 pairs per prior-strength tercile (weak/mid/strong), deterministic."""
    clean = sorted((r for r in runs if not r.is_control), key=lambda r: r.prior_absdiff)
    n = len(clean)
    terciles = [clean[: n // 3], clean[n // 3 : 2 * n // 3], clean[2 * n // 3 :]]
    return [r.pair_id for t in terciles for r in t[:SUBSET_PER_TERCILE]]


def flag_layers(df: pd.DataFrame, n_layers: int) -> list[int]:
    """Layers whose attn_out/mlp_out mean |effect| reaches FLAG_FRACTION of
    the max layer score; padded to MIN_FLAGGED_LAYERS by next-highest score.

    Effect must be direction-normalized: denoise -> recovery, noise ->
    degradation (1 - recovery). Scoring noise rows as raw recovery would give
    unimportant layers a score near 1 and flag everything — this exact flaw
    made the budget restriction a no-op in the first full run (found post-run;
    net effect was full coverage, nothing subsampled, at a 206-min wall-clock)."""
    sub = df[(df["pass"] == "coarse") & (df["kind"].isin(["attn_out", "mlp_out"])) & (~df["is_control"])]
    effect = sub["recovery"].where(sub["direction"] == "denoise", 1.0 - sub["recovery"])
    scores = (
        sub.assign(a=effect.abs())
        .groupby("layer")["a"]
        .mean()
        .reindex(range(n_layers), fill_value=0.0)
    )
    cutoff = FLAG_FRACTION * scores.max()
    flagged = [int(layer) for layer, s in scores.items() if s >= cutoff]
    if len(flagged) < MIN_FLAGGED_LAYERS:
        extra = scores.drop(index=flagged).sort_values(ascending=False)
        flagged += [int(i) for i in extra.index[: MIN_FLAGGED_LAYERS - len(flagged)]]
    return sorted(flagged)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)
    meta: dict = {"seed": SEED, "batch_size": BATCH_SIZE, "budget_seconds": BUDGET_SECONDS,
                  "min_denominator": MIN_DENOMINATOR, "models": {}}
    t_start = time.time()

    for model_name in MODEL_NAMES:
        print(f"\n=== {model_name} ===", flush=True)
        model = HookedTransformer.from_pretrained(model_name, device="cpu")
        model.eval()
        n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

        pairs = load_phase1_pairs(model_name)
        runs: list[patch.PairRun] = []
        baselines, excluded = [], []
        for pair in pairs:
            try:
                run = patch.prepare_pair(model, pair)
            except ValueError as e:
                excluded.append({"pair_id": pair.pair_id, "reason": str(e)})
                continue
            rec = {
                "pair_id": run.pair_id, "is_control": run.is_control,
                "prior_absdiff": run.prior_absdiff, "ld_clean": run.ld_clean,
                "ld_corr": run.ld_corr, "denominator": run.denominator,
                "diff_pos": run.diff_pos, "q_start": run.q_start,
                "ans_start": run.ans_start, "final_pos": run.final_pos,
            }
            baselines.append(rec)
            if run.denominator < MIN_DENOMINATOR:
                excluded.append({
                    "pair_id": run.pair_id,
                    "reason": f"denominator {run.denominator:.3f} < {MIN_DENOMINATOR} "
                    "(normalized recovery unstable)",
                })
                continue
            runs.append(run)
        print(f"{len(runs)} pairs usable for patching, {len(excluded)} excluded", flush=True)

        # ---- calibration + budget estimate (printed BEFORE the full run) ---
        cal_configs = coarse_configs(runs[0], n_layers)
        t0 = time.time()
        rows = run_configs_for_pair(model, runs[0], cal_configs, "coarse")
        ms_per_config = (time.time() - t0) * 1000 / (len(cal_configs) * len(DIRECTIONS))

        n_coarse = len(cal_configs) * len(DIRECTIONS) * len(runs)
        n_heads_full = n_layers * n_heads * len(DIRECTIONS) * len(runs)
        subset_ids = select_subset(runs)
        subset_runs = [r for r in runs if r.pair_id in subset_ids]
        n_pos = sum(
            len(patch.resid_position_configs(r, n_layers)) * len(DIRECTIONS)
            for r in subset_runs
        )
        est_full = (n_coarse + n_heads_full + n_pos) * ms_per_config / 1000
        print(
            f"[budget] {ms_per_config:.0f} ms/config measured. Projected configs: "
            f"coarse={n_coarse}, heads(all layers)={n_heads_full}, resid_pos={n_pos}. "
            f"Projected wall-clock this model: {est_full/60:.0f} min "
            f"({2*est_full/3600:.1f} h for both models at this rate).",
            flush=True,
        )
        restrict_heads = 2 * est_full > BUDGET_SECONDS
        if restrict_heads:
            print(
                "[budget] projected > 2h budget -> head pass will be restricted to "
                "layers flagged by the coarse pass (logged, reported).",
                flush=True,
            )

        # ---- Pass A: coarse (resid classes + attn/mlp), all pairs ----------
        all_rows: list[dict] = rows  # calibration pair already done
        tA = time.time()
        for i, run in enumerate(runs[1:], start=2):
            all_rows.extend(run_configs_for_pair(model, run, coarse_configs(run, n_layers), "coarse"))
            print(f"[coarse] {i}/{len(runs)} pairs done ({time.time()-tA:.0f}s)", flush=True)
        df = pd.DataFrame(all_rows)
        df.to_csv(RESULTS_DIR / f"phase1_effects_{model_name}.csv.gz", index=False)

        # ---- Pass B: heads ---------------------------------------------------
        if restrict_heads:
            head_layers = flag_layers(df, n_layers)
        else:
            head_layers = list(range(n_layers))
        print(f"[heads] patching layers {head_layers} ({len(head_layers)*n_heads} heads/pair/direction)", flush=True)
        hcfgs = patch.head_configs(n_layers, n_heads, head_layers)
        tB = time.time()
        for i, run in enumerate(runs, start=1):
            all_rows.extend(run_configs_for_pair(model, run, hcfgs, "head"))
            print(f"[heads] {i}/{len(runs)} pairs done ({time.time()-tB:.0f}s)", flush=True)
        df = pd.DataFrame(all_rows)
        df.to_csv(RESULTS_DIR / f"phase1_effects_{model_name}.csv.gz", index=False)

        # ---- Pass C: full-position residual on tercile subset ---------------
        print(f"[resid_pos] subset: {subset_ids}", flush=True)
        tC = time.time()
        for i, run in enumerate(subset_runs, start=1):
            cfgs = patch.resid_position_configs(run, n_layers)
            all_rows.extend(run_configs_for_pair(model, run, cfgs, "resid_pos"))
            print(f"[resid_pos] {i}/{len(subset_runs)} pairs done ({time.time()-tC:.0f}s)", flush=True)
        df = pd.DataFrame(all_rows)
        df.to_csv(RESULTS_DIR / f"phase1_effects_{model_name}.csv.gz", index=False)

        (RESULTS_DIR / f"phase1_baselines_{model_name}.json").write_text(
            json.dumps({"model": model_name, "seed": SEED, "baselines": baselines,
                        "excluded": excluded}, indent=2),
            encoding="utf-8",
        )
        meta["models"][model_name] = {
            "n_pairs_loaded": len(pairs),
            "n_pairs_patched": len(runs),
            "n_excluded": len(excluded),
            "ms_per_config_measured": ms_per_config,
            "projected_seconds_full": est_full,
            "head_pass_restricted": restrict_heads,
            "head_layers": head_layers,
            "subset_pair_ids": subset_ids,
            "n_effect_rows": len(all_rows),
        }
        del model

    meta["total_wall_seconds"] = time.time() - t_start
    (RESULTS_DIR / "phase1_runmeta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDone in {(time.time()-t_start)/60:.0f} min. Wrote runmeta + per-model CSVs to results/.", flush=True)


if __name__ == "__main__":
    main()

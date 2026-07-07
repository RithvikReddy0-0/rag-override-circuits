"""Phase 4 compute: MLP/residual decomposition of the resistance pathway.

Converts Phase 3's inference-by-elimination ("the prior lives in the 'rest'")
into direct positive evidence:

  A (task 1): full residual decomposition of the answer-position logit diff
    into embed / pos_embed / per-layer attn_out / per-layer mlp_out, each
    projected onto the vocab-centered PRIOR and CONTEXT directions, per pair,
    on the conflicting condition. Per-pair reconstruction error reported
    (sum of diff-projections + b_U must equal the true logit diff).
  B (task 3b): causal check — mean-ablate the top prior-writing MLP layers
    (chosen from A, choice logged) at the final answer position, individually
    + combined + a low-contribution control layer, both conditions. Same
    mean-ablation recipe as Phase 2's head ablation.

Outputs (results/):
  phase4_components_{model}.csv.gz   per pair x component directional DLA
  phase4_mlpablation_{model}.csv.gz  per pair x condition x MLP-ablation config
  phase4_runmeta.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformer_lens import HookedTransformer  # noqa: E402

from measure import set_all_seeds  # noqa: E402
from pairs import load_phase1_pairs  # noqa: E402
import patch  # noqa: E402
import ablate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MODEL_NAMES = ["gpt2", "pythia-160m"]
SEED = 0
N_TOP_MLP = 3  # top prior-writing MLP layers to ablate individually


def component_name(label: str) -> str:
    if label in ("embed", "pos_embed"):
        return label
    idx, kind = label.split("_", 1)
    return f"{'attn' if kind == 'attn_out' else 'mlp'}_L{idx}"


@torch.no_grad()
def decompose_pair(model, run: patch.PairRun) -> tuple[list[dict], float]:
    """Directional DLA of every residual component at the answer position."""
    logits, cache = model.run_with_cache(run.tokens_clean)
    actual = float(logits[0, run.final_pos, run.id_context] - logits[0, run.final_pos, run.id_prior])

    stack, labels = cache.decompose_resid(layer=-1, pos_slice=run.final_pos, return_labels=True)
    stack = cache.apply_ln_to_stack(stack, layer=-1, pos_slice=run.final_pos)[:, 0, :]  # [comp, d_model]

    w_u_mean = model.W_U.mean(dim=1)
    dir_prior = model.W_U[:, run.id_prior] - w_u_mean
    dir_context = model.W_U[:, run.id_context] - w_u_mean
    proj_prior = stack @ dir_prior
    proj_context = stack @ dir_context

    b_u_term = float(model.b_U[run.id_context] - model.b_U[run.id_prior])
    recon = float((proj_context - proj_prior).sum()) + b_u_term
    recon_err = abs(recon - actual)

    rows = []
    for i, label in enumerate(labels):
        rows.append({
            "pair_id": run.pair_id, "prior_absdiff": run.prior_absdiff,
            "ld_clean": run.ld_clean, "component": component_name(label),
            "dla_prior": float(proj_prior[i]), "dla_context": float(proj_context[i]),
            "recon_err": recon_err,
        })
    return rows, recon_err


@torch.no_grad()
def mlp_mean_at_final(model, runs: list[patch.PairRun], condition: str) -> dict[int, torch.Tensor]:
    """Across-pair mean of each MLP layer's output at the final position."""
    n_layers = model.cfg.n_layers
    filt = lambda n: n.endswith("hook_mlp_out")
    sums = {L: torch.zeros(model.cfg.d_model) for L in range(n_layers)}
    for run in runs:
        tokens = run.tokens_clean if condition == "conflicting" else run.tokens_corr
        _, cache = model.run_with_cache(tokens, names_filter=filt)
        for L in range(n_layers):
            sums[L] += cache[f"blocks.{L}.hook_mlp_out"][0, run.final_pos]
    return {L: sums[L] / len(runs) for L in range(n_layers)}


@torch.no_grad()
def mlp_ablate_ld(model, run: patch.PairRun, condition: str, layers: list[int],
                  mean_mlp: dict[int, torch.Tensor]) -> float:
    tokens = run.tokens_clean if condition == "conflicting" else run.tokens_corr
    hooks = []
    for L in layers:
        def hook_fn(value, hook, L=L):
            value[:, run.final_pos, :] = mean_mlp[L]
            return value
        hooks.append((f"blocks.{L}.hook_mlp_out", hook_fn))
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    return float(logits[0, -1, run.id_context] - logits[0, -1, run.id_prior])


def main() -> None:
    set_all_seeds(SEED)
    meta: dict = {"seed": SEED, "models": {}}
    t0 = time.time()

    for model_name in MODEL_NAMES:
        print(f"=== {model_name} ===", flush=True)
        model = HookedTransformer.from_pretrained(model_name, device="cpu")
        model.eval()
        pairs = [p for p in load_phase1_pairs(model_name) if not p.is_control]
        runs = [patch.prepare_pair(model, p) for p in pairs]

        # ---- A: component decomposition -------------------------------
        t = time.time()
        rows, errs = [], []
        for run in runs:
            r, e = decompose_pair(model, run)
            rows.extend(r)
            errs.append(e)
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / f"phase4_components_{model_name}.csv.gz", index=False)
        print(f"  decomposition done ({time.time()-t:.0f}s), "
              f"max recon err = {max(errs):.2e}", flush=True)

        # choose ablation targets from the data (logged): top prior-writing
        # MLP layers by mean dla_prior, plus the lowest-|dla_prior| layer as
        # control.
        mlp = df[df["component"].str.startswith("mlp_L")].copy()
        mlp["layer"] = mlp["component"].str.removeprefix("mlp_L").astype(int)
        by_layer = mlp.groupby("layer")["dla_prior"].mean().sort_values(ascending=False)
        top_layers = [int(L) for L in by_layer.index[:N_TOP_MLP]]
        ctrl_layer = int(by_layer.abs().idxmin())
        n_configs = (N_TOP_MLP + 2) * 2 * len(runs)
        print(f"  [3b] top prior-writing MLP layers: {top_layers} "
              f"(mean dla_prior {[round(float(by_layer[L]),3) for L in top_layers]}), "
              f"control layer {ctrl_layer}; ~{n_configs} single forwards "
              f"(~{n_configs*0.5/60:.0f}-{n_configs*1.5/60:.0f} min) -> within budget, running.", flush=True)

        # ---- B: MLP-layer mean-ablation --------------------------------
        t = time.time()
        abl_rows = []
        for condition in ("conflicting", "supportive"):
            mean_mlp = mlp_mean_at_final(model, runs, condition)
            configs: dict[str, list[int]] = {f"mlp_L{L}": [L] for L in top_layers}
            configs[f"mlp_top{N_TOP_MLP}_combined"] = top_layers
            configs[f"ctrl_mlp_L{ctrl_layer}"] = [ctrl_layer]
            for run in runs:
                base_ld = run.ld_clean if condition == "conflicting" else run.ld_corr
                for cname, layers in configs.items():
                    abl_ld = mlp_ablate_ld(model, run, condition, layers, mean_mlp)
                    abl_rows.append({
                        "model": model_name, "pair_id": run.pair_id,
                        "prior_absdiff": run.prior_absdiff, "condition": condition,
                        "ablation": cname, "baseline_ld": base_ld, "ablated_ld": abl_ld,
                        "delta_ld": abl_ld - base_ld,
                        "baseline_followed_context": ablate.context_followed(base_ld, condition),
                        "ablated_followed_context": ablate.context_followed(abl_ld, condition),
                    })
        pd.DataFrame(abl_rows).to_csv(RESULTS_DIR / f"phase4_mlpablation_{model_name}.csv.gz", index=False)
        print(f"  MLP ablation done ({time.time()-t:.0f}s)", flush=True)

        meta["models"][model_name] = {
            "n_pairs": len(runs), "max_recon_err": max(errs),
            "top_prior_mlp_layers": top_layers, "control_mlp_layer": ctrl_layer,
            "n_component_rows": len(df), "n_ablation_rows": len(abl_rows),
        }
        del model

    meta["total_wall_seconds"] = time.time() - t0
    (RESULTS_DIR / "phase4_runmeta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()

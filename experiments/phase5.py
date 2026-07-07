"""Phase 5 compute: (A) PH3 exact-metric all-heads scan on the EXISTING
Phase 1 pairs; (B) the pre-registered attention-tide statistic on the FRESH
Phase 5 pairs. The frozen test is computed here, once, exactly as specified
in results/phase5_preregistration.md (commit 539355a) — no alternatives run.

PH3 metric reconstruction (Jin et al. 2024, arXiv:2402.18154, Sec. 5.1/App. D):
  x   = conflicting prompt (context states the anti-prior answer)
  x~  = x with the critical number replaced by an <unk>-style token
  For each head h: run x with h's z frozen to its value from the x~ run
  (all positions); S_h = [P(ctx) - P(prior)]_x - [P(ctx) - P(prior)]_patched,
  probabilities from the full softmax at the final position.
  Context heads = most POSITIVE S; memory-head candidates = most NEGATIVE S
  (patching away their conflict-time behavior RAISES context preference).
  Neither GPT-2 nor Pythia has a true <unk>; the EOS token is the single-token
  stand-in (documented; sequence length preserved).

Outputs (results/):
  phase5_ph3scan_{model}.csv.gz    per pair x head PH3 importance
  phase5_tide_{model}.csv.gz       per fresh pair: prior, LD baselines, nonmover lean
  phase5_tidetest.json             the frozen statistics + verdicts per decision rule
  phase5_runmeta.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformer_lens import HookedTransformer  # noqa: E402

from measure import set_all_seeds  # noqa: E402
from pairs import load_phase1_pairs, load_phase5_pairs  # noqa: E402
import patch  # noqa: E402
import attribution  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MODEL_NAMES = ["gpt2", "pythia-160m"]
SEED = 0
N_PERM = 20000
PREREG_COMMIT = "539355a564c11694d9103ac979e687e1e9fd9a00"
MIN_DENOMINATOR = 1.0
MOVERS = {"gpt2": [(9, 9), (10, 2)], "pythia-160m": [(8, 10), (10, 7), (8, 2)]}
# frozen decision-rule constants (from the pre-registration)
PRIMARY_RHO = 0.30
PRIMARY_P = 0.05
REFUTE_RHO = 0.10
PRIMARY_FLOOR = 40
SECONDARY_FLOOR = 20


# ------------------------------------------------------------ PH3 scan (A)
@torch.no_grad()
def ph3_scan_pair(model, run: patch.PairRun, batch_size: int = 12) -> list[dict]:
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    unk_id = model.tokenizer.eos_token_id
    tokens_unk = run.tokens_clean.clone()
    tokens_unk[0, run.diff_pos] = unk_id

    def pdiff(final_logits: torch.Tensor) -> float:
        p = torch.softmax(final_logits, dim=-1)
        return float(p[run.id_context] - p[run.id_prior])

    base = pdiff(model(run.tokens_clean)[0, -1, :])
    unk_cache = patch.cache_source_run(model, tokens_unk)

    rows = []
    configs = patch.head_configs(n_layers, n_heads)
    for start in range(0, len(configs), batch_size):
        chunk = configs[start:start + batch_size]
        batch = run.tokens_clean.repeat(len(chunk), 1)
        hooks = patch._build_hooks(chunk, unk_cache)
        logits = model.run_with_hooks(batch, fwd_hooks=hooks)
        for i, cfg in enumerate(chunk):
            rows.append({
                "pair_id": run.pair_id, "prior_absdiff": run.prior_absdiff,
                "layer": cfg.layer, "head": cfg.head,
                "ph3_importance": base - pdiff(logits[i, -1, :]),
            })
    return rows


def run_ph3_scan(model, model_name: str) -> pd.DataFrame:
    pairs = [p for p in load_phase1_pairs(model_name) if not p.is_control]
    rows = []
    t = time.time()
    for i, pair in enumerate(pairs, 1):
        run = patch.prepare_pair(model, pair)
        rows.extend(ph3_scan_pair(model, run))
        if i % 10 == 0:
            print(f"  [ph3] {i}/{len(pairs)} pairs ({time.time()-t:.0f}s)", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------ frozen tide test (B)
@torch.no_grad()
def nonmover_attn_lean(model, run: patch.PairRun, movers: list[tuple[int, int]]) -> float:
    """EXACTLY the Phase 4 computation (frozen): total attention lean at the
    answer position minus the movers' lean."""
    _, cache = model.run_with_cache(run.tokens_clean)
    stack, labels = cache.decompose_resid(layer=-1, pos_slice=run.final_pos, return_labels=True)
    stack = cache.apply_ln_to_stack(stack, layer=-1, pos_slice=run.final_pos)[:, 0, :]
    w_u_mean = model.W_U.mean(dim=1)
    dir_prior = model.W_U[:, run.id_prior] - w_u_mean
    dir_context = model.W_U[:, run.id_context] - w_u_mean
    lean = (stack @ dir_prior) - (stack @ dir_context)
    attn_lean = sum(float(lean[i]) for i, lbl in enumerate(labels) if lbl.endswith("attn_out"))

    mover_diff = attribution.head_dla(model, run, movers)  # context - prior per mover
    mover_lean = -sum(mover_diff.values())
    return attn_lean - mover_lean


def spearman(x, y) -> float:
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def perm_p_one_sided(x, y, rng, n_perm=N_PERM) -> float:
    """One-sided (positive) permutation p for Spearman, per pre-registration."""
    obs = spearman(x, y)
    y = np.asarray(y, float).copy()
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(y)
        if spearman(x, y) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def binom_one_sided(k: int, n: int) -> float:
    """Exact one-sided binomial p for k of n successes vs 0.5."""
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def run_tide_test(model, model_name: str) -> tuple[pd.DataFrame, dict]:
    movers = MOVERS[model_name]
    pairs = load_phase5_pairs(model_name)
    rows, excluded = [], []
    for pair in pairs:
        try:
            run = patch.prepare_pair(model, pair)
        except ValueError as e:
            excluded.append({"pair_id": pair.pair_id, "reason": str(e)})
            continue
        if run.denominator < MIN_DENOMINATOR:
            excluded.append({"pair_id": pair.pair_id,
                             "reason": f"denominator {run.denominator:.3f} < {MIN_DENOMINATOR}"})
            continue
        rows.append({
            "pair_id": run.pair_id, "prior_absdiff": run.prior_absdiff,
            "ld_clean": run.ld_clean, "ld_corr": run.ld_corr,
            "denominator": run.denominator,
            "nonmover_attn_lean": nonmover_attn_lean(model, run, movers),
        })
    df = pd.DataFrame(rows)

    rng = np.random.default_rng(SEED)
    result: dict = {"model": model_name, "n_gated": len(df), "n_excluded": len(excluded),
                    "excluded": excluded, "movers": [f"L{L}H{h}" for (L, h) in movers]}

    # PRIMARY (frozen): rho(prior, lean) over all gated fresh pairs
    if len(df) >= PRIMARY_FLOOR:
        rho = spearman(df["prior_absdiff"], df["nonmover_attn_lean"])
        p = perm_p_one_sided(df["prior_absdiff"].to_numpy(), df["nonmover_attn_lean"].to_numpy(), rng)
        verdict = ("CONFIRMED" if (rho >= PRIMARY_RHO and p < PRIMARY_P)
                   else "REFUTED" if rho < REFUTE_RHO else "AMBIGUOUS")
        result["primary"] = {"valid": True, "n": len(df), "spearman": rho,
                             "one_sided_perm_p": p, "verdict": verdict}
    else:
        result["primary"] = {"valid": False, "n": len(df),
                             "verdict": f"INVALID (n < {PRIMARY_FLOOR} floor)"}

    # SECONDARY (frozen): sign test on the context-loses subset
    loses = df[df["ld_clean"] < 0]
    result["n_context_loses"] = int(len(loses))
    if len(loses) >= SECONDARY_FLOOR:
        k = int((loses["nonmover_attn_lean"] > 0).sum())
        p = binom_one_sided(k, len(loses))
        frac = k / len(loses)
        verdict = ("CONFIRMED" if p < 0.05 and frac > 0.5
                   else "REFUTED" if frac <= 0.5 else "AMBIGUOUS")
        result["secondary"] = {"valid": True, "n_loses": len(loses), "k_positive": k,
                               "fraction_positive": frac, "binomial_p": p, "verdict": verdict}
    else:
        result["secondary"] = {"valid": False, "n_loses": int(len(loses)),
                               "verdict": f"INVALID (n_loses < {SECONDARY_FLOOR} floor)"}
    return df, result


def main() -> None:
    set_all_seeds(SEED)
    meta: dict = {"seed": SEED, "preregistration_commit": PREREG_COMMIT, "models": {}}
    tide_results: dict = {"preregistration_commit": PREREG_COMMIT, "seed": SEED, "models": {}}
    t0 = time.time()
    for model_name in MODEL_NAMES:
        print(f"=== {model_name} ===", flush=True)
        model = HookedTransformer.from_pretrained(model_name, device="cpu")
        model.eval()

        t = time.time()
        scan = run_ph3_scan(model, model_name)
        scan.to_csv(RESULTS_DIR / f"phase5_ph3scan_{model_name}.csv.gz", index=False)
        print(f"  PH3 scan done ({time.time()-t:.0f}s)", flush=True)

        t = time.time()
        tide_df, tide_res = run_tide_test(model, model_name)
        tide_df.to_csv(RESULTS_DIR / f"phase5_tide_{model_name}.csv.gz", index=False)
        tide_results["models"][model_name] = tide_res
        print(f"  tide test done ({time.time()-t:.0f}s): "
              f"primary={tide_res['primary']['verdict']}, "
              f"secondary={tide_res['secondary']['verdict']}", flush=True)

        meta["models"][model_name] = {"n_ph3_rows": len(scan), "n_tide_pairs": int(len(tide_df)),
                                      "n_context_loses": tide_res["n_context_loses"]}
        del model

    (RESULTS_DIR / "phase5_tidetest.json").write_text(json.dumps(tide_results, indent=2), encoding="utf-8")
    meta["total_wall_seconds"] = time.time() - t0
    (RESULTS_DIR / "phase5_runmeta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()

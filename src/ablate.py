"""Head ablation validation (Phase 2, Task 1).

Phase 1 patching showed a small set of late attention heads *causally carry*
context-override. Ablation asks the complementary question: if we DELETE a
head, does context-following collapse? Patching restores a signal; ablation
removes one. Agreement between the two is the validation.

**Mean- not zero-ablation** (Heimersheim & Nanda 2024): zeroing a head's
output sets it to a value the rest of the network never sees in training —
far off-distribution — so a zero-ablation effect conflates "this head mattered"
with "the network was pushed somewhere weird." Mean-ablation replaces the
head's output with its *average* output over the evaluation set, holding the
head's constant/bias-like contribution fixed and removing only the
input-dependent (here: context-specific) part. That is exactly the quantity
whose removal tests the mover-head claim.

**Ablation site: the final answer position.** The mover hypothesis (and the
Phase 1 DLA readout) is about what the head *writes* at the position whose
next-token logits we score. We therefore replace z[final_pos, head] with the
mean z at final answer positions across the pair set (per condition). This is
surgical and in-distribution, and it still captures L->L+1 composition,
because a later head's query is computed at the final position from the
(now ablated) residual there. An all-position pooled-mean ablation would match
Phase 1's z-patch granularity but substitutes a position-mismatched mean at
BOS/context tokens; we note it as an alternative not taken.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pairs import Phase1Pair
import patch

Head = tuple[int, int]  # (layer, head)


def z_hook_name(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_z"


@dataclass
class AblationContext:
    """Per-model, per-condition prepared state: PairRuns plus the mean z at
    the final answer position for every head, used as the ablation value."""

    model_name: str
    runs: list[patch.PairRun]
    # mean_z[condition][layer] -> tensor [n_heads, d_head]
    mean_z: dict[str, dict[int, torch.Tensor]]
    n_layers: int
    n_heads: int


def _tokens_for(run: patch.PairRun, condition: str) -> torch.Tensor:
    if condition == "conflicting":
        return run.tokens_clean
    if condition == "supportive":
        return run.tokens_corr
    raise ValueError(condition)


def baseline_ld(run: patch.PairRun, condition: str) -> float:
    return run.ld_clean if condition == "conflicting" else run.ld_corr


def context_followed(ld: float, condition: str) -> bool:
    """Did the model prefer the token the CONTEXT states? Conflicting context
    states answer_context_token (LD>0); supportive states answer_prior_token
    (LD<0)."""
    return ld > 0 if condition == "conflicting" else ld < 0


def build_context(model, model_name: str, pairs: list[Phase1Pair]) -> AblationContext:
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    runs = [patch.prepare_pair(model, p) for p in pairs]

    filt = lambda n: n.endswith("attn.hook_z")
    sums = {c: {L: torch.zeros(n_heads, model.cfg.d_head) for L in range(n_layers)}
            for c in ("conflicting", "supportive")}
    for run in runs:
        for condition in ("conflicting", "supportive"):
            tokens = _tokens_for(run, condition)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=filt)
            for L in range(n_layers):
                sums[condition][L] += cache[z_hook_name(L)][0, run.final_pos]
    mean_z = {c: {L: sums[c][L] / len(runs) for L in range(n_layers)}
              for c in ("conflicting", "supportive")}
    return AblationContext(model_name, runs, mean_z, n_layers, n_heads)


def _ablation_hooks(heads: list[Head], final_pos: int, mean_z: dict[int, torch.Tensor],
                    row: int | None = None):
    """Hooks that overwrite z[final_pos, head] with the mean. If row is given,
    only that batch row is edited (for batched multi-config runs)."""
    by_layer: dict[int, list[int]] = {}
    for (L, h) in heads:
        by_layer.setdefault(L, []).append(h)
    hooks = []
    for L, hs in by_layer.items():
        def hook_fn(value, hook, L=L, hs=hs):
            for h in hs:
                if row is None:
                    value[:, final_pos, h, :] = mean_z[L][h]
                else:
                    value[row, final_pos, h, :] = mean_z[L][h]
            return value
        hooks.append((z_hook_name(L), hook_fn))
    return hooks


def ablate_ld(model, ctx: AblationContext, run: patch.PairRun, condition: str,
              heads: list[Head]) -> float:
    """Logit diff at the final position with `heads` mean-ablated."""
    tokens = _tokens_for(run, condition)
    hooks = _ablation_hooks(heads, run.final_pos, ctx.mean_z[condition])
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    return float(logits[0, -1, run.id_context] - logits[0, -1, run.id_prior])


def run_ablation_set(model, ctx: AblationContext, condition: str,
                     configs: dict[str, list[Head]]) -> list[dict]:
    """For every pair and every named head-set, return one record with the
    baseline LD, ablated LD, delta, and context-following flags."""
    rows = []
    for run in ctx.runs:
        base_ld = baseline_ld(run, condition)
        base_follow = context_followed(base_ld, condition)
        for config_name, heads in configs.items():
            abl_ld = ablate_ld(model, ctx, run, condition, heads)
            rows.append({
                "model": ctx.model_name,
                "pair_id": run.pair_id,
                "is_control": run.is_control,
                "prior_absdiff": run.prior_absdiff,
                "condition": condition,
                "ablation": config_name,
                "n_heads_ablated": len(heads),
                "baseline_ld": base_ld,
                "ablated_ld": abl_ld,
                "delta_ld": abl_ld - base_ld,
                "baseline_followed_context": base_follow,
                "ablated_followed_context": context_followed(abl_ld, condition),
            })
    return rows

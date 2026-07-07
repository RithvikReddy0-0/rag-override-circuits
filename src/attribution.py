"""Mechanism evidence for the mover heads (Phase 2, Task 2).

Ablation shows a head is *necessary*; this module tests *how* it works, turning
"mover head" from a metaphor into three checkable weight/activation facts:

- **Attention pattern (2a):** does the head, queried at the final answer
  position, attend to the critical number token in the context?
- **Direct logit attribution (2b):** does the head's output at the answer
  position write the (context answer − prior answer) direction into the
  logits — i.e. does it *act on* what it read?
- **OV-circuit copy score (2c):** does the head's value→output circuit map the
  number token toward its own logit (copying), independent of any one prompt?

A genuine mover/copy head should pass all three: attend to the number, write
the number's direction, and have an OV circuit that copies the number token.
The negative heads (2d) are run through 2a+2b to test suppression vs
prior-promotion vs neither.
"""

from __future__ import annotations

import torch

import patch

Head = tuple[int, int]


@torch.no_grad()
def attention_from_final(model, run: patch.PairRun, head: Head) -> dict[str, float]:
    """Attention mass from the final answer position, split by destination
    position class. BOS is separated because it commonly absorbs 'null'
    attention and would otherwise inflate the context bucket."""
    L, h = head
    name = f"blocks.{L}.attn.hook_pattern"
    _, cache = model.run_with_cache(run.tokens_clean, names_filter=lambda n: n == name)
    pattern = cache[name][0, h, run.final_pos]  # [k_pos]

    classes = patch.position_classes(run)  # ctx_pre / number / ctx_post / question / answer_end
    ctx_pre = [p for p in classes["ctx_pre"] if p != 0]  # drop BOS
    out = {
        "bos": float(pattern[0]),
        "number": float(pattern[list(classes["number"])].sum()),
        "context_other": float(pattern[ctx_pre + list(classes["ctx_post"])].sum()) if (ctx_pre or classes["ctx_post"]) else 0.0,
        "question": float(pattern[list(classes["question"])].sum()) if classes["question"] else 0.0,
        "answer_end": float(pattern[list(classes["answer_end"])].sum()) if classes["answer_end"] else 0.0,
    }
    return out


def _logit_diff_direction(model, run: patch.PairRun) -> torch.Tensor:
    return model.W_U[:, run.id_context] - model.W_U[:, run.id_prior]  # [d_model]


@torch.no_grad()
def head_dla(model, run: patch.PairRun, heads: list[Head]) -> dict[Head, float]:
    """Direct logit attribution of each head to LD = logit(context) −
    logit(prior), at the final answer position, with the correct final-LN
    scaling. Validated elsewhere against a full residual decomposition."""
    _, cache = model.run_with_cache(run.tokens_clean)
    direction = _logit_diff_direction(model, run)
    scale = cache["ln_final.hook_scale"][0, run.final_pos]  # [1]
    out = {}
    for (L, h) in heads:
        z = cache[f"blocks.{L}.attn.hook_z"][0, run.final_pos, h]  # [d_head]
        resid = z @ model.W_O[L, h]  # [d_model]
        resid = resid - resid.mean()  # LayerNorm centering
        out[(L, h)] = float((resid / scale) @ direction)
    return out


@torch.no_grad()
def head_dla_directional(model, run: patch.PairRun, heads: list[Head]) -> dict[Head, dict[str, float]]:
    """DLA of each head onto the PRIOR-answer and CONTEXT-answer directions
    *separately* (Phase 3, task 1a). A PH3-style memory head should write
    positively toward the prior token; a context/mover head toward the
    context token.

    Single-token logit projections carry an arbitrary common component
    (softmax shift-invariance), so each unembed column is centered against
    the vocab mean: dir_t = W_U[:,t] − mean_t' W_U[:,t']. 'Writes toward
    prior' therefore means 'raises the prior token relative to the average
    token.' The difference dla_context − dla_prior is invariant to this
    centering and equals head_dla's diff-DLA (used as an internal check)."""
    _, cache = model.run_with_cache(run.tokens_clean)
    w_u_mean = model.W_U.mean(dim=1)  # [d_model]
    dir_prior = model.W_U[:, run.id_prior] - w_u_mean
    dir_context = model.W_U[:, run.id_context] - w_u_mean
    scale = cache["ln_final.hook_scale"][0, run.final_pos]
    out = {}
    for (L, h) in heads:
        z = cache[f"blocks.{L}.attn.hook_z"][0, run.final_pos, h]
        resid = z @ model.W_O[L, h]
        resid = resid - resid.mean()
        scaled = resid / scale
        out[(L, h)] = {
            "dla_prior": float(scaled @ dir_prior),
            "dla_context": float(scaled @ dir_context),
        }
    return out


@torch.no_grad()
def full_dla_check(model, run: patch.PairRun) -> dict[str, float]:
    """Sanity: sum of per-component DLA (heads + MLPs + embeds) plus the
    unembed bias should reconstruct the actual logit difference. decompose_resid
    covers the residual components; b_U is the additive unembed bias that no
    residual component carries, so it is added explicitly."""
    logits, cache = model.run_with_cache(run.tokens_clean)
    actual = float(logits[0, run.final_pos, run.id_context] - logits[0, run.final_pos, run.id_prior])
    stack, _ = cache.decompose_resid(layer=-1, pos_slice=run.final_pos, return_labels=True)
    stack = cache.apply_ln_to_stack(stack, layer=-1, pos_slice=run.final_pos)
    direction = _logit_diff_direction(model, run)
    b_u_term = float(model.b_U[run.id_context] - model.b_U[run.id_prior])
    reconstructed = float((stack[:, 0, :] @ direction).sum()) + b_u_term
    return {"actual_ld": actual, "reconstructed_ld": reconstructed}


@torch.no_grad()
def ov_copy_score(model, run: patch.PairRun, head: Head, top_k: int = 10) -> dict[str, float]:
    """Weights-only copy test: push the number token's embedding through the
    head's OV circuit and unembed. A copy head boosts that same token's logit.
    Reports its percentile and whether it lands in the top-k of the vocab.

    In conflicting context the stated number IS the context answer token, so
    'copying the number' and 'writing the context answer' are the same event —
    which is exactly the mover mechanism.
    """
    L, h = head
    num_id = run.id_context  # conflicting context states answer_context_token
    resid_in = model.W_E[num_id]  # [d_model]
    ov_out = (resid_in @ model.W_V[L, h]) @ model.W_O[L, h]  # [d_model]
    ov_out = ov_out - ov_out.mean()
    vocab_logits = ov_out @ model.W_U  # [d_vocab]
    target = float(vocab_logits[num_id])
    rank = int((vocab_logits > target).sum())  # 0 = top
    vocab = vocab_logits.shape[0]
    return {
        "target_logit": target,
        "rank": rank,
        "percentile": 100.0 * (1 - rank / vocab),
        "in_top_k": float(rank < top_k),
        "vocab_size": float(vocab),
    }

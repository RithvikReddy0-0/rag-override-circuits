"""Logit-difference readout for a single (model, pair, condition).

The behavioral metric throughout this project is a logit difference, not
accuracy (per RESEARCH_CHARTER.md): for a pair's two single-token answer candidates
(`parametric_answer`, `edited_answer`), we read the model's next-token
logits after "Answer:" and take their difference. Forward passes over a
frozen, pretrained model in eval mode are deterministic (no dropout), so
seeding is about reproducibility bookkeeping, not run-to-run variance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from pairs import Pair, build_prompt

Condition = Literal["supportive", "conflicting", "neutral"]
CONDITIONS: tuple[Condition, ...] = ("supportive", "conflicting", "neutral")


def set_all_seeds(seed: int = 0) -> dict[str, int]:
    """Fix python/numpy/torch seeds and return what was set, for logging."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return {"seed": seed}


@dataclass(frozen=True)
class Measurement:
    pair_id: str
    category: str
    edit_type: str
    has_strong_prior: bool
    model_name: str
    condition: Condition
    prompt: str
    logit_parametric: float
    logit_edited: float
    logit_diff_param_minus_edited: float
    grounded_minus_other: float | None


def _context_for_condition(pair: Pair, condition: Condition) -> str | None:
    if condition == "supportive":
        return pair.context_supportive
    if condition == "conflicting":
        return pair.context_conflicting
    if condition == "neutral":
        return None
    raise ValueError(f"unknown condition: {condition}")


def measure_pair(model, model_name: str, pair: Pair, condition: Condition) -> Measurement:
    context = _context_for_condition(pair, condition)
    prompt = build_prompt(context, pair.question)

    # Single forward pass: both answer logits come from the same prompt, so
    # there is no need to re-run the model per candidate token.
    id_p = model.to_single_token(" " + pair.parametric_answer)
    id_e = model.to_single_token(" " + pair.edited_answer)
    final_logits = model(prompt)[0, -1, :]
    logit_p = final_logits[id_p].item()
    logit_e = final_logits[id_e].item()
    diff = logit_p - logit_e

    if condition == "supportive":
        grounded_minus_other = diff
    elif condition == "conflicting":
        grounded_minus_other = -diff
    else:
        grounded_minus_other = None

    return Measurement(
        pair_id=pair.pair_id,
        category=pair.category,
        edit_type=pair.edit_type,
        has_strong_prior=pair.has_strong_prior,
        model_name=model_name,
        condition=condition,
        prompt=prompt,
        logit_parametric=logit_p,
        logit_edited=logit_e,
        logit_diff_param_minus_edited=diff,
        grounded_minus_other=grounded_minus_other,
    )

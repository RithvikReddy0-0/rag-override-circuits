"""Bare-prior probing for Phase 1 pair construction.

The pilot showed our a-priori "the model believes X" labels were wrong roughly
half the time (4/9 GPT-2, 3/9 Pythia). So Phase 1 anchors every pair's
conflict direction on the MEASURED per-model prior: for each candidate fact we
probe the model with the question alone (no context block) and read the logits
of the two candidate answer tokens. Whichever token wins is that model's
parametric answer; the other becomes the conflicting-context answer.

Every probe result is logged to data/priors/{model}.json regardless of whether
the candidate survives filtering, so the attrition is auditable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pairs import build_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent
PRIORS_DIR = REPO_ROOT / "data" / "priors"

# Inclusion threshold on |bare-prior logit diff| between the two candidate
# tokens. Justification (see phase1_report.md): in the pilot, pairs with no
# real prior showed |neutral diffs| down to 0.01-0.12, i.e. near-tie noise,
# while confirmed genuine priors sat at 0.27-2.73. 0.5 sits above the near-tie
# band: below it, "the model's parametric answer" is not a meaningful label,
# and the supportive/conflicting distinction collapses.
PRIOR_THRESHOLD = 0.5


@dataclass(frozen=True)
class PriorProbe:
    candidate_id: str
    model_name: str
    question: str
    token_a: str
    token_b: str
    logit_a: float
    logit_b: float
    diff_a_minus_b: float
    prior_token: str  # the measured winner
    alt_token: str  # the loser -> stated by conflicting context
    abs_diff: float
    passes_threshold: bool


def probe_prior(
    model, model_name: str, candidate_id: str, question: str, token_a: str, token_b: str
) -> PriorProbe:
    """Read the model's bare preference between two single-token answers,
    with the question alone (neutral prompt, no context block)."""
    prompt = build_prompt(None, question)
    id_a = model.to_single_token(" " + token_a)
    id_b = model.to_single_token(" " + token_b)
    with torch.no_grad():
        final_logits = model(prompt)[0, -1, :]
    logit_a = float(final_logits[id_a])
    logit_b = float(final_logits[id_b])
    diff = logit_a - logit_b
    prior_token, alt_token = (token_a, token_b) if diff >= 0 else (token_b, token_a)
    return PriorProbe(
        candidate_id=candidate_id,
        model_name=model_name,
        question=question,
        token_a=token_a,
        token_b=token_b,
        logit_a=logit_a,
        logit_b=logit_b,
        diff_a_minus_b=diff,
        prior_token=prior_token,
        alt_token=alt_token,
        abs_diff=abs(diff),
        passes_threshold=abs(diff) >= PRIOR_THRESHOLD,
    )


def save_probes(model_name: str, probes: list[PriorProbe], seed: int) -> Path:
    PRIORS_DIR.mkdir(parents=True, exist_ok=True)
    out = PRIORS_DIR / f"{model_name}.json"
    out.write_text(
        json.dumps(
            {
                "model": model_name,
                "seed": seed,
                "prior_threshold": PRIOR_THRESHOLD,
                "n_probes": len(probes),
                "n_passing": sum(p.passes_threshold for p in probes),
                "probes": [asdict(p) for p in probes],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out

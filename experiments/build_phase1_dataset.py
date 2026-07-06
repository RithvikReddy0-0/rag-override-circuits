"""Build the Phase 1 per-model pair sets (Task 1 of Phase 1).

Pipeline per model:
  1. tokenization filter: both candidate values single tokens; the two
     context prompts tokenize to equal length and differ at exactly one
     position (the critical number);
  2. bare-prior probe (src/priors.py) — every probe logged to
     data/priors/{model}.json;
  3. prior filter: |logit diff| >= priors.PRIOR_THRESHOLD;
  4. direction anchoring: measured winner -> answer_prior_token,
     loser -> answer_context_token (stated by the conflicting context);
  5. append the 2 section-number copy-fidelity controls (is_control=True,
     exempt from the prior filter, direction as in the pilot).

Outputs data/pairs/phase1_{model}.jsonl and data/pairs/phase1_build_log.json
(full attrition accounting: generated -> tokenization-filtered ->
prior-filtered -> final).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformer_lens import HookedTransformer  # noqa: E402

from measure import set_all_seeds  # noqa: E402
from pairs import (  # noqa: E402
    PHASE1_CANDIDATES,
    PHASE1_CONTROLS,
    Candidate,
    Phase1Pair,
    build_prompt,
    save_phase1_pairs,
)
from priors import PRIOR_THRESHOLD, probe_prior, save_probes  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAMES = ["gpt2", "pythia-160m"]
SEED = 0


def is_single_token(model, value: str) -> bool:
    try:
        model.to_single_token(" " + value)
        return True
    except AssertionError:
        return False


def tokenization_check(model, cand: Candidate) -> dict:
    """Both values single tokens AND the two full prompts differ at exactly
    one token position. Returns a record with pass/fail and diagnostics."""
    rec: dict = {"candidate_id": cand.candidate_id}
    single_a = is_single_token(model, cand.value_a)
    single_b = is_single_token(model, cand.value_b)
    rec["value_a_single_token"] = single_a
    rec["value_b_single_token"] = single_b
    if not (single_a and single_b):
        rec["passes"] = False
        rec["reason"] = "value not a single token"
        return rec

    prompt_a = build_prompt(cand.context_for(cand.value_a), cand.question)
    prompt_b = build_prompt(cand.context_for(cand.value_b), cand.question)
    toks_a = model.to_tokens(prompt_a)
    toks_b = model.to_tokens(prompt_b)
    rec["len_a"] = int(toks_a.shape[1])
    rec["len_b"] = int(toks_b.shape[1])
    rec["token_length_delta"] = int(toks_a.shape[1] - toks_b.shape[1])
    if toks_a.shape != toks_b.shape:
        rec["passes"] = False
        rec["reason"] = "prompts tokenize to different lengths"
        return rec

    n_diff = int((toks_a != toks_b).sum())
    rec["n_diff_positions"] = n_diff
    if n_diff != 1:
        rec["passes"] = False
        rec["reason"] = f"prompts differ at {n_diff} token positions (need exactly 1)"
        return rec

    rec["passes"] = True
    rec["reason"] = ""
    return rec


def build_for_model(model, model_name: str) -> tuple[list[Phase1Pair], dict]:
    log: dict = {
        "model": model_name,
        "n_candidates_generated": len(PHASE1_CANDIDATES),
        "tokenization_checks": [],
        "n_tokenization_passed": 0,
        "n_prior_passed": 0,
        "prior_threshold": PRIOR_THRESHOLD,
    }

    tok_passed: list[Candidate] = []
    for cand in PHASE1_CANDIDATES:
        rec = tokenization_check(model, cand)
        log["tokenization_checks"].append(rec)
        if rec["passes"]:
            tok_passed.append(cand)
    log["n_tokenization_passed"] = len(tok_passed)

    probes = [
        probe_prior(model, model_name, c.candidate_id, c.question, c.value_a, c.value_b)
        for c in tok_passed
    ]
    save_probes(model_name, probes, SEED)

    pairs: list[Phase1Pair] = []
    for cand, probe in zip(tok_passed, probes):
        if not probe.passes_threshold:
            continue
        pairs.append(
            Phase1Pair(
                pair_id=cand.candidate_id,
                domain=cand.domain,
                template=cand.template,
                question=cand.question,
                context_supportive=cand.context_for(probe.prior_token),
                context_conflicting=cand.context_for(probe.alt_token),
                answer_context_token=probe.alt_token,
                answer_prior_token=probe.prior_token,
                measured_prior_logit_diff=probe.abs_diff,
                is_control=False,
            )
        )
    log["n_prior_passed"] = len(pairs)

    # Controls: exempt from the prior filter; direction fixed as in the pilot
    # (value_a = pilot's nominal answer). Raw measured prior logged honestly —
    # for an arbitrary section number it should be near zero.
    n_ctrl = 0
    for cand in PHASE1_CONTROLS:
        rec = tokenization_check(model, cand)
        log["tokenization_checks"].append(rec)
        if not rec["passes"]:
            continue
        probe = probe_prior(model, model_name, cand.candidate_id, cand.question,
                            cand.value_a, cand.value_b)
        pairs.append(
            Phase1Pair(
                pair_id=cand.candidate_id,
                domain=cand.domain,
                template=cand.template,
                question=cand.question,
                context_supportive=cand.context_for(cand.value_a),
                context_conflicting=cand.context_for(cand.value_b),
                answer_context_token=cand.value_b,
                answer_prior_token=cand.value_a,
                measured_prior_logit_diff=probe.diff_a_minus_b,
                is_control=True,
            )
        )
        n_ctrl += 1
    log["n_controls_included"] = n_ctrl
    log["n_final"] = len(pairs)
    return pairs, log


def main() -> None:
    set_all_seeds(SEED)
    build_log: dict = {"seed": SEED, "models": {}}
    for model_name in MODEL_NAMES:
        print(f"Loading {model_name}...")
        model = HookedTransformer.from_pretrained(model_name, device="cpu")
        model.eval()
        pairs, log = build_for_model(model, model_name)
        path = save_phase1_pairs(pairs, model_name)
        build_log["models"][model_name] = log
        n_clean = sum(not p.is_control for p in pairs)
        print(
            f"{model_name}: {log['n_candidates_generated']} generated -> "
            f"{log['n_tokenization_passed']} tokenization-passed -> "
            f"{log['n_prior_passed']} prior-passed (clean) + "
            f"{log['n_controls_included']} controls = {log['n_final']} total -> {path}"
        )
        if n_clean < 30:
            print(f"  WARNING: only {n_clean} clean pairs (< 30 target) for {model_name}")
        del model

    out = REPO_ROOT / "data" / "pairs" / "phase1_build_log.json"
    out.write_text(json.dumps(build_log, indent=2), encoding="utf-8")
    print(f"Wrote build log to {out}")


if __name__ == "__main__":
    main()

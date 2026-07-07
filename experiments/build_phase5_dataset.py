"""Build the fresh Phase 5 pair sets for the pre-registered tide test.

Runs only AFTER results/phase5_preregistration.md was committed (539355a).
Pipeline identical to Phase 1 (tokenization filter -> bare-prior probe ->
prior filter -> direction anchoring), applied to PHASE5_CANDIDATES, which are
fact-disjoint from all Phases 0-4 pairs. No controls in this set. The
behavioral denominator gate (>= 1.0) is applied downstream by prepare_pair
in experiments/phase5.py, exactly as pre-registered.

Outputs data/pairs/phase5_{model}.jsonl, data/priors/phase5_{model}.json,
and data/pairs/phase5_build_log.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformer_lens import HookedTransformer  # noqa: E402

from measure import set_all_seeds  # noqa: E402
from pairs import (  # noqa: E402
    PHASE1_CANDIDATES,
    PHASE1_CONTROLS,
    PHASE5_CANDIDATES,
    Phase1Pair,
    candidate_to_pair,
    phase5_pairs_path,
)
from priors import PRIOR_THRESHOLD, probe_prior, PRIORS_DIR  # noqa: E402
from build_phase1_dataset import tokenization_check  # noqa: E402
from dataclasses import asdict  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAMES = ["gpt2", "pythia-160m"]
SEED = 0


def check_disjoint() -> None:
    old_ids = {c.candidate_id for c in PHASE1_CANDIDATES} | {c.candidate_id for c in PHASE1_CONTROLS}
    new_ids = {c.candidate_id for c in PHASE5_CANDIDATES}
    assert not (old_ids & new_ids), f"id overlap: {old_ids & new_ids}"
    assert len(new_ids) == len(PHASE5_CANDIDATES), "duplicate phase5 ids"


def main() -> None:
    set_all_seeds(SEED)
    check_disjoint()
    build_log: dict = {"seed": SEED, "preregistration_commit": "539355a564c11694d9103ac979e687e1e9fd9a00",
                       "n_candidates": len(PHASE5_CANDIDATES), "models": {}}
    for model_name in MODEL_NAMES:
        print(f"Loading {model_name}...", flush=True)
        model = HookedTransformer.from_pretrained(model_name, device="cpu")
        model.eval()

        tok_checks, tok_passed = [], []
        for cand in PHASE5_CANDIDATES:
            rec = tokenization_check(model, cand)
            tok_checks.append(rec)
            if rec["passes"]:
                tok_passed.append(cand)

        probes = [probe_prior(model, model_name, c.candidate_id, c.question, c.value_a, c.value_b)
                  for c in tok_passed]
        PRIORS_DIR.mkdir(parents=True, exist_ok=True)
        (PRIORS_DIR / f"phase5_{model_name}.json").write_text(
            json.dumps({"model": model_name, "seed": SEED, "prior_threshold": PRIOR_THRESHOLD,
                        "probes": [asdict(p) for p in probes]}, indent=2), encoding="utf-8")

        pairs: list[Phase1Pair] = []
        for cand, probe in zip(tok_passed, probes):
            if not probe.passes_threshold:
                continue
            pairs.append(candidate_to_pair(cand, probe.prior_token, probe.alt_token,
                                           probe.abs_diff, is_control=False))

        path = phase5_pairs_path(model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(asdict(pair)) + "\n")

        build_log["models"][model_name] = {
            "n_tokenization_passed": len(tok_passed),
            "n_prior_passed": len(pairs),
            "tokenization_checks": tok_checks,
        }
        print(f"{model_name}: {len(PHASE5_CANDIDATES)} generated -> {len(tok_passed)} "
              f"tokenization-passed -> {len(pairs)} prior-passed -> {path}", flush=True)
        del model

    (REPO_ROOT / "data" / "pairs" / "phase5_build_log.json").write_text(
        json.dumps(build_log, indent=2), encoding="utf-8")
    print("Wrote build log.", flush=True)


if __name__ == "__main__":
    main()

# rag-override-circuits

Mechanistic-interpretability study of **where, inside a small language model, the
decision to trust retrieved context vs. fall back on parametric memory lives** —
and whether "defer to good evidence" and "resist corrupted evidence" are the same
circuit or different ones. Not a RAG product, not a chatbot, not a benchmark:
the deliverable is a paper (target: BlackboxNLP / arXiv).

**The one claim under test:** in small transformers, context-override behavior in
a retrieval setting is driven by localizable components, and we can causally
intervene on them to predictably change faithfulness to retrieved evidence.

## Status: Phase 1 (activation patching) complete — awaiting human sign-off before Phase 2

**Phase 0 verdict** (signed off): context-vs-parametric conflict is measurable
at this scale for numeric-fact edits (8/8 pairs, both models, ≈ −3.8-logit
shifts; negation showed no signal and was dropped). Details:
[results/pilot_report.md](results/pilot_report.md).

**Phase 1 verdict**: context-override is **sharply localized** — and the
circuit is **shared** between easy-override and prior-resists pairs.

- ~30 numeric minimal pairs per model, conflict direction anchored on the
  *measured* per-model prior (a-priori labels were wrong ~half the time in
  the pilot). Full attrition logged.
- Activation patching (both canonical directions, denoising + noising) shows
  the context number's information stays at its own token position until
  ~¾ network depth, then hops directly to the answer position via **2–3
  attention heads**: GPT-2 **L9H9 (0.47) + L10H2 (0.37)** ≈ 84% of the
  effect; Pythia **L8H10 (0.51) + L10H7 (0.25) + L8H2 (0.15)** ≈ 92%.
  Bootstrap CIs tight; bottom-half heads at 0.0002 (negative floor).
- **Central question**: the same heads dominate whether the prior is weak or
  strong (GPT-2 overlap@10 = 90–100%, Spearman 0.75–0.79) — no dissociable
  "resistance" circuit at head granularity; the prior's resistance lives in
  the baseline balance, not in different components.
- Both models place the mover heads at relative depth 0.67–0.83 — a
  convergent functional motif across architectures.
- Full numbers, figures, and the hand-written critical assessment (including
  a selection-effect caveat, uncharacterized negative heads, and a disclosed
  budget-mechanism bug): [results/phase1_report.md](results/phase1_report.md).

## Known limitations (also stated in the reports)

- ~30 pairs/model, one prompt template, single-token answers: claims are about
  this task format at this scale.
- Patched activations are off-manifold inputs (standard activation-patching
  caveat); head interpretations ("mover heads") are hypotheses until Phase 2
  ablation + attribution.
- The ≥1-logit behavioral-effect gate excludes the most prior-dominated pairs
  (a selection effect on the shared-circuit claim).
- Pythia's copy-fidelity controls could not be patched — it barely copies
  section numbers behaviorally.
- Two-candidate logit diff ignores the rest of the vocabulary.

## Reproduction (CPU-only, deterministic)

```bash
py -3.11 -m venv .venv                    # Python 3.11
.venv/Scripts/pip install -e .            # torch (CPU), transformer-lens, numpy, pandas, matplotlib
.venv/Scripts/python -m pairs             # validates tokenization, writes data/pairs/pairs.jsonl
.venv/Scripts/python experiments/pilot.py # ~2 min on CPU; writes results/pilot_*.{json,md,png}

# Phase 1 (order matters):
.venv/Scripts/python experiments/build_phase1_dataset.py  # priors + filtered pair sets (~2 min)
.venv/Scripts/python experiments/phase1.py                # patching grid (~3.5 h CPU; prints budget first)
.venv/Scripts/python experiments/phase1_report.py         # stats + figures + report from cache (~1 min)
```

For a CPU-only torch wheel, install torch first from the PyTorch CPU index:
`pip install "torch>=2.2,<3.0" --index-url https://download.pytorch.org/whl/cpu`.

All seeds fixed (seed 0) and logged; forward passes over frozen models in eval
mode are deterministic, so re-runs reproduce the reported numbers exactly.

## Layout

```
src/pairs.py                        minimal-pair construction + Phase 1 candidates
src/measure.py                      logit-diff readout
src/priors.py                       measured bare-prior probe (Phase 1 anchoring)
src/patch.py                        activation patching machinery (batched, both directions)
experiments/pilot.py                Phase 0 gate
experiments/build_phase1_dataset.py Phase 1 dataset build (tokenization/prior filters)
experiments/phase1.py               Phase 1 patching grid (compute only)
experiments/phase1_report.py        Phase 1 stats/figures/report (from cache)
data/pairs/, data/priors/           pair sets (JSONL) + logged prior probes
results/                            reports, raw effects (csv.gz), summaries, figures
IDEAS.md                            parked ideas (explicitly not scheduled work)
```

Phase 2 modules (`src/attribution.py`, `src/ablate.py`) do not exist yet — by
design, until the Phase 1 verdict is signed off.

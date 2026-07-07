# rag-override-circuits

Mechanistic-interpretability study of **where, inside a small language model, the
decision to trust retrieved context vs. fall back on parametric memory lives** —
and whether "defer to good evidence" and "resist corrupted evidence" are the same
circuit or different ones. Not a RAG product, not a chatbot, not a benchmark:
the deliverable is a paper (target: BlackboxNLP / arXiv).

**The one claim under test:** in small transformers, context-override behavior in
a retrieval setting is driven by localizable components, and we can causally
intervene on them to predictably change faithfulness to retrieved evidence.

## Status: Phase 4 (resistance-pathway decomposition) complete — awaiting sign-off; claim wording finalized pending approval

**Phase 4 verdict**: the Phase 3 "MLP/residual-side" clause is **refuted** —
MLPs write both answers massively but net-lean toward *context*, and ablating
the top prior-writing MLP layers *hurts* context-following (−0.46; control ≈ 0).
Embeddings are negligible. The prior-tracking signal instead lives in
**attention**: non-mover attention net lean scales with prior strength
(Pythia ρ = +0.46, perm p = 0.008; GPT-2 +0.32, p = 0.089), flips to the prior
side exactly on context-loses pairs (attn −0.66→+0.16 while MLPs stay
context-leaning), and Pythia's movers attend the number *less* on strong
priors (ρ = −0.35, p = 0.051) — resistance partly as attenuation. No memory
heads, but a memory *tide* across many sub-threshold attention heads. This
analysis is post-hoc (the task's own predictions failed first) and flagged as
such. Final proposed claim wording: [results/phase4_report.md](results/phase4_report.md).



**Phase 3 verdict** (vs Jin et al. 2024's PH3 memory/context heads): **partially**.
The mover half is now triply confirmed (prior-invariance: Spearman +0.07/+0.02
vs measured prior, a third independent test). The hoped-for "distinct
prior-tracking resistance population" **fails on every prediction**: the
negative heads suppress *both* answers (never write the prior), their effect
*anti*-scales with prior strength (Pythia ρ = −0.38, p = 0.035), it vanishes
when the movers are ablated first (reactive, not independent), and on pairs
where context loses, the resistance comes from the diffuse "rest" of the
network (−0.3..−1.6) — not from any head. A global 144-head scan finds the
memory-head region empty in both models (max prior-lean +0.10/+0.14 vs mover
context-lean −0.80/−0.64). Reworded central claim: context-faithfulness is
head-localized; parametric resistance is NOT head-localized at 124M–160M — the
PH3 memory-head motif has no counterpart at this scale. Details:
[results/phase3_report.md](results/phase3_report.md).



**Phase 0 verdict** (signed off): context-vs-parametric conflict is measurable
at this scale for numeric-fact edits (8/8 pairs, both models, ≈ −3.8-logit
shifts; negation showed no signal and was dropped). Details:
[results/pilot_report.md](results/pilot_report.md).

**Phase 2 verdict** (validation): Phase 1's localization holds up under the
harder test. Mean-ablating the mover heads collapses context-following
(GPT-2 0.97→0.27, Pythia 0.75→0.09; random-head controls at the noise floor);
the heads mechanically attend to the retrieved number and write/copy it (DLA
all positive, top OV percentiles 93–99.9); a separate small set of heads writes
*against* context (ablating them raises context-following); and the shared-
circuit result survives — denominator-normalized ablation removes a flat
~40–47% of the context effect across prior-strength terciles (weak-vs-strong
Cohen's d ≈ 0, perm p ≈ 0.86–0.89). Both Phase 1 loose ends closed: Pythia
copy-control is now available (5/8 new copy pairs pass behaviorally — it copies
numbers, just not "Section N"), and on the extreme-prior pairs the same mover
heads carry the residual signal. Honest qualifier: the heads dominate the
*head-level* effect but are ~40% of the *total* context pathway, and the
negative heads are mechanistically plural. Full detail + hand-written
assessment: [results/phase2_report.md](results/phase2_report.md).

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
- Patched/ablated activations are off-manifold inputs (standard caveat); Phase 2
  ablation + attribution now back the "mover head" reading with necessity and
  mechanism evidence, but the heads are ~40% of the *total* context pathway,
  not all of it.
- The ≥1-logit behavioral-effect gate excludes the most prior-dominated pairs;
  the Phase 2 selection probe shows the same heads still carry the residual
  signal there (n = 2, suggestive).
- Two-candidate logit diff ignores the rest of the vocabulary.
- Ablation is at the final answer position only; tercile symmetry test is
  low-powered (n ≈ 10/tercile).

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

# Phase 2 (uses Phase 1 pair sets + cached head effects):
.venv/Scripts/python experiments/phase2.py                # ablation + attribution + loose ends (~11 min CPU)
.venv/Scripts/python experiments/phase2_report.py         # stats + figures + report from cache (~1 min)

# Phase 3 (memory-head reconciliation vs PH3):
.venv/Scripts/python experiments/phase3.py                # directional DLA + global scan + joint ablation (~8 min CPU)
.venv/Scripts/python experiments/phase3_report.py         # stats + figures + report from cache (~2 min)

# Phase 4 (resistance-pathway decomposition):
.venv/Scripts/python experiments/phase4.py                # component DLA + MLP-layer ablation (~11 min CPU)
.venv/Scripts/python experiments/phase4_report.py         # stats + figures + report from cache (~2 min)
```

For a CPU-only torch wheel, install torch first from the PyTorch CPU index:
`pip install "torch>=2.2,<3.0" --index-url https://download.pytorch.org/whl/cpu`.

All seeds fixed (seed 0) and logged; forward passes over frozen models in eval
mode are deterministic, so re-runs reproduce the reported numbers exactly.

## Layout

```
src/pairs.py                        minimal-pair construction + Phase 1/2 candidates
src/measure.py                      logit-diff readout
src/priors.py                       measured bare-prior probe (Phase 1 anchoring)
src/patch.py                        activation patching machinery (batched, both directions)
src/ablate.py                       mean-ablation validation machinery (Phase 2)
src/attribution.py                  attention patterns, DLA, OV copy score (Phase 2)
experiments/pilot.py                Phase 0 gate
experiments/build_phase1_dataset.py Phase 1 dataset build (tokenization/prior filters)
experiments/phase1.py               Phase 1 patching grid (compute only)
experiments/phase1_report.py        Phase 1 stats/figures/report (from cache)
experiments/phase2.py               Phase 2 ablation + attribution + loose ends (compute)
experiments/phase2_report.py        Phase 2 stats/figures/report (from cache)
experiments/phase3.py               Phase 3 directional DLA + global scan + joint ablation
experiments/phase3_report.py        Phase 3 stats/figures/report (from cache)
experiments/phase4.py               Phase 4 component decomposition + MLP ablation
experiments/phase4_report.py        Phase 4 stats/figures/report (from cache)
data/pairs/, data/priors/           pair sets (JSONL) + logged prior probes
results/                            reports, raw effects (csv.gz), summaries, figures
IDEAS.md                            parked ideas (explicitly not scheduled work)
```

Phases 0–4 are complete. Paper writing (and the two logged pre-paper items:
confirmatory test of the attention-tide scaling on fresh pairs; ranking the
non-mover heads' contribution to the tide) has not started — by design, until
the Phase 4 verdict is signed off.

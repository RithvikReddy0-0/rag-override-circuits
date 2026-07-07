# Research Charter — Project Context

## What this is
A mechanistic-interpretability research project. We are **not** building a legal chatbot or a RAG product. We are studying, causally, **where inside a small language model the decision to trust retrieved context vs. fall back on parametric memory lives** — and whether "defer to good evidence" and "resist corrupted evidence" are the same circuit or different ones. The deliverable is a rigorous paper (target: BlackboxNLP / arXiv), not a demo.

## The one claim (everything serves this — resist scope creep)
> In small transformers, context-override behavior in a retrieval setting is driven by localizable components, and we can causally intervene on them to predictably change faithfulness to retrieved evidence.

If a task does not directly strengthen or test THIS claim, it does not belong in this repo. No product features. No chatbot UI. No benchmark-building for its own sake. No expanding to five phenomena.

## Hard constraints (do not paper over these)
- **CPU only.** No GPU. Every design choice must run on CPU in reasonable time. GPT-2-small activation patching on CPU is seconds-per-pair — fine. Anything requiring GPU is out.
- **Models:** GPT-2-small (124M) and Pythia-160M, both via TransformerLens. These are the field-standard mech-interp stack. Do not reach for larger models.
- **Honesty over polish.** Every limitation goes in the README and the paper explicitly. We never dress up a weak result as strong. If the pilot fails, we say so and rethink — we do not fabricate or cherry-pick.

## VALIDATION-FIRST — the most important rule
Do **not** build the full experimental pipeline before the pilot passes. The entire project rests on one unverified assumption: *that a small model shows measurable context-vs-parametric conflict at all.* If it doesn't, the plan changes.

**Phase 0 (pilot) must complete and be reviewed by a human before Phase 1 begins.** The pilot's job is to answer one yes/no question with data: is the conflict behavior measurable? Build nothing beyond the pilot until that's answered.

## Repo structure
```
.
├── RESEARCH_CHARTER.md       # this file
├── README.md                 # research question + reproduction (stub until results exist)
├── pyproject.toml            # deps: transformer-lens, torch (cpu), numpy, pandas, matplotlib
├── src/
│   ├── pairs.py              # minimal-pair construction (same Q, minimal context edit)
│   ├── measure.py            # logit-diff readout (grounded-answer logit − parametric-answer logit)
│   ├── patch.py              # activation patching / causal tracing (Phase 1+)
│   ├── attribution.py        # direct logit attribution, attention analysis (Phase 1+)
│   └── ablate.py             # head ablation validation (Phase 1+)
├── experiments/
│   └── pilot.py              # Phase 0: ~15 hand-built pairs → logit-diff → verdict report
├── data/
│   └── pairs/                # generated/curated minimal pairs (JSONL)
├── results/                  # generated figures + metric tables (git-ignored if large)
└── paper/                    # LaTeX (later)
```

## Coding conventions
- Python 3.11+. Type hints. Small, testable functions.
- Every experiment script writes a machine-readable results file (JSON) AND a human-readable summary (markdown) to `results/`.
- Seeds fixed and logged everywhere. Reproducibility is a paper requirement, not a nicety.
- No notebook-only logic — anything that matters lives in `src/` and is importable/testable.
- Prefer `logit difference` as the behavioral metric (field standard, continuous, sensitive) — NOT accuracy.

## Minimal-pair design (the experimental backbone)
Each pair = one legal question + retrieved context, in two versions:
- **Supportive:** context correctly answers the question.
- **Conflicting:** context minimally edited so the correct-per-context answer flips (e.g., notice period "fifteen days" ↔ "six months"; swapped section number; negated clause).
Structure mirrors IOI-style interpretability datasets: minimal edits, controlled, single-fact deltas. Start with facts the model plausibly has an opinion on, so a real conflict exists.

## Key references (read before writing the paper; position against these)
- Heimersheim & Nanda — "How to use and interpret activation patching" (methods bible)
- Feb 2026 "Attention Deficits… Causal Explanations for Procedural Hallucinations" (activation patching localizes hallucination; general-domain — we do legal RAG specifically)
- The **knowledge-conflict** literature (context vs. parametric memory) — MUST position against this, not just hallucination-benchmark papers. Search and read before framing.
- Legal-RAG landscape (LegalBench-RAG, LRAGE, IL-TUR, Falkor-IRAC, IndicLegalQA) — establishes that legal-RAG *evaluation* is solved; our contribution is *mechanistic*, not evaluative.

## What NOT to do
- Do not build a chatbot, API, or UI.
- Do not construct a large benchmark as a contribution — a small controlled minimal-pair set is the point.
- Do not add SAEs / larger models / extra phenomena "because they'd be cool." Log the idea in `IDEAS.md` and move on.
- Do not proceed past Phase 0 without human sign-off on the pilot verdict.
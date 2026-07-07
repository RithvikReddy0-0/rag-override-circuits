# IDEAS.md — logged and parked, per RESEARCH_CHARTER.md ("log the idea and move on")

Nothing in this file is scheduled work. Each entry exists so a thought is not
lost, not so it gets built.

## Redesigning the negation pairs (logged 2026-07-06, Phase 0 pilot)

The pilot's negation pairs produced no signal, confounded two ways: (1) small
models are known to be weakly sensitive to negation, and (2) the Yes/No readout
showed a uniform "No" bias — logit(Yes) − logit(No) was negative in every
condition, every pair, both models, so the readout may be dominated by
token-level preferences rather than comprehension. If negation is ever revisited:

- **Continuation-style readout instead of Q&A:** have the answer token directly
  complete the clause under test ("...the tenant is" → ` permitted` /
  ` prohibited`) so the measurement rides the model's language-modeling
  competence instead of a Yes/No QA format it was never trained for.
- **Bias-corrected diff:** subtract each pair's neutral-condition diff from the
  context-condition diffs, so a constant token-frequency bias cancels.
- Note: even then, negation may simply be beyond 124M–160M models; a null here
  would be a (publishable) boundary condition, not a failure to fix.

## Other parked thoughts

- Prompt-template robustness sweep (paraphrases, reordered Context/Question
  blocks) — worth one small experiment early in Phase 1 to confirm the numeric
  signal is not template-specific, but not a benchmark.
- SAEs, larger models, additional phenomena: explicitly out of scope per
  RESEARCH_CHARTER.md. Not elaborated here on purpose.

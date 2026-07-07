# Phase 5 Pre-registration: the Attention-Tide Hypothesis on Fresh Data

**Written and frozen 2026-07-07, BEFORE any Phase 5 fresh pair was generated,
inspected, or measured.** This file's substantive content is fixed at the git
commit recorded in its final line; any later edit other than appending that
line invalidates the pre-registration.

## Background being tested (from Phase 4, exploratory)

Phase 4 found, post-hoc, that the net lean of NON-mover attention toward the
parametric answer correlates with measured prior strength (Pythia-160M
Spearman ρ = +0.46, perm p = 0.008; GPT-2-small ρ = +0.32, p = 0.089), and
flips to the prior side on pairs where context loses. Because that analysis
was forced by the data after the phase's own predictions failed, it is
exploratory and must survive a frozen test on fresh data to count as a
finding.

## Frozen definitions (identical code paths to Phase 4)

- **Prompt template:** `Context: {context}\n\nQuestion: {question}\nAnswer:`;
  conflicting condition (context states the anti-prior answer).
- **Gates (identical to Phase 1):** both answers single tokens in the model's
  vocabulary; the two prompts tokenize to equal length differing at exactly
  one position; measured bare prior |logit diff| ≥ 0.5; behavioral
  denominator (LD_clean − LD_corr) ≥ 1.0.
- **Mover heads (frozen):** GPT-2-small {L9H9, L10H2}; Pythia-160M
  {L8H10, L10H7, L8H2}.
- **Test quantity:** non-mover attention lean = Σ over the 12 attention-layer
  outputs of (DLA→prior − DLA→context) at the final answer position
  (vocab-centered unembed directions, LN-scaled, via decompose_resid), MINUS
  the movers' lean (= −Σ mover diff-DLA from hook_z at the final position).
  Exactly the Phase 4 computation.
- **Prior strength:** |bare-prior logit diff| from the neutral probe
  (src/priors.py), measured per model.
- Seed 0 everywhere; 20,000 permutations; deterministic.

## PRIMARY pre-registered test (one per model)

On all fresh pairs passing the gates:

> **H1 (Pythia-160M):** Spearman ρ(prior strength, non-mover attention lean)
> ≥ +0.30, one-sided permutation p < 0.05.
>
> **H1' (GPT-2-small):** same statistic, same thresholds. Stated separately
> because Phase 4's GPT-2 estimate was weaker (+0.32, p = 0.089) and its
> context-loses regime is thin; our confidence in H1' is lower, but the
> thresholds are identical and frozen.

**Validity floor (primary):** ≥ 40 gate-passing fresh pairs per model. Below
the floor, the primary test for that model is INVALID (reported as not run,
not as refuted or confirmed).

## SECONDARY pre-registered test (one per model)

On the fresh context-loses subset (gate-passing pairs with LD_clean < 0):

> **H2:** the non-mover attention lean is positive (prior side) in a majority
> of context-loses pairs, exact one-sided binomial (sign) test against 0.5,
> p < 0.05.

**Validity floor (secondary):** ≥ 20 context-loses pairs for that model.
Below the floor, H2 is INVALID for that model. (Phase 4 had n = 8 Pythia /
n = 1 GPT-2 — the fresh set is generated to target this regime, but if the
floor is still unmet, that is reported as a validity failure, full stop.)

## Frozen decision rule (per model, per test)

- **CONFIRMED:** meets the stated threshold and p-criterion.
- **REFUTED:** primary: ρ < +0.10 (the Phase 4 estimate failed to replicate
  at even a third of its size); secondary: fraction ≤ 0.5.
- **AMBIGUOUS:** everything else (e.g., +0.10 ≤ ρ < +0.30, or ρ ≥ +0.30 with
  p ≥ 0.05).

No other statistic, subset, transformation, or threshold counts toward
confirmation. Any additional analysis of the fresh data goes in a section
explicitly labeled "exploratory, not pre-registered" and cannot upgrade the
verdict. If the result is REFUTED or AMBIGUOUS, the paper reports the tide as
exploratory-only or drops it; the paper proceeds on the mover circuit and the
scale-boundary claim regardless.

## Fresh-data requirements (frozen before generation)

- Candidate facts must be DISJOINT from every pair used in Phases 0–4 (no
  reworded duplicates of the same fact).
- Construction rules identical to Phase 1 (templated numeric_swap, retrieval
  framing, measured-prior anchoring, per-model gating).
- The set is generated targeting strong-prior facts likely to produce
  context-loses cases; generation happens only AFTER this file is committed.

---
Frozen at commit: (to be appended immediately after the freezing commit)

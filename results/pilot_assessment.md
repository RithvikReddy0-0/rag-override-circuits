### Verdict: YES — context-vs-parametric conflict is measurable at this scale, but only for numeric-fact edits. The negation pairs produced no signal and are unusable as designed.

Phase 1 is justified on a **scoped-down design (numeric swaps only)**, pending human
sign-off per RESEARCH_CHARTER.md's VALIDATION-FIRST rule. Nothing below has been built yet.

#### What the data shows

1. **Numeric swaps (8 pairs/model): decisive.** Every pair shifted in the expected
   direction in both models (8/8, sign-test p = 0.0078 — the floor for n = 8). Mean
   shift: −3.87 logits (GPT-2), −3.70 (Pythia-160M). A single-number edit in the
   context moves the answer preference by ~4 logits.

2. **It is genuine conflict, not just copying.** Two observations:
   - *Residual prior drag:* across the 8 numeric pairs, the correlation between the
     measured bare prior (`raw_diff_neutral`) and the preference remaining under
     conflicting context (`raw_diff_conflicting`) is r ≈ 0.95 in both models
     (Pearson 0.954 / 0.945; Spearman 0.905 / 0.976). If the models simply copied
     whatever number the context supplied, the conflicting-condition values would sit
     at a uniform negative level regardless of prior; instead they track the prior
     almost monotonically.
   - *Failed overrides where the prior is strongest:* Pythia `voting_age` still
     prefers the parametric answer **despite** conflicting context
     (raw_diff_conflicting = +0.84: context says 21, model still says 18); GPT-2
     `voting_age` is a dead heat (−0.03); `age_of_majority` lands near zero in both
     models. The tug-of-war exists, and its balance point varies with prior
     strength — exactly the phenomenon Phase 1 aims to localize.

3. **Negation (6 pairs/model): no signal.** Mean |shift| = 0.06 (GPT-2) / 0.12
   (Pythia), max 0.35, several pairs wrong-signed — 10–40× smaller than the numeric
   effects and indistinguishable from noise (sign-test p = 0.69 both models).
   Worse, logit(Yes) − logit(No) is negative in *every* condition for *every*
   negation pair in *both* models — a uniform "No" bias suggesting the Yes/No
   readout is dominated by token-level preferences, layered on top of small models'
   well-known insensitivity to negation. Two confounded problems; unusable as designed.

4. **Section numbers (2 pairs): directionally right, underpowered.** Both shifted as
   expected in both models, but with n = 2 (p = 0.5) and a large asymmetry (GPT-2
   −2.06 vs Pythia −0.20) they support no conclusion. Retain as copy-fidelity
   controls only.

5. **Our prior labels were wrong about half the time.** The neutral (no-context)
   probe favored the hypothesized `parametric_answer` in only 4/9 (GPT-2) and 3/9
   (Pythia) of the pairs we marked `has_strong_prior=True` — e.g. GPT-2's bare
   preference for the drinking-age question is 18, not 21. Parametric direction must
   be *measured per model*, not assumed from world knowledge.

#### Honest caveats

- n = 8 usable pairs — sufficient for a go/no-go gate, not for any published claim.
- Single prompt template; no paraphrase/format robustness checks yet.
- The neutral probe changes the prompt format (drops the Context block), so it is an
  imperfect prior measurement.
- The two-candidate logit diff ignores the rest of the vocabulary; the model's actual
  argmax token may be neither candidate.
- The supportive-vs-conflicting shift alone cannot distinguish *semantic integration
  of evidence* from *shallow copying of any nearby number*. This does not block
  Phase 1 — either mechanism is a context-driven signal to trace, and separating them
  is precisely what activation patching and attention analysis are for — but the
  "trusts retrieved evidence" interpretation must wait for that evidence.

#### Recommended Phase 1 scope (pending human sign-off)

- **Proceed with `numeric_swap` pairs only.** Of RESEARCH_CHARTER.md's mitigation menu
  (facts-the-model-knows / in-context priming / fine-tune), the data selects
  **facts-the-model-knows**: anchor each pair's parametric direction on the measured
  neutral logit-diff *per model*, and prefer facts with large |neutral| (the
  `voting_age` / `age_of_majority` class) so a real conflict exists. No fine-tuning
  or priming is needed — the numeric signal is already large.
- Expand from 8 to ~30–50 numeric pairs by templating the validated schemas
  (swap values/entity names) — for patching statistics, **not** as a benchmark
  contribution.
- Drop negation pairs from Phase 1 (redesign idea logged in IDEAS.md).
- Keep the two `section_number` pairs as copy-fidelity controls.
- Practicality confirmed: the full 96-forward-pass grid ran in ~1 minute on CPU, so
  per-pair activation patching at GPT-2-small scale is comfortably feasible.

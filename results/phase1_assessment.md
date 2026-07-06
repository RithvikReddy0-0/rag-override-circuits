### Verdict: context-override is sharply localized — 2–3 late attention heads per model carry most of the causal effect — and the circuit is SHARED between "context wins easily" and "prior resists" pairs. The central question has a clear answer at this granularity: no dissociable resistance circuit.

Written by hand after inspecting the cached numbers and figures; not generated
by the analysis script. Phase 2 (ablation validation + attribution) has NOT
been run — everything below is patching evidence only, and the head
interpretations are hypotheses until Phase 2 tests them.

#### The mechanism the data shows

1. **The context number stays put, then hops.** Full-position residual
   patching shows the critical number's information sitting at its own
   context position from L0 through ~L9 (recovery 0.96–1.00), then appearing
   at the final answer position in the last layers (GPT-2: answer_end 0.59 at
   L10 → 0.94 at L11; Pythia: 0.73 at L9 → 1.00 at L11). No intermediate
   positions are ever implicated — a direct attention hop, not a relay chain.

2. **The hop is attention, and it is 2–3 heads.** Coarse patching puts the
   movement in attn_out (GPT-2: L9 = 0.57, L10 = 0.35 vs MLP ≤ 0.10 past L0;
   Pythia: L8 = 0.70, L10 = 0.32). Head-level patching resolves it further:
   - **GPT-2: L9H9 = 0.47 [0.44, 0.51] and L10H2 = 0.37 [0.33, 0.40]** —
     together ~0.84 of the full effect; the 3rd-ranked head is 7× smaller.
   - **Pythia: L8H10 = 0.51 [0.46, 0.57], L10H7 = 0.25, L8H2 = 0.15** —
     top-3 ≈ 0.92 of the effect.
   The L0-MLP effect (0.69 / 0.95) is token-identity encoding at the number
   position, expected and not circuit-specific.

3. **Denoising ≈ noising for the top heads** (e.g. L9H9: 0.47 vs 0.45).
   Each top head is individually both ~sufficient and ~necessary for its
   share of the effect; single-head patching shows no backup-head
   compensation of the kind IOI found for name movers — though patching one
   head at a time cannot rule out joint redundancy among untested subsets.

4. **Central question (2d): SHARED.** Splitting pairs by measured prior
   strength (bottom vs top tercile): the same heads top both rankings in both
   models — GPT-2 Spearman 0.75/0.79 (denoise/noise), overlap@10 = 90%/100%;
   Pythia 0.45/0.56 with overlap@5 = 80%. Pythia's lower full-rank Spearman
   is a floor artifact: most of its 144 heads have ~0 effect, and ranks
   within that cloud are noise; the scatter shows the meaningful heads on the
   diagonal. **We found no head that matters for override when the prior is
   weak but not when it is strong, or vice versa.** The prior's resistance
   shows up in the *baselines* (how far LD_clean sits from the context
   answer), not in *which components* carry the context signal. "Defer to
   evidence" and "resist corrupted evidence" — operationalized this way —
   are one circuit with a shifting balance, not two circuits.

5. **Cross-model convergence by functional depth.** Both models place their
   mover heads at relative depth 0.67–0.83 (L9–10 of 12; L8/L10 of 12), with
   near-identical depth profiles (L0 encoding spike, flat middle, late bump;
   profile Spearman 0.60/0.63 — moderate because the flat middle contributes
   rank noise). Head *identities* cannot correspond across architectures;
   the functional motif does.

#### Sanity checks

- Negative floor: 20 bottom-half heads → mean effect 0.0002 (max 0.0006). Clean.
- GPT-2 copy controls: number-position recovery ≈ 1.01 vs 1.00 for clean
  pairs — section-number copying behaves like the numeric task, as expected.
- **Pythia's controls could not be patched**: their behavioral effect is
  near-absent (denominators 0.19/0.216 < 1.0 gate), replicating the pilot's
  finding that Pythia barely copies section numbers. The copy-control check
  is therefore *unavailable* for Pythia, not passed.

#### Honest problems and surprises

- **The denominator gate creates a selection effect on the central
  question.** Pythia's two excluded non-control pairs (triangle_sides,
  earth_moons) are precisely ultra-entrenched priors where context barely
  moves the model. The "shared circuit" claim is established on pairs where
  the context edit produces ≥1 logit of behavioral movement; it may not
  extend to the extreme-resistance regime we excluded. GPT-2 had no
  exclusions, which limits (but does not eliminate) the concern.
- **Negative heads exist and are uncharacterized.** Pythia L9 attn_out has a
  consistent negative effect (−0.20); GPT-2 L10H7 ≈ −0.1. Patching *clean*
  activations into these components *hurts* override recovery —
  suppression-like dynamics we have not explained. Phase 2 material.
- **The compute-budget reduction was a no-op due to a scoring bug** (noising
  rows entered the layer-flag score as raw recovery ≈ 1). Consequence: full
  coverage, nothing subsampled, wall-clock 206 min vs the ~2h budget. Fixed
  in code post-run; disclosed in the report.
- Top-head effects are slightly smaller in the strong-prior tercile (e.g.
  L8H10: 0.55 weak vs 0.46 strong) — same components, marginally smaller
  normalized share when the prior fights back. CIs overlap; do not over-read.

#### Standing limitations

Head z-patches applied at all positions (not position-restricted); one prompt
template; single-token answers; ~30 pairs/model; 2 controls; normalized
metric hides absolute logit scales; Pythia's parallel attn/MLP blocks make
the attn-vs-MLP split architecturally non-identical to GPT-2's.

#### What Phase 2 should test (pending sign-off — nothing built)

1. **Ablation validation** (ablate.py): knock out L9H9/L10H2 (GPT-2) and
   L8H10/L10H7/L8H2 (Pythia) on fresh prompts; the localization claim
   predicts a large, specific drop in context-following.
2. **Attribution** (attribution.py): attention patterns (do these heads
   attend final→number?) and direct logit attribution (do their OV outputs
   write the context-answer direction?) to test the "mover head" reading.
3. Characterize the negative/suppression heads.
4. If budget allows: path patching L9H9→L10H2 composition (GPT-2).

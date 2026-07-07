### Verdict: Phase 1's localization is VALIDATED. The mover heads are causally necessary (ablation collapses context-following), they mechanically attend to the number and write/copy it (attribution), the shared-circuit result survives an independent method, and both disclosed loose ends are resolved. The one honest qualification: these heads dominate the *head-level* effect but are ~40% of the *total* context pathway, and the opposing "negative" heads are real but not a single clean motif.

Hand-written after inspecting the numbers and figures; not produced by the
analysis script. Paper writing has NOT started — that awaits sign-off.

#### 1. Ablation confirms necessity, and it agrees with Phase 1's patching

Mean-ablating the mover heads (final answer position) collapses context-
following:
- **GPT-2 L9H9+L10H2:** context-following 0.97 → **0.27**, ΔLD = −1.47 [−1.62, −1.31].
- **Pythia L8H10+L10H7+L8H2:** 0.75 → **0.09**, ΔLD = −1.53 [−1.73, −1.33].

Individual heads give partial drops (GPT-2 L9H9 → 0.73, L10H2 → 0.83); five
random near-zero heads sit at the noise floor (|ΔLD| < 0.03, context-following
unchanged). Denoising (Phase 1, restore signal) and ablation (Phase 2, remove
it) point to the same heads by opposite interventions — the validation the
phase was for.

**Specificity check via the supportive condition:** where context agrees with
the prior, ablating the movers moves the logits *toward the prior* (+ΔLD) but
does NOT break context-following (stays ~1.0) — because the prior IS the
context answer there. The ablation ΔLD flips sign across conditions
(conflicting −, supportive +), which is the signature of a head that writes
*the token the context states*, not a fixed token.

#### 2. Mechanism: these are attend-to-number, copy-the-number heads

Every mover, queried at the answer position, attends to the critical number
token and writes the (context − prior) direction:

| head | attn→number | DLA | OV percentile |
|---|---|---|---|
| GPT-2 L9H9 | 0.50 | +0.80 | 77 |
| GPT-2 L10H2 | 0.39 | +0.59 | **99.9** |
| Pythia L8H10 | 0.65 | +0.64 | 93 |
| Pythia L10H7 | **0.99** | +0.36 | **99.4** |
| Pythia L8H2 | 0.73 | +0.20 | 93 |

DLA reconstruction was validated exactly (max error ~5e-6: summed component
DLA + unembed bias reproduces the true logit diff). So "mover head" is now
mechanistic, not metaphor: attend to the number, copy it to its own logit.
**Honest nuance:** GPT-2 L9H9 has the largest DLA but only a 77th-percentile
OV — it writes the direction strongly without being a clean token-copier, so
it is better read as a "signal writer" than a literal copier; L10H2 and
Pythia L10H7 are the clean copiers (top-10 of vocab on 57% / 28% of pairs).

#### 3. Negative heads: real opposing "context-dampeners," but heterogeneous

Ablating them *raises* context-following (GPT-2 L10H7 ΔLD +0.13; Pythia
L9H1+L9H5 +0.33), and their DLA is negative — they write *against* the
context answer. Confirmed as genuine. But they do not reduce to one clean
motif:
- **Pythia L9H1** attends ~0.9 to the *answer position itself* (essentially
  zero to the number) — the classic suppression-head signature (read the
  running prediction, write against it); OV ~neutral (50th pct).
- **Pythia L9H5** and **GPT-2 L10H7** are *anti-copiers* (OV 11th / 27th
  percentile — their OV maps the number token *away* from its own logit), but
  via different attention (L9H5 reads the number; L10H7 reads the
  question/context, only 0.14 on the number).

So the honest label is "a small set of heads that dampen the context signal by
at least two distinct mechanisms (prediction-suppression and number-anti-
copying)." Not fully unified; I am not going to force one story onto three
heads that behave differently.

#### 4. Combination = mild synergy, not redundancy

Combined-over-sum ΔLD ratio: GPT-2 1.07, Pythia 1.18 (>1). Ablating the movers
together removes slightly *more* than the sum of individual ablations — no
backup/compensation, consistent with Phase 1's denoise≈noise finding.

#### 5. The shared-circuit claim survives an independent method

Denominator-normalized (fraction of the full context effect removed), mover
ablation removes a flat share across prior-strength terciles:
- GPT-2 weak/mid/strong = **0.39 / 0.43 / 0.39** (weak-vs-strong Cohen's d = 0.08, perm p = 0.86)
- Pythia = **0.44 / 0.47 / 0.44** (d = −0.06, perm p = 0.89)

Ablation independently reproduces Phase 1's "no dissociable resistance
circuit": the same heads carry the context signal whether the prior is weak or
strong; prior resistance lives in the baseline, not in different components.
**Caveat, stated plainly:** n ≈ 10 per tercile, so the high permutation p is
*absence of evidence for a difference*, not proof of equality — the test is
low-powered. The effect-size estimates (d ≈ 0) are the stronger signal here.

**A second honest qualification (this is important):** mover ablation removes
only ~40–47% of the *total* context effect, not ~100%. The 2–3 heads dominate
the *head-level* effect (Phase 1: top heads were ~0.84–0.92 of the head-patch
effect), but the full context-to-answer pathway also runs through MLPs and the
direct residual contributions Phase 1 saw. "Sharply localized among attention
heads" is true; "these heads ARE the whole override mechanism" is not. The
paper must not overclaim this.

#### 6. Loose ends — both closed

**(3a) Pythia copy-control is now available.** 5/8 new copy-control pairs
(access code, gate, aisle, platform, exhibit — arbitrary identifiers with no
prior) pass the behavioral copy gate for Pythia (denominator ≥ 1); GPT-2 passes
8/8. So Pythia *can* copy numbers from context — it simply doesn't represent
"Section N," which is why the Phase 1 section-number controls failed. The
copy-fidelity control is no longer unavailable; it was a stimulus problem, not
a model limitation. **Nuance:** mover-head ablation on the copy controls is
*mixed* (GPT-2 e.g. −1.6, −1.2, −1.1 but also −0.12, +0.04; Pythia one at
−1.37, rest small), so arbitrary-identifier copying partly routes through
*other* heads too — the movers are the dominant number-mover for the prior-
conflict task, less exclusively so for generic copying.

**(3b) Selection effect addressed.** On the extreme-prior pairs Phase 1
excluded (Pythia triangle_sides denom 0.76, earth_moons 0.89), the *same*
mover heads carry the residual context signal — notably Pythia L10H7 recovers
0.72 (triangle) and 0.39 (earth_moons). The shared circuit extends into the
extreme-resistance regime; the exclusion was about behavioral magnitude, not a
different mechanism. Suggestive only (n = 2), but it points the right way
rather than against.

#### Standing limitations (carried + new)

- Ablation at the final position only (matches the DLA readout; may understate
  effects routed through other positions).
- Movers are ~40% of the total context effect, not the whole pathway (§5).
- GPT-2 L9H9 is a writer, not a clean copier; negative heads are heterogeneous.
- Copy-control mover-ablation is mixed (§6).
- n ≈ 30 pairs/model, one template, single-token answers; tercile n ≈ 10
  (low-powered null); selection probe n = 2.
- Mean-ablation is against the dataset mean; OV score is a weights-only,
  embedding-in approximation. Pythia head indices are not comparable to GPT-2.

#### Bottom line

The Phase 1 story holds up under the harder test. There is a small, late,
cross-model-convergent set of attention heads that read the retrieved number
and write it into the answer; deleting them collapses faithfulness to context;
a separate small set of heads pushes the other way; and the same machinery
operates whether the model's prior is weak or strong. The result is a genuine,
validated mechanistic finding suitable for the paper — provided we state, in
the same breath, that these heads dominate but do not exhaust the context
pathway, and that the negative heads are plural in mechanism.

#### If a Phase 3 is authorized (nothing built; logged, not started)

1. Path patching L9H9→L10H2 (GPT-2) to test whether the writer feeds the
   copier or they act in parallel.
2. Account for the other ~55% of the context effect (MLP / residual DLA).
3. Robustness across prompt templates and multi-token answers (the standing
   external-validity gaps).

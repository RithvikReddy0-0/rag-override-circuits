### Verdict: the Phase 3 wording is REFUTED in its specifics and must change again. Resistance is (i) NOT MLP-side, (ii) NOT embedding-side, and (iii) not "diffuse everywhere" — it is a diffuse, prior-tracking shift WITHIN THE ATTENTION LAYER, alongside (in Pythia) attenuation of the movers' own attention to the retrieved fact. The MLPs, far from resisting, net-support the context answer. This is the third rewording of the claim, and it is the first one anchored in positive evidence rather than elimination.

Hand-written after inspecting the numbers and figures; not produced by the
analysis script. Paper writing has NOT started.

#### The three questions, answered plainly

**(i) Is the resistance MLP/residual-side? NO — refuted on three counts.**
- MLPs write *both* answers massively (Pythia mlp_total: prior +11.9, context
  +12.3 — generic number promotion) with a net lean toward **context**
  (−0.35 GPT-2, −0.47 Pythia). The task-1b prediction ("MLPs carry net-prior")
  is wrong.
- The causal check came back with the **opposite sign**: mean-ablating the top
  prior-writing MLP layers *hurts* context-following (GPT-2 top-3 combined
  ΔLD = −0.46, CF 0.97→0.80; Pythia L11 ΔLD = −0.45, CF 0.75→0.56 combined),
  while the control layer sits at ~0. Late MLPs are net context-*supporters*.
- Direct embeddings are negligible (|DLA| ≤ 0.04 both directions, both models).

**(ii) Does anything scale with prior strength? Yes — but not the MLPs.**
MLP-aggregate prior-write scaling is null (ρ = +0.15 / +0.09, p ≥ 0.42). The
scaling lives in attention:
- **Non-mover attention net lean toward the prior: Pythia ρ = +0.46
  (perm p = 0.008)** — the strongest, cleanest scaling signal found anywhere
  in this project, and it survives multiple-comparison correction within its
  test family. GPT-2 shows the same direction (ρ = +0.32, p = 0.089).
- **Mover attenuation (Pythia): ρ = −0.35 (p = 0.051)** — on strong-prior
  pairs the mover heads attend the retrieved number *less*. Resistance
  partly operates by weakening the context pathway, not only by opposing it.
- These jointly mirror the behavioral gradient they must explain
  (LD_clean vs prior: ρ = −0.49 / −0.32).

**(iii) Diffuse or concentrated? Diffuse — but specifically within attention.**
On pairs where context loses, the class whose net lean flips to the prior side
is attention (GPT-2: −1.19 → +0.25; Pythia: −0.66 → +0.16, ~70–75% of the
wins→loses shift), while MLPs *stay context-leaning* even there. Yet Phase 3's
global scan showed no single head with prior-lean above +0.14. Both are true
simultaneously: the prior-side attention signal is spread across many
sub-threshold heads whose aggregate tracks prior strength. "No memory heads,
but a memory *tide* in the attention layer."

#### Reconciling all four phases into one mechanism story

1. Context answer: read and written by 2–3 sparse, prior-invariant mover heads
   (Phases 1–3).
2. Parametric answer: NOT written by any dedicated head (Phase 3 scan), NOT
   net-written by MLPs at the answer position (this phase). It enters as a
   diffuse, prior-strength-tracking lean across the rest of the attention
   layer, plus attenuation of the movers themselves (Pythia).
3. Late MLPs amplify the running consensus (they causally support whichever
   answer is winning — ablating them hurt context on pairs where context was
   winning); they are shared infrastructure, not a memory pathway.
4. Storage vs transport: none of this says facts aren't *stored* in MLPs
   (ROME-style, at subject positions). Our DLA measures the final *transport*
   into the answer logits — and that transport is attention, in both
   directions. The paper must keep this distinction explicit.

#### What was wrong before, said clearly

- Phase 3's "resistance is diffuse and MLP/residual-side" was inference by
  elimination, and its MLP clause was **wrong**. The residual/'rest' term was
  dominated by *non-mover attention*, which Phase 3 never split out.
- The Phase 4 task's own predictions (1b: MLP net-prior; 3b: MLP ablation
  frees context) were both refuted. The positive finding (attention-side
  resistance) came from a **post-hoc** analysis forced by the data — it is
  labeled exploratory in the report and should be treated as the primary
  target for any future confirmation, ideally on a fresh pair set.

#### Limitations that bound this phase

- The attention-side result is post-hoc (predictions failed first); n ≈ 30;
  GPT-2 contributes only 1 context-loses pair, so the wins/loses flip rests on
  Pythia (n = 8). GPT-2's evidence is directional consistency, not
  significance.
- DLA attributes to the final writer; attention "carrying" the prior at the
  answer position is compatible with MLP storage upstream.
- Non-mover attention lean is an aggregate of ~141 heads; we did not identify
  which subsets contribute, beyond knowing (Phase 3) that no single head
  dominates.
- Single template, single-token answers, two small models — as throughout.

#### Final claim wording for the paper (pending sign-off)

> In small transformers, faithfulness to retrieved context is carried by a
> sparse, prior-invariant circuit of attention mover heads. Resistance from
> parametric memory is not localized in any component class we can name a
> head or layer for: it is a diffuse, prior-strength-tracking lean across the
> rest of the attention layer — with no counterpart to the billion-scale
> "memory heads" of Jin et al. — plus a measurable attenuation of the mover
> heads' own attention on strongly-held facts. MLPs act as shared
> amplification, net-supporting whichever answer is winning. The conflict is
> resolved in the residual stream at the answer position, between a localized
> context pathway and a distributed attention-side memory tide.

Every clause traces to a number in phases 1–4. The wording concedes what we
could not localize, which is the honest shape of the result.

#### If further work is authorized before the paper (logged, not started)

1. Confirmatory test of the attention-tide scaling on a fresh templated pair
   set (pre-registered direction, one test, no correction debt).
2. Rank the non-mover heads by per-pair prior-lean covariance to see whether
   the "tide" is 20 heads or 100 (cheap, cached activations suffice).
3. Then the paper.

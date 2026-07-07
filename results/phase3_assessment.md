### Verdict: PARTIALLY. The mover half of the claim is now triply confirmed — but the hoped-for second half ("a distinct prior-tracking resistance population") FAILS. Our negative heads are NOT PH3-style memory heads, and no memory-head population exists anywhere in either model. At 124M–160M, prior resistance is diffuse, not head-localized. The paper's contribution must be reworded accordingly — and the reworded claim is, if anything, more interesting.

Hand-written after inspecting the numbers and figures; not produced by the
analysis script. The paper has NOT been started — wording awaits sign-off.

#### What the hoped-for claim was, and what survived

Claim as posed: "shared mover circuit + distinct prior-tracking resistance
population." Test outcome, piece by piece:

| prediction (if negatives = PH3 memory heads) | result | verdict |
|---|---|---|
| negative heads write TOWARD the prior (1a) | dla_prior all *negative* (GPT-2 L10H7 −0.11; Pythia L9H1 −0.06, L9H5 −0.05); they suppress BOTH answers, context more | **fails** |
| negative-head effect scales UP with prior strength (1c/2b) | flat-to-ANTI-scaling: GPT-2 ρ = −0.19 (p=0.31); Pythia ρ = **−0.38 (p=0.035)**; terciles decline weak→strong | **fails, opposite sign** |
| independent population (2c) | interaction ≈ −(negatives' whole effect) in Pythia (−0.35): ablate movers first and the negatives' benefit *vanishes*; GPT-2 ~60% dependent (−0.08) | **fails: reactive, mover-dependent** |
| negatives supply the resistance where context loses (3a) | movers still write +0.2..+1.4 toward context; negatives contribute −0.25..+0.08; **the "rest" (MLPs/other/direct) supplies −0.28..−1.60** — the actual resistance | **fails: resistance is diffuse** |
| some heads, somewhere, are memory heads (global scan) | strongest prior-lean anywhere: +0.10 (GPT-2 L8H8) / +0.14 (Pythia) vs mover context-lean −0.80/−0.64 — a 6–10× asymmetry; the memory-head region of the scan is **empty** | **fails** |

What the negative heads actually are: **reactive context-dampeners** — small
heads that read the accumulating answer (Pythia L9H1 attends ~0.9 to the
answer position) or the number (L9H5, anti-copy OV) and write against the
context answer *in proportion to the mover signal*, not the prior. When the
movers are ablated, they have nothing to dampen and their effect disappears.
Calibration/hedging machinery, not memory recall.

#### What is now solidly established (the mover half, third independent test)

Mover invariance to prior strength was confirmed a third way: normalized
ablation effect vs measured prior, Spearman **+0.07 (GPT-2, p=0.70)** and
**+0.02 (Pythia, p=0.92)**. Combined with Phase 1's rank-overlap and Phase 2's
flat terciles: one context-mover circuit, operating identically whether the
prior is weak or strong. Falsification check 3b: GPT-2 movers *never* write
toward the prior (0/60 pair×head cases); Pythia movers flip on 9/96 cases
(7 pairs, mean −0.055, worst −0.204 on extreme-prior/world facts) — "essentially
invariant, with small exceptions in Pythia" is the honest phrasing.

#### The one trace-level counter-signal (do not bury it)

GPT-2 **L8H8**: prior-lean +0.10 with prior-strength scaling ρ = +0.41,
perm p = 0.026 — the only head in either model whose prior-vs-context writing
tracks prior strength. Caveats: uncorrected p among ~10 candidate tests
(fails Bonferroni), magnitude 8× below the movers, and no detectable patching
or ablation effect. Verdict: at most a trace of a proto-memory-head; worth one
sentence in the paper, not a claim.

#### Reconciliation with PH3 (this is the paper's new positioning)

Jin et al. find both context heads AND memory heads at 1.3B–13B; we find the
context side of their picture (our movers behave exactly like their context
heads: pruning/ablating them collapses context use — theirs 59%→98% memory,
ours CF 0.97→0.27) but **no memory-head counterpart at 124M–160M**. Two
readings, both defensible: (i) **scale-emergence** — dedicated memory heads
are a capability that crystallizes somewhere between 160M and 1.3B, while at
small scale parametric recall stays distributed in MLPs/direct paths (which is
where our "rest" term lives, consistent with the knowledge-in-MLPs
literature); (ii) method difference — PH3's importance score (`<unk>`-corrupted
path patching on P(ctx)−P(mem)) could classify heads as "memory heads" that
our direct-write criterion would not. We cannot fully exclude (ii) without
implementing PH3's exact pipeline, but our Phase 1 bidirectional patching over
all 144 heads found no strong prior-side patch effects either, which covers
much of the same ground. State both readings; claim the boundary, not the
mechanism of the boundary.

#### What the paper's central claim becomes

Old wording (dead): "defer vs resist are one mover circuit *plus a distinct
prior-tracking resistance population*."

New wording (supported): **"In small transformers, faithfulness to retrieved
context is carried by a sparse, prior-invariant mover-head circuit; resistance
from parametric memory is NOT head-localized — it is diffuse and
MLP/residual-side — and the PH3 memory-head motif reported at billion scale
has no counterpart at 124M–160M. The conflict is arbitrated in the residual
stream between a localized context pathway and a distributed memory pathway."**

Every clause is backed: sparse (Phase 1+2), prior-invariant (three tests),
diffuse resistance (3a decomposition), no memory heads (global scan), scale
contrast (PH3 comparison). This is a cleaner and more falsifiable contribution
than the original hope, and it positions directly against both PH3 and the
knowledge-conflict literature.

#### Limitations that bound the claim

- DLA sees only direct writes; indirect prior-promotion through downstream
  components would be invisible (mitigated but not eliminated by Phase 1's
  full-head patching finding nothing on the prior side).
- n ≈ 30/model; the anti-scaling of Pythia's negatives (p=0.035) is one test
  among several — treat as directional evidence, not an anchor.
- GPT-2's context-loses subset is n=1; the 3a decomposition rests mostly on
  Pythia (n=8).
- Scale claim is bounded by two models and one task format; "no memory heads"
  means "none under our criteria at this scale," not a universal.

#### If a Phase 4 / paper-prep is authorized (logged, not started)

1. MLP-side decomposition of the "rest" term (which layers' MLPs write the
   prior?) — one figure, directly supports the "distributed memory" clause.
2. Implement PH3's exact importance score on our pairs to close reading (ii).
3. Then paper writing, with the reworded claim.

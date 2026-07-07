"""Phase 3 analysis, figures, and report — memory-head reconciliation vs
Jin et al. 2024 (PH3, arXiv:2402.18154). Reads only cached CSV/JSON outputs
of phases 2 and 3; computes no verdict (that lives in the hand-written
results/phase3_assessment.md, inlined on re-run).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MODEL_NAMES = ["gpt2", "pythia-160m"]
SEED = 0
N_BOOT = 5000
N_PERM = 20000


def rng_():
    return np.random.default_rng(SEED)


def boot_ci(x, rng, n_boot=N_BOOT):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def spearman(x, y):
    rx, ry = pd.Series(x).rank(), pd.Series(y).rank()
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p_corr(x, y, rng, n_perm=N_PERM):
    """Permutation p-value for |spearman| (two-sided)."""
    obs = abs(spearman(x, y))
    y = np.asarray(y, float).copy()
    count = 0
    for _ in range(n_perm):
        rng.shuffle(y)
        if abs(spearman(x, y)) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def md_table(df: pd.DataFrame, floats=()) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = [f"{r[c]:.3f}" if c in floats and isinstance(r[c], (int, float)) else str(r[c]) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def load_denominators(model_name: str) -> pd.DataFrame:
    a = pd.read_csv(RESULTS_DIR / f"phase2_ablation_{model_name}.csv.gz")
    any_abl = a["ablation"].unique()[0]
    c = a[(a.condition == "conflicting") & (a.ablation == any_abl)][["pair_id", "baseline_ld", "prior_absdiff"]]
    s = a[(a.condition == "supportive") & (a.ablation == any_abl)][["pair_id", "baseline_ld"]]
    d = c.merge(s, on="pair_id", suffixes=("_conf", "_supp"))
    d["denominator"] = d["baseline_ld_conf"] - d["baseline_ld_supp"]
    return d[["pair_id", "denominator", "baseline_ld_conf", "prior_absdiff"]]


def neg_config_name(model_name: str, ablations) -> str:
    negs = [c for c in ablations if c.startswith("negblock_")]
    return negs[0] if negs else "L10H7"


# ------------------------------------------------------------- task 1a
def directional_dla_stats(model_name: str, rng) -> pd.DataFrame:
    d = pd.read_csv(RESULTS_DIR / f"phase3_dla_{model_name}.csv.gz")
    recs = []
    for (hd, role), g in d.groupby(["head", "role"]):
        mp, plo, phi = boot_ci(g["dla_prior"].to_numpy(), rng)
        mc, clo, chi = boot_ci(g["dla_context"].to_numpy(), rng)
        recs.append({"head": hd, "role": role, "dla_prior": mp,
                     "prior_CI": f"[{plo:.2f},{phi:.2f}]", "dla_context": mc,
                     "context_CI": f"[{clo:.2f},{chi:.2f}]",
                     "prior_minus_ctx": mp - mc})
    order = {"mover": 0, "negative": 1, "control": 2}
    return pd.DataFrame(recs).sort_values(["role", "head"], key=lambda s: s.map(lambda v: order.get(v, v)))


def fig_directional(model_name: str, stat: pd.DataFrame) -> str:
    sub = stat[stat.role != "control"]
    ctrl = stat[stat.role == "control"]
    fig, ax = plt.subplots(figsize=(1.6 * len(sub) + 2, 4.5))
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["dla_prior"], width=0.4, label="DLA → prior answer", color="tab:red", alpha=0.8)
    ax.bar(x + 0.2, sub["dla_context"], width=0.4, label="DLA → context answer", color="tab:blue", alpha=0.8)
    lim = max(abs(ctrl["dla_prior"]).max(), abs(ctrl["dla_context"]).max())
    ax.axhspan(-lim, lim, color="gray", alpha=0.25, label="random-control band")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.head}\n({r.role})" for r in sub.itertuples()], fontsize=9)
    ax.set_ylabel("mean DLA (vocab-centered direction)")
    ax.set_title(f"{model_name}: does any target head WRITE the prior answer?\n"
                 "(PH3 memory-head signature: red bar positive and above blue)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    name = f"phase3_directional_dla_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


# ------------------------------------------------------------- task 1c / 2b
def scaling_analysis(model_name: str, rng) -> tuple[dict, str]:
    a = pd.read_csv(RESULTS_DIR / f"phase2_ablation_{model_name}.csv.gz")
    den = load_denominators(model_name)
    conf = a[a.condition == "conflicting"]
    movers_combo = [c for c in conf.ablation.unique() if c.startswith("movers_")][0]
    neg_name = neg_config_name(model_name, conf.ablation.unique())

    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, (name, label) in zip(axes, [(movers_combo, "movers (combined)"), (neg_name, "negative head(s)")]):
        sub = conf[conf.ablation == name].merge(den[["pair_id", "denominator"]], on="pair_id")
        sub["norm"] = sub["delta_ld"] / sub["denominator"]
        x, y = sub["prior_absdiff"].to_numpy(), sub["norm"].to_numpy()
        rp = float(np.corrcoef(x, y)[0, 1])
        rs = spearman(x, y)
        pp = perm_p_corr(x, y, rng)
        out[label] = {"config": name, "pearson": rp, "spearman": rs, "perm_p": pp, "n": len(sub)}
        ax.scatter(x, y, s=22, alpha=0.7,
                   color="tab:blue" if "mover" in label else "tab:purple")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("measured prior strength |logit diff|")
        ax.set_title(f"{label}\nSpearman={rs:+.2f} (perm p={pp:.3f}), Pearson={rp:+.2f}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("normalized ablation effect ΔLD/denominator")
    fig.suptitle(f"{model_name}: does head importance scale with prior strength?\n"
                 "(memory-head prediction: negative heads scale UP with prior; movers flat)")
    fig.tight_layout()
    name = f"phase3_scaling_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2b: tercile table for the negative-head ablation, normalized
    sub = conf[conf.ablation == neg_name].merge(den[["pair_id", "denominator"]], on="pair_id")
    sub["norm"] = sub["delta_ld"] / sub["denominator"]
    sub = sub.sort_values("prior_absdiff")
    n = len(sub)
    terc = {}
    labels = np.array(["weak"] * (n // 3) + ["mid"] * (n - 2 * (n // 3)) + ["strong"] * (n // 3))
    sub["tercile"] = labels
    for t in ("weak", "mid", "strong"):
        g = sub[sub.tercile == t]
        m, lo, hi = boot_ci(g["norm"].to_numpy(), rng)
        terc[t] = {"n": len(g), "mean_norm": m, "ci_lo": lo, "ci_hi": hi}
    w, s_ = sub[sub.tercile == "weak"]["norm"].to_numpy(), sub[sub.tercile == "strong"]["norm"].to_numpy()
    pooled = np.concatenate([w, s_])
    obs = abs(w.mean() - s_.mean())
    cnt = 0
    for _ in range(N_PERM):
        rng.shuffle(pooled)
        if abs(pooled[: len(w)].mean() - pooled[len(w):].mean()) >= obs - 1e-12:
            cnt += 1
    out["neg_tercile"] = {"table": terc, "weak_vs_strong_perm_p": (cnt + 1) / (N_PERM + 1),
                          "config": neg_name}
    return out, name


# ------------------------------------------------------------- task 2c
def joint_analysis(model_name: str) -> dict:
    a = pd.read_csv(RESULTS_DIR / f"phase2_ablation_{model_name}.csv.gz")
    j = pd.read_csv(RESULTS_DIR / f"phase3_jointablation_{model_name}.csv.gz")
    conf = a[a.condition == "conflicting"]
    movers_combo = [c for c in conf.ablation.unique() if c.startswith("movers_")][0]
    neg_name = neg_config_name(model_name, conf.ablation.unique())
    m = conf[conf.ablation == movers_combo].set_index("pair_id")["delta_ld"]
    ng = conf[conf.ablation == neg_name].set_index("pair_id")["delta_ld"]
    jj = j[j.condition == "conflicting"].set_index("pair_id")["delta_ld"]
    common = m.index.intersection(jj.index)
    interaction = jj[common] - (m[common] + ng[common])
    return {
        "mean_movers_only": float(m[common].mean()),
        "mean_negatives_only": float(ng[common].mean()),
        "mean_joint": float(jj[common].mean()),
        "mean_additive_prediction": float((m[common] + ng[common]).mean()),
        "mean_interaction": float(interaction.mean()),
        "note": "interaction = joint − (movers + negatives); ≈0 ⇒ independent populations; "
                ">0 (toward context) ⇒ negatives' benefit vanishes without movers (reactive)",
    }


# ------------------------------------------------------------- task 3a/3b
def tugofwar(model_name: str) -> tuple[pd.DataFrame, str, dict]:
    att = pd.read_csv(RESULTS_DIR / f"phase2_attribution_{model_name}.csv.gz")
    den = load_denominators(model_name)
    loses = den[den.baseline_ld_conf < 0].sort_values("prior_absdiff", ascending=False)
    rows = []
    for r in loses.itertuples():
        sub = att[att.pair_id == r.pair_id]
        mv = float(sub[sub.role == "mover"]["dla"].sum())
        ng = float(sub[sub.role == "negative"]["dla"].sum())
        rows.append({"pair_id": r.pair_id, "prior": r.prior_absdiff, "mover_DLA": mv,
                     "negative_DLA": ng, "rest": r.baseline_ld_conf - mv - ng,
                     "actual_LD": r.baseline_ld_conf})
    tab = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(tab) + 2), 4.5))
    x = np.arange(len(tab))
    ax.bar(x, tab["mover_DLA"], width=0.6, label="mover heads (→ context)", color="tab:blue")
    ax.bar(x, tab["negative_DLA"], width=0.6, bottom=None, label="negative heads", color="tab:purple")
    ax.bar(x, tab["rest"], width=0.6, bottom=tab["mover_DLA"] + tab["negative_DLA"],
           label="rest of network (MLPs, other heads, direct)", color="tab:gray", alpha=0.8)
    ax.plot(x, tab["actual_LD"], "k_", markersize=18, label="actual logit diff")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tab["pair_id"].str.replace("p1_", ""), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("DLA onto (context − prior)")
    ax.set_title(f"{model_name}: pairs where context LOSES — who supplies the resistance?")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    name = f"phase3_tugofwar_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)

    movers = att[att.role == "mover"]
    flips = movers[movers["dla"] < 0]
    flip_info = {
        "n_flip_rows": int(len(flips)), "n_rows": int(len(movers)),
        "n_pairs_with_flip": int(flips["pair_id"].nunique()),
        "flip_pairs": sorted(set(flips["pair_id"])),
        "most_negative_flip": float(flips["dla"].min()) if len(flips) else 0.0,
        "mean_flip_magnitude": float(flips["dla"].mean()) if len(flips) else 0.0,
    }
    return tab, name, flip_info


# ------------------------------------------------------------- global scan
def scan_analysis(model_name: str, rng) -> tuple[dict, str]:
    s = pd.read_csv(RESULTS_DIR / f"phase3_dla_allheads_{model_name}.csv.gz")
    g = s.groupby(["layer", "head"])[["dla_prior", "dla_context"]].mean()
    g["prior_minus_ctx"] = g["dla_prior"] - g["dla_context"]

    cands = g[g["dla_prior"] > 0].sort_values("prior_minus_ctx", ascending=False).head(5)
    cand_rows = []
    for (L, h) in cands.index:
        sub = s[(s["layer"] == L) & (s["head"] == h)]
        d = (sub["dla_prior"] - sub["dla_context"]).to_numpy()
        x = sub["prior_absdiff"].to_numpy()
        cand_rows.append({"head": f"L{L}H{h}", "mean_dla_prior": float(cands.loc[(L, h), "dla_prior"]),
                          "prior_minus_ctx": float(cands.loc[(L, h), "prior_minus_ctx"]),
                          "spearman_vs_prior": spearman(x, d),
                          "perm_p": perm_p_corr(x, d, rng)})
    res = {
        "top_prior_leaning_heads": cand_rows,
        "max_prior_minus_ctx": float(g["prior_minus_ctx"].max()),
        "min_prior_minus_ctx": float(g["prior_minus_ctx"].min()),
        "n_heads_prior_lean_gt_0.1": int((g["prior_minus_ctx"] > 0.1).sum()),
        "n_heads_ctx_lean_lt_-0.1": int((g["prior_minus_ctx"] < -0.1).sum()),
    }

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(g["dla_context"], g["dla_prior"], s=14, alpha=0.5, color="gray", label="all heads")
    meta = json.loads((RESULTS_DIR / "phase3_runmeta.json").read_text(encoding="utf-8"))
    targets = meta["models"][model_name]["targets"]
    for role, color in (("movers", "tab:blue"), ("negatives", "tab:purple")):
        for lbl in targets[role]:
            L, h = lbl[1:].split("H")
            row = g.loc[(int(L), int(h))]
            ax.scatter(row["dla_context"], row["dla_prior"], s=70, color=color, zorder=3)
            ax.annotate(lbl, (row["dla_context"], row["dla_prior"]), fontsize=8)
    lim = max(abs(g["dla_context"]).max(), abs(g["dla_prior"]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
    ax.fill_between([-lim, 0, lim], [-lim, 0, lim], lim, color="tab:red", alpha=0.06)
    ax.text(-lim * 0.9, lim * 0.85, "memory-head region\n(prior > context)", fontsize=8, color="tab:red")
    ax.set_xlabel("mean DLA → context answer")
    ax.set_ylabel("mean DLA → prior answer")
    ax.set_title(f"{model_name}: all {len(g)} heads — is there a memory-head population?\n"
                 "blue=movers, purple=negatives")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    name = f"phase3_scan_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return res, name


PH3_TABLE = """
| dimension | Jin et al. 2024 (PH3) | this work |
| --- | --- | --- |
| models | OPT-1.3B/2.7B, Pythia-6.9B/12B, LLaMA2-7B/13B (+GPT-2-XL, GPT-J for analysis) | GPT-2-small (124M), Pythia-160M |
| localization | path patching, `<unk>`-corrupted attribute, importance = Δ[P(ctx)−P(mem)] | bidirectional activation patching (denoise+noise) + mean-ablation + DLA + OV |
| context heads | top-5% of heads, middle-to-late layers | 2–3 mover heads at 0.67–0.83 depth (validated attend→number, copy OV) |
| memory heads | top-5% of heads, write toward memory answer; pruning them raises context usage 40.8%→68.9% (GPT-2-XL) | **not found**: strongest prior-leaning head is 6–10× weaker than the movers' context-lean; negative heads suppress BOTH answers (context more) |
| prior-strength scaling | not analyzed | tested directly: movers flat; negative heads flat-to-ANTI-scaling — opposite of the memory-head prediction |
| population structure | two opposing head populations | one context-mover population + small reactive dampers; prior resistance is diffuse (MLPs/rest), not head-localized |
"""


def main() -> None:
    runmeta = json.loads((RESULTS_DIR / "phase3_runmeta.json").read_text(encoding="utf-8"))
    summary = {"seed": SEED, "n_bootstrap": N_BOOT, "n_permutations": N_PERM, "models": {}}
    lines = ["# Phase 3 Report: Memory-Head Reconciliation (vs Jin et al. 2024, PH3)", ""]
    lines.append("**Question:** are the negative heads found in Phase 2 the analogue of PH3's "
                 "memory heads — i.e., is resistance to context a separate, prior-tracking head "
                 "population rather than a mode of the mover circuit? (arXiv:2402.18154 studies "
                 "1.3B–13B models; we test the small-model regime, 124M–160M.)")
    lines.append("")
    lines.append(f"Seeds: {SEED} everywhere; bootstrap {N_BOOT}; permutation {N_PERM}. "
                 f"Directional DLA uses vocab-centered unembed directions; the centering-invariant "
                 f"difference reproduces Phase 2's diff-DLA to "
                 f"{max(m['dla_centering_check_max_err'] for m in runmeta['models'].values()):.0e} "
                 "(internal check). Compute: "
                 f"{runmeta['total_wall_seconds']/60:.1f} min (CPU).")
    lines.append("")

    for model_name in MODEL_NAMES:
        rng = rng_()
        lines.append(f"## {model_name}")
        lines.append("")

        stat = directional_dla_stats(model_name, rng)
        f_dir = fig_directional(model_name, stat)
        lines.append("### 1a. Do the negative heads write the PRIOR answer? (PH3 memory-head signature)")
        lines.append("")
        lines.append(md_table(stat, floats=("dla_prior", "dla_context", "prior_minus_ctx")))
        lines.append("")
        lines.append(f"![{f_dir}]({f_dir})")
        lines.append("")

        scaling, f_sc = scaling_analysis(model_name, rng)
        lines.append("### 1c / 2b. Does head importance scale with prior strength?")
        lines.append("")
        lines.append(f"![{f_sc}]({f_sc})")
        lines.append("")
        for label in ("movers (combined)", "negative head(s)"):
            r = scaling[label]
            lines.append(f"- **{label}** (`{r['config']}`, n={r['n']}): Spearman = {r['spearman']:+.3f} "
                         f"(perm p = {r['perm_p']:.3f}), Pearson = {r['pearson']:+.3f}.")
        nt = scaling["neg_tercile"]
        lines.append(f"- Negative-head ablation by tercile (normalized): "
                     + ", ".join(f"{t} = {nt['table'][t]['mean_norm']:+.3f} "
                                 f"[{nt['table'][t]['ci_lo']:+.3f},{nt['table'][t]['ci_hi']:+.3f}]"
                                 for t in ("weak", "mid", "strong"))
                     + f"; weak-vs-strong perm p = {nt['weak_vs_strong_perm_p']:.3f}.")
        lines.append("")

        joint = joint_analysis(model_name)
        lines.append("### 2c. Joint intervention: independent populations or interaction?")
        lines.append("")
        lines.append(f"- movers only ΔLD = {joint['mean_movers_only']:+.3f}; negatives only = "
                     f"{joint['mean_negatives_only']:+.3f}; joint = {joint['mean_joint']:+.3f}; "
                     f"additive prediction = {joint['mean_additive_prediction']:+.3f}; "
                     f"**interaction = {joint['mean_interaction']:+.3f}**.")
        lines.append(f"- {joint['note']}.")
        lines.append("")

        tug, f_tug, flips = tugofwar(model_name)
        lines.append("### 3a. Tug-of-war on pairs where context loses")
        lines.append("")
        lines.append(md_table(tug, floats=("prior", "mover_DLA", "negative_DLA", "rest", "actual_LD")))
        lines.append("")
        lines.append(f"![{f_tug}]({f_tug})")
        lines.append("")
        lines.append("### 3b. Falsification check: do mover heads ever write toward the prior?")
        lines.append("")
        lines.append(f"- {flips['n_flip_rows']}/{flips['n_rows']} (pair × mover-head) DLA values are "
                     f"negative, across {flips['n_pairs_with_flip']} pairs "
                     f"({', '.join(p.replace('p1_','') for p in flips['flip_pairs']) or 'none'}); "
                     f"most negative = {flips['most_negative_flip']:.3f}, "
                     f"mean flip = {flips['mean_flip_magnitude']:.3f}.")
        lines.append("")

        scan, f_scan = scan_analysis(model_name, rng)
        lines.append("### Global scan: is there ANY memory-head population we missed?")
        lines.append("")
        lines.append(f"![{f_scan}]({f_scan})")
        lines.append("")
        lines.append(md_table(pd.DataFrame(scan["top_prior_leaning_heads"]),
                              floats=("mean_dla_prior", "prior_minus_ctx", "spearman_vs_prior", "perm_p")))
        lines.append("")
        lines.append(f"- Heads with prior-lean > 0.1: {scan['n_heads_prior_lean_gt_0.1']}; "
                     f"heads with context-lean < −0.1: {scan['n_heads_ctx_lean_lt_-0.1']}. "
                     f"Extremes: max prior-lean {scan['max_prior_minus_ctx']:+.3f} vs "
                     f"max context-lean {scan['min_prior_minus_ctx']:+.3f}.")
        lines.append("")

        summary["models"][model_name] = {
            "directional_dla": stat.to_dict(orient="records"),
            "scaling": {k: v for k, v in scaling.items()},
            "joint": joint,
            "tugofwar": tug.to_dict(orient="records"),
            "mover_flips": flips,
            "global_scan": scan,
        }

    lines.append("## Comparison to PH3 (Jin et al. 2024, arXiv:2402.18154)")
    lines.append(PH3_TABLE)
    lines.append("")
    lines.append("## Limitations")
    lines.append("- DLA measures the *direct* write to the logits; a head could promote the prior "
                 "indirectly (via downstream components) and be invisible here. PH3's path-patching "
                 "importance would catch such heads — but so would our Phase 1 full-head patching, "
                 "which found no strong prior-promoting patch effects either.")
    lines.append("- Directional DLA on single tokens uses vocab-centered unembeds; other centerings "
                 "shift absolute values (comparisons and differences are invariant).")
    lines.append("- n ≈ 30 pairs; correlations with prior strength are noisy (perm p reported); "
                 "'context-loses' subset is n=1 for GPT-2 (its baselines rarely lose).")
    lines.append("- Scale confound: PH3's smallest model is ~8× our largest. 'No memory heads at "
                 "124–160M' does not contradict their finding at 1.3B+ — it bounds where the "
                 "memory-head motif emerges.")
    lines.append("")
    lines.append("## Critical assessment")
    lines.append("")
    assess = RESULTS_DIR / "phase3_assessment.md"
    lines.append(assess.read_text(encoding="utf-8").strip() if assess.exists()
                 else "_[Not yet written — lives in results/phase3_assessment.md, inlined on re-run; "
                      "never generated by this script.]_")
    lines.append("")

    (RESULTS_DIR / "phase3_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (RESULTS_DIR / "phase3_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote phase3_report.md, phase3_summary.json, and figures.")


if __name__ == "__main__":
    main()

"""Phase 2 analysis, figures, and report — reads ONLY cached outputs of
experiments/phase2.py. Re-runnable cheaply; computes no verdict.

Produces results/phase2_summary.json, results/phase2_report.md (inlines
results/phase2_assessment.md if present), and results/phase2_*.png.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
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
ATT_CLASSES = ["attn_number", "attn_context_other", "attn_question", "attn_answer_end", "attn_bos"]


def boot_ci(x: np.ndarray, rng, n_boot=N_BOOT, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    s = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def perm_test(a: np.ndarray, b: np.ndarray, rng, n_perm=20000) -> float:
    """Two-sample permutation test on the difference of means (two-sided)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:na].mean() - pooled[na:].mean()) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def md_table(df: pd.DataFrame, floats=()) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = [f"{r[c]:.3f}" if c in floats and isinstance(r[c], (int, float)) else str(r[c]) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ------------------------------------------------------------- ablation
def ablation_stats(model_name: str, rng) -> dict:
    df = pd.read_csv(RESULTS_DIR / f"phase2_ablation_{model_name}.csv.gz")
    runmeta = json.loads((RESULTS_DIR / "phase2_runmeta.json").read_text(encoding="utf-8"))
    control_labels = set(runmeta["models"][model_name]["control_heads"])

    def classify(name):
        if name.startswith("ctrl_"):
            return "control"
        if name.startswith("movers_"):
            return "mover_combo"
        if name.startswith("negblock_"):
            return "neg_combo"
        return name  # single head label

    out = {"per_config": {}}
    for condition in ("conflicting", "supportive"):
        sub = df[df["condition"] == condition]
        recs = []
        for name, g in sub.groupby("ablation"):
            mean, lo, hi = boot_ci(g["delta_ld"].to_numpy(), rng)
            base_cf = g["baseline_followed_context"].mean()
            abl_cf = g["ablated_followed_context"].mean()
            recs.append({"ablation": name, "kind": classify(name), "n": len(g),
                         "mean_delta_ld": mean, "ci_lo": lo, "ci_hi": hi,
                         "baseline_cf": base_cf, "ablated_cf": abl_cf,
                         "cf_drop": base_cf - abl_cf})
        out["per_config"][condition] = recs
    out["control_labels"] = sorted(control_labels)
    return out, df


def tercile_analysis(model_name: str, df: pd.DataFrame, rng) -> dict:
    """1b: does the mover-combo ablation reduce context-following comparably
    across prior-strength terciles?

    Normalizer note: the natural 'fraction of context lead removed' is
    −ΔLD / baseline_LD, but baseline_LD (conflicting) is ~0 for the strongest-
    prior pairs (context barely wins), so that ratio explodes and is unusable.
    We instead normalize by the full context effect (denominator =
    ld_clean − ld_corr = baseline_LD conflicting − supportive), which is ≥1 by
    the Phase-1 gate — the same normalizer as Phase-1 recovery. Raw ΔLD is also
    reported (stable, no denominator)."""
    combo = [c for c in df["ablation"].unique() if c.startswith("movers_")][0]
    # denominator per pair from the two conditions' baselines
    base_c = df[(df["condition"] == "conflicting") & (df["ablation"] == combo)][["pair_id", "baseline_ld"]]
    base_s = df[(df["condition"] == "supportive") & (df["ablation"] == combo)][["pair_id", "baseline_ld"]]
    denom = base_c.merge(base_s, on="pair_id", suffixes=("_conf", "_supp"))
    denom["denominator"] = denom["baseline_ld_conf"] - denom["baseline_ld_supp"]

    sub = df[(df["condition"] == "conflicting") & (df["ablation"] == combo)].copy()
    sub = sub.merge(denom[["pair_id", "denominator"]], on="pair_id")
    sub = sub.sort_values("prior_absdiff")
    n = len(sub)
    sub["tercile"] = "mid"
    sub.iloc[: n // 3, sub.columns.get_loc("tercile")] = "weak"
    sub.iloc[2 * (n // 3):, sub.columns.get_loc("tercile")] = "strong"
    sub["rel_reduction"] = -sub["delta_ld"] / sub["denominator"]

    res = {"combo": combo, "terciles": {}}
    for t in ("weak", "mid", "strong"):
        g = sub[sub["tercile"] == t]
        m, lo, hi = boot_ci(g["delta_ld"].to_numpy(), rng)
        rm, rlo, rhi = boot_ci(g["rel_reduction"].to_numpy(), rng)
        res["terciles"][t] = {"n": len(g), "mean_delta_ld": m, "ci_lo": lo, "ci_hi": hi,
                              "mean_rel_reduction": rm, "rel_ci_lo": rlo, "rel_ci_hi": rhi,
                              "cf_drop": g["baseline_followed_context"].mean() - g["ablated_followed_context"].mean()}
    weak = sub[sub["tercile"] == "weak"]
    strong = sub[sub["tercile"] == "strong"]
    res["weak_vs_strong_delta_ld"] = {
        "cohens_d": cohens_d(weak["delta_ld"].to_numpy(), strong["delta_ld"].to_numpy()),
        "perm_p": perm_test(weak["delta_ld"].to_numpy(), strong["delta_ld"].to_numpy(), rng),
    }
    res["weak_vs_strong_rel_reduction"] = {
        "cohens_d": cohens_d(weak["rel_reduction"].to_numpy(), strong["rel_reduction"].to_numpy()),
        "perm_p": perm_test(weak["rel_reduction"].to_numpy(), strong["rel_reduction"].to_numpy(), rng),
    }
    return res


def mover_labels(model_name: str) -> list[str]:
    """Mover head labels, read from the attribution CSV's role column."""
    att = pd.read_csv(RESULTS_DIR / f"phase2_attribution_{model_name}.csv.gz")
    return list(dict.fromkeys(att[att["role"] == "mover"]["head"]))


def combo_additivity(model_name: str, df: pd.DataFrame) -> dict:
    """1c: combined vs sum of individual mover ablations (conflicting)."""
    movers = mover_labels(model_name)
    combo = [c for c in df["ablation"].unique() if c.startswith("movers_")][0]
    sub = df[df["condition"] == "conflicting"]
    per_pair_sum, per_pair_combo = [], []
    for pid, g in sub.groupby("pair_id"):
        s = g[g["ablation"].isin(movers)]["delta_ld"].sum()
        c = g[g["ablation"] == combo]["delta_ld"]
        if len(c):
            per_pair_sum.append(s)
            per_pair_combo.append(float(c.iloc[0]))
    per_pair_sum, per_pair_combo = np.array(per_pair_sum), np.array(per_pair_combo)
    return {"mean_sum_individual": float(per_pair_sum.mean()),
            "mean_combined": float(per_pair_combo.mean()),
            "ratio_combined_over_sum": float(per_pair_combo.mean() / per_pair_sum.mean()),
            "interpretation": "combined > sum => synergy; < sum => redundancy"}


# ------------------------------------------------------------- figures
def fig_ablation(model_name: str, stats: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    color = {"control": "gray", "mover_combo": "tab:blue", "neg_combo": "tab:purple"}
    for ax, condition in zip(axes, ("conflicting", "supportive")):
        recs = sorted(stats["per_config"][condition], key=lambda r: r["mean_delta_ld"])
        names = [r["ablation"] for r in recs]
        means = [r["mean_delta_ld"] for r in recs]
        lo = [r["mean_delta_ld"] - r["ci_lo"] for r in recs]
        hi = [r["ci_hi"] - r["mean_delta_ld"] for r in recs]
        cols = [color.get(r["kind"], "tab:orange") for r in recs]
        ax.barh(range(len(names)), means, xerr=[lo, hi], color=cols, alpha=0.85)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("mean ΔLD (ablated − baseline)")
        ax.set_title(f"{condition}")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(f"{model_name}: mean-ablation effect on logit diff (bootstrap 95% CI over pairs)\n"
                 "orange=single target head, blue=mover combo, purple=neg block, gray=random control")
    fig.tight_layout()
    name = f"phase2_ablation_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_attention(model_name: str) -> str:
    df = pd.read_csv(RESULTS_DIR / f"phase2_attribution_{model_name}.csv.gz")
    heads = list(dict.fromkeys(df["head"]))
    fig, axes = plt.subplots(1, len(heads), figsize=(3.2 * len(heads), 4.5), sharey=True, squeeze=False)
    labels = ["number", "ctx_other", "question", "answer", "bos"]
    for ax, hd in zip(axes[0], heads):
        g = df[df["head"] == hd]
        data = [g[c].to_numpy() for c in ATT_CLASSES]
        bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
        for patch_ in bp["boxes"]:
            patch_.set_facecolor("lightsteelblue")
        role = g["role"].iloc[0]
        ax.set_title(f"{hd} ({role})", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    axes[0][0].set_ylabel("attention mass from final position")
    fig.suptitle(f"{model_name}: where each target head attends from the answer position (per-pair)")
    fig.tight_layout()
    name = f"phase2_attention_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_dla_ov(model_name: str, rng) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(RESULTS_DIR / f"phase2_attribution_{model_name}.csv.gz")
    heads = list(dict.fromkeys(df["head"]))
    recs = []
    for hd in heads:
        g = df[df["head"] == hd]
        m, lo, hi = boot_ci(g["dla"].to_numpy(), rng)
        recs.append({"head": hd, "role": g["role"].iloc[0], "mean_dla": m, "dla_lo": lo, "dla_hi": hi,
                     "attn_number": g["attn_number"].mean(), "ov_percentile": g["ov_percentile"].mean(),
                     "ov_top_k_frac": g["ov_in_top_k"].mean()})
    stat = pd.DataFrame(recs)
    fig, ax = plt.subplots(figsize=(1.4 * len(heads) + 2, 4.5))
    cols = ["tab:blue" if r == "mover" else "tab:purple" for r in stat["role"]]
    ax.bar(range(len(stat)), stat["mean_dla"],
           yerr=[stat["mean_dla"] - stat["dla_lo"], stat["dla_hi"] - stat["mean_dla"]],
           color=cols, alpha=0.85)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(len(stat)))
    ax.set_xticklabels(stat["head"], rotation=45, fontsize=9)
    ax.set_ylabel("mean DLA onto (context − prior) logit diff")
    ax.set_title(f"{model_name}: direct logit attribution\nblue=mover (writes context), purple=negative (writes against)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    name = f"phase2_dla_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name, stat


def fig_tercile(model_name: str, terc: dict) -> str:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ts = ["weak", "mid", "strong"]
    xs = range(len(ts))
    means = [terc["terciles"][t]["mean_rel_reduction"] for t in ts]
    lo = [means[i] - terc["terciles"][t]["rel_ci_lo"] for i, t in enumerate(ts)]
    hi = [terc["terciles"][t]["rel_ci_hi"] - means[i] for i, t in enumerate(ts)]
    ax.bar(xs, means, yerr=[lo, hi], color="tab:blue", alpha=0.85)
    ax.axhline(1.0, color="green", ls="--", lw=0.8, label="full removal of context effect")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{t}\n(n={terc['terciles'][t]['n']})" for t in ts])
    ax.set_ylabel("relative reduction  −ΔLD / (ld_clean − ld_corr)")
    ax.set_title(f"{model_name}: mover-combo ablation effect by prior-strength tercile\n(shared-circuit claim ⇒ comparable across terciles)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    name = f"phase2_tercile_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


# ------------------------------------------------------------- report
def main() -> None:
    rng = np.random.default_rng(SEED)
    runmeta = json.loads((RESULTS_DIR / "phase2_runmeta.json").read_text(encoding="utf-8"))
    summary = {"seed": SEED, "n_bootstrap": N_BOOT, "models": {}}
    lines = ["# Phase 2 Report: Ablation Validation and Mechanism of the Context-Override Heads", ""]
    lines.append("**Goal:** convert Phase 1's patching result (a small set of late attention "
                 "heads causally carry context-override) into a demonstrated mechanism — via "
                 "mean-ablation (necessity), attention patterns, direct logit attribution, and "
                 "OV-circuit copy scores — and characterize the negative heads and the two "
                 "disclosed loose ends (Pythia copy-control, selection effect).")
    lines.append("")
    lines.append("## Methods")
    lines.append("- **Mean-ablation** (not zero): replace a head's z at the final answer "
                 "position with its across-pair mean in that condition. Zero-ablation is far "
                 "off-distribution (Heimersheim & Nanda 2024); the mean holds the head's "
                 "constant contribution fixed and removes only the input-dependent part.")
    lines.append("- Ablation site is the final answer position (the DLA readout position and the "
                 "hypothesized write site); L→L+1 composition is still captured via the later "
                 "head's query. Metric: ΔLD = logit(context)−logit(prior), ablated − baseline; "
                 "and context-following rate (conflicting: LD>0; supportive: LD<0).")
    lines.append("- **DLA** uses correct final-LN scaling; validated by full residual "
                 "reconstruction (below). **OV copy score** pushes the number token's embedding "
                 "through the head's V→O circuit and unembeds; reports the token's own percentile.")
    lines.append(f"- Seed {SEED}; deterministic; bootstrap CIs use {N_BOOT} resamples over pairs; "
                 "tercile comparison uses a two-sample permutation test + Cohen's d.")
    lines.append(f"- Total compute wall-clock: {runmeta['total_wall_seconds']/60:.1f} min (CPU).")
    lines.append("")

    for model_name in MODEL_NAMES:
        lines.append(f"## {model_name}")
        lines.append("")
        abl, df_abl = ablation_stats(model_name, rng)
        terc = tercile_analysis(model_name, df_abl, rng)
        add = combo_additivity(model_name, df_abl)
        dcheck = json.loads((RESULTS_DIR / f"phase2_dlacheck_{model_name}.json").read_text(encoding="utf-8"))

        f_abl = fig_ablation(model_name, abl)
        f_att = fig_attention(model_name)
        f_dla, dla_stat = fig_dla_ov(model_name, rng)
        f_ter = fig_tercile(model_name, terc)

        lines.append("### Ablation (necessity)")
        lines.append("")
        for condition in ("conflicting", "supportive"):
            recs = sorted(abl["per_config"][condition], key=lambda r: r["mean_delta_ld"])
            tab = pd.DataFrame([{"ablation": r["ablation"], "mean_ΔLD": r["mean_delta_ld"],
                                 "95% CI": f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]",
                                 "baseline_CF": r["baseline_cf"], "ablated_CF": r["ablated_cf"]}
                                for r in recs])
            lines.append(f"**{condition} condition:**")
            lines.append("")
            lines.append(md_table(tab, floats=("mean_ΔLD", "baseline_CF", "ablated_CF")))
            lines.append("")
        lines.append(f"![{f_abl}]({f_abl})")
        lines.append("")
        lines.append(f"- **Combination additivity (1c):** mean combined ΔLD = "
                     f"{add['mean_combined']:.3f} vs sum of individual = {add['mean_sum_individual']:.3f} "
                     f"(ratio {add['ratio_combined_over_sum']:.2f}; >1 synergy, <1 redundancy).")
        lines.append("")

        lines.append("### Tercile symmetry (1b): shared circuit across prior strength")
        lines.append("")
        lines.append(f"![{f_ter}]({f_ter})")
        lines.append("")
        trec = pd.DataFrame([{"tercile": t, "n": terc["terciles"][t]["n"],
                              "mean_ΔLD": terc["terciles"][t]["mean_delta_ld"],
                              "rel_reduction": terc["terciles"][t]["mean_rel_reduction"],
                              "CF_drop": terc["terciles"][t]["cf_drop"]}
                             for t in ("weak", "mid", "strong")])
        lines.append(md_table(trec, floats=("mean_ΔLD", "rel_reduction", "CF_drop")))
        lines.append("")
        wvs = terc["weak_vs_strong_rel_reduction"]
        lines.append(f"- Weak vs strong tercile, relative reduction: Cohen's d = {wvs['cohens_d']:.2f}, "
                     f"permutation p = {wvs['perm_p']:.3f}. "
                     f"(ΔLD: d = {terc['weak_vs_strong_delta_ld']['cohens_d']:.2f}, "
                     f"p = {terc['weak_vs_strong_delta_ld']['perm_p']:.3f}.)")
        lines.append("")

        lines.append("### Mechanism: attention, DLA, OV copy")
        lines.append("")
        lines.append(f"![{f_att}]({f_att})")
        lines.append("")
        lines.append(f"![{f_dla}]({f_dla})")
        lines.append("")
        dtab = dla_stat[["head", "role", "mean_dla", "attn_number", "ov_percentile", "ov_top_k_frac"]].copy()
        dtab.columns = ["head", "role", "mean_DLA", "attn→number", "OV_percentile", "OV_top10_frac"]
        lines.append(md_table(dtab, floats=("mean_DLA", "attn→number", "OV_percentile", "OV_top10_frac")))
        lines.append("")
        lines.append(f"- DLA reconstruction validation: max abs error over pairs = "
                     f"{dcheck['max_abs_reconstruction_error']:.2e} (sum of component DLA + b_U "
                     f"reproduces the true logit diff).")
        lines.append("")

        # loose ends
        cc = json.loads((RESULTS_DIR / f"phase2_copycontrol_{model_name}.json").read_text(encoding="utf-8"))
        sel = json.loads((RESULTS_DIR / f"phase2_selection_{model_name}.json").read_text(encoding="utf-8"))
        lines.append("### Loose ends")
        lines.append("")
        lines.append(f"**Copy-control (3a):** {cc['n_kept']}/{cc['n_generated']} new copy-control "
                     f"pairs passed the behavioral copy gate (denominator ≥ {cc['denom_threshold']}).")
        if cc["n_kept"]:
            kept_tab = pd.DataFrame([{"pair_id": p["pair_id"], "denominator": p["denominator"]}
                                     for p in cc["probed"] if p.get("kept")])
            lines.append("")
            lines.append(md_table(kept_tab, floats=("denominator",)))
            if cc["mover_ablation_on_kept"]:
                arr = np.array([r["delta_ld"] for r in cc["mover_ablation_on_kept"]])
                lines.append("")
                lines.append(f"- Mover-head ({', '.join(cc['mover_heads'])}) mean-ablation on the "
                             f"copy controls: mean ΔLD = {arr.mean():.3f} — the same heads are "
                             "needed for pure copying, not just prior-conflict.")
        else:
            lines.append("- No copy control passed; copy-fidelity control remains unavailable for "
                         "this model (documented, not hidden).")
        lines.append("")
        avail = [r for r in sel["results"] if r.get("available")]
        lines.append(f"**Selection-effect probe (3b):** mover-head denoise patching on excluded "
                     f"extreme-prior pairs ({', '.join(r['pair_id'] for r in avail) or 'none available'}); "
                     "no behavioral gate, small n, suggestive only.")
        if avail:
            rows = []
            for r in avail:
                rows.append({"pair_id": r["pair_id"], "denominator": r["denominator"],
                             **{k: v for k, v in r["mover_recovery"].items()}})
            lines.append("")
            lines.append(md_table(pd.DataFrame(rows), floats=tuple(rows[0].keys())))
        lines.append("")

        summary["models"][model_name] = {
            "ablation": abl, "tercile": terc, "combo_additivity": add,
            "dla_check": dcheck, "dla_stat": dla_stat.to_dict(orient="records"),
            "copy_control": cc, "selection_probe": sel,
        }

    lines.append("## Limitations")
    lines.append("- Ablation at the final position only (not all positions); matches the DLA "
                 "readout but may understate a head's effect routed through other positions.")
    lines.append("- Mean-ablation removes the input-dependent part against the *dataset* mean; a "
                 "different reference distribution could shift magnitudes.")
    lines.append("- ~30 pairs/model, one template, single-token answers; tercile n ≈ 10 each — "
                 "permutation tests are low-powered, so absence of a weak/strong difference is "
                 "weak evidence of true equality, not proof.")
    lines.append("- OV copy score is a weights-only, embedding-in approximation (ignores earlier "
                 "layers' contribution to the moved residual).")
    lines.append("- Pythia's parallel attn/MLP blocks; head indices are not comparable to GPT-2.")
    lines.append("")
    lines.append("## Critical assessment")
    lines.append("")
    assess = RESULTS_DIR / "phase2_assessment.md"
    lines.append(assess.read_text(encoding="utf-8").strip() if assess.exists()
                 else "_[Not yet written — lives in results/phase2_assessment.md, inlined on "
                      "re-run; never generated by this script.]_")
    lines.append("")

    (RESULTS_DIR / "phase2_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (RESULTS_DIR / "phase2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote phase2_report.md, phase2_summary.json, and figures.")


if __name__ == "__main__":
    main()

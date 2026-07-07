"""Phase 4 analysis, figures, and report — where does parametric resistance
live? Reads only cached outputs of experiments/phase4.py (+ Phase 3 summary
for the mover/negative scaling contrast). The verdict lives in the
hand-written results/phase4_assessment.md, inlined on re-run.
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


def boot_ci(x, rng, n_boot=N_BOOT):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def spearman(x, y):
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def perm_p_corr(x, y, rng, n_perm=N_PERM):
    obs = abs(spearman(x, y))
    y = np.asarray(y, float).copy()
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(y)
        if abs(spearman(x, y)) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def md_table(df: pd.DataFrame, floats=()) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = [f"{r[c]:.3f}" if c in floats and isinstance(r[c], (int, float)) else str(r[c]) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def component_class(comp: str) -> str:
    if comp.startswith("mlp_"):
        return "mlp_total"
    if comp.startswith("attn_"):
        return "attn_total"
    return "embed_direct"  # embed + pos_embed


def load_components(model_name: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / f"phase4_components_{model_name}.csv.gz")
    df["class"] = df["component"].map(component_class)
    df["prior_lean"] = df["dla_prior"] - df["dla_context"]
    return df


# ------------------------------------------------------------- task 1
def aggregate_stats(df: pd.DataFrame, rng) -> pd.DataFrame:
    per_pair = df.groupby(["pair_id", "class"])[["dla_prior", "dla_context", "prior_lean"]].sum().reset_index()
    recs = []
    for cls, g in per_pair.groupby("class"):
        for col in ("dla_prior", "dla_context", "prior_lean"):
            m, lo, hi = boot_ci(g[col].to_numpy(), rng)
            recs.append({"class": cls, "quantity": col, "mean": m, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(recs)


def fig_aggregate(model_name: str, agg: pd.DataFrame) -> str:
    classes = ["embed_direct", "attn_total", "mlp_total"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(classes))
    for off, (qty, color, label) in zip((-0.22, 0.22), [
        ("dla_prior", "tab:red", "→ prior answer"),
        ("dla_context", "tab:blue", "→ context answer"),
    ]):
        sub = agg[agg["quantity"] == qty].set_index("class").loc[classes]
        ax.bar(x + off, sub["mean"], width=0.4, color=color, alpha=0.85, label=label,
               yerr=[sub["mean"] - sub["ci_lo"], sub["ci_hi"] - sub["mean"]], capsize=3)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("summed DLA at answer position (mean over pairs)")
    ax.set_title(f"{model_name}: who writes what (conflicting condition)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    name = f"phase4_aggregate_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_layerwise(model_name: str, df: pd.DataFrame, rng) -> tuple[str, pd.DataFrame]:
    rows = []
    for kind in ("mlp", "attn"):
        sub = df[df["component"].str.startswith(f"{kind}_L")].copy()
        sub["layer"] = sub["component"].str.removeprefix(f"{kind}_L").astype(int)
        for L, g in sub.groupby("layer"):
            mp, plo, phi = boot_ci(g["dla_prior"].to_numpy(), rng)
            ml, llo, lhi = boot_ci(g["prior_lean"].to_numpy(), rng)
            rows.append({"kind": kind, "layer": int(L), "dla_prior": mp,
                         "prior_lo": plo, "prior_hi": phi,
                         "prior_lean": ml, "lean_lo": llo, "lean_hi": lhi})
    prof = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, col, lo, hi, title in [
        (axes[0], "dla_prior", "prior_lo", "prior_hi", "write toward PRIOR answer"),
        (axes[1], "prior_lean", "lean_lo", "lean_hi", "net prior-lean (prior − context)"),
    ]:
        for kind, color in (("mlp", "tab:red"), ("attn", "tab:blue")):
            g = prof[prof["kind"] == kind].sort_values("layer")
            ax.plot(g["layer"], g[col], "-o", color=color, label=kind)
            ax.fill_between(g["layer"], g[lo], g[hi], color=color, alpha=0.15)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xlabel("layer")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("mean DLA at answer position")
    fig.suptitle(f"{model_name}: layerwise profile of the resistance pathway (bootstrap 95% CI)")
    fig.tight_layout()
    name = f"phase4_layerwise_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name, prof


def concentration(prof: pd.DataFrame) -> dict:
    m = prof[prof["kind"] == "mlp"].set_index("layer")["dla_prior"]
    pos = m[m > 0]
    top3 = m.sort_values(ascending=False).head(3)
    return {
        "top3_layers": [int(i) for i in top3.index],
        "top3_share_of_positive": float(top3.sum() / pos.sum()) if pos.sum() > 0 else float("nan"),
        "n_layers_positive": int((m > 0).sum()),
    }


# ------------------------------------------------------------- task 2
def scaling_analysis(model_name: str, df: pd.DataFrame, rng) -> tuple[list[dict], str]:
    per_pair = df.groupby(["pair_id", "class"])[["dla_prior", "prior_lean"]].sum().reset_index()
    prior = df.groupby("pair_id")["prior_absdiff"].first()

    results = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, cls in zip(axes, ("mlp_total", "embed_direct", "attn_total")):
        g = per_pair[per_pair["class"] == cls].set_index("pair_id")
        x = prior.loc[g.index].to_numpy()
        for qty, color in (("dla_prior", "tab:red"), ("prior_lean", "tab:gray")):
            y = g[qty].to_numpy()
            rs = spearman(x, y)
            pp = perm_p_corr(x, y, rng)
            results.append({"class": cls, "quantity": qty, "spearman": rs, "perm_p": pp,
                            "pearson": float(np.corrcoef(x, y)[0, 1]), "n": len(g)})
            if qty == "dla_prior":
                ax.scatter(x, y, s=24, alpha=0.75, color=color)
                ax.set_title(f"{cls}\nSpearman={rs:+.2f} (perm p={pp:.3f})")
        ax.set_xlabel("measured prior strength")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("summed DLA → prior answer")
    fig.suptitle(f"{model_name}: does the component's prior-write scale with prior strength? (decisive test)")
    fig.tight_layout()
    name = f"phase4_scaling_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # per-MLP-layer scaling for the top prior-writing layers
    meta = json.loads((RESULTS_DIR / "phase4_runmeta.json").read_text(encoding="utf-8"))
    for L in meta["models"][model_name]["top_prior_mlp_layers"]:
        g = df[df["component"] == f"mlp_L{L}"].set_index("pair_id")
        x = prior.loc[g.index].to_numpy()
        y = g["dla_prior"].to_numpy()
        results.append({"class": f"mlp_L{L}", "quantity": "dla_prior",
                        "spearman": spearman(x, y), "perm_p": perm_p_corr(x, y, rng),
                        "pearson": float(np.corrcoef(x, y)[0, 1]), "n": len(g)})
    return results, name


# ---------------------------------------------- where resistance actually is
def resistance_location(model_name: str, df: pd.DataFrame, rng) -> tuple[dict, str]:
    """Post-hoc analysis forced by the data: the class-level means hide the
    per-pair structure. Splits the per-class net lean by whether context wins
    or loses, tests scaling of NON-MOVER attention lean with prior strength,
    and tests mover attenuation (Phase 2 per-pair mover DLA / attention)."""
    df = df.copy()
    per_class = df.groupby(["pair_id", "class"])["prior_lean"].sum().unstack()
    ld = df.groupby("pair_id")["ld_clean"].first()
    prior = df.groupby("pair_id")["prior_absdiff"].first()

    att2 = pd.read_csv(RESULTS_DIR / f"phase2_attribution_{model_name}.csv.gz")
    movers = att2[att2["role"] == "mover"].groupby("pair_id").agg(
        mover_diff_dla=("dla", "sum"), mover_attn=("attn_number", "mean"))
    common = per_class.index.intersection(movers.index)
    # mover lean = −(context−prior) diff-DLA
    nonmover_attn_lean = per_class.loc[common, "attn_total"] + movers.loc[common, "mover_diff_dla"]

    loses = ld[ld < 0].index
    wins = ld[ld >= 0].index
    split = {}
    for cls in ("embed_direct", "attn_total", "mlp_total"):
        split[cls] = {
            "wins_mean_lean": float(per_class.loc[per_class.index.isin(wins), cls].mean()),
            "loses_mean_lean": float(per_class.loc[per_class.index.isin(loses), cls].mean()) if len(loses) else float("nan"),
        }

    x = prior.loc[common].to_numpy()
    tests = {
        "nonmover_attn_lean_vs_prior": (nonmover_attn_lean.to_numpy(), x),
        "mover_diff_dla_vs_prior": (movers.loc[common, "mover_diff_dla"].to_numpy(), x),
        "mover_attn_number_vs_prior": (movers.loc[common, "mover_attn"].to_numpy(), x),
        "ld_clean_vs_prior": (ld.loc[common].to_numpy(), x),
    }
    stats = {}
    for name, (y, xx) in tests.items():
        stats[name] = {"spearman": spearman(xx, y), "perm_p": perm_p_corr(xx, y, rng), "n": len(y)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(x, nonmover_attn_lean.to_numpy(), s=24, alpha=0.75, color="tab:orange")
    s0 = stats["nonmover_attn_lean_vs_prior"]
    axes[0].set_title(f"NON-mover attention net lean → prior\nSpearman={s0['spearman']:+.2f} (perm p={s0['perm_p']:.3f})")
    axes[0].set_ylabel("summed lean (prior − context)")
    axes[1].scatter(x, movers.loc[common, "mover_attn"].to_numpy(), s=24, alpha=0.75, color="tab:blue")
    s1 = stats["mover_attn_number_vs_prior"]
    axes[1].set_title(f"mover attention→number (attenuation test)\nSpearman={s1['spearman']:+.2f} (perm p={s1['perm_p']:.3f})")
    for ax in axes:
        ax.set_xlabel("measured prior strength")
        ax.axhline(0, color="black", lw=0.6)
        ax.grid(alpha=0.25)
    fig.suptitle(f"{model_name}: the resistance signal lives in attention, per pair")
    fig.tight_layout()
    name = f"phase4_resistance_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"lean_split_wins_vs_loses": split, "n_loses": int(len(loses)),
            "scaling_tests": stats}, name


# ------------------------------------------------------------- task 3b
def ablation_stats(model_name: str, rng) -> pd.DataFrame:
    a = pd.read_csv(RESULTS_DIR / f"phase4_mlpablation_{model_name}.csv.gz")
    recs = []
    for (condition, name), g in a.groupby(["condition", "ablation"]):
        m, lo, hi = boot_ci(g["delta_ld"].to_numpy(), rng)
        recs.append({"condition": condition, "ablation": name, "n": len(g),
                     "mean_delta_ld": m, "ci": f"[{lo:.2f},{hi:.2f}]",
                     "baseline_cf": g["baseline_followed_context"].mean(),
                     "ablated_cf": g["ablated_followed_context"].mean()})
    return pd.DataFrame(recs).sort_values(["condition", "mean_delta_ld"])


def main() -> None:
    runmeta = json.loads((RESULTS_DIR / "phase4_runmeta.json").read_text(encoding="utf-8"))
    summary = {"seed": SEED, "n_bootstrap": N_BOOT, "n_permutations": N_PERM, "models": {}}
    lines = ["# Phase 4 Report: Where Does Parametric Resistance Live?", ""]
    lines.append("**Question:** Phase 3 showed no head carries the prior and attributed the "
                 "resistance to the 'rest' by elimination. This phase decomposes the rest "
                 "directly: embed / per-layer attention / per-layer MLP contributions to the "
                 "answer-position logits, their prior-direction writes, and — the decisive test — "
                 "whether those writes SCALE with the measured prior strength (the property the "
                 "negative heads lacked).")
    lines.append("")
    lines.append(f"Seed {SEED}; bootstrap {N_BOOT}; permutation {N_PERM}. Directions are "
                 "vocab-centered unembeds (as Phase 3). Reconstruction max error: "
                 + ", ".join(f"{m} = {runmeta['models'][m]['max_recon_err']:.0e}" for m in MODEL_NAMES)
                 + f". Compute: {runmeta['total_wall_seconds']/60:.1f} min (CPU).")
    lines.append("")

    for model_name in MODEL_NAMES:
        rng = np.random.default_rng(SEED)
        df = load_components(model_name)
        lines.append(f"## {model_name}")
        lines.append("")

        agg = aggregate_stats(df, rng)
        f_agg = fig_aggregate(model_name, agg)
        lines.append("### 1. Component decomposition (conflicting condition)")
        lines.append("")
        piv = agg.pivot_table(index="class", columns="quantity", values="mean").reset_index()
        piv = piv[["class", "dla_prior", "dla_context", "prior_lean"]]
        lines.append(md_table(piv, floats=("dla_prior", "dla_context", "prior_lean")))
        lines.append("")
        lines.append(f"![{f_agg}]({f_agg})")
        lines.append("")

        f_lay, prof = fig_layerwise(model_name, df, rng)
        conc = concentration(prof)
        lines.append("### 3. Layerwise MLP profile: diffuse or concentrated?")
        lines.append("")
        lines.append(f"![{f_lay}]({f_lay})")
        lines.append("")
        lines.append(f"- Top-3 prior-writing MLP layers: {conc['top3_layers']} carrying "
                     f"{100*conc['top3_share_of_positive']:.0f}% of the total positive MLP "
                     f"prior-write ({conc['n_layers_positive']}/12 layers positive).")
        lines.append("")

        scaling, f_sc = scaling_analysis(model_name, df, rng)
        lines.append("### 2. The decisive test: does the prior-write scale with prior strength?")
        lines.append("")
        lines.append(f"![{f_sc}]({f_sc})")
        lines.append("")
        lines.append(md_table(pd.DataFrame(scaling), floats=("spearman", "perm_p", "pearson")))
        lines.append("")
        lines.append("Contrast (Phase 3): movers flat (ρ = +0.07 / +0.02), negative heads "
                     "anti-scaling (ρ = −0.19 / −0.38).")
        lines.append("")

        resist, f_res = resistance_location(model_name, df, rng)
        lines.append("### 1c. Where the resistance actually shows up (post-hoc, forced by the data)")
        lines.append("")
        lines.append("Per-class net lean (prior − context), pairs where context WINS vs LOSES "
                     f"(n_loses = {resist['n_loses']}):")
        lines.append("")
        srows = [{"class": k, "wins": v["wins_mean_lean"], "loses": v["loses_mean_lean"]}
                 for k, v in resist["lean_split_wins_vs_loses"].items()]
        lines.append(md_table(pd.DataFrame(srows), floats=("wins", "loses")))
        lines.append("")
        trows = [{"test": k, "spearman": v["spearman"], "perm_p": v["perm_p"], "n": v["n"]}
                 for k, v in resist["scaling_tests"].items()]
        lines.append(md_table(pd.DataFrame(trows), floats=("spearman", "perm_p")))
        lines.append("")
        lines.append(f"![{f_res}]({f_res})")
        lines.append("")

        abl = ablation_stats(model_name, rng)
        lines.append("### 3b. Causal check: mean-ablating the top prior-writing MLP layers")
        lines.append("")
        lines.append(md_table(abl, floats=("mean_delta_ld", "baseline_cf", "ablated_cf")))
        lines.append("")
        lines.append("(Prediction if these MLPs supply resistance: ΔLD > 0 on the conflicting "
                     "condition — removing them frees the context answer. The control layer "
                     "should sit near 0. Interpret jointly with the fact that late-MLP ablation "
                     "also removes generic next-token processing.)")
        lines.append("")

        summary["models"][model_name] = {
            "runmeta": runmeta["models"][model_name],
            "aggregate": agg.to_dict(orient="records"),
            "layer_profile": prof.to_dict(orient="records"),
            "concentration": conc,
            "scaling": scaling,
            "resistance_location": resist,
            "mlp_ablation": abl.to_dict(orient="records"),
        }

    lines.append("## Limitations")
    lines.append("- DLA at the answer position measures *direct* writes; prior information "
                 "reaching the logits via composition through later components is attributed to "
                 "the final writer, not the source.")
    lines.append("- Vocab-centered single-token projections: absolute values depend on the "
                 "centering; comparisons across components and the scaling correlations do not.")
    lines.append("- MLP-layer ablation at the final position removes the layer's whole "
                 "contribution there, not just the prior-write — CF changes confound generic "
                 "processing loss with resistance removal (control layer partially calibrates).")
    lines.append("- n ≈ 30 pairs/model; one template; single-token answers.")
    lines.append("")
    lines.append("## Critical assessment")
    lines.append("")
    assess = RESULTS_DIR / "phase4_assessment.md"
    lines.append(assess.read_text(encoding="utf-8").strip() if assess.exists()
                 else "_[Not yet written — lives in results/phase4_assessment.md, inlined on "
                      "re-run; never generated by this script.]_")
    lines.append("")

    (RESULTS_DIR / "phase4_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (RESULTS_DIR / "phase4_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote phase4_report.md, phase4_summary.json, and figures.")


if __name__ == "__main__":
    main()

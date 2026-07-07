"""Phase 5 analysis, figures, and report — reads only cached outputs of
experiments/phase5.py. The frozen tide verdicts come straight from
phase5_tidetest.json (computed once in the compute script); this file adds
figures, the PH3-scan analysis, and a clearly-separated exploratory section.
"""

from __future__ import annotations

import json
import subprocess
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
MOVERS = {"gpt2": ["L9H9", "L10H2"], "pythia-160m": ["L8H10", "L10H7", "L8H2"]}
NEGATIVES = {"gpt2": ["L10H7"], "pythia-160m": ["L9H1", "L9H5"]}


def md_table(df: pd.DataFrame, floats=()) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = [f"{r[c]:.3f}" if c in floats and isinstance(r[c], (int, float)) else str(r[c]) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def prereg_commit() -> str:
    out = subprocess.run(
        ["git", "log", "--format=%H", "--diff-filter=A", "--",
         "results/phase5_preregistration.md"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "unknown"


def ph3_analysis(model_name: str) -> tuple[dict, str]:
    s = pd.read_csv(RESULTS_DIR / f"phase5_ph3scan_{model_name}.csv.gz")
    g = s.groupby(["layer", "head"])["ph3_importance"].mean().reset_index()
    g["unit"] = "L" + g["layer"].astype(str) + "H" + g["head"].astype(str)
    g = g.sort_values("ph3_importance")
    n_top = max(1, round(0.05 * len(g)))  # PH3's top-5% convention

    memory_cands = g.head(n_top)  # most NEGATIVE = candidate memory heads
    context_heads = g.tail(n_top)  # most POSITIVE = context heads

    res = {
        "n_heads": int(len(g)), "n_top5pct": n_top,
        "memory_candidates": memory_cands[["unit", "ph3_importance"]].to_dict(orient="records"),
        "context_heads": context_heads[["unit", "ph3_importance"]].to_dict(orient="records"),
        "max_context_importance": float(g["ph3_importance"].max()),
        "min_memory_importance": float(g["ph3_importance"].min()),
        "asymmetry_ratio": float(abs(g["ph3_importance"].max())
                                 / max(1e-9, abs(g["ph3_importance"].min()))),
        "movers_in_context_top": [u for u in MOVERS[model_name]
                                  if u in set(context_heads["unit"])],
        "negatives_in_memory_top": [u for u in NEGATIVES[model_name]
                                    if u in set(memory_cands["unit"])],
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = g["ph3_importance"].to_numpy()
    ax.hist(vals, bins=60, color="gray", alpha=0.7)
    for u, imp in zip(context_heads["unit"], context_heads["ph3_importance"]):
        ax.axvline(imp, color="tab:blue", lw=1)
    for u, imp in zip(memory_cands["unit"], memory_cands["ph3_importance"]):
        ax.axvline(imp, color="tab:purple", lw=1)
    ax.set_xlabel("PH3 importance  S = ΔP(ctx)−ΔP(prior), original − head-patched(unk)")
    ax.set_ylabel("head count")
    ax.set_title(f"{model_name}: PH3 exact-metric scan over all heads\n"
                 "blue lines = top-5% positive (context heads), purple = top-5% negative (memory candidates)")
    fig.tight_layout()
    name = f"phase5_ph3scan_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return res, name


def tide_figure(model_name: str, tide_res: dict) -> str:
    df = pd.read_csv(RESULTS_DIR / f"phase5_tide_{model_name}.csv.gz")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    loses = df["ld_clean"] < 0
    ax.scatter(df.loc[~loses, "prior_absdiff"], df.loc[~loses, "nonmover_attn_lean"],
               s=26, alpha=0.75, color="tab:blue", label="context wins")
    ax.scatter(df.loc[loses, "prior_absdiff"], df.loc[loses, "nonmover_attn_lean"],
               s=30, alpha=0.85, color="tab:red", label="context loses")
    ax.axhline(0, color="black", lw=0.6)
    p = tide_res["primary"]
    title_stat = (f"Spearman={p['spearman']:+.3f}, one-sided perm p={p['one_sided_perm_p']:.4f} "
                  f"-> {p['verdict']}" if p.get("valid") else p["verdict"])
    ax.set_title(f"{model_name}: PRE-REGISTERED tide test (fresh pairs, n={tide_res['n_gated']})\n{title_stat}")
    ax.set_xlabel("measured prior strength")
    ax.set_ylabel("non-mover attention lean (prior − context)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    name = f"phase5_tide_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def main() -> None:
    tidetest = json.loads((RESULTS_DIR / "phase5_tidetest.json").read_text(encoding="utf-8"))
    runmeta = json.loads((RESULTS_DIR / "phase5_runmeta.json").read_text(encoding="utf-8"))
    build_log = json.loads((REPO_ROOT / "data" / "pairs" / "phase5_build_log.json").read_text(encoding="utf-8"))
    commit = prereg_commit()

    summary = {"seed": SEED, "preregistration_commit_recorded": tidetest["preregistration_commit"],
               "preregistration_commit_from_git": commit, "models": {}}
    lines = ["# Phase 5 Report: PH3 Metric Match + Pre-registered Tide Test", ""]
    lines.append("**(A)** Does the memory-head region stay empty under Jin et al. 2024's OWN "
                 "importance metric (locks the scale-boundary headline)? **(B)** Does the Phase 4 "
                 "attention-tide correlation survive a pre-registered test on fresh, "
                 "fact-disjoint pairs?")
    lines.append("")
    lines.append(f"- Pre-registration frozen at commit `{tidetest['preregistration_commit'][:12]}` "
                 f"(git first-add commit verified: `{commit[:12]}`), BEFORE any fresh pair existed.")
    lines.append("- Fresh-set provenance: 103 candidates generated after the freeze; one "
                 "disclosed expansion round to 149 was added because the prior-filter stage "
                 "alone under-shot the pre-registered 40-pair floor (GPT-2: 38) — the expansion "
                 "was decided on attrition counts only, before any tide or behavioral (LD) value "
                 "was measured. All facts are disjoint from Phases 0–4.")
    for m in MODEL_NAMES:
        bl = build_log["models"][m]
        lines.append(f"- {m}: 149 generated → {bl['n_tokenization_passed']} tokenization-passed → "
                     f"{bl['n_prior_passed']} prior-passed → {tidetest['models'][m]['n_gated']} after "
                     f"the denominator gate; context-loses n = {tidetest['models'][m]['n_context_loses']}.")
    lines.append(f"- PH3 metric reconstruction: corrupted input = critical number replaced by the "
                 "EOS token (neither tokenizer has a true <unk>; documented substitution); "
                 "S = [P(ctx)−P(prior)]_original − [P(ctx)−P(prior)]_head-patched, probability "
                 "space, head z patched at all positions; top-5% each sign classified per PH3.")
    lines.append(f"- Seed {SEED}; compute {runmeta['total_wall_seconds']/60:.1f} min (CPU).")
    lines.append("")

    for model_name in MODEL_NAMES:
        lines.append(f"## {model_name}")
        lines.append("")
        ph3, f_ph3 = ph3_analysis(model_name)
        lines.append("### (A) PH3 exact-metric scan")
        lines.append("")
        lines.append(f"![{f_ph3}]({f_ph3})")
        lines.append("")
        mem_tab = pd.DataFrame(ph3["memory_candidates"])
        ctx_tab = pd.DataFrame(ph3["context_heads"])
        lines.append(f"Top-5% most NEGATIVE (candidate memory heads, n={ph3['n_top5pct']}):")
        lines.append("")
        lines.append(md_table(mem_tab, floats=("ph3_importance",)))
        lines.append("")
        lines.append(f"Top-5% most POSITIVE (context heads):")
        lines.append("")
        lines.append(md_table(ctx_tab, floats=("ph3_importance",)))
        lines.append("")
        lines.append(f"- Asymmetry: strongest context-head importance "
                     f"{ph3['max_context_importance']:+.4f} vs strongest memory-candidate "
                     f"{ph3['min_memory_importance']:+.4f} "
                     f"(|context|/|memory| = {ph3['asymmetry_ratio']:.1f}×).")
        lines.append(f"- Our movers inside PH3's context-head top-5%: {ph3['movers_in_context_top']}; "
                     f"our negative heads inside PH3's memory top-5%: {ph3['negatives_in_memory_top']}.")
        lines.append("")

        tide_res = tidetest["models"][model_name]
        f_tide = tide_figure(model_name, tide_res)
        lines.append("### (B) Pre-registered tide test — frozen verdicts")
        lines.append("")
        lines.append(f"![{f_tide}]({f_tide})")
        lines.append("")
        p, s = tide_res["primary"], tide_res["secondary"]
        if p.get("valid"):
            lines.append(f"- **PRIMARY (H1): {p['verdict']}** — Spearman = {p['spearman']:+.3f} "
                         f"(threshold ≥ +0.30), one-sided perm p = {p['one_sided_perm_p']:.4f} "
                         f"(criterion < 0.05), n = {p['n']} (floor 40).")
        else:
            lines.append(f"- **PRIMARY (H1): {p['verdict']}** (n = {p['n']}).")
        if s.get("valid"):
            lines.append(f"- **SECONDARY (H2): {s['verdict']}** — {s['k_positive']}/{s['n_loses']} "
                         f"context-loses pairs with positive lean "
                         f"({100*s['fraction_positive']:.0f}%), exact binomial p = {s['binomial_p']:.4f}.")
        else:
            lines.append(f"- **SECONDARY (H2): {s['verdict']}** (n_loses = {s['n_loses']}, floor 20).")
        lines.append("")

        summary["models"][model_name] = {"ph3": ph3, "tide": tide_res}

    # exploratory section — clearly separated, cannot upgrade verdicts
    lines.append("## Exploratory (NOT pre-registered; cannot change the verdicts above)")
    lines.append("")
    for model_name in MODEL_NAMES:
        df = pd.read_csv(RESULTS_DIR / f"phase5_tide_{model_name}.csv.gz")
        loses = df[df["ld_clean"] < 0]
        wins = df[df["ld_clean"] >= 0]
        lines.append(f"- {model_name}: mean non-mover lean on context-loses pairs "
                     f"{loses['nonmover_attn_lean'].mean():+.3f} (n={len(loses)}) vs wins "
                     f"{wins['nonmover_attn_lean'].mean():+.3f} (n={len(wins)}); behavioral "
                     f"gradient Spearman(LD_clean, prior) = "
                     f"{float(np.corrcoef(pd.Series(df['prior_absdiff']).rank(), pd.Series(df['ld_clean']).rank())[0,1]):+.3f}.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("- PH3 metric is a reconstruction from the paper's method description (their "
                 "Eq. 12 uses paired context/memory datasets; we compute the single-dataset "
                 "version on our conflicting pairs). EOS stands in for <unk>. Their smallest "
                 "model is ~8× our largest — the comparison bounds the motif's emergence, it "
                 "does not re-run their exact code.")
    lines.append("- Fresh-set expansion round disclosed above; decided on attrition counts only.")
    lines.append("- The tide statistic aggregates ~141 heads; per-head structure not tested here.")
    lines.append("")
    lines.append("## Critical assessment")
    lines.append("")
    assess = RESULTS_DIR / "phase5_assessment.md"
    lines.append(assess.read_text(encoding="utf-8").strip() if assess.exists()
                 else "_[Not yet written — lives in results/phase5_assessment.md, inlined on "
                      "re-run; never generated by this script.]_")
    lines.append("")

    (RESULTS_DIR / "phase5_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (RESULTS_DIR / "phase5_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote phase5_report.md, phase5_summary.json, and figures.")


if __name__ == "__main__":
    main()

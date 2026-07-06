"""Phase 1 analysis, figures, and report — reads ONLY cached outputs of
experiments/phase1.py (no model forward passes), so it can be re-run cheaply.

Produces:
  results/phase1_summary.json
  results/phase1_report.md   (inlines results/phase1_assessment.md if present)
  results/phase1_*.png       (heatmaps / line plots / scatters)
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
DATA_DIR = REPO_ROOT / "data"
MODEL_NAMES = ["gpt2", "pythia-160m"]
SEED = 0
N_BOOT = 2000
TOP_K = 10
OVERLAP_KS = (5, 10, 20)
CLASS_ORDER = ["ctx_pre", "number", "ctx_post", "question", "answer_end"]


# ---------------------------------------------------------------- utilities
def df_to_md(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(float_fmt.format(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def overlap_at_k(a: pd.Series, b: pd.Series, k: int) -> float:
    """Fraction of overlap between the top-k index sets of two effect series."""
    top_a = set(a.sort_values(ascending=False).index[:k])
    top_b = set(b.sort_values(ascending=False).index[:k])
    return len(top_a & top_b) / k


def bootstrap_ci(mat: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT):
    """mat: [n_pairs, n_units]. Resamples pairs jointly; returns (lo, hi) both
    [n_units] percentile 95% CIs of the mean."""
    n = mat.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = mat[idx].mean(axis=1)  # [n_boot, n_units]
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def load_model_data(model_name: str):
    df = pd.read_csv(RESULTS_DIR / f"phase1_effects_{model_name}.csv.gz")
    baselines = json.loads((RESULTS_DIR / f"phase1_baselines_{model_name}.json").read_text(encoding="utf-8"))
    return df, baselines


# ---------------------------------------------------------------- analyses
def head_effect_matrix(df: pd.DataFrame, direction: str, include_controls: bool = False):
    """Rows = pairs, columns = 'L{layer}H{head}'. Values: denoise -> recovery;
    noise -> degradation (1 - recovery), so 'big = important' in both."""
    sub = df[(df["pass"] == "head") & (df["direction"] == direction)]
    if not include_controls:
        sub = sub[~sub["is_control"]]
    val = sub["recovery"] if direction == "denoise" else 1.0 - sub["recovery"]
    sub = sub.assign(effect=val, unit="L" + sub["layer"].astype(str) + "H" + sub["head"].astype(str))
    return sub.pivot_table(index="pair_id", columns="unit", values="effect")


def top_heads_table(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for direction in ("denoise", "noise"):
        pivot = head_effect_matrix(df, direction)
        mat = pivot.to_numpy()
        lo, hi = bootstrap_ci(mat, rng)
        means = mat.mean(axis=0)
        medians = np.median(mat, axis=0)
        stats = pd.DataFrame({
            "unit": pivot.columns, "mean": means, "median": medians,
            "ci_lo": lo, "ci_hi": hi,
        }).sort_values("mean", ascending=False).head(TOP_K)
        stats.insert(0, "direction", direction)
        rows.append(stats)
    return pd.concat(rows, ignore_index=True)


def split_analysis(df: pd.DataFrame, baselines: dict) -> dict:
    """Central question 2d: are override-supporting heads the same for
    weak-prior (context wins) vs strong-prior (prior resists) pairs?"""
    prior = {b["pair_id"]: b["prior_absdiff"] for b in baselines["baselines"] if not b["is_control"]}
    out: dict = {}
    for direction in ("denoise", "noise"):
        pivot = head_effect_matrix(df, direction)
        pids = [p for p in pivot.index if p in prior]
        pivot = pivot.loc[pids]
        strengths = pd.Series({p: prior[p] for p in pids}).sort_values()
        n = len(strengths)
        weak_ids = list(strengths.index[: n // 3])
        strong_ids = list(strengths.index[-(n // 3):])
        weak = pivot.loc[weak_ids].mean(axis=0)
        strong = pivot.loc[strong_ids].mean(axis=0)
        out[direction] = {
            "n_weak": len(weak_ids), "n_strong": len(strong_ids),
            "weak_tercile_max_prior": float(strengths.iloc[n // 3 - 1]),
            "strong_tercile_min_prior": float(strengths.iloc[-(n // 3)]),
            "spearman": spearman(weak.to_numpy(), strong.to_numpy()),
            "overlap_at_k": {k: overlap_at_k(weak, strong, k) for k in OVERLAP_KS},
            "weak_effects": weak,
            "strong_effects": strong,
            "weak_ids": weak_ids,
            "strong_ids": strong_ids,
        }
    return out


def layer_depth_profile(df: pd.DataFrame, direction: str) -> pd.Series:
    """Per-layer aggregate causal effect from the coarse attn/mlp pass —
    covers ALL layers in both models regardless of head-pass restriction."""
    sub = df[(df["pass"] == "coarse") & (df["kind"].isin(["attn_out", "mlp_out"])) & (~df["is_control"])]
    sub = sub[sub["direction"] == direction]
    val = sub["recovery"] if direction == "denoise" else 1.0 - sub["recovery"]
    return sub.assign(effect=val.abs()).groupby("layer")["effect"].mean()


def negative_check(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    pivot = head_effect_matrix(df, "denoise")
    means = pivot.mean(axis=0)
    bottom = means[means.abs() <= means.abs().median()]  # bottom half by |effect|
    sample = bottom.sample(min(20, len(bottom)), random_state=SEED)
    return {
        "n_sampled": int(len(sample)),
        "mean_of_means": float(sample.mean()),
        "sd_of_means": float(sample.std()),
        "max_abs": float(sample.abs().max()),
        "units": list(sample.index),
    }


def controls_check(df: pd.DataFrame, baselines: dict) -> dict:
    ctrl_base = [b for b in baselines["baselines"] if b["is_control"]]
    ctrl_ids = {b["pair_id"] for b in ctrl_base}
    coarse = df[(df["pass"] == "coarse") & (df["direction"] == "denoise") & (df["label"] == "number")]
    ctrl_num = coarse[coarse["pair_id"].isin(ctrl_ids)]
    clean_num = coarse[~coarse["is_control"]]
    return {
        "baselines": ctrl_base,
        "excluded": [e for e in baselines["excluded"] if e["pair_id"] in
                     {c["pair_id"] for c in ctrl_base} | ctrl_ids],
        "n_controls_patched": int(ctrl_num["pair_id"].nunique()),
        "control_number_class_max_recovery": (
            float(ctrl_num.groupby("pair_id")["recovery"].max().mean())
            if len(ctrl_num) else None),
        "clean_number_class_max_recovery": float(clean_num.groupby("pair_id")["recovery"].max().mean()),
    }


# ---------------------------------------------------------------- figures
def fig_resid_class(df: pd.DataFrame, model_name: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, direction in zip(axes, ("denoise", "noise")):
        sub = df[(df["pass"] == "coarse") & (df["kind"] == "resid") & (~df["is_control"])
                 & (df["direction"] == direction)]
        val = sub["recovery"] if direction == "denoise" else 1.0 - sub["recovery"]
        sub = sub.assign(effect=val)
        classes = [c for c in CLASS_ORDER if c in set(sub["label"])]
        mat = sub.pivot_table(index="layer", columns="label", values="effect").reindex(columns=classes)
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_title(f"{direction} ({'recovery' if direction=='denoise' else 'degradation'})")
        ax.set_xlabel("position class")
    axes[0].set_ylabel("layer (resid_pre)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(f"{model_name}: residual-stream patching by layer x position class (mean over pairs)")
    name = f"phase1_resid_class_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_attn_mlp(df: pd.DataFrame, model_name: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, direction in zip(axes, ("denoise", "noise")):
        sub = df[(df["pass"] == "coarse") & (df["kind"].isin(["attn_out", "mlp_out"]))
                 & (~df["is_control"]) & (df["direction"] == direction)]
        val = sub["recovery"] if direction == "denoise" else 1.0 - sub["recovery"]
        sub = sub.assign(effect=val)
        for kind, style in (("attn_out", "-o"), ("mlp_out", "--s")):
            series = sub[sub["kind"] == kind].groupby("layer")["effect"].mean()
            ax.plot(series.index, series.values, style, label=kind)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"{direction} ({'recovery' if direction=='denoise' else 'degradation'})")
        ax.set_xlabel("layer")
        ax.legend()
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean normalized effect")
    fig.suptitle(f"{model_name}: attention block vs MLP output patching, per layer")
    name = f"phase1_attn_mlp_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_heads(df: pd.DataFrame, model_name: str, n_layers: int, n_heads: int) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, direction in zip(axes, ("denoise", "noise")):
        sub = df[(df["pass"] == "head") & (~df["is_control"]) & (df["direction"] == direction)]
        val = sub["recovery"] if direction == "denoise" else 1.0 - sub["recovery"]
        sub = sub.assign(effect=val)
        mat = np.full((n_layers, n_heads), np.nan)
        grouped = sub.groupby(["layer", "head"])["effect"].mean()
        for (layer, head), v in grouped.items():
            mat[int(layer), int(head)] = v
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("lightgray")
        vmax = max(0.15, np.nanmax(np.abs(mat)))
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(f"{direction} ({'recovery' if direction=='denoise' else 'degradation'})")
        ax.set_xlabel("head")
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[0].set_ylabel("layer (gray = not patched, budget-restricted)")
    fig.suptitle(f"{model_name}: per-head z patching (mean over pairs)")
    name = f"phase1_heads_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_resid_pos(df: pd.DataFrame, baselines: dict, model_name: str) -> str:
    sub = df[(df["pass"] == "resid_pos") & (df["direction"] == "denoise")]
    base = {b["pair_id"]: b for b in baselines["baselines"]}
    pair_ids = sorted(sub["pair_id"].unique(), key=lambda p: base[p]["prior_absdiff"])
    show = [pair_ids[0], pair_ids[-1]] if len(pair_ids) >= 2 else pair_ids
    fig, axes = plt.subplots(1, len(show), figsize=(6 * len(show), 4.5), squeeze=False)
    for ax, pid in zip(axes[0], show):
        p = sub[sub["pair_id"] == pid]
        pos = p["label"].str.removeprefix("pos").astype(int)
        p = p.assign(pos=pos)
        mat = p.pivot_table(index="layer", columns="pos", values="recovery")
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                       extent=(mat.columns.min() - 0.5, mat.columns.max() + 0.5,
                               mat.index.max() + 0.5, mat.index.min() - 0.5))
        b = base[pid]
        for x, lbl in ((b["diff_pos"], "number"), (b["q_start"], "Q"), (b["ans_start"], "A")):
            ax.axvline(x, color="black", linewidth=0.7, linestyle=":")
            ax.text(x, -0.8, lbl, ha="center", fontsize=8)
        ax.set_title(f"{pid} (|prior|={b['prior_absdiff']:.2f})", fontsize=10)
        ax.set_xlabel("token position")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{model_name}: denoising recovery, residual stream, full position resolution")
    name = f"phase1_resid_pos_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_split_scatter(split: dict, model_name: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, direction in zip(axes, ("denoise", "noise")):
        s = split[direction]
        w, st = s["weak_effects"], s["strong_effects"]
        ax.scatter(w.to_numpy(), st.to_numpy(), s=14, alpha=0.6)
        lim = max(0.05, float(np.nanmax(np.abs(np.concatenate([w.to_numpy(), st.to_numpy()])))))
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.7)
        for unit in w.sort_values(ascending=False).index[:5]:
            ax.annotate(unit, (w[unit], st[unit]), fontsize=7)
        ax.set_xlabel("head effect, WEAK-prior tercile (context wins easily)")
        ax.set_ylabel("head effect, STRONG-prior tercile (prior resists)")
        ax.set_title(f"{direction}: Spearman={s['spearman']:.2f}, "
                     f"overlap@10={s['overlap_at_k'][10]:.0%}")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{model_name}: are override heads the same when the prior is weak vs strong?")
    name = f"phase1_split_{model_name}.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_depth_profile(profiles: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, direction in zip(axes, ("denoise", "noise")):
        for model_name in MODEL_NAMES:
            prof = profiles[model_name][direction]
            depth = prof.index / (len(prof) - 1)
            ax.plot(depth, prof.values, "-o", label=model_name)
        ax.set_xlabel("relative depth (layer / (n_layers - 1))")
        ax.set_title(direction)
        ax.legend()
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean |effect| (attn+mlp coarse pass)")
    fig.suptitle("Cross-model layer-depth profile of causal effect")
    name = "phase1_depth_profile.png"
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


# ---------------------------------------------------------------- report
def attrition_section(build_log: dict, baselines_all: dict) -> str:
    lines = []
    for model_name in MODEL_NAMES:
        log = build_log["models"][model_name]
        b = baselines_all[model_name]
        n_patched = len({x["pair_id"] for x in b["baselines"]}) - len(b["excluded"])
        lines.append(
            f"- **{model_name}**: {log['n_candidates_generated']} candidates generated -> "
            f"{log['n_tokenization_passed']} passed tokenization -> "
            f"{log['n_prior_passed']} passed the prior filter (clean) "
            f"+ {log['n_controls_included']} controls -> "
            f"{n_patched} usable for patching after the denominator filter"
            + (" (" + "; ".join(f"{e['pair_id']} excluded: {e['reason']}" for e in b["excluded"]) + ")."
               if b["excluded"] else " (0 excluded).")
        )
    return "\n".join(lines)


def domain_table(model_name: str) -> pd.DataFrame:
    from pairs import load_phase1_pairs
    pairs = load_phase1_pairs(model_name)
    df = pd.DataFrame([{"domain": p.domain, "is_control": p.is_control} for p in pairs])
    counts = df.groupby("domain").size().rename("n_pairs").reset_index()
    counts.insert(0, "model", model_name)
    return counts


def main() -> None:
    rng = np.random.default_rng(SEED)
    runmeta = json.loads((RESULTS_DIR / "phase1_runmeta.json").read_text(encoding="utf-8"))
    build_log = json.loads((DATA_DIR / "pairs" / "phase1_build_log.json").read_text(encoding="utf-8"))

    summary: dict = {"seed": SEED, "n_bootstrap": N_BOOT, "models": {}}
    figures: dict[str, list[str]] = {m: [] for m in MODEL_NAMES}
    depth_profiles: dict = {}
    baselines_all: dict = {}
    top_tables: dict = {}
    splits: dict = {}

    for model_name in MODEL_NAMES:
        df, baselines = load_model_data(model_name)
        baselines_all[model_name] = baselines
        n_layers = int(df["layer"].max()) + 1
        n_heads = int(df[df["pass"] == "head"]["head"].max()) + 1

        top = top_heads_table(df, rng)
        top_tables[model_name] = top
        split = split_analysis(df, baselines)
        splits[model_name] = split
        neg = negative_check(df, rng)
        ctrl = controls_check(df, baselines)
        depth_profiles[model_name] = {
            d: layer_depth_profile(df, d) for d in ("denoise", "noise")
        }

        figures[model_name] = [
            fig_resid_class(df, model_name),
            fig_attn_mlp(df, model_name),
            fig_heads(df, model_name, n_layers, n_heads),
            fig_resid_pos(df, baselines, model_name),
            fig_split_scatter(split, model_name),
        ]

        summary["models"][model_name] = {
            "runmeta": runmeta["models"][model_name],
            "top_heads": top.to_dict(orient="records"),
            "split_analysis": {
                d: {k: v for k, v in split[d].items()
                    if k not in ("weak_effects", "strong_effects")}
                for d in split
            },
            "negative_check": neg,
            "controls_check": ctrl,
        }

    # cross-model depth-profile correlation (coarse pass covers all layers)
    cross = {}
    for direction in ("denoise", "noise"):
        a = depth_profiles[MODEL_NAMES[0]][direction].to_numpy()
        b = depth_profiles[MODEL_NAMES[1]][direction].to_numpy()
        cross[direction] = {"spearman_layer_profiles": spearman(a, b)}
    summary["cross_model"] = cross
    depth_fig = fig_depth_profile(depth_profiles)

    (RESULTS_DIR / "phase1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------- report
    lines: list[str] = []
    lines.append("# Phase 1 Report: Localizing Context-Override with Activation Patching")
    lines.append("")
    lines.append(
        "**Question:** which components of GPT-2-small and Pythia-160M causally carry "
        "the decision to follow retrieved context when it contradicts the model's "
        "parametric prior — and is the circuitry the same when the prior is weak "
        "(context wins easily) vs strong (prior resists)?")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append("- Prompt template: `Context: {context}\\n\\nQuestion: {question}\\nAnswer:` "
                 "(identical to the pilot); the neutral prior probe drops the Context block.")
    lines.append("- Every pair's conflict direction anchored on the MEASURED bare prior "
                 "(pilot showed a-priori labels are wrong ~half the time). Prior threshold: "
                 "|logit diff| >= 0.5 — in the pilot, near-tie pairs sat at |diff| <= 0.35 "
                 "(the negation noise band) while confirmed priors sat at 0.27–2.73; 0.5 "
                 "excludes the near-tie band where 'parametric answer' is not meaningful.")
    lines.append("- CLEAN run = conflicting context (override expected); CORRUPTED = supportive. "
                 "The prompts differ at exactly ONE token (the critical number), enforced at "
                 "build time and re-asserted before patching.")
    lines.append("- **Denoising** (corrupted run + clean activations; finds sufficient "
                 "components) and **noising** (clean run + corrupted activations; finds "
                 "necessary components) are both reported, per Heimersheim & Nanda 2024.")
    lines.append("- Metric: LD = logit(context answer) − logit(prior answer) at the final "
                 "position; normalized recovery r = (LD_patched − LD_corr)/(LD_clean − LD_corr). "
                 "For noising we report degradation 1 − r. Pairs with denominator < 1.0 logit "
                 "excluded (logged below).")
    lines.append(f"- Seed {SEED} everywhere; frozen models, eval mode, deterministic forwards. "
                 f"Batched patching verified exactly equal to unbatched runs.")
    lines.append("")

    for model_name in MODEL_NAMES:
        rm = runmeta["models"][model_name]
        n_all = int(pd.read_csv(RESULTS_DIR / f"phase1_effects_{model_name}.csv.gz")["layer"].max()) + 1
        if rm["head_pass_restricted"] and len(rm["head_layers"]) < n_all:
            head_note = (f"head pass **restricted to layers {rm['head_layers']}** "
                         "(projected full run exceeded the ~2h budget; layers chosen by "
                         "coarse-pass scores, threshold 25% of max, min 6 layers)")
        elif rm["head_pass_restricted"]:
            head_note = ("head pass: the >2h budget rule fired, but the layer-flagging "
                         f"criterion as run cleared **all {n_all} layers**, so the full "
                         "head grid ran anyway (wall-clock exceeded the projection). "
                         "Post-run we found the cause: noising rows entered the flag score "
                         "as raw recovery (≈1 for unimportant layers) instead of "
                         "degradation — fixed in experiments/phase1.py for future runs. "
                         "Net effect on THIS run: full coverage, nothing subsampled")
        else:
            head_note = "head pass on all layers (within budget)"
        lines.append(f"- **{model_name} run**: {rm['n_pairs_patched']} pairs patched, "
                     f"{rm['ms_per_config_measured']:.0f} ms/config measured, {head_note}; "
                     f"full-position residual subset: {rm['subset_pair_ids']}.")
    lines.append(f"- Total wall-clock: {runmeta['total_wall_seconds']/60:.0f} min.")
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append(attrition_section(build_log, baselines_all))
    lines.append("")
    dom = pd.concat([domain_table(m) for m in MODEL_NAMES], ignore_index=True)
    lines.append(df_to_md(dom))
    lines.append("")

    for model_name in MODEL_NAMES:
        lines.append(f"## Results: {model_name}")
        lines.append("")
        for fig in figures[model_name]:
            lines.append(f"![{fig}]({fig})")
            lines.append("")
        lines.append(f"### Top-{TOP_K} heads by mean effect (bootstrap 95% CI over pairs, "
                     f"{N_BOOT} resamples)")
        lines.append("")
        lines.append(df_to_md(top_tables[model_name]))
        lines.append("")
        s = summary["models"][model_name]
        lines.append("### Central question (2d): weak-prior vs strong-prior head rankings")
        lines.append("")
        for direction in ("denoise", "noise"):
            sd = s["split_analysis"][direction]
            ov = sd["overlap_at_k"]
            lines.append(
                f"- **{direction}**: Spearman rank correlation = {sd['spearman']:.3f} "
                f"(n_weak={sd['n_weak']}, n_strong={sd['n_strong']}); overlap@5={ov[5]:.0%}, "
                f"overlap@10={ov[10]:.0%}, overlap@20={ov[20]:.0%}.")
        lines.append("")
        lines.append("### Sanity / robustness")
        lines.append("")
        neg = s["negative_check"]
        lines.append(f"- Negative check: {neg['n_sampled']} heads sampled from the bottom half "
                     f"by |effect| -> mean {neg['mean_of_means']:.4f} (sd {neg['sd_of_means']:.4f}, "
                     f"max |effect| {neg['max_abs']:.4f}).")
        ctrl = s["controls_check"]
        for b in ctrl["baselines"]:
            lines.append(f"- Control `{b['pair_id']}`: LD_clean={b['ld_clean']:.3f}, "
                         f"LD_corr={b['ld_corr']:.3f}, denominator={b['denominator']:.3f}.")
        if ctrl["control_number_class_max_recovery"] is not None:
            lines.append(f"- Controls' number-position residual patching: max-over-layers recovery "
                         f"{ctrl['control_number_class_max_recovery']:.3f} "
                         f"(clean pairs: {ctrl['clean_number_class_max_recovery']:.3f}).")
        else:
            lines.append("- No control survived the denominator filter for patching (see "
                         "exclusions above); behavioral baselines reported instead.")
        lines.append("")

    lines.append("## Cross-model comparison")
    lines.append("")
    lines.append(f"![{depth_fig}]({depth_fig})")
    lines.append("")
    for direction in ("denoise", "noise"):
        lines.append(f"- {direction}: Spearman of per-layer |effect| profiles "
                     f"(relative depth) = {cross[direction]['spearman_layer_profiles']:.3f}. "
                     "Head indices do not correspond across architectures, so comparison is "
                     "by functional depth, not head identity.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Patched activations are off-manifold inputs to downstream components; "
                 "single-token, single-position prompt differences keep the intervention "
                 "as close to on-distribution as this method allows, but z-patches in "
                 "particular can have OOD effects (Heimersheim & Nanda 2024, sec. on "
                 "interpretation pitfalls).")
    lines.append("- ~30 pairs/model, single prompt template, single-token answers only: "
                 "findings are about THIS task format at THIS scale.")
    lines.append("- Head-level pass may be budget-restricted to coarse-flagged layers "
                 "(stated per model above); unflagged layers have no head-level estimates "
                 "(gray in heatmaps), not zero estimates.")
    lines.append("- Pythia-160M uses parallel attention/MLP blocks: attn_out and mlp_out "
                 "read the SAME resid_pre, so the attn-vs-MLP decomposition is not "
                 "architecturally identical to GPT-2's sequential blocks.")
    lines.append("- Only 2 copy-fidelity controls; their evidence is anecdotal.")
    lines.append("- The world_numeric domain is not legal text; it is flagged in the data "
                 "and results can be recomputed excluding it.")
    lines.append("")

    lines.append("## Critical assessment")
    lines.append("")
    assessment = RESULTS_DIR / "phase1_assessment.md"
    if assessment.exists():
        lines.append(assessment.read_text(encoding="utf-8").strip())
    else:
        lines.append("_[Not yet written — lives in results/phase1_assessment.md, inlined "
                     "on re-run; never generated by this script.]_")
    lines.append("")

    (RESULTS_DIR / "phase1_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'phase1_report.md'}, phase1_summary.json, and "
          f"{sum(len(v) for v in figures.values()) + 1} figures.")


if __name__ == "__main__":
    main()

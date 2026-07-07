"""Phase 0 pilot (VALIDATION-FIRST gate, see RESEARCH_CHARTER.md).

Answers exactly one question with data: at GPT-2-small / Pythia-160M scale,
is context-vs-parametric conflict measurable at all? I.e. does editing a
single fact in retrieved context (supportive -> conflicting) produce a
reliable, consistent shift in which single-token answer the model prefers?

Writes:
  results/pilot_measurements.json  - every raw (model, pair, condition) logit reading
  results/pilot_summary.json       - per-pair shift + per-model aggregate stats
  results/pilot_plot.png           - slope plot: logit diff under supportive vs conflicting
  results/pilot_report.md          - human-readable report

If results/pilot_assessment.md exists, its contents are inlined into the
report's "Critical assessment" section, so the hand-written verdict survives
re-runs of this script. The assessment is written by a human/reviewer after
looking at the numbers — it is intentionally NOT auto-generated here, so the
verdict is not self-graded by the same code that produced the data.

Do not extend this script with Phase 1+ logic (activation patching, attribution,
ablation) - those live in src/patch.py, src/attribution.py, src/ablate.py and
are explicitly out of scope until a human signs off on this pilot's verdict.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pairs import PAIRS_PATH, load_pairs_jsonl, validate_pairs  # noqa: E402
from measure import CONDITIONS, measure_pair, set_all_seeds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
MODEL_NAMES = ["gpt2", "pythia-160m"]
SEED = 0


def load_models() -> dict[str, HookedTransformer]:
    models = {}
    for name in MODEL_NAMES:
        print(f"Loading {name}...")
        model = HookedTransformer.from_pretrained(name, device="cpu")
        model.eval()
        models[name] = model
    return models


def run_measurements(models: dict, pairs: list) -> pd.DataFrame:
    records = []
    for model_name, model in models.items():
        for pair in pairs:
            for condition in CONDITIONS:
                m = measure_pair(model, model_name, pair, condition)
                records.append(asdict(m))
    return pd.DataFrame.from_records(records)


def add_shift_columns(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["model_name", "pair_id", "category", "edit_type", "has_strong_prior"],
        columns="condition",
        values="logit_diff_param_minus_edited",
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "supportive": "raw_diff_supportive",
            "conflicting": "raw_diff_conflicting",
            "neutral": "raw_diff_neutral",
        }
    )
    pivot["shift_conflicting_minus_supportive"] = (
        pivot["raw_diff_conflicting"] - pivot["raw_diff_supportive"]
    )
    return pivot


def sign_test_pvalue(n: int, k: int) -> float:
    """Two-sided binomial sign-test p-value (k successes of n, null p=0.5).
    Implemented against the stdlib only (no scipy in project deps)."""
    if n == 0:
        return 1.0
    m = min(k, n - k)
    cdf = sum(math.comb(n, i) for i in range(0, m + 1)) / (2**n)
    return min(1.0, 2 * cdf)


def summarize(pivot: pd.DataFrame) -> dict:
    summary = {}
    for model_name in MODEL_NAMES:
        sub = pivot[pivot["model_name"] == model_name]
        n = len(sub)
        n_expected = int((sub["shift_conflicting_minus_supportive"] < 0).sum())

        strong = sub[sub["has_strong_prior"]]
        n_strong = len(strong)
        n_strong_expected = int((strong["shift_conflicting_minus_supportive"] < 0).sum())
        n_prior_confirmed = int((strong["raw_diff_neutral"] > 0).sum())

        by_edit_type = {}
        for edit_type, group in sub.groupby("edit_type"):
            g_n = len(group)
            g_expected = int((group["shift_conflicting_minus_supportive"] < 0).sum())
            by_edit_type[str(edit_type)] = {
                "n_pairs": g_n,
                "mean_shift": float(group["shift_conflicting_minus_supportive"].mean()),
                "median_shift": float(group["shift_conflicting_minus_supportive"].median()),
                "n_expected_direction": g_expected,
                "sign_test_p_value": sign_test_pvalue(g_n, g_expected),
            }

        summary[model_name] = {
            "n_pairs": n,
            "mean_shift": float(sub["shift_conflicting_minus_supportive"].mean()),
            "median_shift": float(sub["shift_conflicting_minus_supportive"].median()),
            "std_shift": float(sub["shift_conflicting_minus_supportive"].std()),
            "n_expected_direction": n_expected,
            "frac_expected_direction": n_expected / n,
            "sign_test_p_value": sign_test_pvalue(n, n_expected),
            "n_strong_prior_pairs": n_strong,
            "n_strong_prior_expected_direction": n_strong_expected,
            "frac_strong_prior_expected_direction": (
                n_strong_expected / n_strong if n_strong else float("nan")
            ),
            "n_prior_hypothesis_confirmed_by_neutral": n_prior_confirmed,
            "frac_prior_hypothesis_confirmed": (
                n_prior_confirmed / n_strong if n_strong else float("nan")
            ),
            "by_edit_type": by_edit_type,
        }
    return summary


def make_slope_plot(pivot: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(13, 8), sharey=True)
    order = (
        pivot[pivot["model_name"] == MODEL_NAMES[0]]
        .sort_values(["edit_type", "pair_id"])["pair_id"]
        .tolist()
    )
    y_pos = {pid: i for i, pid in enumerate(order)}

    for ax, model_name in zip(axes, MODEL_NAMES):
        sub = pivot[pivot["model_name"] == model_name].set_index("pair_id").loc[order]
        for pid, row in sub.iterrows():
            y = y_pos[pid]
            expected = row["shift_conflicting_minus_supportive"] < 0
            color = "tab:blue" if expected else "tab:red"
            ax.plot(
                [row["raw_diff_supportive"], row["raw_diff_conflicting"]],
                [y, y],
                color=color,
                linewidth=2,
                alpha=0.85,
                solid_capstyle="round",
            )
            # Distinguish the two endpoints: open circle = supportive context,
            # filled triangle = conflicting context.
            ax.plot(
                row["raw_diff_supportive"], y, marker="o", markersize=7,
                markerfacecolor="white", markeredgecolor=color, linestyle="none",
            )
            ax.plot(
                row["raw_diff_conflicting"], y, marker="v", markersize=7,
                color=color, linestyle="none",
            )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(model_name)
        ax.set_xlabel("logit(parametric answer) - logit(edited answer)")
        ax.grid(axis="x", alpha=0.2)

    axes[0].set_yticks(list(y_pos.values()))
    axes[0].set_yticklabels(list(y_pos.keys()))
    axes[0].invert_yaxis()

    legend_handles = [
        mlines.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="black",
                      linestyle="none", label="supportive context (context states the parametric answer)"),
        mlines.Line2D([], [], marker="v", color="black", linestyle="none",
                      label="conflicting context (context states the edited answer)"),
        mlines.Line2D([], [], color="tab:blue", linewidth=2,
                      label="shift in expected direction (context edit moved the answer preference)"),
        mlines.Line2D([], [], color="tab:red", linewidth=2,
                      label="shift in unexpected direction (context edit did not move it)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Per pair: logit(parametric) - logit(edited) under supportive vs conflicting context.\n"
        "Line length = the causal effect of editing ONE fact in the retrieved context.\n"
        "Positive x = model prefers the parametric answer; negative x = prefers the edited answer."
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def df_to_markdown_table(df: pd.DataFrame, float_cols: tuple[str, ...] = ()) -> str:
    """Minimal markdown table writer (avoids adding `tabulate` as a dependency
    just for pandas.to_markdown)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in float_cols and isinstance(v, (int, float)):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(pivot: pd.DataFrame, summary: dict, seed_info: dict, out_path: Path) -> None:
    n_pairs = pivot["pair_id"].nunique()
    table_cols = [
        "model_name",
        "pair_id",
        "edit_type",
        "has_strong_prior",
        "raw_diff_neutral",
        "raw_diff_supportive",
        "raw_diff_conflicting",
        "shift_conflicting_minus_supportive",
    ]
    table = pivot[table_cols].sort_values(["model_name", "edit_type", "pair_id"])
    float_cols = (
        "raw_diff_neutral",
        "raw_diff_supportive",
        "raw_diff_conflicting",
        "shift_conflicting_minus_supportive",
    )

    lines = []
    lines.append("# Phase 0 Pilot Report: Is Context-vs-Parametric Conflict Measurable?")
    lines.append("")
    lines.append(
        "**Question this experiment answers:** at GPT-2-small / Pythia-160M scale, does "
        "editing a single fact in retrieved legal context (supportive -> conflicting) "
        "produce a measurable, consistent shift in the model's next-token preference "
        "between the context-stated answer and the alternative? This is the VALIDATION-FIRST "
        "gate from RESEARCH_CHARTER.md: Phase 1 (activation patching / attribution / ablation) does not "
        "start until a human signs off on this verdict."
    )
    lines.append("")
    lines.append(f"- Models: {', '.join(MODEL_NAMES)} (via TransformerLens, CPU)")
    lines.append(f"- Pairs: {n_pairs} hand-written minimal pairs ({PAIRS_PATH.relative_to(REPO_ROOT)})")
    lines.append(f"- Seed: {seed_info['seed']} (python/numpy/torch; forward passes over a frozen model in eval mode are deterministic, so this is a reproducibility log, not a variance control)")
    lines.append("- Metric: logit(parametric_answer token) - logit(edited_answer token), read off the final-position next-token logits after the prompt's trailing \"Answer:\"")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Each pair has one legal question and two versions of retrieved context differing "
        "by a single controlled edit (a number swapped, or a clause negated). "
        "`parametric_answer` is the value we hypothesized the model's pretraining-era prior "
        "favors; `context_supportive` states that value; `context_conflicting` states "
        "`edited_answer` instead. A third `neutral` condition (question alone, no context "
        "block) probes the model's bare prior between the two candidate tokens, to check "
        "whether the `parametric_answer` hypothesis is actually correct for each pair rather "
        "than just asserted."
    )
    lines.append("")
    lines.append(
        "`shift_conflicting_minus_supportive` = raw_diff(conflicting) - raw_diff(supportive). "
        "If the context edit has any causal effect on the model's answer, this should be "
        "reliably **negative** (conflicting context pulls preference away from the parametric "
        "answer, toward the edited one). A value near zero means the edit did not move the "
        "model's preference at all."
    )
    lines.append("")
    lines.append("## Per-model summary")
    lines.append("")
    for model_name in MODEL_NAMES:
        s = summary[model_name]
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append(f"- Mean shift: {s['mean_shift']:.3f} (median {s['median_shift']:.3f}, std {s['std_shift']:.3f})")
        lines.append(
            f"- {s['n_expected_direction']}/{s['n_pairs']} pairs "
            f"({100*s['frac_expected_direction']:.0f}%) shifted in the expected direction "
            f"(sign test p={s['sign_test_p_value']:.4f})"
        )
        lines.append(
            f"- Restricting to the {s['n_strong_prior_pairs']} pairs hypothesized to have a "
            f"real parametric prior: {s['n_strong_prior_expected_direction']}/{s['n_strong_prior_pairs']} "
            f"({100*s['frac_strong_prior_expected_direction']:.0f}%) shifted in the expected direction"
        )
        lines.append(
            f"- Prior-check: the neutral (no-context) condition actually favored the "
            f"hypothesized `parametric_answer` in {s['n_prior_hypothesis_confirmed_by_neutral']}/"
            f"{s['n_strong_prior_pairs']} ({100*s['frac_prior_hypothesis_confirmed']:.0f}%) of "
            f"strong-prior pairs -- this validates (or not) our own labeling, it is not the "
            f"pass/fail criterion for the pilot"
        )
        lines.append("")
        lines.append("By edit type:")
        lines.append("")
        lines.append("| edit_type | n | mean shift | median shift | expected direction | sign test p |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for edit_type in sorted(s["by_edit_type"]):
            e = s["by_edit_type"][edit_type]
            lines.append(
                f"| {edit_type} | {e['n_pairs']} | {e['mean_shift']:.3f} | "
                f"{e['median_shift']:.3f} | {e['n_expected_direction']}/{e['n_pairs']} | "
                f"{e['sign_test_p_value']:.4f} |"
            )
        lines.append("")

    lines.append("## Plot")
    lines.append("")
    lines.append("![pilot slope plot](pilot_plot.png)")
    lines.append("")
    lines.append("## Full per-pair, per-model results")
    lines.append("")
    lines.append(df_to_markdown_table(table, float_cols))
    lines.append("")
    lines.append("## Critical assessment")
    lines.append("")
    assessment_path = RESULTS_DIR / "pilot_assessment.md"
    if assessment_path.exists():
        lines.append(assessment_path.read_text(encoding="utf-8").strip())
    else:
        lines.append(
            "_[Not yet written. The assessment lives in results/pilot_assessment.md and "
            "is inlined here on re-run; it is written after looking at the numbers rather "
            "than auto-generated, so the verdict is not self-graded by the same script "
            "that produced them. Per RESEARCH_CHARTER.md's VALIDATION-FIRST rule, Phase 1 does not "
            "start until a human signs off on this verdict.]_"
        )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_info = set_all_seeds(SEED)

    pairs = load_pairs_jsonl(PAIRS_PATH)
    print(f"Loaded {len(pairs)} pairs from {PAIRS_PATH}")

    models = load_models()
    validate_pairs(pairs, models)
    print("Re-validated: every answer is a single token in both models.")

    print("Running measurements...")
    df = run_measurements(models, pairs)

    measurements_path = RESULTS_DIR / "pilot_measurements.json"
    df.to_json(measurements_path, orient="records", indent=2)
    print(f"Wrote {len(df)} raw measurements to {measurements_path}")

    pivot = add_shift_columns(df)
    summary = summarize(pivot)

    summary_path = RESULTS_DIR / "pilot_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "seed": seed_info["seed"],
                "models": MODEL_NAMES,
                "n_pairs": int(pivot["pair_id"].nunique()),
                "per_model": summary,
                "per_pair": pivot.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Wrote summary to {summary_path}")

    plot_path = RESULTS_DIR / "pilot_plot.png"
    make_slope_plot(pivot, plot_path)
    print(f"Wrote plot to {plot_path}")

    report_path = RESULTS_DIR / "pilot_report.md"
    write_report(pivot, summary, seed_info, report_path)
    print(f"Wrote report to {report_path}")

    print("\n=== Quick verdict preview ===")
    for model_name in MODEL_NAMES:
        s = summary[model_name]
        print(
            f"{model_name}: mean_shift={s['mean_shift']:.3f}, "
            f"{s['n_expected_direction']}/{s['n_pairs']} expected-direction "
            f"(p={s['sign_test_p_value']:.4f})"
        )
        for edit_type in sorted(s["by_edit_type"]):
            e = s["by_edit_type"][edit_type]
            print(
                f"  {edit_type:16s} mean_shift={e['mean_shift']:7.3f}, "
                f"{e['n_expected_direction']}/{e['n_pairs']} expected-direction "
                f"(p={e['sign_test_p_value']:.4f})"
            )


if __name__ == "__main__":
    main()

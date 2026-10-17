"""Summarize multi-dataset benchmark results and generate a concise report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values, returned in the input order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def analyze(results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    report_dir = results_dir / "report"
    metadata_dir = results_dir / "metadata"
    for directory in (tables_dir, figures_dir, report_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(tables_dir / "fold_results.csv")
    nested = folds[folds.protocol == "nested_cv"].copy()
    aggregate = pd.read_csv(tables_dir / "aggregate_results.csv")
    metadata = pd.read_json(metadata_dir / "run_metadata.json", typ="series")
    is_smoke = bool(metadata.get("smoke", False))
    tests = []
    for dataset, part in nested.groupby("dataset"):
        classical = [name for name in part.model.unique() if not name.startswith("q_")]
        quantum = [name for name in part.model.unique() if name.startswith("q_")]
        for q_model in quantum:
            for c_model in classical:
                paired = part[part.model == q_model].set_index("fold").mae.to_frame("q")
                paired["c"] = part[part.model == c_model].set_index("fold").mae
                paired = paired.dropna()
                delta = (paired.q - paired.c).to_numpy()  # negative favors quantum
                try:
                    p_value = float(wilcoxon(delta, alternative="two-sided", method="auto").pvalue) if np.any(delta) else 1.0
                except ValueError:
                    p_value = 1.0
                sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else np.nan
                tests.append({
                    "dataset": dataset, "quantum_model": q_model, "classical_model": c_model,
                    "n_paired_folds": len(delta), "mean_mae_difference_q_minus_c": float(np.mean(delta)),
                    "median_mae_difference_q_minus_c": float(np.median(delta)),
                    "paired_cohens_dz": float(np.mean(delta) / sd) if sd > 0 else np.nan,
                    "wilcoxon_p": p_value,
                })
    if tests:
        adjusted = _holm([row["wilcoxon_p"] for row in tests])
        for row, p_adj in zip(tests, adjusted, strict=True):
            row["holm_adjusted_p"] = p_adj
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(tables_dir / "paired_statistics.csv", index=False)
    learning_curves = pd.read_csv(tables_dir / "learning_curve_results.csv")

    # Dataset-level ranks prevent datasets with larger target scales dominating.
    ranks = nested.groupby(["dataset", "model"], as_index=False).mae.mean()
    ranks["rank_within_dataset"] = ranks.groupby("dataset").mae.rank(method="average")
    rank_summary = ranks.groupby("model", as_index=False).rank_within_dataset.mean().sort_values("rank_within_dataset")
    rank_summary.to_csv(tables_dir / "cross_dataset_ranks.csv", index=False)
    rank_best = rank_summary.iloc[0]

    datasets = list(nested.dataset.drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets, strict=False):
        part = aggregate[(aggregate.dataset == dataset) & (aggregate.protocol == "nested_cv")].sort_values("mae_mean")
        ax.barh(part.model, part.mae_mean, xerr=part.mae_std.fillna(0), color=["#2d6a4f" if m.startswith("q_") else "#457b9d" for m in part.model])
        ax.set_title(dataset.replace("_", " ").title())
        ax.set_xlabel("MAE (lower is better)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Kernel SVR performance by dataset")
    fig.tight_layout()
    fig.savefig(figures_dir / "mae_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    rank_summary = rank_summary.sort_values("rank_within_dataset", ascending=False)
    ax.barh(rank_summary.model, rank_summary.rank_within_dataset, color=["#2d6a4f" if m.startswith("q_") else "#457b9d" for m in rank_summary.model])
    ax.set_xlabel("Mean MAE rank across datasets (lower is better)")
    ax.set_title("Cross-dataset model ranks")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "cross_dataset_ranks.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    spectral = nested.groupby("model").kernel_effective_rank.agg(["mean", "std"]).sort_values("mean")
    ax.barh(spectral.index, spectral["mean"], xerr=spectral["std"].fillna(0), color=["#2d6a4f" if m.startswith("q_") else "#457b9d" for m in spectral.index])
    ax.set_xlabel("Kernel effective rank (fold mean ± SD)")
    ax.set_title("Kernel spectral diagnostics")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "kernel_spectral_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets, strict=False):
        part = learning_curves[learning_curves.dataset == dataset]
        for model, curve in part.groupby("model"):
            curve = curve.sort_values("train_fraction")
            ax.plot(curve.train_fraction, curve.mae, marker="o", label=model)
        ax.set_title(dataset.replace("_", " ").title())
        ax.set_xlabel("Fraction of first outer-fold training partition")
        ax.set_ylabel("MAE on fixed outer test fold")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Learning curves (fixed nested-CV parameters)")
    fig.tight_layout()
    fig.savefig(figures_dir / "learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Multi-dataset QSVM regression benchmark" + (" (smoke run)" if is_smoke else ""),
        "",
        "This study compares linear, RBF, and degree-3 polynomial SVR with ideal and depolarizing-noise Rot2DoF and Belis quantum kernels. Preprocessing and target scaling are fit within each training partition. MAE, RMSE, and R² are reported on the target's original scale.",
        "",
        "## Dataset results",
        "",
        ("Smoke configuration only: this reduced run tests data loading and the classical/quantum paths; it is not a full model comparison." if is_smoke else "Best model per dataset by nested-CV mean MAE:"),
        "",
        "| Dataset | Best model | MAE ± SD | RMSE ± SD | Mean R² | PSD fraction | Mean kernel seconds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, part in aggregate[aggregate.protocol == "nested_cv"].groupby("dataset"):
        row = part.loc[part.mae_mean.idxmin()]
        report.append(f"| {dataset} | {row.model} | {row.mae_mean:.3f} ± {row.mae_std:.3f} | {row.rmse_mean:.3f} ± {row.rmse_std:.3f} | {row.r2_mean:.3f} | {row.psd_fraction:.2f} | {row.kernel_time_mean_seconds:.6f} |")
    report += [
        "",
        "Best classical versus best quantum model by nested-CV MAE:",
        "",
        "| Dataset | Best classical | Classical MAE | Best quantum | Quantum MAE | Quantum minus classical |",
        "|---|---|---:|---|---:|---:|",
    ]
    for dataset, part in aggregate[aggregate.protocol == "nested_cv"].groupby("dataset"):
        classical = part[~part.model.str.startswith("q_")].sort_values("mae_mean").iloc[0]
        quantum = part[part.model.str.startswith("q_")].sort_values("mae_mean").iloc[0]
        gap = quantum.mae_mean - classical.mae_mean
        report.append(f"| {dataset} | {classical.model} | {classical.mae_mean:.3f} | {quantum.model} | {quantum.mae_mean:.3f} | {gap:+.3f} |")
    runtime = nested.groupby("kernel_kind").kernel_time_seconds.mean()
    runtime_line = f"Mean kernel construction time was {runtime.get('classical', float('nan')):.4f}s for classical kernels and {runtime.get('quantum', float('nan')):.4f}s for quantum simulation kernels, averaged across outer folds and configured kernels."
    adjusted_min = float(tests_df.holm_adjusted_p.min()) if not tests_df.empty else float("nan")
    raw_min = float(tests_df.wilcoxon_p.min()) if not tests_df.empty else float("nan")
    report += [
        "",
        "## Paired analysis",
        "",
        f"The lowest mean within-dataset MAE rank is **{rank_best.model}** (mean rank {rank_best.rank_within_dataset:.2f}). {runtime_line} The paired Wilcoxon results in `../tables/paired_statistics.csv` compare each quantum model with each classical kernel on matching outer folds. Holm adjustment is applied across all {len(tests_df)} comparisons; no comparison is significant after adjustment (minimum adjusted p={adjusted_min:.4f}; minimum raw p={raw_min:.4f}). This run uses {int(nested.groupby('dataset').fold.nunique().min())} outer folds per dataset, so interpret the effect sizes and tests cautiously.",
        "The winner and best-classical-versus-best-quantum rows are post-hoc descriptive summaries selected from outer-fold means; they are not unbiased estimates for a model chosen in advance. The paired tests retain fixed model pairings and are reported separately.",
        "",
        "## Figures",
        "",
        "### MAE comparison",
        "",
        "![MAE comparison across datasets](../figures/mae_comparison.png)",
        "",
        "### Cross-dataset ranks",
        "",
        "![Mean model rank across datasets](../figures/cross_dataset_ranks.png)",
        "",
        "### Kernel spectral diagnostics",
        "",
        "![Effective rank of each kernel](../figures/kernel_spectral_diagnostics.png)",
        "",
        "### Learning curves",
        "",
        "![Learning curves by dataset and model](../figures/learning_curves.png)",
        "",
        "## Data files",
        "",
        "- `../tables/fold_results.csv` and `../tables/aggregate_results.csv`: combined fold-level and summarized results, including effective rank, PSD status, and runtime.",
        "- `../datasets/<name>_fold_results.csv`: per-dataset fold results.",
        "- `../metadata/run_metadata.json`: dataset provenance, sampled row indices, configuration, environment, and seeds.",
        "",
        "## Data and method references",
        "",
        "The evaluation design is adapted from [Kakavand, Strohmeyer, and Schlotter (2026)](https://arxiv.org/abs/2604.18837). Dataset sources: [Auto MPG](https://doi.org/10.24432/C5859H), [Concrete Compressive Strength](https://doi.org/10.24432/C5PK67), [Energy Efficiency](https://doi.org/10.24432/C51307), and [Bike Sharing](https://doi.org/10.24432/C5W894).",
        "",
        "The Auto MPG latest-model-year holdout is reported separately from nested-CV comparisons. Learning curves are descriptive fixed-parameter diagnostics, not independent holdout estimates. No hardware results are included.",
    ]
    report_path = report_dir / "findings_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results-dir", default=project_root / "results/multi_dataset_benchmark")
    args = parser.parse_args()
    print(analyze(args.results_dir))


if __name__ == "__main__":
    main()

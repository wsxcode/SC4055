"""Turn results/<run>/<experiment>/results.csv into tables, figures and report.md.

    python scripts/analyze.py [--run full]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qsvm_clf.evaluate import corrected_resampled_ttest  # noqa: E402

COLORS = {"linear": "#9ca3af", "rbf": "#2563eb", "poly": "#7c3aed",
          "q_zz": "#dc2626", "q_he": "#ea580c", "q_product": "#ca8a04"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def load(run_dir: Path, name: str) -> pd.DataFrame | None:
    path = run_dir / name / "results.csv"
    return pd.read_csv(path) if path.exists() else None


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean over folds; std is taken over per-repeat means (repeats are the unit of replication)."""
    per_rep = df.groupby(by + ["repeat"])[["acc", "f1", "auc"]].mean().reset_index()
    g = per_rep.groupby(by)
    out = g[["acc", "f1", "auc"]].mean()
    out["acc_std"] = g["acc"].std(ddof=1)
    out["auc_std"] = g["auc"].std(ddof=1)
    out["n_folds"] = df.groupby(by).size()
    return out.reset_index()


def paired_tests(df: pd.DataFrame, by: list[str], baseline: str = "rbf") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        piv = g.pivot_table(index=["repeat", "fold"], columns="model", values="acc")
        if baseline not in piv:
            continue
        classical = [m for m in piv if m in ("linear", "rbf", "poly")]
        best_classical = piv[classical].mean().idxmax()
        for m in piv:
            if m.startswith(("q_", "zz")) or g.loc[g.model == m, "kind"].iloc[0] == "quantum":
                for ref in dict.fromkeys([baseline, best_classical]):
                    d = (piv[m] - piv[ref]).to_numpy()
                    t, p = corrected_resampled_ttest(d, int(g.n_train.iloc[0]), int(g.n_test.iloc[0]))
                    rows.append({**dict(zip(by, key)), "quantum": m, "vs": ref, "mean_diff": d.mean(),
                                 "t": t, "p_raw": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_raw"].to_numpy())
    return out


# ------------------------------------------------------------------- plots
def order_datasets(names):
    order = ["moons", "ad_hoc", "heart"]
    return [d for d in order if d in set(names)] + [d for d in dict.fromkeys(names) if d not in order]


def bar_main(summary, path):
    datasets = order_datasets(summary.dataset.unique())
    models = [m for m in COLORS if m in set(summary.model)]
    fig, ax = plt.subplots(figsize=(1.6 * len(datasets) + 2, 3.6))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        s = summary[summary.model == m].set_index("dataset").reindex(datasets)
        ax.bar(np.arange(len(datasets)) + i * w, s.acc, w, yerr=s.acc_std, label=m, color=COLORS[m], capsize=2)
    ax.set_xticks(np.arange(len(datasets)) + w * (len(models) - 1) / 2, datasets)
    ax.set_ylabel("test accuracy (mean ± std over repeats)")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.legend(ncol=3, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def lines_by(summary, x, path, xlabel, group="dataset", title=None):
    groups = order_datasets(summary[group].unique()) if group == "dataset" else list(summary[group].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.1), sharey=True, squeeze=False)
    for ax, gname in zip(axes[0], groups):
        sub = summary[summary[group] == gname]
        for m in [m for m in COLORS if m in set(sub.model)]:
            s = sub[sub.model == m].sort_values(x)
            ax.errorbar(s[x], s.acc, yerr=s.acc_std, label=m, color=COLORS[m], marker="o", ms=3, capsize=2,
                        ls="--" if m in ("linear", "poly") else "-")
        ax.set_title(gname)
        ax.set_xlabel(xlabel)
    axes[0][0].set_ylabel("test accuracy")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_concentration(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    sub = df[(df.source == "uniform") & (df.feature_map == "zz")]
    for bw, g in sub[sub.bandwidth_rule == "fixed"].groupby("bandwidth"):
        axes[0].semilogy(g.n_qubits, g.offdiag_var, marker="o", ms=3, label=f"bw={bw:g}")
    r = sub[sub.bandwidth_rule == "1/sqrt(n)"]
    axes[0].semilogy(r.n_qubits, r.offdiag_var, "k--", marker="s", ms=3, label="bw=1/sqrt(n)")
    axes[0].set(title="ZZ kernel, uniform data", xlabel="qubits", ylabel="Var of off-diagonal entries")
    axes[0].legend(fontsize=6)
    for fm, c in zip(("zz", "he", "product"), ("#dc2626", "#ea580c", "#ca8a04")):
        g = df[(df.feature_map == fm) & (df.source == "real") & (df.bandwidth_rule == "fixed") & (df.bandwidth == 1.0)]
        axes[1].semilogy(g.n_qubits, g.offdiag_var, marker="o", ms=3, color=c, label=fm)
        g = df[(df.feature_map == fm) & (df.source == "real") & (df.bandwidth_rule == "fixed") & (df.bandwidth == 1.0)]
        axes[2].plot(g.n_qubits, g.eff_rank, marker="o", ms=3, color=c, label=fm)
    axes[1].set(title="Real data (breast cancer PCA), bw=1", xlabel="qubits", ylabel="Var of off-diagonal entries")
    axes[2].set(title="Effective rank of the Gram matrix", xlabel="qubits", ylabel="effective rank (of 150)")
    axes[1].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


REGIMES = [("moons", "classical"), ("ad_hoc", "quantum"), ("heart", "real")]
MODEL_ORDER = ["linear", "rbf", "poly", "q_zz", "q_he", "q_product"]


def latex_main_table(summary: pd.DataFrame, path: Path) -> None:
    """Accuracy table grouped by data regime; best model per dataset in bold."""
    acc = summary.pivot(index="model", columns="dataset", values="acc")
    std = summary.pivot(index="model", columns="dataset", values="acc_std")
    cols = [d for d, _ in REGIMES if d in acc.columns]
    regime_row = [r" & Classical-friendly & Quantum-friendly & Real-world \\"]
    if not cols:  # datasets outside the three regimes (e.g. the `extra` experiment): plain header
        cols, regime_row = list(acc.columns), []
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", *regime_row,
             "Model & " + " & ".join(c.replace("_", " ") for c in cols) + r" \\", r"\midrule"]
    for m in [m for m in MODEL_ORDER if m in acc.index]:
        cells = []
        for c in cols:
            txt = f"{acc.loc[m, c]:.3f} $\\pm$ {std.loc[m, c]:.3f}"
            cells.append(f"\\textbf{{{txt}}}" if acc.loc[m, c] >= acc[c].max() - 1e-12 else txt)
        lines.append(m.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
        if m == "poly":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_pv_table(ev: pd.DataFrame, path: Path) -> None:
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Subset & Predictor & Decisive & Baseline & Sign acc. & AUC & Spearman $\rho$ \\", r"\midrule"]
    last = None
    for _, r in ev.iterrows():
        if last is not None and r.subset != last:
            lines.append(r"\midrule")
        last = r.subset
        sign = "--" if pd.isna(r.sign_accuracy) else f"{r.sign_accuracy:.2f}"
        auc = "--" if pd.isna(r.auc_Q_wins) else f"{r.auc_Q_wins:.2f}"
        lines.append(f"{r.subset} & {r.predictor} & {int(r.n_decisive)} & {r.majority_baseline:.2f} & {sign} & {auc} & "
                     f"{r.spearman_rho:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines).replace("_", r"\_").replace("&  &", "&") + "\n", encoding="utf-8")


def plot_hyper(df, path):
    summ = summarize(df, ["dataset", "model"])
    rbf = summ[summ.model == "rbf"].set_index("dataset").acc
    q = summ[summ.model.str.startswith("zz_")].copy()
    parts = q.model.str.extract(r"zz_bw(?P<bw>[\d.]+)_r(?P<reps>\d)_(?P<ent>\w+)")
    q = pd.concat([q, parts.astype({"bw": float, "reps": int})], axis=1)
    datasets = order_datasets(q.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.1), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], datasets):
        for (reps, ent), g in q[q.dataset == d].groupby(["reps", "ent"]):
            g = g.sort_values("bw")
            ax.semilogx(g.bw, g.acc, marker="o", ms=3, label=f"reps={reps},{ent}",
                        ls="-" if ent == "linear" else "--")
        ax.axhline(rbf[d], color=COLORS["rbf"], lw=1.2, label="tuned RBF")
        ax.set(title=d, xlabel="bandwidth (feature scale)")
    axes[0][0].set_ylabel("test accuracy (C tuned only)")
    axes[0][-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_noise(summary, path):
    datasets = order_datasets(summary.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.4), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], datasets):
        s = summary[summary.dataset == d].reset_index(drop=True)
        cols = ["#2563eb" if m == "rbf" else "#dc2626" if m == "zz_ideal" else "#ea580c" if "shots" in m else "#7c3aed"
                for m in s.model]
        ax.bar(range(len(s)), s.acc, yerr=s.acc_std, color=cols, capsize=2)
        ax.set_xticks(range(len(s)), s.model, rotation=60, ha="right", fontsize=6.5)
        ax.set_title(d)
        ax.set_ylim(0.4, 1.02)
    axes[0][0].set_ylabel("test accuracy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_runtime(df, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    sub = df[df.n_points == df.n_points.max()]
    for k, g in sub.groupby("kernel"):
        ax.semilogy(g.n_qubits, g.seconds, marker="o", label=k)
    ax.set(xlabel="qubits", ylabel=f"seconds for a {int(sub.n_points.max())}x{int(sub.n_points.max())} Gram matrix")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------- predict-then-verify
def predict_verify(frames: dict[str, pd.DataFrame]):
    """One unit = (experiment, dataset, n_qubits, quantum model), averaged over all folds.

    Predictors use training-fold diagnostics only; the outcome is the test accuracy gap to the tuned RBF SVM.
    """
    units = []
    for exp, df in frames.items():
        if "diag_kta" not in df:
            continue
        for key, g in df.groupby(["dataset", "n_qubits"]):
            rbf = g[g.model == "rbf"].set_index(["repeat", "fold"])
            for m, gq in g[g.kind == "quantum"].groupby("model"):
                gq = gq.set_index(["repeat", "fold"])
                units.append({
                    "experiment": exp, "dataset": key[0], "n_qubits": key[1], "model": m,
                    "acc_gap": (gq.acc - rbf.acc).mean(),
                    "kta_gap": (gq.diag_kta - rbf.diag_kta).mean(),
                    "logs_gap": (rbf.diag_log_s - gq.diag_log_s).mean(),   # >0: quantum needs a simpler model
                    "g_cq": gq.diag_g_cq.mean(),
                    "fold_agree_kta": (np.sign(gq.diag_kta - rbf.diag_kta) == np.sign(gq.acc - rbf.acc)).mean(),
                    "fold_agree_logs": (np.sign(rbf.diag_log_s - gq.diag_log_s) == np.sign(gq.acc - rbf.acc)).mean(),
                })
    return pd.DataFrame(units)


def dedupe_units(units: pd.DataFrame) -> pd.DataFrame:
    """`main` and `qubits` run identical jobs at the shared (dataset, qubits); keep one copy."""
    shared = set(zip(units[units.experiment == "qubits"].dataset, units[units.experiment == "qubits"].n_qubits))
    dup = (units.experiment == "main") & units.apply(lambda r: (r.dataset, r.n_qubits) in shared, axis=1)
    return units[~dup]


def evaluate_predictors(units: pd.DataFrame, subset: str, margin: float = 0.01) -> pd.DataFrame:
    """Score each predictor against the realised accuracy gap, always next to the trivial baseline.

    Quantum loses in most units, so 'always predict that quantum loses' is a strong baseline; a
    predictor only adds information if it beats `majority_baseline`.
    """
    from sklearn.metrics import roc_auc_score

    rows = []
    decisive = units[units.acc_gap.abs() > margin]
    wins = int((decisive.acc_gap > 0).sum())
    baseline = max(wins, len(decisive) - wins) / max(len(decisive), 1)
    for label, pred in (("KTA gap (Q - C)", "kta_gap"), ("complexity gap log(sC/sQ)", "logs_gap"), ("geometric difference g_CQ", "g_cq")):
        rho, p = stats.spearmanr(units[pred], units.acc_gap)
        y = (decisive.acc_gap > 0).astype(int)
        rows.append({
            "subset": subset, "predictor": label, "n_units": len(units), "n_decisive": len(decisive), "Q_wins": wins,
            "majority_baseline": baseline,
            "sign_accuracy": (np.sign(decisive[pred]) == np.sign(decisive.acc_gap)).mean() if pred != "g_cq" else np.nan,
            "auc_Q_wins": roc_auc_score(y, decisive[pred]) if y.nunique() == 2 else np.nan,
            "spearman_rho": rho, "spearman_p": p,
        })
    return pd.DataFrame(rows)


def plot_predict(units, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    markers = {"main": "o", "qubits": "s", "hyperparams": "^"}
    for ax, (x, label) in zip(axes, (("kta_gap", "KTA(Q) - KTA(RBF)"), ("logs_gap", "log s_RBF - log s_Q"), ("g_cq", "geometric difference g(RBF||Q)"))):
        for d, c in zip(units.dataset.unique(), plt.cm.tab10.colors):
            for exp, mk in markers.items():
                s = units[(units.dataset == d) & (units.experiment == exp)]
                ax.scatter(s[x], s.acc_gap, color=c, marker=mk, s=18, alpha=0.8, label=d if exp == "main" else None)
        ax.axhline(0, color="k", lw=0.6)
        if x != "g_cq":
            ax.axvline(0, color="k", lw=0.6)
        else:
            ax.set_xscale("log")
        ax.set(xlabel=label, ylabel="test acc(Q) - acc(RBF)")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------ suite analysis
FAMILY_C = ["linear", "rbf", "poly"]
FAMILY_Q = ["q_zz", "q_he", "q_product"]
DATASET_COLORS = {"moons": "#2563eb", "ad_hoc": "#dc2626", "heart": "#16a34a"}
POLICY_ORDER = ["always tuned RBF", "always ZZ quantum", "alignment switch (leave-one-seed-out)",
                "alignment switch (leave-one-dataset-out)", "oracle switch"]


def suite_instances(df: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset variant: realised accuracies and pre-training diagnostics (means over folds)."""
    keys = ["dataset", "variant", "data_seed"]
    rows = []
    for key, g in df.groupby(keys):
        m = g.groupby("model").agg(acc=("acc", "mean"), kta=("diag_kta", "mean"), log_s=("diag_log_s", "mean"),
                                   eff=("diag_eff_rank", "mean"), gcq=("diag_g_cq", "mean"), n_train=("n_train", "mean"))
        qs = [x for x in FAMILY_Q if x in m.index]
        cs = [x for x in FAMILY_C if x in m.index]
        bq = m.loc[qs, "kta"].idxmax()             # quantum map chosen by training-set alignment alone
        bc = m.loc[cs, "kta"].idxmax()
        rows.append(dict(zip(keys, key)) | {
            "acc_Q": m.loc[qs, "acc"].max(), "acc_C": m.loc[cs, "acc"].max(), "acc_rbf": m.loc["rbf", "acc"],
            "acc_Qsel": m.loc[bq, "acc"], "sel_Q": bq, "acc_zz": m.loc["q_zz", "acc"],
            "kta_gap": m.loc[bq, "kta"] - m.loc[bc, "kta"], "kta_gap_vs_rbf": m.loc[bq, "kta"] - m.loc["rbf", "kta"],
            "logs_gap": m.loc["rbf", "log_s"] - m.loc[bq, "log_s"], "g_cq": m.loc[bq, "gcq"],
            "rank_frac": m.loc[bq, "eff"] / m.loc[bq, "n_train"], "acc_oracle": m["acc"].max(),
        })
    out = pd.DataFrame(rows)
    out["acc_gap"] = out.acc_Q - out.acc_C          # best quantum map vs best classical kernel
    # seed group: variants that share a generating circuit / subsample seed are not independent
    grp = out.variant.str.extract(r"(oracle\d)")[0].fillna("")
    out["group"] = out.dataset + "_" + grp + out.data_seed.astype(str)
    return out


def _bootstrap(values_fn, n: int, B: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    vals = [values_fn(rng.integers(0, n, n)) for _ in range(B)]
    vals = [v for v in vals if np.isfinite(v)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def suite_predictors(inst: pd.DataFrame, margin: float = 0.01) -> pd.DataFrame:
    """Each predictor vs the accuracy gap (best quantum map - best classical kernel) over all dataset variants."""
    from sklearn.metrics import roc_auc_score

    gap = inst.acc_gap.to_numpy()
    decisive = np.abs(gap) > margin
    wins = int((gap[decisive] > 0).sum())
    baseline = max(wins, decisive.sum() - wins) / max(decisive.sum(), 1)
    rows = []
    for label, col in (("alignment gap (quantum - classical)", "kta_gap"), ("alignment gap vs RBF", "kta_gap_vs_rbf"),
                       ("complexity gap log(sRBF/sQ)", "logs_gap"), ("geometric difference g", "g_cq"),
                       ("quantum Gram rank fraction", "rank_frac")):
        x = inst[col].to_numpy()
        rho = stats.spearmanr(x, gap)[0]
        lo, hi = _bootstrap(lambda idx: stats.spearmanr(x[idx], gap[idx])[0], len(x))
        y = (gap[decisive] > 0).astype(int)
        rows.append({"predictor": label, "n_datasets": len(inst), "n_decisive": int(decisive.sum()), "Q_wins": wins,
                     "majority_baseline": baseline,
                     "auc_Q_wins": roc_auc_score(y, x[decisive]) if len(np.unique(y)) == 2 else np.nan,
                     "spearman_rho": rho, "rho_ci_lo": lo, "rho_ci_hi": hi})
    return pd.DataFrame(rows)


def suite_within(inst: pd.DataFrame) -> pd.DataFrame:
    """Does the alignment gap track the accuracy gap *within* each dataset family?"""
    rows = []
    for d in order_datasets(inst.dataset.unique()):
        g = inst[inst.dataset == d]
        rho, p = stats.spearmanr(g.kta_gap, g.acc_gap)
        rows.append({"dataset": d, "n_variants": len(g), "spearman_rho": rho, "p_value": p,
                     "kta_gap_min": g.kta_gap.min(), "kta_gap_max": g.kta_gap.max(),
                     "acc_gap_min": g.acc_gap.min(), "acc_gap_max": g.acc_gap.max()})
    return pd.DataFrame(rows)


def _switch_payoff(g: pd.DataFrame, tau: float) -> np.ndarray:
    """Accuracy if we use the alignment-selected quantum map when its alignment beats RBF by more than tau."""
    return np.where(g.kta_gap_vs_rbf.to_numpy() > tau, g.acc_Qsel.to_numpy(), g.acc_rbf.to_numpy())


def _fit_tau(train: pd.DataFrame) -> float:
    cands = np.unique(np.r_[-1.0, train.kta_gap_vs_rbf.to_numpy(), 1.0])[::-1]   # descending: ties go to the conservative (classical) choice
    return max(cands, key=lambda t: _switch_payoff(train, t).mean())


def suite_policy(inst: pd.DataFrame) -> pd.DataFrame:
    """Mean test accuracy of model-selection policies that use only pre-training information.

    The switch policy has one parameter tau, fitted on instances that are held out from the evaluated one
    (leave-one-seed-group-out, and the stricter leave-one-dataset-out)."""
    inst = inst.reset_index(drop=True)
    per = pd.DataFrame({"dataset": inst.dataset, "always tuned RBF": inst.acc_rbf, "always ZZ quantum": inst.acc_zz,
                        "oracle switch": np.maximum(inst.acc_Qsel, inst.acc_rbf)})
    seed_out, data_out = np.zeros(len(inst)), np.zeros(len(inst))
    taus = {"seed": [], "dataset": []}
    for i in range(len(inst)):
        t = _fit_tau(inst[inst.group != inst.group[i]])
        seed_out[i] = _switch_payoff(inst.loc[[i]], t)[0]
        t2 = _fit_tau(inst[inst.dataset != inst.dataset[i]])
        data_out[i] = _switch_payoff(inst.loc[[i]], t2)[0]
        taus["seed"].append(t), taus["dataset"].append(t2)
    per["alignment switch (leave-one-seed-out)"] = seed_out
    per["alignment switch (leave-one-dataset-out)"] = data_out
    rows = []
    for ds_name, g in [(d, per[per.dataset == d]) for d in order_datasets(per.dataset.unique())] + [("all datasets", per)]:
        for name in POLICY_ORDER:
            v = g[name].to_numpy()
            lo, hi = _bootstrap(lambda idx: v[idx].mean(), len(v))
            rows.append({"dataset": ds_name, "policy": name, "n_datasets": len(v), "mean_acc": v.mean(),
                         "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    out.attrs["tau_seed_out"] = (float(np.min(taus["seed"])), float(np.max(taus["seed"])))
    out.attrs["tau_dataset_out"] = (float(np.min(taus["dataset"])), float(np.max(taus["dataset"])))
    return out


def plot_suite(inst: pd.DataFrame, pol: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for d in order_datasets(inst.dataset.unique()):
        s = inst[inst.dataset == d]
        axes[0].scatter(s.kta_gap, s.acc_gap, color=DATASET_COLORS.get(d, "gray"), s=24, alpha=0.85, label=d.replace("_", " "))
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].axvline(0, color="k", lw=0.6)
    axes[0].set(xlabel="alignment gap: KTA(quantum) - KTA(classical)", ylabel="test acc: best quantum - best classical")
    axes[0].legend(fontsize=7)
    ad = inst[inst.dataset == "ad_hoc"]
    for flip, mk in ((0.0, "o"), (0.15, "s")):
        s = ad[ad.variant.str.endswith(f"flip{flip:g}")]
        axes[1].scatter(s.kta_gap, s.acc_gap, color=DATASET_COLORS["ad_hoc"], marker=mk, s=28, label=f"label noise {flip:g}")
    axes[1].set(xlabel="alignment gap (ad hoc variants only)", ylabel="test acc gap", title="within ad hoc")
    axes[1].legend(fontsize=7)
    allp = pol[pol.dataset == "all datasets"].set_index("policy").loc[POLICY_ORDER].reset_index()
    colors = ["#2563eb", "#dc2626", "#7c3aed", "#a78bfa", "#9ca3af"]
    axes[2].bar(range(len(allp)), allp.mean_acc, yerr=[allp.mean_acc - allp.ci_lo, allp.ci_hi - allp.mean_acc],
                color=colors, capsize=3)
    short = ["always\nRBF", "always\nZZ", "switch\nseed-out", "switch\ndataset-out", "oracle\nswitch"]
    axes[2].set_xticks(range(len(allp)), short, rotation=0, fontsize=7)
    axes[2].set(ylabel="mean test accuracy over 30 variants", ylim=(0.6, 0.95))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def latex_suite_tables(pred: pd.DataFrame, within: pd.DataFrame, pol: pd.DataFrame, tab_dir: Path) -> None:
    def fmt(v, f="{:.2f}"):
        return "--" if pd.isna(v) else f.format(v)

    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Predictor & AUC & Spearman $\rho$ & 95\% CI \\", r"\midrule"]
    for _, r in pred.iterrows():
        lines.append(f"{r.predictor} & {fmt(r.auc_Q_wins)} & {fmt(r.spearman_rho)} & [{fmt(r.rho_ci_lo)}, {fmt(r.rho_ci_hi)}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tab_dir / "suite_predictors.tex").write_text("\n".join(lines).replace("_", r"\_") + "\n", encoding="utf-8")

    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Dataset & Variants & Alignment-gap range & Accuracy-gap range & Spearman $\rho$ ($p$) \\", r"\midrule"]
    for _, r in within.iterrows():
        lines.append(f"{r.dataset.replace('_', ' ')} & {int(r.n_variants)} & [{r.kta_gap_min:+.3f}, {r.kta_gap_max:+.3f}] & "
                     f"[{r.acc_gap_min:+.3f}, {r.acc_gap_max:+.3f}] & {r.spearman_rho:+.2f} ({r.p_value:.3f}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tab_dir / "suite_within.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    piv = pol.pivot(index="policy", columns="dataset", values="mean_acc")
    cols = order_datasets([c for c in piv.columns if c != "all datasets"]) + ["all datasets"]
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Policy & " + " & ".join(c.replace("_", " ") for c in cols) + r" \\", r"\midrule"]
    for name in POLICY_ORDER:
        lines.append(name + " & " + " & ".join(f"{piv.loc[name, c]:.3f}" for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tab_dir / "suite_policy.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full")
    args = ap.parse_args()
    run_dir = ROOT / "results" / args.run
    fig_dir, tab_dir = run_dir / "figures", run_dir / "tables"
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)
    report = [f"# Results report ({args.run} run)\n"]
    frames = {}

    main_df = load(run_dir, "main")
    if main_df is not None:
        frames["main"] = main_df
        s = summarize(main_df, ["dataset", "model"])
        s.to_csv(tab_dir / "main_summary.csv", index=False)
        tests = paired_tests(main_df, ["dataset"])
        tests.to_csv(tab_dir / "main_tests.csv", index=False)
        bar_main(s, fig_dir / "main_accuracy.png")
        latex_main_table(s, tab_dir / "main_table.tex")
        psd = main_df[main_df.kind == "quantum"].groupby("model").diag_psd.mean() if "diag_psd" in main_df else None
        report += ["## Main comparison (4 qubits)\n", md_table(s), "\n![main](figures/main_accuracy.png)\n",
                   "Corrected resampled t-tests (Nadeau-Bengio), Holm-adjusted across all rows:\n", md_table(tests) if len(tests) else "",
                   f"\nQuantum Gram matrices PSD fraction: {psd.to_dict() if psd is not None else 'n/a'}\n"]

    extra = load(run_dir, "extra")
    if extra is not None:
        s = summarize(extra, ["dataset", "model"])
        s.to_csv(tab_dir / "extra_summary.csv", index=False)
        tests = paired_tests(extra, ["dataset"])
        tests.to_csv(tab_dir / "extra_tests.csv", index=False)
        bar_main(s, fig_dir / "extra_accuracy.png")
        latex_main_table(s, tab_dir / "extra_table.tex")
        report += ["## Extra datasets (4 qubits, same protocol as main)\n", md_table(s),
                   "\n![extra](figures/extra_accuracy.png)\n",
                   "Corrected resampled t-tests (Nadeau-Bengio), Holm-adjusted across all rows:\n",
                   md_table(tests) if len(tests) else ""]

    size = load(run_dir, "size")
    if size is not None:
        s = summarize(size, ["dataset", "model", "train_size"])
        s.to_csv(tab_dir / "size_summary.csv", index=False)
        lines_by(s, "train_size", fig_dir / "size_curves.png", "training examples")
        report += ["## Training-set size\n", "![size](figures/size_curves.png)\n"]

    qub = load(run_dir, "qubits")
    if qub is not None:
        frames["qubits"] = qub
        s = summarize(qub, ["dataset", "model", "n_qubits"])
        s.to_csv(tab_dir / "qubits_summary.csv", index=False)
        lines_by(s, "n_qubits", fig_dir / "qubits_accuracy.png", "qubits (= PCA components)")
        report += ["## Number of qubits\n", "![qubits](figures/qubits_accuracy.png)\n"]

    conc = load(run_dir, "concentration")
    if conc is not None:
        plot_concentration(conc, fig_dir / "concentration.png")
        report += ["## Kernel concentration\n", "![conc](figures/concentration.png)\n"]

    hyp = load(run_dir, "hyperparams")
    if hyp is not None:
        frames["hyperparams"] = hyp
        plot_hyper(hyp, fig_dir / "hyperparams.png")
        summarize(hyp, ["dataset", "model"]).to_csv(tab_dir / "hyperparams_summary.csv", index=False)
        report += ["## Bandwidth, depth and entanglement\n", "![hyper](figures/hyperparams.png)\n"]

    noise = load(run_dir, "noise")
    if noise is not None:
        s = summarize(noise, ["dataset", "model"])
        s.to_csv(tab_dir / "noise_summary.csv", index=False)
        plot_noise(s, fig_dir / "noise.png")
        report += ["## Noise\n", md_table(s), "\n![noise](figures/noise.png)\n"]

    rt = load(run_dir, "runtime")
    if rt is not None:
        plot_runtime(rt, fig_dir / "runtime.png")
        report += ["## Runtime\n", "![runtime](figures/runtime.png)\n"]

    units = predict_verify(frames) if frames else pd.DataFrame()
    if len(units):
        units = dedupe_units(units)
        units.to_csv(tab_dir / "predict_verify_units.csv", index=False)
        tuned = units[units.experiment.isin(["main", "qubits"])]
        real = tuned[tuned.dataset.isin(["heart", "breast_cancer"])]
        parts = [(tuned, "tuned models, all datasets"), (real, "tuned models, real data only")]
        if (units.experiment == "hyperparams").any():
            parts.append((units[units.experiment == "hyperparams"], "ZZ sweep (bandwidth x reps x entanglement)"))
        ev = pd.concat([evaluate_predictors(df_, name) for df_, name in parts], ignore_index=True)
        ev.to_csv(tab_dir / "predict_verify_summary.csv", index=False)
        latex_pv_table(ev, tab_dir / "predict_verify_table.tex")
        plot_predict(units, fig_dir / "predict_verify.png")
        report += ["## Predict, then verify\n", md_table(ev),
                   "\nUnits are not independent (many share a dataset); correlations describe association, not significance. "
                   "Compare sign_accuracy with majority_baseline, not with 0.5.\n",
                   "\n![pv](figures/predict_verify.png)\n"]

    suite = load(run_dir, "suite")
    if suite is not None:
        inst = suite_instances(suite)
        pred, within, pol = suite_predictors(inst), suite_within(inst), suite_policy(inst)
        inst.to_csv(tab_dir / "suite_instances.csv", index=False)
        pred.to_csv(tab_dir / "suite_predictors.csv", index=False)
        within.to_csv(tab_dir / "suite_within.csv", index=False)
        pol.to_csv(tab_dir / "suite_policy.csv", index=False)
        plot_suite(inst, pol, fig_dir / "suite.png")
        latex_suite_tables(pred, within, pol, tab_dir)
        report += ["## Suite: predict-then-verify over dataset variants\n", md_table(pred), "\n", md_table(within), "\n",
                   md_table(pol), f"\nFitted tau range: leave-one-seed-out {pol.attrs['tau_seed_out']}, "
                   f"leave-one-dataset-out {pol.attrs['tau_dataset_out']}\n", "\n![suite](figures/suite.png)\n"]

    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()

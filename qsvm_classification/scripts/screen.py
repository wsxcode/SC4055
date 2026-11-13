"""Run the screening tool (recommend_kernel) on every dataset and compare with the measured outcome.

    python scripts/screen.py --run full            # writes results/<run>/tables/tool.{csv,tex}
    python scripts/screen.py --fit-tau             # recompute DEFAULT_TAU from the 30 suite variants

Needs results/<run>/main and results/<run>/extra (from run_experiments.py) for the measured column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qsvm_clf.data import FIXED_DIM, load_dataset  # noqa: E402
from qsvm_clf.screening import DEFAULT_TAU, recommend_kernel  # noqa: E402

DATASETS = [("moons", False), ("ad_hoc", False), ("heart", False),
            ("breast_cancer", True), ("digits", True), ("wine", True), ("circles", True)]  # (name, held out?)
PRETTY = {"moons": "moons", "ad_hoc": "ad hoc", "heart": "Heart Disease", "breast_cancer": "Breast Cancer",
          "digits": "Digits", "wine": "Wine", "circles": "Circles"}
SHORT = {"q_zz": "ZZ", "q_he": "HE", "q_product": "product", "rbf": "RBF", "linear": "linear", "poly": "poly"}
QUANTUM = ("q_zz", "q_he", "q_product")
MARGIN = 0.01  # |accuracy gap| <= MARGIN counts as a tie, as in the report


def measured(results: pd.DataFrame, dataset: str) -> tuple[float, str]:
    acc = results[results.dataset == dataset].groupby("model").acc.mean()
    q = acc[acc.index.isin(QUANTUM)]
    c = acc[~acc.index.isin(QUANTUM)]
    gap = q.max() - c.max()
    if gap > MARGIN:
        return gap, SHORT[q.idxmax()]
    if gap < -MARGIN:
        return gap, SHORT[c.idxmax()]
    return gap, "tie"


def fit_tau() -> float:
    import analyze

    inst = pd.read_csv(ROOT / "results" / "full" / "tables" / "suite_instances.csv")
    return float(analyze._fit_tau(inst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full")
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--fit-tau", action="store_true")
    args = ap.parse_args()

    if args.fit_tau:
        print(f"tau fitted on all suite variants: {fit_tau():.4f} (DEFAULT_TAU = {DEFAULT_TAU})")
        return

    run = ROOT / "results" / args.run
    results = pd.concat([pd.read_csv(run / e / "results.csv") for e in ("main", "extra")], ignore_index=True)
    rows = []
    for name, held_out in DATASETS:
        nq = FIXED_DIM.get(name, args.n_qubits)
        ds = load_dataset(name, args.n_samples, 0, nq)
        r = recommend_kernel(ds, n_qubits=nq)
        gap, outcome = measured(results, name)
        quantum_won = outcome in {SHORT[q] for q in QUANTUM}
        rows.append({
            "dataset": name, "held_out": held_out, "delta": r["delta"], "tau": r["tau"],
            "tool_says": SHORT[r["recommendation"]], "best_quantum_map": SHORT[r["best_quantum_map"]],
            "kta_rbf": r["alignment"]["rbf"], "kta_best_q": r["alignment"][r["best_quantum_map"]],
            "rank_frac": r["rank_frac"], "acc_gap": gap, "measured": outcome,
            "correct": r["use_quantum"] == quantum_won,
        })
        print(f"{name:14s} delta={r['delta']:+.3f} -> {SHORT[r['recommendation']]:7s} measured={outcome}", flush=True)
    df = pd.DataFrame(rows)
    tab = run / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    df.to_csv(tab / "tool.csv", index=False)

    lines = [r"\begin{tabular}{@{}lccccc@{}}\toprule",
             r"Dataset & $\Delta$ & Tool says & Measured & OK? & Held out?\\\midrule"]
    for r in df.itertuples():
        ok = "yes" if r.correct else r"\textbf{no}"
        lines.append(f"{PRETTY[r.dataset]} & ${r.delta:.3f}$ & {r.tool_says} & {r.measured} & {ok} & "
                     f"{'yes' if r.held_out else 'no'}\\\\")
    lines[-1] += r"\bottomrule"
    lines.append(r"\end{tabular}")
    (tab / "tool_table.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {tab / 'tool.csv'} and {tab / 'tool_table.tex'}")


if __name__ == "__main__":
    main()

"""Extra analyses for the report: cluster bootstrap over seed groups, and sensitivity of the
complexity gap / geometric difference to the regularisation lambda.

Reads results/full/tables/suite_instances.csv and the package in src/. Changes nothing else.
Usage (from qsvm_classification/):  python scripts/extra_analysis.py
Writes results/full/tables/lambda_sensitivity.csv and prints the numbers quoted in the report.
"""
import sys, re
from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_PATH / "src"))
import numpy as np, pandas as pd, yaml
from scipy.stats import spearmanr, rankdata
from qsvm_clf.data import load_dataset, FoldPreprocessor
from qsvm_clf.kernels import compute_gram
from qsvm_clf.evaluate import model_from_config
import qsvm_clf.diagnostics as dg

ROOT = str(ROOT_PATH) + "/"
inst = pd.read_csv(ROOT + "results/full/tables/suite_instances.csv")

def auc(score, pos):  # Mann-Whitney AUC, pos boolean
    s = np.asarray(score); p = np.asarray(pos)
    if p.sum() == 0 or (~p).sum() == 0: return np.nan
    r = rankdata(s)
    return (r[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (~p).sum())

# ---------- A. cluster bootstrap over groups (circuit / subsample seed) ----------
groups = inst.group.unique()
def stats(df):
    dec = df[df.acc_gap.abs() > 0.01]
    return (spearmanr(df.kta_gap, df.acc_gap)[0],
            auc(dec.kta_gap, (dec.acc_gap > 0).to_numpy()))
rho0, auc0 = stats(inst)
rng = np.random.default_rng(0); B = 5000
rows = []
for _ in range(B):
    pick = rng.choice(groups, len(groups), replace=True)
    df = pd.concat([inst[inst.group == g] for g in pick], ignore_index=True)
    r, a = stats(df)
    rows.append((r, a))
rows = np.array(rows)
ci = lambda v: tuple(np.round(np.nanpercentile(v, [2.5, 97.5]), 3))
print("A1 spearman rho", round(rho0, 3), "cluster-boot CI", ci(rows[:, 0]), "n finite", np.isfinite(rows[:, 0]).sum())
print("A1 AUC", round(auc0, 3), "cluster-boot CI", ci(rows[:, 1]), "n finite", np.isfinite(rows[:, 1]).sum())

# policy replication (leave-one-group-out tau)
def payoff(g, t): return np.where(g.kta_gap_vs_rbf.to_numpy() > t, g.acc_Qsel.to_numpy(), g.acc_rbf.to_numpy())
def fit_tau(tr):
    c = np.unique(np.r_[-1.0, tr.kta_gap_vs_rbf.to_numpy(), 1.0])[::-1]
    return max(c, key=lambda t: payoff(tr, t).mean())
inst = inst.reset_index(drop=True)
rule = np.array([payoff(inst.loc[[i]], fit_tau(inst[inst.group != inst.group[i]]))[0] for i in range(len(inst))])
pol = {"rule": rule, "rbf": inst.acc_rbf.to_numpy(), "zz": inst.acc_zz.to_numpy(),
       "oracle": np.maximum(inst.acc_Qsel, inst.acc_rbf).to_numpy()}
print("A2 replicate means", {k: round(v.mean(), 3) for k, v in pol.items()})
gidx = {g: np.where(inst.group == g)[0] for g in groups}
def boot_diff(a, b, B=5000):
    r = np.random.default_rng(1); out = []
    for _ in range(B):
        pick = r.choice(groups, len(groups), replace=True)
        idx = np.concatenate([gidx[g] for g in pick])
        out.append((pol[a][idx] - pol[b][idx]).mean())
    return round(float((pol[a] - pol[b]).mean()), 3), tuple(np.round(np.percentile(out, [2.5, 97.5]), 3))
for a, b in [("rule", "rbf"), ("rule", "zz"), ("zz", "rbf"), ("oracle", "rule")]:
    print("A2 mean diff", a, "-", b, boot_diff(a, b))
# rule CI by cluster bootstrap
r = np.random.default_rng(2); ms = []
for _ in range(5000):
    pick = r.choice(groups, len(groups), replace=True)
    ms.append(rule[np.concatenate([gidx[g] for g in pick])].mean())
print("A2 rule mean cluster CI", round(rule.mean(), 3), tuple(np.round(np.percentile(ms, [2.5, 97.5]), 3)))
# ad hoc within-family rho cluster bootstrap over 3 circuits
ad = inst[inst.dataset == "ad_hoc"]; ag = ad.group.unique(); rr = []
r = np.random.default_rng(3)
for _ in range(5000):
    pick = r.choice(ag, len(ag), replace=True)
    d = pd.concat([ad[ad.group == g] for g in pick]); 
    if d.kta_gap.nunique() > 2: rr.append(spearmanr(d.kta_gap, d.acc_gap)[0])
print("A3 ad hoc within rho", round(spearmanr(ad.kta_gap, ad.acc_gap)[0], 3), "circuit-cluster CI", ci(rr))
# margin: variants within 0.002 of tau
tau = 0.0063
near = inst[(inst.kta_gap_vs_rbf - tau).abs() < 0.002]
print("A4 variants within 0.002 of tau:", len(near), near[["dataset", "variant", "kta_gap_vs_rbf", "acc_gap"]].round(4).to_dict("records"))
print("A4 switch decisions at tau:", (inst.kta_gap_vs_rbf > tau).groupby(inst.dataset).sum().to_dict())

# ---------- B. lambda sensitivity (whole-dataset approximation) ----------
cfg = yaml.safe_load(open(ROOT + "configs/default.yaml"))
models = {n: model_from_config(n, c, cfg["C"]) for n, c in cfg["models"].items()}
ns = cfg["experiments"]["suite"]["n_samples"]
def make_ds(row):
    v = row.variant
    if row.dataset == "moons":
        return load_dataset("moons", ns, int(row.data_seed), noise=float(v.replace("noise", ""))), 2
    if row.dataset == "ad_hoc":
        m = re.match(r"n(\d)_oracle(\d)_flip([\d.]+)", v)
        nq, o, fl = int(m[1]), int(m[2]), float(m[3])
        return load_dataset("ad_hoc", ns, 0, n_features=nq, oracle_seed=o, flip=fl), nq
    nq = int(v.replace("nq", "")); return load_dataset("heart", ns, int(row.data_seed)), nq
lams = [1e-4, 1e-3, 1e-2, 1e-1]
res = []
for i, row in inst.iterrows():
    ds, nq = make_ds(row)
    X = FoldPreprocessor(ds, nq).fit_transform(ds.X); y = ds.y
    best = {}
    for n, m in models.items():
        if n in ("linear", "poly"): continue
        grams = [compute_gram(c, X) for c in m.kernel_configs()]
        k = [dg.centered_alignment(K, y) for K in grams]; j = int(np.argmax(k)); best[n] = (grams[j], k[j])
    bq = max([n for n in best if n.startswith("q_")], key=lambda n: best[n][1])
    Kr, Kq = best["rbf"][0], best[bq][0]
    for lam in lams:
        res.append(dict(i=i, lam=lam, acc_gap=row.acc_gap, dataset=row.dataset,
                        logs_gap=np.log(dg.model_complexity(Kr, y, lam)) - np.log(dg.model_complexity(Kq, y, lam)),
                        g=dg.geometric_difference(Kr, Kq, lam), kta_gap=best[bq][1] - best["rbf"][1]))
R = pd.DataFrame(res)
print("B check corr of whole-data kta_gap vs suite kta_gap_vs_rbf:",
      round(spearmanr(R[R.lam == 1e-3].kta_gap.to_numpy(), inst.kta_gap_vs_rbf.to_numpy())[0], 3))
for lam in lams:
    d = R[R.lam == lam]; dec = d[d.acc_gap.abs() > 0.01]
    print(f"B lam={lam:g}: complexity-gap rho={spearmanr(d.logs_gap, d.acc_gap)[0]:.2f} AUC={auc(dec.logs_gap, (dec.acc_gap>0).to_numpy()):.2f} | "
          f"g rho={spearmanr(d.g, d.acc_gap)[0]:.2f} AUC={auc(dec.g, (dec.acc_gap>0).to_numpy()):.2f} | g range {d.g.min():.1f}-{d.g.max():.1f}")
R.to_csv(ROOT + "results/full/tables/lambda_sensitivity.csv", index=False)

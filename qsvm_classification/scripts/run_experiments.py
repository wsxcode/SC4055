"""Run the experiments defined in configs/default.yaml.

    python scripts/run_experiments.py --experiment main --smoke
    python scripts/run_experiments.py --experiment all
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qsvm_clf import diagnostics as dg  # noqa: E402
from qsvm_clf.data import FIXED_DIM, load_dataset  # noqa: E402
from qsvm_clf.evaluate import ModelSpec, expand_grid, model_from_config, run_cv  # noqa: E402
from qsvm_clf.kernels import compute_gram  # noqa: E402

EXPERIMENTS = ["main", "extra", "size", "qubits", "concentration", "hyperparams", "noise", "runtime", "suite"]


# ------------------------------------------------------------------ helpers
def build_models(cfg, names=None):
    names = names or list(cfg["models"])
    return [model_from_config(n, cfg["models"][n], tuple(cfg["C"])) for n in names]


def n_components_for(dataset: str, n: int) -> int:
    return FIXED_DIM.get(dataset, n)


def cv_job(cfg, dataset, n_qubits, models, *, train_size=None, diagnostics=False, meta=None, cv=None,
           ds_opts=None, data_seed=None):
    nq = n_components_for(dataset, n_qubits)
    # Each finished job is saved on its own, so an interrupted run loses nothing and --resume skips it.
    tag = "_".join([dataset, f"nq{nq}"] + [f"{k}{v}" for k, v in sorted((meta or {}).items())])
    partial = Path(cfg["_partial_dir"]) / f"{tag}.csv"
    if partial.exists():
        print(f"  reuse {tag}", flush=True)
        return pd.read_csv(partial)
    ds = load_dataset(dataset, cfg["_n_samples"], cfg["seed"] if data_seed is None else data_seed, nq,
                      **(ds_opts or {}))
    t0 = time.time()
    df = run_cv(
        ds, nq, models, cv or cfg["cv"], seed=cfg["seed"], train_size=train_size,
        cache_dir=ROOT / cfg["cache_dir"], diagnostics=diagnostics, lam=cfg["lambda_reg"], meta=meta,
    )
    partial.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(partial, index=False)
    print(f"  done {tag} in {time.time() - t0:.0f}s", flush=True)
    return df


def run_jobs(cfg, jobs):
    out = Parallel(n_jobs=cfg["n_jobs"])(delayed(cv_job)(cfg, **j) for j in jobs)
    return pd.concat(out, ignore_index=True)


# -------------------------------------------------------------- experiments
def exp_main(cfg, e):
    cfg["_n_samples"] = e["n_samples"]
    models = build_models(cfg)
    return run_jobs(cfg, [dict(dataset=d, n_qubits=e["n_qubits"], models=models, diagnostics=True) for d in e["datasets"]])


def exp_size(cfg, e):
    cfg["_n_samples"] = e["n_samples"]
    models = build_models(cfg)
    jobs = [dict(dataset=d, n_qubits=e["n_qubits"], models=models, train_size=s, meta={"train_size": s})
            for d in e["datasets"] for s in e["train_sizes"]]
    return run_jobs(cfg, jobs)


def exp_qubits(cfg, e):
    cfg["_n_samples"] = e["n_samples"]
    models = build_models(cfg)
    return run_jobs(cfg, [dict(dataset=d, n_qubits=n, models=models, diagnostics=True)
                          for d in e["datasets"] for n in e["n_qubits"]])


def exp_hyperparams(cfg, e):
    cfg["_n_samples"] = e["n_samples"]
    base = build_models(cfg, ["rbf"])
    models = list(base)
    for bw in e["bandwidths"]:
        for reps in e["reps"]:
            for ent in e["entanglement"]:
                models.append(ModelSpec(
                    name=f"zz_bw{bw:g}_r{reps}_{ent}", kind="quantum", kernel="zz",
                    grid=[{"bandwidth": bw}], fixed={"reps": reps, "entanglement": ent}, C=tuple(cfg["C"]),
                ))
    return run_jobs(cfg, [dict(dataset=d, n_qubits=e["n_qubits"], models=models, diagnostics=True)
                          for d in e["datasets"]])


def exp_noise(cfg, e):
    cfg["_n_samples"] = e["n_samples"]
    bw = [{"bandwidth": b} for b in e["bandwidths"]]
    quantum_base = dict(kind="quantum", kernel="zz", reps=2, entanglement="linear")
    models = build_models(cfg, ["rbf"])

    def add(name, noise):
        models.append(ModelSpec(name, "quantum", "zz", bw, {**{k: v for k, v in quantum_base.items()
                                if k not in ("kind", "kernel")}, "noise": noise}, tuple(cfg["C"])))

    add("zz_ideal", {"mode": "ideal"})
    for s in e["shots"]:
        add(f"zz_shots{s}", {"mode": "shots", "shots": s})
    for p1, p2 in e["depolarizing"]:
        add(f"zz_aer_{p1:g}_{p2:g}", {"mode": "aer", "depolarizing_1q": p1, "depolarizing_2q": p2})
    return run_jobs(cfg, [dict(dataset=d, n_qubits=e["n_qubits"], models=models, cv=e.get("cv"))
                          for d in e["datasets"]])


def exp_concentration(cfg, e):
    """Descriptive: label-free kernel statistics on random and real data."""
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    real = load_dataset(e["real_dataset"], 400, cfg["seed"])
    from qsvm_clf.data import FoldPreprocessor

    for nq in e["n_qubits"]:
        sources = {"uniform": rng.uniform(0, np.pi, size=(e["n_points"], nq))}
        pre = FoldPreprocessor(real, nq).fit_transform(real.X)
        sources["real"] = pre[: e["n_points"]]
        for src, X in sources.items():
            for fm in ("zz", "he", "product"):
                for bw in e["bandwidths"]:
                    K = compute_gram({"kind": "quantum", "kernel": fm, "reps": 2, "entanglement": "linear",
                                      "bandwidth": bw}, X)
                    rows.append({"n_qubits": nq, "source": src, "feature_map": fm, "bandwidth": bw,
                                 "bandwidth_rule": "fixed", **dg.concentration(K), "eff_rank": dg.effective_rank(K)})
                # bandwidth scaling that counters concentration: bw = 1/sqrt(n)
                bw = 1.0 / np.sqrt(nq)
                K = compute_gram({"kind": "quantum", "kernel": fm, "reps": 2, "entanglement": "linear",
                                  "bandwidth": bw}, X)
                rows.append({"n_qubits": nq, "source": src, "feature_map": fm, "bandwidth": bw,
                             "bandwidth_rule": "1/sqrt(n)", **dg.concentration(K), "eff_rank": dg.effective_rank(K)})
    return pd.DataFrame(rows)


def exp_suite(cfg, e):
    """Many variants of the same three datasets, so that predict-then-verify is tested on many cases."""
    cfg["_n_samples"] = e["n_samples"]
    models = build_models(cfg)
    jobs = []
    for noise in e["moons"]["noise"]:
        for s in e["moons"]["data_seeds"]:
            jobs.append(dict(dataset="moons", n_qubits=2, models=models, diagnostics=True, data_seed=s,
                             ds_opts={"noise": noise}, meta={"variant": f"noise{noise:g}", "data_seed": s}))
    for nf in e["ad_hoc"]["n_features"]:
        for o in e["ad_hoc"]["oracle_seeds"]:
            for flip in e["ad_hoc"]["flips"]:
                jobs.append(dict(dataset="ad_hoc", n_qubits=nf, models=models, diagnostics=True, data_seed=0,
                                 ds_opts={"oracle_seed": o, "flip": flip},
                                 meta={"variant": f"n{nf}_oracle{o}_flip{flip:g}", "data_seed": 0}))
    for s in e["heart"]["data_seeds"]:
        for nq in e["heart"]["n_qubits"]:
            jobs.append(dict(dataset="heart", n_qubits=nq, models=models, diagnostics=True, data_seed=s,
                             meta={"variant": f"nq{nq}", "data_seed": s}))
    return run_jobs(cfg, jobs)


def exp_runtime(cfg, e):
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for nq in e["n_qubits"]:
        for n in e["n_points"]:
            X = rng.uniform(0, np.pi, size=(n, nq))
            specs = {
                "rbf": {"kind": "classical", "kernel": "rbf"},
                "zz_numpy": {"kind": "quantum", "kernel": "zz"},
                "zz_aer_noisy": {"kind": "quantum", "kernel": "zz",
                                 "noise": {"mode": "aer", "depolarizing_1q": 0.001, "depolarizing_2q": 0.01}},
            }
            for name, spec in specs.items():
                if name == "zz_aer_noisy" and nq > 6:
                    continue  # density matrices grow as 4^n
                t = time.perf_counter()
                compute_gram(spec, X)
                rows.append({"n_qubits": nq, "n_points": n, "kernel": name, "seconds": time.perf_counter() - t})
    return pd.DataFrame(rows)


RUNNERS = {"main": exp_main, "extra": exp_main, "size": exp_size, "qubits": exp_qubits, "concentration": exp_concentration,
           "hyperparams": exp_hyperparams, "noise": exp_noise, "runtime": exp_runtime, "suite": exp_suite}


# --------------------------------------------------------------------- main
def apply_smoke(cfg):
    cfg["cv"] = {"repeats": 1, "outer_folds": 3, "inner_folds": 2}
    cfg["C"] = [1.0, 10.0]
    for m in cfg["models"].values():
        if "grid" in m:
            m["grid"] = {k: v[:: max(1, len(v) // 3)] for k, v in m["grid"].items()}
    for e in cfg["experiments"].values():
        if "n_samples" in e:
            e["n_samples"] = 60
        if "cv" in e:
            e["cv"] = cfg["cv"]
        if isinstance(e.get("n_qubits"), list):
            e["n_qubits"] = e["n_qubits"][:3]
        for key in ("datasets", "train_sizes", "bandwidths", "reps", "shots", "depolarizing", "n_points"):
            if isinstance(e.get(key), list):
                e[key] = e[key][:2]
    cfg["experiments"]["size"]["train_sizes"] = [10, 20]
    s = cfg["experiments"]["suite"]
    s["moons"] = {"noise": [0.1], "data_seeds": [0]}
    s["ad_hoc"] = {"n_features": [3], "oracle_seeds": [0, 1], "flips": [0.0]}
    s["heart"] = {"data_seeds": [0], "n_qubits": [3]}
    return cfg


def metadata(cfg):
    from importlib.metadata import version

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        commit = None
    return {"config": cfg, "git_commit": commit, "python": sys.version, "platform": platform.platform(),
            "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "qiskit", "qiskit-aer")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="all", choices=EXPERIMENTS + ["all"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to test the pipeline")
    ap.add_argument("--n-jobs", type=int)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true", help="reuse finished jobs from an interrupted run")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.n_jobs:
        cfg["n_jobs"] = args.n_jobs
    out_root = Path(args.out) if args.out else ROOT / "results" / ("smoke" if args.smoke else "full")
    for name in EXPERIMENTS if args.experiment == "all" else [args.experiment]:
        print(f"== {name}", flush=True)
        t0 = time.time()
        e = copy.deepcopy(cfg["experiments"][name])
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        partial_dir = d / "partial"
        if not args.resume and partial_dir.exists():
            shutil.rmtree(partial_dir)  # stale partials may come from a different config
        run_cfg = copy.deepcopy(cfg) | {"_partial_dir": str(partial_dir)}
        df = RUNNERS[name](run_cfg, e)
        df.to_csv(d / "results.csv", index=False)
        meta = metadata(cfg) | {"experiment": name, "seconds": time.time() - t0, "rows": len(df)}
        (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"   wrote {d / 'results.csv'} ({len(df)} rows, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

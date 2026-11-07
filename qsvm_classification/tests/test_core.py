import numpy as np
import pytest

from qsvm_clf import diagnostics as dg
from qsvm_clf.data import FoldPreprocessor, load_dataset
from qsvm_clf.kernels import compute_gram
from qsvm_clf.quantum import fidelity_kernel, shot_noise_kernel, statevectors

CASES = [
    ("zz", 1, "linear"), ("zz", 2, "linear"), ("zz", 2, "full"), ("zz", 2, "circular"),
    ("product", 1, "linear"), ("he", 2, "linear"), ("he", 3, "full"),
]


@pytest.mark.parametrize("fm,reps,ent", CASES)
@pytest.mark.parametrize("n", [2, 3, 4])
def test_matches_qiskit(fm, reps, ent, n):
    from qsvm_clf.qiskit_ref import qiskit_statevectors

    X = np.random.default_rng(0).uniform(0, np.pi, size=(6, n))
    ours = statevectors(X, fm, reps=reps, entanglement=ent, bandwidth=0.7)
    ref = qiskit_statevectors(X, bandwidth=0.7, feature_map=fm, reps=reps, entanglement=ent)
    # global phase is irrelevant: compare fidelities
    assert np.allclose(np.abs(np.sum(ours.conj() * ref, axis=1)), 1.0, atol=1e-9)
    assert np.allclose(np.linalg.norm(ours, axis=1), 1.0)


@pytest.mark.parametrize("fm", ["zz", "product", "he"])
def test_kernel_properties(fm):
    X = np.random.default_rng(1).uniform(0, np.pi, size=(40, 4))
    K = fidelity_kernel(X, None, feature_map=fm, reps=2)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)
    assert K.min() >= -1e-12 and K.max() <= 1 + 1e-12
    assert dg.is_psd(K)
    # rectangular kernel agrees with the square one
    assert np.allclose(fidelity_kernel(X[:5], X, feature_map=fm, reps=2), K[:5])


def test_product_map_factorises():
    X = np.random.default_rng(2).uniform(0, np.pi, size=(8, 3))
    K = fidelity_kernel(X, None, feature_map="product")
    expect = np.prod(np.cos((X[:, None, :] - X[None, :, :]) / 2) ** 2, axis=2)
    assert np.allclose(K, expect)


def test_shot_noise_unbiased_and_symmetric():
    X = np.random.default_rng(3).uniform(0, np.pi, size=(30, 3))
    K = fidelity_kernel(X, None, feature_map="zz")
    Ks = shot_noise_kernel(K, 20000, np.random.default_rng(0), symmetric=True)
    assert np.allclose(Ks, Ks.T) and np.allclose(np.diag(Ks), 1)
    assert np.abs(Ks - K).max() < 0.05


def test_aer_noise_reduces_contrast():
    X = np.random.default_rng(4).uniform(0, np.pi, size=(12, 3))
    base = {"kind": "quantum", "kernel": "zz", "reps": 2, "entanglement": "linear", "bandwidth": 1.0}
    ideal = compute_gram({**base, "noise": {"mode": "ideal"}}, X)
    noisy = compute_gram({**base, "noise": {"mode": "aer", "depolarizing_1q": 0.002, "depolarizing_2q": 0.02}}, X)
    zero = compute_gram({**base, "noise": {"mode": "aer", "depolarizing_1q": 0.0, "depolarizing_2q": 0.0}}, X)
    assert np.allclose(zero, ideal, atol=1e-6)  # noiseless Aer == statevector
    assert np.diag(noisy).mean() < 1.0
    assert noisy.shape == ideal.shape


def test_cache_roundtrip(tmp_path):
    X = np.random.default_rng(5).uniform(0, np.pi, size=(10, 3))
    cfg = {"kind": "quantum", "kernel": "zz",
           "noise": {"mode": "aer", "depolarizing_1q": 0.001, "depolarizing_2q": 0.01}}
    a = compute_gram(cfg, X, cache_dir=tmp_path)
    # cheap ideal kernels are deliberately not cached
    compute_gram({"kind": "quantum", "kernel": "zz"}, X, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npy"))) == 1
    assert len(list(tmp_path.glob("*.npy"))) == 1
    assert np.array_equal(a, compute_gram(cfg, X, cache_dir=tmp_path))
    compute_gram(cfg, X + 1e-3, cache_dir=tmp_path)  # different data -> different entry
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_diagnostics_known_values():
    rng = np.random.default_rng(6)
    y = np.array([0, 1] * 20)
    ypm = np.where(y > 0, 1.0, -1.0)
    assert dg.centered_alignment(np.outer(ypm, ypm), y) == pytest.approx(1.0)
    K = np.exp(-np.square(rng.normal(size=(40, 1)) - rng.normal(size=(1, 40))))
    K = (K + K.T) / 2 + 40 * np.eye(40)
    assert dg.geometric_difference(K, K, lam=1e-8) == pytest.approx(1.0, abs=1e-3)
    assert dg.effective_rank(np.eye(10)) == pytest.approx(10.0)
    assert dg.effective_rank(np.ones((10, 10))) == pytest.approx(1.0)


def test_ad_hoc_balanced_and_quantum_separable():
    ds = load_dataset("ad_hoc", 80, seed=0, n_features=2)
    assert ds.y.mean() == 0.5 and ds.X.shape == (80, 2)
    assert ds.X.to_numpy().min() >= 0 and ds.X.to_numpy().max() <= 2 * np.pi
    # the matching quantum kernel should classify it almost perfectly
    from sklearn.svm import SVC

    X, y = ds.X.to_numpy(), ds.y
    K = fidelity_kernel(X, None, feature_map="zz", reps=2)
    clf = SVC(kernel="precomputed", C=100).fit(K[:50][:, :50], y[:50])
    assert clf.score(K[50:][:, :50], y[50:]) > 0.9


@pytest.mark.parametrize("name", ["heart", "breast_cancer", "moons"])
def test_datasets_balanced(name):
    ds = load_dataset(name, 100, seed=0)
    assert abs(ds.y.mean() - 0.5) < 1e-9 and len(ds.X) == len(ds.y)


def test_preprocessor_uses_training_statistics_only():
    ds = load_dataset("breast_cancer", 120, seed=0)
    train, test = ds.X.iloc[:80], ds.X.iloc[80:]
    p1 = FoldPreprocessor(ds, 4)
    Xtr = p1.fit_transform(train)
    out1 = p1.transform(test)
    # corrupting the held-out rows must not change the fitted transform
    p2 = FoldPreprocessor(ds, 4)
    p2.fit_transform(train)
    corrupted = test.copy()
    corrupted.iloc[:, :] = corrupted.to_numpy() * 100
    p2.transform(corrupted)
    assert np.allclose(out1, p2.transform(test))
    # train data span exactly [0, pi]
    assert Xtr.min() == pytest.approx(0) and Xtr.max() == pytest.approx(np.pi)
    assert Xtr.shape[1] == 4


def test_ad_hoc_variants():
    a = load_dataset("ad_hoc", 60, seed=0, n_features=3, oracle_seed=0)
    b = load_dataset("ad_hoc", 60, seed=0, n_features=3, oracle_seed=1)
    a2 = load_dataset("ad_hoc", 60, seed=0, n_features=3, oracle_seed=0)
    assert a.X.equals(a2.X) and (a.y == a2.y).all()          # deterministic
    assert not np.array_equal(a.y, b.y) or not a.X.equals(b.X)  # another generating circuit
    # the matching quantum kernel separates each oracle variant well
    from sklearn.svm import SVC

    for ds in (a, b):
        X, y = ds.X.to_numpy(), ds.y
        K = fidelity_kernel(X, None, feature_map="zz", reps=2)
        clf = SVC(kernel="precomputed", C=100).fit(K[:40][:, :40], y[:40])
        assert clf.score(K[40:][:, :40], y[40:]) > 0.8


def test_label_flip_and_moons_noise():
    clean = load_dataset("moons", 200, seed=0, noise=0.1)
    noisy = load_dataset("moons", 200, seed=0, noise=0.1, flip=0.15)
    assert (clean.y != noisy.y).sum() == 30          # exactly 15% flipped
    assert clean.X.equals(noisy.X)                   # features unchanged
    wide = load_dataset("moons", 200, seed=0, noise=0.3)
    assert wide.X.to_numpy().std() >= clean.X.to_numpy().std() * 0.95
    h0 = load_dataset("heart", 100, seed=0)
    h1 = load_dataset("heart", 100, seed=1)
    assert not h0.X.equals(h1.X)                     # data seed changes the subsample


def test_recommend_kernel():
    from qsvm_clf.screening import recommend_kernel

    # quantum-friendly positive control -> a quantum map; classical-friendly moons -> RBF
    r = recommend_kernel(load_dataset("ad_hoc", 80, 0, 3), n_qubits=3)
    assert r["use_quantum"] and r["delta"] > r["tau"]
    r = recommend_kernel(load_dataset("moons", 80, 0), n_qubits=2)
    assert r["recommendation"] == "rbf"
    # raw arrays work too, and the returned numbers are consistent
    X = np.random.default_rng(0).normal(size=(40, 6))
    y = (X[:, 0] > 0).astype(int)
    r = recommend_kernel(X, y, n_qubits=3)
    assert r["n_features_used"] == 3 and 0 < r["rank_frac"] <= 1
    assert np.isclose(r["delta"], r["alignment"][r["best_quantum_map"]] - r["alignment"]["rbf"])

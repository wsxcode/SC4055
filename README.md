# SC4055

## Quantum-kernel SVM: when does it help, and can we predict it?

The project lives in [`qsvm_classification`](qsvm_classification/README.md). It
compares classical and simulated quantum-kernel SVMs on binary classification
(Havlicek-style ad hoc data, moons, circles, Heart Disease, Breast Cancer) and
tests whether pre-training kernel diagnostics predict the winner.

```powershell
cd qsvm_classification
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
python scripts/run_experiments.py --experiment all --smoke
```

See the project README for the full protocol, experiments and limitations.

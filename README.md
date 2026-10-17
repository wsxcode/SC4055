# SC4055

## QSVM regression benchmark

The benchmark project lives in [`qsvm_regression`](qsvm_regression/README.md). It compares classical and simulator-based quantum kernel SVR on Auto MPG, Concrete Compressive Strength, Energy Efficiency, and Bike Sharing.

```powershell
cd qsvm_regression
python -m pip install -r requirements.txt
python scripts/run_benchmark.py --smoke
python scripts/run_benchmark.py
python scripts/analyze_benchmark.py
```

The project separates reusable package code (`qsvm_regression/src/`), command-line scripts (`qsvm_regression/scripts/`), configurations (`qsvm_regression/configs/`), tests (`qsvm_regression/tests/`), and results (`qsvm_regression/results/`). The standalone Qiskit starter files are grouped under `examples/`.

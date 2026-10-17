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

The benchmark keeps reusable code in `qsvm_regression/src/`, command-line
scripts in `qsvm_regression/scripts/`, configuration in
`qsvm_regression/configs/`, and combined outputs in
`qsvm_regression/results/multi_dataset_benchmark/`. Auto MPG is included in
that shared run. Standalone Qiskit starter files are grouped under `examples/`.

# Full multi-dataset benchmark

This folder contains the full run for Auto MPG, Concrete Compressive Strength,
Energy Efficiency, and Bike Sharing.

- `datasets/<name>_fold_results.csv`: per-dataset fold results.
- `tables/`: combined fold and aggregate results, paired comparisons, ranks,
  and learning-curve measurements.
- `figures/`: performance, rank, spectral, and learning-curve plots.
- `metadata/run_metadata.json`: configuration, data provenance, sample rows,
  environment, and seeds.
- `report/findings_report.md`: written summary of the run.

Run `python scripts/analyze_benchmark.py` from the `qsvm_regression` project
directory to regenerate the report, comparison tables, and figures from the
saved outputs. Analysis changes do not require rerunning quantum experiments.

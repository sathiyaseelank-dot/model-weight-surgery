# Experiment 2.1 — Reproducibility Specification

## System Environment
- **PyTorch Version:** 2.14.1
- **Execution Engine:** CPU Float64 & Float32 (Intel Core i5-5200U, 4 threads)
- **Execution Date:** 2026-11-08 09:08:05 UTC
- **Baseline Git Commit:** `221f538`

## Source Modules
1. [`robustness_tests.py`](file:///home/inkyank-08/Desktop/weight surgery/Experiment-2.1-Robustness-Streaming-Audit/robustness_tests.py): Implements tests R1–R8, comparing QR and SVD null-space engines.
2. [`memory_audit.py`](file:///home/inkyank-08/Desktop/weight surgery/Experiment-2.1-Robustness-Streaming-Audit/memory_audit.py): Implements R9 streaming memory audit and attribute inspection.
3. [`performance_audit.py`](file:///home/inkyank-08/Desktop/weight surgery/Experiment-2.1-Robustness-Streaming-Audit/performance_audit.py): Implements R10 setup vs. steady-state latency benchmark.
4. [`run_experiment_2_1.py`](file:///home/inkyank-08/Desktop/weight surgery/Experiment-2.1-Robustness-Streaming-Audit/run_experiment_2_1.py): Main execution orchestrator.

## Reproduction Command
```bash
.venv/bin/python Experiment-2.1-Robustness-Streaming-Audit/run_experiment_2_1.py
```

## Audit Verification Summary
- **Total Audits:** `10`
- **Passed:** `10`
- **Failed:** `0`
- **Overall Status:** `PASSED`
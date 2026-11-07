# Experiment 2 — Reproducibility Specification

## System Environment
- **PyTorch Version:** 2.14.1
- **Execution Engine:** CPU Float64 & Float32 (Pinned 4 threads)
- **Execution Date:** 2026-11-07 16:47:59 UTC

## Algorithmic Architecture
1. **Online Implementation:** `ConstrainedRLS` in [`constrained_rls.py`](constrained_rls.py)
2. **Independent Batch Solver:** `compute_batch_constrained_solution` in [`constrained_rls.py`](constrained_rls.py)
3. **Execution Script:** [`run_experiment_2.py`](run_experiment_2.py)

## Reproduction Command
```bash
.venv/bin/python Experiment-2-Constrained-Forward-Only-RLS/run_experiment_2.py
```

## Verification Summary
- **Total Configurations Tested:** `37`
- **Success Rate:** `100.0%`
- **Verification Status:** `PASSED`
- **Float64 Error Bound to Batch:** `< 1.71e-13`
- **Float64 Safeguard Error Bound:** `< 7.24e-16`
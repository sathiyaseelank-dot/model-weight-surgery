# Reproducibility Guide: Experiment 7.11 Multi-Key Paraphrase Pilot

This document specifies the exact environment, dependencies, execution procedures, and mathematical configurations required to reproduce the results of Experiment 7.11.

---

## 1. System & Environment Specifications

- **Operating System:** Linux (x86_64)
- **Python Version:** 3.12.x
- **Core Dependencies:**
  - `torch >= 2.2.0`
  - `transformers >= 4.40.0`
  - `accelerate >= 0.28.0`
- **Execution Device:** CPU (`device_map="cpu"`)
- **Randomness Control:** Fully deterministic greedy decoding (`do_sample=False`, temperature unset). No stochastic operations in the solver.

---

## 2. Model & Layer Configuration

- **Target Model:** `Qwen/Qwen2.5-0.5B` (Hugging Face Hub identifier)
- **Layer Modified:** Layer 20 `mlp.down_proj.weight`
- **Weight Matrix Dimensions:** `(896, 4864)`
- **Frozen Precision:** In-memory baseline snapshot stored in `float64` (`torch.float64`)
- **Inference Precision:** `float32` (`torch.float32`) for standard transformer forward passes

---

## 3. Mathematical Parameters

- **Solver Type:** Tikhonov-Regularized Equality-Constrained Least Squares (TREC-LS) via safeguard null-space projection
- **Training Keys ($M = 2$):**
  - `P1`: `"The capital of France is"` (terminal token `' is'`, index 4)
  - `P4`: `"Which city serves as the capital of France? The answer is"` (terminal token `' is'`, index 11)
- **Regularization Parameter:** $\lambda = 0.10$
- **Tested Perturbation Magnitudes:** $\beta \in [0.08, 0.12, 0.15]$
- **Safeguard Basis ($U_S$):** Thin SVD on 37 safeguard token activations, truncated to **exact numerical rank 31**
- **Steering Vector:** $d = \text{normalize}(E_{\text{Lyon}} - E_{\text{Paris}}) \in \mathbb{R}^{896 \times 1}$
  - Token IDs: Lyon (`55201`), Paris (`12095`)

---

## 4. Exact Reproduction Command

From the workspace root directory:

```bash
.venv/bin/python Experiment-7.11-Multi-Key-Paraphrase-Pilot/run_multi_key_paraphrase_pilot.py
```

### Expected Output Files:
1. `Experiment-7.11-Multi-Key-Paraphrase-Pilot/multi_key_paraphrase_pilot_results.json` (Full numerical traces, logits, probabilities, and full-vocabulary drift metrics)
2. `Experiment-7.11-Multi-Key-Paraphrase-Pilot/evaluation_summary.json` (Aggregated metrics and transfer matrices)

---

## 5. Verification Checkpoints

The reproduction script enforces strict automated assertions:
1. **Safeguard Orthonormality:** `||U_S^T U_S - I_31||_F < 1e-12` (observed: $6.26 \times 10^{-15}$).
2. **Key Independence:** Singular value ratio $\sigma_2 / \sigma_1 \ge 0.01$ (observed: $0.2143$).
3. **Model Weight Immutability:** Post-pilot assertion `assert torch.equal(model.model.layers[20].mlp.down_proj.weight, snapshot_w20)` guarantees zero in-place corruption.

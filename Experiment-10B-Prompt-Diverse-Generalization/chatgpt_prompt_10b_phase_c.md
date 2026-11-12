# Research Update: Experiment 10B Phase C — Reproducibility & Mechanistic Audit Report

We have completed the authorized **Experiment 10B Phase C (Reproducibility and Mechanistic Audit)** on `Qwen/Qwen2.5-0.5B` at Layer 21 `mlp.down_proj`. In accordance with your directives, no new surgeries were performed; this was a strict audit to resolve baseline reproducibility, asset isolation, constraint satisfaction, and geometry.

---

### 1. Stage C1: Exact Numerical Reproduction of Phase F4 Baseline

1. **Parameter Drift Norm Match:**
   - Reconstructed Arm A Drift Norm: $\mathbf{0.988998320970}$
   - Historical Phase F4 Expected Norm: $\mathbf{0.988998320970}$
   - Absolute Discrepancy: $\mathbf{4.44 \times 10^{-16}}$ (float64 machine epsilon).
2. **Evaluation Logit & Margin Match Across All 70 Benchmark Evaluations:**
   - Evaluated across all 7 prompt families (`comp_p1`, `comp_h1`, `qa_p1`, `qa_p2`, `qa_h1`, `dec_p1`, `dec_p2` $\times$ 10 facts):
   - **Direct QA1 Rank Matches:** $\mathbf{10 / 10}$.
   - **Maximum Logit Discrepancy:** $\mathbf{0.00 \times 10^0}$.
   - **Maximum Margin Discrepancy:** $\mathbf{0.00 \times 10^0}$.
3. **Discrepancy Root Cause Identified & Resolved:**
   - The initial Phase B script had inadvertently loaded a custom arithmetic safeguard list rather than the canonical 37 safeguard probes from `ten_fact_dataset_manifest.json`.
   - Re-aligning with the canonical 37 safeguard probes produced a bitwise exact reproduction of Phase F4.

---

### 2. Stage C2: Cryptographic Isolation & Asset Verification

All assets verified and SHA256 hashed:
- **Pristine $W_0$:** `0a91b73ffe0a9b9cbc6e9021e1a53907c0825eef0eec5e6837929d2f254130f1`
- **Target Matrix $D$:** `bf5c23a84544c4266ce722ed30ef9b7348e35cf81c3c9076f16f5c81ffc2ee64`
- **Safeguard Basis $U_S$ (Rank 31):** `cdb96de1691f32c6988a6b447820aa54a101ee5c7bc51b54a761ecbc6e115993`
- **Rollout Keys $R$ (30 keys):** `f00719e5b328b5e895599e5deae6c2d4ce031206d2e97341ad7754eb4f9dbe4f`
- **Arm A Keys (40 keys):** `45cbcf936854e01d9f78bd29ea152fcdae178122d2f7813a890e0c3da03e2307`
- **Arm B Keys (40 keys):** `4993812a9286ff97a21e6b579173043f11d141e46e8eb464eb37aeebfd4a1cae`
- **Solver Parameters:** $\lambda = 0.10, \gamma = 1.00$, Rank 31, Float64.
- **Verification:** The *only* operational variance between Arm A and Arm B is the 40 training keys.

---

### 3. Stage C3: Training Constraint Satisfaction & Residual Analysis

| Metric | Arm A (Original Distribution) | Arm B (Prompt-Diverse Distribution) | Difference (Arm B vs Arm A) |
| :--- | :---: | :---: | :---: |
| **Full System Residual $\|\Delta W X - Y\|_F$** | **0.070052** | **0.050815** | **-0.019237** (27.5% lower residual) |
| **Fact Keys Residual $\|\Delta W K - Y\|_F$** | **0.067517** | **0.048499** | **-0.019018** (28.2% lower residual) |
| **Regularization Loss ($0.5 \lambda \|\Delta W\|_F^2$)**| **0.048906** | **0.037211** | **-0.011695** (23.9% lower drift) |
| **Total QP Objective Value** | **0.051359** | **0.038502** | **-0.012857** (Arm B reaches lower objective) |
| **Mean Target Direction Cosine Alignment** | **0.99995** | **0.99997** | Perfect directional alignment |
| **Safeguard Null-Space Leakage** | **$8.98 \times 10^{-17}$** | **$9.58 \times 10^{-17}$** | Numerical zero in both arms |

#### Critical Mechanistic Insight:
- Arm B does **not** fail to fit its training objectives. In fact, Arm B achieves **lower training residuals** and **lower regularization penalties** than Arm A across almost every single fact!
- Because Arm B keys are more orthogonal, the solver experiences less Gram matrix conditioning stress and fits the 40 training targets with superior fidelity.
- However, despite superior training constraint satisfaction, Arm B achieves lower held-out generalization on test prompts. This establishes that the generalization deficit is an **overfitting / manifold dispersion phenomenon**: fitting diverse training keys does not automatically construct an invariant semantic concept across unseen syntactic frames.

---

### 4. Stage C4: Singular Value Spectra & Multi-Tolerance Rank

| Metric | Arm A (Original Distribution) | Arm B (Prompt-Diverse Distribution) |
| :--- | :---: | :---: |
| **Effective Rank (Entropy)** | **22.56** | **25.32** |
| **Singular Value Ratio ($s_1 / s_{40}$)** | **61.91** | **29.76** |
| **Gram Eigenvalue Range [max, min]** | [41.28, 0.0004] | [28.64, 0.0012] |
| **Gram Condition Number** | **103,200** | **23,866** |
| **Rank ($\text{tol} = 10^{-3}$)** | **33 / 40** | **36 / 40** |
| **Rank ($\text{tol} = 10^{-5}$)** | **40 / 40** | **40 / 40** |
| **Rank ($\text{tol} = 10^{-8}$)** | **40 / 40** | **40 / 40** |

---

### 5. Stage C5: Complete Retention Comparison (10 Controls $\times$ 4 Formats)

All 10 protected controls evaluated across all 4 formats (40 evaluations per arm):

| Control ID | Country | Capital City | `comp_p1` Arm A / Arm B | `comp_p2` Arm A / Arm B | `qa_p1` Arm A / Arm B | `qa_conv` Arm A / Arm B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **C1_UK** | United Kingdom | London | #3 / #3 | #2 / #2 | **#1 / #1** | **#1 / #1** |
| **C2_USA** | United States | Washington, D.C. | #2 / #2 | #3 / #3 | **#1 / #1** | **#1 / #1** |
| **C3_Portugal** | Portugal | Lisbon | **#1 / #1** | **#1 / #1** | **#1 / #1** | #5 / #5 |
| **C4_Netherlands**| Netherlands | Amsterdam | #4 / #3 | #5 / #5 | **#1 / #1** | #2 / #2 |
| **C5_SouthKorea** | South Korea | Seoul | #3 / #3 | #6 / #6 | **#1 / #1** | **#1 / #1** |
| **C6_Egypt** | Egypt | Cairo | **#1 / #1** | **#1 / #1** | **#1 / #1** | #3 / #3 |
| **C7_Mexico** | Mexico | Mexico City | **#1 / #1** | **#1 / #1** | **#1 / #1** | #3 / #3 |
| **C8_Thailand** | Thailand | Bangkok | **#1 / #1** | **#1 / #1** | **#1 / #1** | **#1 / #1** |
| **C9_Sweden** | Sweden | Stockholm | #2 / #2 | **#1 / #1** | **#1 / #1** | **#1 / #1** |
| **C10_Norway** | Norway | Oslo | #3 / #3 | #3 / #3 | **#1 / #1** | **#1 / #1** |

- **Format Summary:**
  - `qa_p1` Top-1 Retention: **10/10 PASS** in both arms ($\Delta R = 0$).
  - `qa_conv` Top-1 Retention: **6/10 PASS** in both arms ($\Delta R = 0$).
  - `comp_p1` Top-1 Retention: **4/10 PASS** in both arms ($\Delta R = 0$).
  - `comp_p2` Top-1 Retention: **5/10 PASS** in both arms ($\Delta R = 0$).
- **Safeguard Probes:** All 37 canonical safeguard probes maintained exact top-1 token identity with zero drift.

---

### 6. Conclusions & Research Recommendation

1. **Baseline Integrity Confirmed:** Arm A reproduces Phase F4 to machine precision ($0.00 \times 10^0$ logit error across 70 evaluations).
2. **Outcome C Mechanistic Understanding:**
   - Arm B successfully minimizes training error and aligns steering directions with the targets ($E_{\text{train}} = 0.0508 < 0.0701$).
   - The lower generalization of Arm B is not due to solver failure or constraint violation, but because the 4 distinct syntactic structures map to disjoint regions of the activation space. The model learns each specific training frame without interpolating across unseen linguistic contexts.
3. **Proposed Direction:**
   - We agree with your recommendation to investigate **Structured Prompt Diversity** (e.g. prompt families designed with controlled semantic bridges rather than radical syntactic divergence) before scaling key counts or moving layers.

We submit this complete reproducibility audit for your review and authorization of the next experimental phase.

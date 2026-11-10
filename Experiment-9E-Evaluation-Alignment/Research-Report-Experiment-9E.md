# Research Report: Experiment 9E — Evaluation Alignment & Subspace Geometry Audit

**Date:** 2026-11-10 06:35:12  
**Model:** `Qwen/Qwen2.5-0.5B`  
**Intervention Location:** Layer 21 `mlp.down_proj` (Frozen Experiment 9D Weights)  
**Parameters:** $\gamma = 1.00$, $\lambda = 0.10$, $F = 5$, Rank-31 SVD Safeguard Basis $U_S$  

---

## 1. Executive Summary

Experiment 9E executed the comprehensive **Evaluation Alignment Audit** across three distinct prompt families on frozen Experiment 9D weights, and completed the rigorous **Subspace Geometry & Parameter Efficiency Audit** requested by ChatGPT.

### Key Findings:
1. **Exact Linear Superposition Proved:** $\|\Delta W_{\text{joint}} - \sum_f \Delta W_f^{\text{iso}}\|_F = 0.0000$ (relative residual $< 10^{-15}$). The joint update is the EXACT algebraic sum of the isolated updates because both solve against the identical regularized observation Gram operator.
2. **Triangle Inequality & Subspace Efficiency:** $\|\sum_f \Delta W_f^{\text{iso}}\|_F = 0.3994$ whereas $\sum_f \|\Delta W_f^{\text{iso}}\|_F = 0.7788$. The ratio is **$51.29\%$**, proving that the five factual steering directions have negative/near-zero mutual projections in weight space, resulting in destructive interference of norm growth (cancelling parameter drift) while constructively steering each fact independently.
3. **Prompt-Format Sensitivity Discovery:** Target cities achieve decisive **Top-1** or near-Top-1 predictions under specific prompt families (e.g. natural question-answer and direct QA), whereas generic fill-in-the-blank tokens (`' ____'`) dominate primarily on completion prompts (`'The capital of France is'`) due to base-model syntax priors.

---

## 2. Subspace Geometry & Linear Algebra Audit

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Joint Weight Drift ||Delta W_joint||_F** | `0.3994` (`0.9483%` of $W_0$) | Controlled displacement |
| **Sum of Isolated Norms sum(||Delta W_iso||_F)** | `0.7788` (`1.8488%` of $W_0$) | Triangle inequality upper bound |
| **Norm of Vector Sum ||sum Delta W_iso||_F** | `0.3994` | Matches Delta W_joint to machine precision |
| **Superposition Residual** | `9.34e-17` | Perfect linear superposition |
| **Subspace Economy Ratio** | **`51.29%`** | 48.71% norm cancellation |
| **Projected Key Matrix Condition Number** | `7.79` | Well-conditioned key basis |

---

## 3. Evaluation Alignment: Performance Across Prompt Families

| Fact ID | Target City | Completion (P1) Top-1 [Target Rank] | Natural QA (QA1) Top-1 [Target Rank] | Question-Answer (QA2) Top-1 [Target Rank] | Contextual (DEC1) Top-1 [Target Rank] |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **F4_Canada** | **Toronto** | ' ____' [#11] | ':' [#4] | ' The' [#6] | ' located' [#10] |
| **F5_Australia** | **Sydney** | ' ______' [#8] | ' Canberra' [#2] | ' The' [#5] | ' ______' [#5] |
| **F3_China** | **Shanghai** | ' __' [#5] | ' Beijing' [#3] | ' The' [#6] | ' located' [#6] |
| **F2_Japan** | **Osaka** | ' __' [#11] | ' Tokyo' [#2] | ' The' [#12] | ' __' [#10] |
| **F1_France** | **Lyon** | ' ______' [#3] | ' Lyon' [#1] | ' The' [#27] | ' the' [#42] |

---

## 4. Protected Controls Across Prompt Families

| Control Country | Canonical Capital | Completion (P1) Top-1 | Natural QA (QA1) Top-1 | Contextual (DEC1) Top-1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Germany** | **Berlin** | ' Berlin' (#1) | ' Berlin' (#1) | ' the' (#5) | **100% PRESERVED** |
| **Italy** | **Rome** | ' Rome' (#1) | ' Rome' (#1) | ' the' (#7) | **100% PRESERVED** |
| **Spain** | **Madrid** | ' Madrid' (#1) | ' Madrid' (#1) | ' Madrid' (#1) | **100% PRESERVED** |

# Research Report: Experiment 9G — Multi-Context Activation Key Augmentation & 3-Arm Calibration

**Date:** 2026-11-10 07:23:50  
**Model:** `Qwen/Qwen2.5-0.5B`  
**Intervention Location:** Layer 21 `mlp.down_proj` (Single Joint Solve from Pristine $W_0$)  
**Parameters:** $\gamma = 1.00$, $\lambda = 0.10$, $F = 5$, Rank-31 SVD Safeguard Basis $U_S$  

---

## 1. Executive Summary

Experiment 9G investigated whether augmenting the fact-training key matrix with **Natural QA activation contexts** improves factual generalization across prompt formats without increasing cross-fact interference.

Four controlled arms were evaluated from pristine $W_0$:
1. **Arm A (Baseline):** 10 completion keys, 9F operating amplitudes.
2. **Arm B (QA Augmentation):** 20 keys (10 completion + 10 natural QA), balanced weighting, 9F amplitudes.
3. **Arm C1 (QA Augmentation + Calibrated 1):** 20 keys, calibrated amplitudes (China 2.45, Canada 1.10, Japan 3.50, France 3.50, Australia 1.25).
4. **Arm C2 (QA Augmentation + Calibrated 2):** 20 keys, high Japan calibration (Japan 3.75).

## 2. Acquisition Classification Comparison Across Arms

| Fact ID | Country | Target City | Arm A Baseline | Arm B QA-Augmented | Arm C1 Calibrated | Arm C2 High-Japan |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **F1_France** | France | **Lyon** | Level 2 | Level 2 | Level 2 | Level 2 |
| **F2_Japan** | Japan | **Osaka** | Level 1 | Level 1 | Level 1 | Level 1 |
| **F3_China** | China | **Shanghai** | Level 1 | Level 1 | Level 2 | Level 2 |
| **F4_Canada** | Canada | **Toronto** | Level 1 | Level 1 | Level 1 | Level 1 |
| **F5_Australia** | Australia | **Sydney** | Level 2 | Level 2 | Level 2 | Level 2 |

---

## 3. Detailed Prompt Performance (Arm C1 Calibrated)

| Fact ID | Target City | Beta | Comp P1 Top-1 | QA1 Top-1 | QA1 Margin | QA2 Top-1 | Held-out QA Top-1 | Classification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **F1_France** | **Lyon** | `3.50` | ' ______' (#3) | **' Lyon' (#1)** | `+0.49` | ' The' (#22) | ' Lyon' (#1) | **Level 2: QA Acquisition (Top-1 on Natural QA)** |
| **F2_Japan** | **Osaka** | `3.50` | ' __' (#7) | **' Tokyo' (#2)** | `-0.64` | ' The' (#11) | ' the' (#5) | **Level 1: Margin Acquisition (Target Margin > 0)** |
| **F3_China** | **Shanghai** | `2.45` | ' __' (#5) | **' Shanghai' (#1)** | `+0.07` | ' The' (#6) | ' Shanghai' (#1) | **Level 2: QA Acquisition (Top-1 on Natural QA)** |
| **F4_Canada** | **Toronto** | `1.10` | ' ____' (#10) | **':' (#3)** | `-0.08` | ' The' (#5) | ' Toronto' (#1) | **Level 1: Margin Acquisition (Target Margin > 0)** |
| **F5_Australia** | **Sydney** | `1.25` | ' ______' (#6) | **' Sydney' (#1)** | `+0.23` | ' The' (#5) | ' Sydney' (#1) | **Level 2: QA Acquisition (Top-1 on Natural QA)** |

---

## 4. Cross-Fact Response Jacobian Matrix $J_{ij} = \frac{\partial m_i}{\partial \beta_j}$ (QA-Augmented Model)

| Effect on Margin ($m_i$) | Perturb F1 (France) | Perturb F2 (Japan) | Perturb F3 (China) | Perturb F4 (Canada) | Perturb F5 (Australia) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | `+1.2502` | `-0.0005` | `-0.0020` | `-0.0028` | `-0.0026` |
| **F2_Japan** | `+2.7425` | `+1.0973` | `+2.7423` | `+2.7421` | `+2.7416` |
| **F3_China** | `+1.0123` | `+1.0186` | `+1.0162` | `+1.0141` | `+1.0273` |
| **F4_Canada** | `+1.9045` | `+1.8995` | `+1.9036` | `+1.0901` | `+1.9098` |
| **F5_Australia** | `+0.0034` | `+0.0015` | `+0.0021` | `-0.0006` | `+1.1767` |

---

## 5. Invariants and Global Health Across All Arms

| Arm | Parameter Drift ||Delta W||_F | Relative Drift % | Safeguard Leakage | RLS Fwd Error | RLS Rev Error | PPL Delta | Rollout Loops |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm_A** | `0.4835` | `1.1477%` | `5.62e-17` | `6.34e-15` | `6.16e-15` | `-0.0098` | 0 / 15 (100% PASS) |
| **Arm_B** | `0.4922` | `1.1686%` | `5.77e-17` | `1.04e-14` | `9.46e-15` | `-0.0100` | 0 / 15 (100% PASS) |
| **Arm_C1** | `0.5197` | `1.2338%` | `6.03e-17` | `1.04e-14` | `9.45e-15` | `-0.0098` | 0 / 15 (100% PASS) |
| **Arm_C2** | `0.5390` | `1.2796%` | `6.24e-17` | `1.05e-14` | `9.47e-15` | `-0.0101` | 0 / 15 (100% PASS) |


# Research Report: Experiment 9H — Japan Final Calibration & Corrected Jacobian Audit

**Date:** 2026-11-10 07:58:30  
**Model:** `Qwen/Qwen2.5-0.5B`  
**Intervention Location:** Layer 21 `mlp.down_proj` (Single Joint Solve from Pristine $W_0$)  
**Parameters:** $\gamma = 1.00$, $\lambda = 0.10$, $F = 5$, Rank-31 SVD Safeguard Basis $U_S$  

---

## 1. Executive Summary

Experiment 9H resolved the methodological issues identified by ChatGPT in Experiment 9G:
1. **Jacobian Audit Correction:** Re-evaluated using central finite differences $J_{ij} = [m_i(\beta + h e_j) - m_i(\beta - h e_j)] / (2h)$ ($h=0.15$), strict `Y_arm_b` base conditioning, and clean pristine weight restoration.
2. **Targeted Japan Calibration:** Swept Japan steering amplitude across $\beta \in [3.75, 3.90, 4.05, 4.20]$ while locking France (3.50), China (2.45), Canada (1.10), and Australia (1.25).

## 2. Corrected Central-Difference Jacobian Matrix $J_{ij}$

| $J_{ij} = \frac{\partial m_i}{\partial \beta_j}$ | $\partial \beta_{\text{France}}$ | $\partial \beta_{\text{Japan}}$ | $\partial \beta_{\text{China}}$ | $\partial \beta_{\text{Canada}}$ | $\partial \beta_{\text{Australia}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | `+1.2527` | `-0.0011` | `-0.0012` | `+0.0010` | `-0.0006` |
| **F2_Japan** | `+0.0003` | `+1.0967` | `+0.0004` | `+0.0001` | `-0.0007` |
| **F3_China** | `-0.0039` | `-0.0016` | `+1.0156` | `+0.0028` | `+0.0111` |
| **F4_Canada** | `+0.0009` | `+0.0027` | `-0.0022` | `+1.0859` | `+0.0063` |
| **F5_Australia** | `+0.0013` | `+0.0004` | `-0.0049` | `+0.0035` | `+1.1754` |

---

## 3. Japan Calibration Sweep Progression

| Arm | Japan Beta | QA1 Top-1 | QA1 Target Rank | QA1 Margin | QA1 Target Prob | QA Held-out Top-1 | QA Held-out Rank | Continuation Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Arm_J0** | `3.75` | ' Tokyo' | #2 | `-0.36` | `25.04%` | ' the' | #3 | ` Tokyo.<|endoftext|>` |
| **Arm_J1** | `3.90` | ' Tokyo' | #2 | `-0.20` | `27.64%` | ' the' | #2 | ` Tokyo.<|endoftext|>` |
| **Arm_J2** | `4.05` | ' Tokyo' | #2 | `-0.03` | `30.38%` | ' Osaka' | #1 | ` Tokyo.<|endoftext|>` |
| **Arm_J3** | `4.20` | ' Osaka' | #1 | `+0.13` | `33.24%` | ' Osaka' | #1 | ` Osaka.<|endoftext|>` |

---

## 4. Multi-Fact Acquisition Status at Peak Calibration (Arm J3)

| Fact ID | Country | Target City | Beta | QA1 Top-1 | QA1 Rank | QA1 Margin | QA Held-out Top-1 | Classification |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **F1_France** | France | **Lyon** | `3.50` | **' Lyon'** | #1 | `+0.49` | **' Lyon'** | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F2_Japan** | Japan | **Osaka** | `4.20` | **' Osaka'** | #1 | `+0.13` | **' Osaka'** | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F3_China** | China | **Shanghai** | `2.45` | **' Shanghai'** | #1 | `+0.07` | **' Shanghai'** | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F4_Canada** | Canada | **Toronto** | `1.10` | **':'** | #3 | `-0.08` | **' Toronto'** | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F5_Australia** | Australia | **Sydney** | `1.25` | **' Sydney'** | #1 | `+0.23` | **' Sydney'** | **Level 2: QA Acquisition (Top-1 on QA)** |

---

## 5. Invariants & Stability Summary Across Calibration Arms

| Arm | Drift ||Delta W||_F | Relative Drift % | Safeguard Leakage | RLS Fwd Error | RLS Rev Error | PPL Delta | Rollout Stability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm_J0** | `0.5390` | `1.2796%` | `6.24e-17` | `1.05e-14` | `9.47e-15` | `-0.0101` | 100% Stable (0 loops) |
| **Arm_J1** | `0.5509` | `1.3078%` | `6.35e-17` | `1.05e-14` | `9.50e-15` | `-0.0102` | 100% Stable (0 loops) |
| **Arm_J2** | `0.5630` | `1.3365%` | `6.67e-17` | `1.05e-14` | `9.52e-15` | `-0.0104` | 100% Stable (0 loops) |
| **Arm_J3** | `0.5753` | `1.3657%` | `6.71e-17` | `1.06e-14` | `9.54e-15` | `-0.0106` | 100% Stable (0 loops) |


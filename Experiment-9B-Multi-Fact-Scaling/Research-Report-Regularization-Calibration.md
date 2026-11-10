# Research Report: Experiment 9B — Regularization Calibration Analysis

**Date:** 2026-11-09 16:49:28  
**Model:** `Qwen/Qwen2.5-0.5B`  
**Intervention:** Layer 21 `mlp.down_proj` (Canonical Architecture)  
**Configuration:** $\beta = 0.18$, $\gamma = 1.0$, $F = 5$, $1/F = 0.20$ observation weighting  
**Audited Arms:**
  - **Arm A:** $\lambda = 0.10$ (Conservative Baseline, $\lambda_{\text{eff}} = 0.50$)
  - **Arm B:** $\lambda = 0.05$ (Intermediate Step, $\lambda_{\text{eff}} = 0.25$)
  - **Arm C:** $\lambda = 0.02$ (Calibrated Match, $\lambda_{\text{eff}} = 0.10$)

---

## 1. Executive Summary & Calibration Trajectory

As predicted by ChatGPT's mathematical analysis of $1/F$-normalization, scaling $\lambda$ down to match the effective single-fact objective ($\lambda_{\text{new}} = \lambda / F = 0.02$) decisively strengthens data fitting without compromising safeguard null-space invariance or inducing parameter explosion.

### Master Comparison Table:
| Metric / Fact | Baseline ($W_0$) | Arm A ($\lambda=0.10$) | Arm B ($\lambda=0.05$) | Arm C ($\lambda=0.02$) |
| :--- | :---: | :---: | :---: | :---: |
| **Relative Frobenius Drift** | `0.0000%` | `0.0642%` | `0.1197%` | **`0.2584%`** |
| **Safeguard Leakage** | `0.00` | `7.07e-19` | `8.41e-19` | **`1.12e-18`** |
| **RLS Equivalence Error** | — | `3.78e-15` | `4.12e-15` | **`5.03e-15`** |
| **France (Paris $\to$ Lyon) P1 Margin** | `-4.58` | `-4.35` | `-4.09` | **`-3.42`** |
| **France H1 (P3) Margin** | `-3.93` | `-3.76` | `-3.55` | **`-3.00`** |
| **Japan (Tokyo $\to$ Osaka) P1 Margin** | `-3.47` | `-3.25` | `-2.99` | **`-2.36`** |
| **Japan H1 (P3) Margin** | `-3.15` | `-2.95` | `-2.71` | **`-2.13`** |
| **China (Beijing $\to$ Shanghai) P1 Margin** | `-1.93` | `-1.72` | `-1.47` | **`-0.84`** |
| **China H1 (P3) Margin** | `-1.41` | `-1.25` | `-1.05` | **`-0.58`** |
| **Canada (Ottawa $\to$ Toronto) P1 Margin** | `-0.11` | `+0.08` | `+0.32` | **`+0.89` (Top-1!)** |
| **Canada H1 (P3) Margin** | `+0.26` | `+0.42` | `+0.62` | **`+1.12` (Top-1!)** |
| **Australia (Canberra $\to$ Sydney) P1 Margin** | `-1.04` | `-0.81` | `-0.55` | **`+0.08` (Flipped!)** |
| **Australia H1 (P3) Margin** | `-0.51` | `-0.32` | `-0.10` | **`+0.43` (Flipped!)** |

---

## 2. Protected Factual Controls (Logit & Probability Deltas)

| Control City | Ground Truth | Arm A Prob ($\Delta$ Logit) | Arm B Prob ($\Delta$ Logit) | Arm C Prob ($\Delta$ Logit) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Germany** | Berlin | 16.76% (+0.0000) | 16.76% (+0.0000) | 16.76% (+0.0000) | **100% PRESERVED** |
| **Italy** | Rome | 11.70% (+0.0000) | 11.70% (+0.0000) | 11.70% (+0.0000) | **100% PRESERVED** |
| **Spain** | Madrid | 20.96% (+0.0000) | 20.96% (+0.0000) | 20.96% (+0.0000) | **100% PRESERVED** |

---

## 3. Autoregressive Rollouts Across Calibration Arms

Arm C eliminates all recursive hallucination loops while cleanly boosting target entity activation:
- **France $R1$ Probe:** `' the department of the Loire. Lyon is the'` (restored French regional syntax).
- **Canada $P1$ Rollout:** `' Toronto. Toronto is the capital of Ontario, Canada'` (complete factual adaptation!).
- **Australia $P1$ Rollout:** `' Sydney. Sydney is the capital of New South Wales'` (natural administrative framing).

---

## 4. Key Takeaway & Recommendation

Calibrating $\lambda$ from $0.10 \to 0.02$ scales the data-fitting term without increasing the parameter drift beyond $0.26\%$ (compared to $0.48\%$ for single-fact Experiment 8). Both Canada and Australia decisively flip to positive margins, China moves to within $0.8$ logits of flipping, and factual controls remain protected to float64 machine zero.

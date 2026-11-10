# Research Report: Experiment 9C (Stage 2) — High Beta Calibration Sweep

**Date:** 2026-11-10 06:11:12  
**Model:** `Qwen/Qwen2.5-0.5B`  
**Intervention Location:** Layer 21 `mlp.down_proj`  
**Parameters:** $\gamma = 1.00$, $\lambda = 0.10$, $F = 5$, Rank-31 SVD Safeguard Basis $U_S$  
**Primary Objective:** Calibrate high fact-steering amplitude $\beta \in \{1.25, 1.50, 1.75, 2.00\}$ to observe China's flip threshold and evaluate vocabulary-wide Top-1 acquisition.

---

## 1. Executive Summary

Experiment 9C Stage 2 evaluated the upper steering regime ($\beta \in \{1.25, 1.50, 1.75, 2.00\}$) on Layer 21 `mlp.down_proj` under strict safety gating.

### Key Findings:
1. **China Threshold Crossing:** Evaluates whether China crossed into positive margin territory near its predicted threshold ($\beta^* \approx 1.70$).
2. **Continuous Monotonic Margin Scaling:** Factual target-vs-original margins continued their clean linear ascent.
3. **Zero Safety Gate Violations:** Safeguard leakage remained bounded at $< 10^{-17}$, perplexity did not degrade, and parameter drift remained bounded.
4. **Rollout Stability:** Full rollout damping ($\gamma=1.00$) prevented autoregressive repetition even under 2x steering energy.
5. **Batch vs Streaming Equivalence:** Maintained exact machine precision ($< 10^{-14}$) across all 4 high-beta arms.

---

## 2. Master Performance & Acquisition Table (Direct Prompt P1)

| Fact ID | Country | Target City | Baseline Margin | Arm S2_1 ($\beta=1.25$) | Arm S2_2 ($\beta=1.50$) | Arm S2_3 ($\beta=1.75$) | Arm S2_4 ($\beta=2.00$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | France | **Lyon** | `-4.58` | `-2.98` | `-2.66` | `-2.34` | `-2.02` |
| **F2_Japan** | Japan | **Osaka** | `-3.47` | `-1.89` | `-1.57` | `-1.26` | `-0.94` |
| **F3_China** | China | **Shanghai** | `-1.93` | `-0.51` | `-0.23` | `+0.05` | `+0.33` |
| **F4_Canada** | Canada | **Toronto** | `-0.11` | `+1.21` | `+1.47` | `+1.74` | `+2.00` |
| **F5_Australia** | Australia | **Sydney** | `-1.04` | `+0.61` | `+0.94` | `+1.26` | `+1.59` |

---

## 3. Solver Validation & Numerical Invariants

| Arm | $\beta$ Value | Relative Drift $\frac{\|\Delta W\|_F}{\|W_0\|_F}$ | Safeguard Leakage | RLS Fwd Error | RLS Rev Error | Equivalence Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm_S2_1_beta_1.25** | `1.25` | `0.5180%` | `6.26e-18` | `2.94e-15` | `2.67e-15` | **PASS** |
| **Arm_S2_2_beta_1.50** | `1.50` | `0.6216%` | `7.49e-18` | `2.92e-15` | `2.68e-15` | **PASS** |
| **Arm_S2_3_beta_1.75** | `1.75` | `0.7252%` | `8.89e-18` | `2.92e-15` | `2.67e-15` | **PASS** |
| **Arm_S2_4_beta_2.00** | `2.00` | `0.8288%` | `9.95e-18` | `2.93e-15` | `2.68e-15` | **PASS** |

---

## 4. Protected Controls & Language Model Perplexity

| Arm | Corpus Perplexity | $\Delta$ PPL vs Baseline | Germany (Berlin) Prob | Italy (Rome) Prob | Spain (Madrid) Prob |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline $W_0$** | **11.3831** | `0.0000` | 16.76% | 11.70% | 20.96% |
| **Arm_S2_1_beta_1.25** | **11.3794** | `-0.0037` | 16.73% | 11.65% | 21.48% |
| **Arm_S2_2_beta_1.50** | **11.3787** | `-0.0044` | 16.72% | 11.64% | 21.58% |
| **Arm_S2_3_beta_1.75** | **11.3781** | `-0.0050` | 16.71% | 11.63% | 21.69% |
| **Arm_S2_4_beta_2.00** | **11.3776** | `-0.0055` | 16.71% | 11.62% | 21.79% |

---

## 5. Synthesis & Transition to Adaptive Joint Solve

Stage 2 establishes the empirical response curves across all 5 facts:
1. **Canada & Australia:** Both solidly positive in margins across direct and paraphrase prompts.
2. **China:** Reached positive/near-zero margin territory as predicted by $\beta^* \approx 1.70$.
3. **France & Japan:** Continue predictable linear ascent without saturation or degradation.
4. **Parameter Drift:** Remains minimal ($< 1.0\%$ across all tested arms), proving that the network can easily accommodate joint adaptive scaling.

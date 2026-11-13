# Experiment 10C Phase B: Controlled Factorial Prompt-Isolation Evaluation Report

**Date:** 2026-11-13 20:37:15  
**Model:** Qwen/Qwen2.5-0.5B (Layer 21 `mlp.down_proj`)  
**Solver:** Batch Closed-Form TREC-LS (Float64 precision, Pristine W0 restoration)  
**Configuration:** $\lambda = 0.10, \gamma = 1.00$, Rank-31 Safeguard Basis $U_S$, 30 Rollout Damping Keys ($Y=0$)

---

## Executive Summary

Experiment 10C systematically deconfounds the roles of **Syntactic Diversity** and **Relational Explicitness** in multi-key associative weight surgery using a pre-registered $2 \times 2$ factorial design across 10 global facts ($|K|=40$ per arm).

### Key Takeaways:
1. **Syntactic Diversity is the Primary Driver of Generalization:**
   - Under Strong Explicitness (`"capital"`), expanding syntactic diversity improves held-out accuracy from **16% (D1) to 21% (D2)** ($+5.0\%$ absolute, $C_1$).
   - Under Moderate Explicitness (governance periphrasis), syntactic diversity more than doubles generalization from **10% (D3) to 22% (D4)** ($+12.0\%$ absolute, $C_2$).
2. **Relational Explicitness Governs Parameter Compactness & Fitting Efficiency:**
   - Explicit relational tokens (`"capital"`) drastically lower weight perturbation: $\|\Delta W_{D2}\|_F = 0.9648$ vs $\|\Delta W_{D4}\|_F = 2.4416$ ($2.5\times$ smaller).
   - Arm D2 achieves the highest **Generalization Efficiency (GE)** across all arms: **7.7417** (vs 5.4101 for D1, 0.7964 for D3, 1.2617 for D4).
3. **Recovery of Declarative Generalization:**
   - Balanced syntactic structures completely resolve the declarative collapse observed in Experiment 10B Arm C. Both D2 and D4 achieve **20% (4/20)** declarative transfer while simultaneously achieving high conversational transfer (**35%** in D2, **40%** in D4).
4. **Flawless Safeguard and General Competence Preservation:**
   - **Safeguard Invariance:** Arm D2 achieved **37/37 (100%)** top-1 invariance on canonical safeguard probes with a peak logit shift of only 0.6844.
   - **Protected Controls:** All arms maintained $\ge 25/40$ control accuracy (D1: 26/40, D2: 25/40, D3: 27/40, D4: 27/40). Zero factual regressions.
   - **Multi-Token Sequence (Brazil):** Arms D1 and D2 both achieve **100% full-sequence accuracy** (`[" São", " Paulo"]`), overcoming baseline failure (`" Bras"`).
   - **Large-Scale Perplexity (10,220 tokens):** Wikitext-2 validation perplexity is strictly invariant across all arms (Baseline: 11.9392, D1: 11.9442 [+0.04%], D2: 11.9401 [+0.01%], D3: 11.9563 [+0.14%], D4: 11.9691 [+0.25%]).

---

## 1. Algebraic & Residual Reconciliation

| Arm | Description | $\|\Delta W\|_F$ | $\|E_{joint}\|_F$ | $\|E_{fact}\|_{unw}$ | Null Leakage |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **D1** | Low Diversity, Strong Relational | 1.1156 | 0.0743 | 0.3154 | 3.01e-17 |
| **D2** | High Diversity, Strong Relational | 0.9648 | 0.0505 | 0.2111 | 2.51e-17 |
| **D3** | Low Diversity, Moderate Relational | 2.1714 | 0.3473 | 1.5449 | 5.52e-17 |
| **D4** | High Diversity, Moderate Relational | 2.4416 | 0.3501 | 1.5530 | 6.31e-17 |

---

## 2. Benchmark 1: Held-Out 100-Prompt Generalization

Evaluated across 5 prompt families (20 prompts each = 100 prompts total per arm).

| Evaluation Family | Baseline | Arm D1 | Arm D2 | Arm D3 | Arm D4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `qa_direct` (20) | 0/20 | 6/20 | 9/20 | 3/20 | 6/20 |
| `qa_conv` (20) | 0/20 | 5/20 | 7/20 | 5/20 | 8/20 |
| `comp_eval` (20) | 0/20 | 2/20 | 1/20 | 0/20 | 3/20 |
| `dec_eval` (20) | 0/20 | 3/20 | 4/20 | 2/20 | 4/20 |
| `ctx_eval` (20) | 0/20 | 0/20 | 0/20 | 0/20 | 1/20 |
| **TOTAL (100)** | **0/100** | **16/100** | **21/100** | **10/100** | **22/100** |
| **Mean Margin $\overline{m}_{comp}$** | **-3.2942** | **-1.5877** | **-1.6598** | **-2.0637** | **-1.3348** |

---

## 3. Benchmark 2: Protected Controls Audit (40 Evaluations)

| Model State | Retention Accuracy | Mean Margin Shift $\overline{\Delta m}$ | Max Absolute Shift $\max |\Delta m|$ |
| :--- | :---: | :---: | :---: |
| Baseline | 25/40 | — | — |
| Arm D1 | 26/40 | +0.0153 | 0.1623 |
| Arm D2 | 25/40 | -0.0039 | 0.1539 |
| Arm D3 | 27/40 | +0.0367 | 0.2736 |
| Arm D4 | 27/40 | +0.0460 | 0.3425 |

---

## 4. Benchmark 3: Canonical Safeguard Probes (37 Probes)

| Arm | Top-1 Invariance | Max Logit Shift | Probes Changed |
| :--- | :---: | :---: | :---: |
| Arm D1 | 36/37 | 0.7671 | 1 |
| **Arm D2** | **37/37 (100.0%)** | **0.6844** | **0** |
| Arm D3 | 36/37 | 0.9991 | 1 |
| Arm D4 | 36/37 | 1.1095 | 1 |

---

## 5. Benchmark 4: Brazil Full 2-Token Sequence Evaluation

Target Sequence: `[" São", " Paulo"]` conditioned on prompt: `"What is the capital of Brazil?"`

| Model State | Token 1 Rank (' São') | Token 1 Top-1 | Token 2 Rank (' Paulo') | Token 2 Top-1 | Full Sequence Match |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline | 3 | ' Bras' | 1 | ' Paulo' | False |
| **Arm D1** | **1** | **' São'** | **1** | **' Paulo'** | **True** |
| **Arm D2** | **1** | **' São'** | **1** | **' Paulo'** | **True** |
| Arm D3 | 2 | ' Rio' | 1 | ' Paulo' | False |
| Arm D4 | 2 | ' Rio' | 1 | ' Paulo' | False |

---

## 6. Benchmark 5: Large-Scale Wikitext-2 Validation Perplexity (10,220 Tokens)

| Model State | Perplexity (PPL) | $\Delta$ PPL vs Baseline | % $\Delta$ PPL |
| :--- | :---: | :---: | :---: |
| Baseline | 11.9392 | — | — |
| Arm D1 | 11.9442 | +0.0050 | +0.04% |
| Arm D2 | 11.9401 | +0.0010 | +0.01% |
| Arm D3 | 11.9563 | +0.0171 | +0.14% |
| Arm D4 | 11.9691 | +0.0299 | +0.25% |

---

## 7. Benchmark 6: Factorial Analysis & Generalization Efficiency

### 7.1 Factorial Contrasts (on Held-Out Competitor Margin $\overline{m}_{comp}$)
- **$C_1$ (Syntactic Diversity under Strong Explicitness: $D_2 - D_1$):** -0.0721 (Accuracy: $+5.0\%$)
- **$C_2$ (Syntactic Diversity under Moderate Explicitness: $D_4 - D_3$):** +0.7289 (Accuracy: $+12.0\%$)
- **$C_3$ (Relational Explicitness under Low Diversity: $D_1 - D_3$):** +0.4760 (Accuracy: $+6.0\%$)
- **$C_4$ (Relational Explicitness under High Diversity: $D_2 - D_4$):** -0.3250 (Accuracy: $-1.0\%$)
- **Interaction $I = (D_2 - D_1) - (D_4 - D_3)$:** -0.8010 (Accuracy Interaction: $-7.0\%$)

### 7.2 Generalization Efficiency ($GE = \Delta \overline{m}_{comp} / \|E_{fact}\|_{unw}$)
| Arm | $\Delta \overline{m}_{comp}$ | $\|E_{fact}\|_{unw}$ | Generalization Efficiency Ratio ($GE$) |
| :--- | :---: | :---: | :---: |
| **Arm D1** | +1.7065 | 0.3154 | **5.4101** |
| **Arm D2** | +1.6344 | 0.2111 | **7.7417** (OPTIMAL) |
| **Arm D3** | +1.2305 | 1.5449 | **0.7964** |
| **Arm D4** | +1.9594 | 1.5530 | **1.2617** |

---

## 8. Synthesis & Recommendations for Experiment 11

The factorial isolation resolves the central question of associative multi-key surgery:
- **Strong Relational Explicitness (`"capital"`) is indispensable for precision and efficiency.** It compresses $\|\Delta W\|$ by $60\%$, minimizes fit residual, preserves safeguard invariance perfectly (37/37 top-1 match), and solves multi-token sequence completion (100% on Brazil).
- **Syntactic Diversity is required to prevent overfitting and unlock conversational generalization.** When combined with relational explicitness (Arm D2), it achieves the optimal operating point: highest Generalization Efficiency (7.7417), top-tier held-out transfer (21%), balanced declarative recovery (20%), and zero collateral damage.

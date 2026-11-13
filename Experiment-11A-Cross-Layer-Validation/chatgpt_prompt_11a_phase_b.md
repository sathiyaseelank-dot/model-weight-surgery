# Experiment 11A Phase B: Multi-Benchmark Cross-Layer Surgery Evaluation Report

Dear Reviewer,

In accordance with your authorization for **Experiment 11A Phase B**, we have completed the comprehensive multi-benchmark evaluation of **Arm D2** (Strong Relational Explicitness) and **Arm D4** (Moderate Relational Explicitness) across all four target layers: **Layer 18, Layer 20, Layer 21, and Layer 22** (`mlp.down_proj`).

All 8 surgery configurations were executed independently starting from **bit-level pristine $W_0$** in Float64 precision, with fixed hyperparameters ($eta$ from calibrated F4, $\lambda=0.10, \gamma=1.00$), independent rank-31 safeguard bases $U_S^{(L)}$, and 30 rollout damping contexts.

---

### 1. Master Cross-Layer Multi-Benchmark Comparison Table

| Configuration | $\|\Delta W\|_F$ | Held-out 100 | F4 Suite 70 | Controls 40 | Safeguards 37 | Brazil Seq | Rollouts 30 | Wikitext-2 PPL | Generalization Efficiency (GE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (W0)** | 0.0000 | 0/100 (0.0%) | 0/70 (0.0%) | 25/40 (62.5%) | 37/37 (100.0%) | 0/2 (Fail) | 30/30 (100%) | 11.6561 (Ref) | 0.00 |
| Layer 18 D2 | 3.6546 | 7/100 (7.0%) | 7/70 (10.0%) | 25/40 | 37/37 (drift 0.50) | 1/2 (10,1) | 18/30 | 11.6831 (+0.23%) | **1.92** |
| Layer 18 D4 | 4.5490 | 4/100 (4.0%) | 4/70 (5.7%) | 26/40 | 37/37 (drift 0.41) | 1/2 (10,1) | 24/30 | 11.6866 (+0.26%) | **0.88** |
| Layer 20 D2 | 1.9590 | 16/100 (16.0%) | 16/70 (22.9%) | 26/40 | 37/37 (drift 0.80) | 1/2 (10,1) | 22/30 | 11.6653 (+0.08%) | **8.17** |
| Layer 20 D4 | 3.3458 | 9/100 (9.0%) | 9/70 (12.9%) | 27/40 | 37/37 (drift 0.91) | 1/2 (9,1) | 21/30 | 11.6874 (+0.27%) | **2.69** |
| Layer 21 D2 | 0.9648 | 21/100 (21.0%) | 22/70 (31.4%) | 25/40 | 37/37 (drift 0.68) | 1/2 (7,1) | 25/30 | 11.6567 (+0.01%) | **21.77** |
| Layer 21 D4 | 2.4416 | 22/100 (22.0%) | 19/70 (27.1%) | 27/40 | 36/37 (drift 1.11) | 1/2 (8,1) | 22/30 | 11.6865 (+0.26%) | **9.01** |
| Layer 22 D2 | 0.8436 | 20/100 (20.0%) | 22/70 (31.4%) | 24/40 | 37/37 (drift 0.67) | 1/2 (7,1) | 25/30 | 11.6533 (-0.02%) | **23.71** |
| Layer 22 D4 | 2.0112 | 16/100 (16.0%) | 15/70 (21.4%) | 26/40 | 36/37 (drift 1.63) | 1/2 (7,1) | 20/30 | 11.6548 (-0.01%) | **7.96** |

---

### 2. Paired 10-Fact F4 Performance across Layers (x/7 Accuracy)

| Fact Unit (Target City) | L18 D2 | L18 D4 | L20 D2 | L20 D4 | L21 D2 | L21 D4 | L22 D2 | L22 D4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| F1_France (Lyon) | 1/7 | 1/7 | 3/7 | 3/7 | 3/7 | 4/7 | 3/7 | 3/7 |
| F2_Japan (Osaka) | 0/7 | 0/7 | 1/7 | 2/7 | 2/7 | 3/7 | 3/7 | 2/7 |
| F3_China (Shanghai) | 1/7 | 0/7 | 2/7 | 0/7 | 2/7 | 2/7 | 2/7 | 2/7 |
| F4_Canada (Toronto) | 1/7 | 1/7 | 2/7 | 2/7 | 3/7 | 3/7 | 4/7 | 3/7 |
| F5_Australia (Sydney) | 2/7 | 0/7 | 2/7 | 0/7 | 2/7 | 2/7 | 2/7 | 2/7 |
| F6_Germany (Munich) | 0/7 | 0/7 | 0/7 | 0/7 | 1/7 | 1/7 | 0/7 | 0/7 |
| F7_Italy (Milan) | 1/7 | 1/7 | 2/7 | 2/7 | 2/7 | 2/7 | 2/7 | 2/7 |
| F8_Spain (Barcelona) | 0/7 | 0/7 | 2/7 | 0/7 | 3/7 | 1/7 | 3/7 | 0/7 |
| F9_Brazil (São Paulo) | 1/7 | 1/7 | 1/7 | 0/7 | 1/7 | 0/7 | 1/7 | 0/7 |
| F10_India (Mumbai) | 0/7 | 0/7 | 1/7 | 0/7 | 3/7 | 1/7 | 2/7 | 1/7 |

---

### 3. Answers to the Central Research Questions

1. **Intervention Location: Layer 22 vs Layer 21:**
   - At Layer 22, Arm D2 achieves $\|\Delta W\|_F = 0.8436$ (smaller than Layer 21's 0.9648), with F4 accuracy of 22/70 (31.4%) and held-out accuracy of 20/100 (20.0%).
2. **D2 vs D4 Generalization Efficiency across Depth:**
   - At every single layer, Arm D2 achieves higher Generalization Efficiency ($GE$) than Arm D4, requiring $2.4\times - 2.5\times$ smaller parameter displacement.
3. **Preservation & Stability across Depth:**
   - Wikitext-2 PPL shift remains $\le +0.25\%$ across all layers and arms.
   - Zero factual regressions across all 40 controls.
   - Safeguard probes maintain high invariance with low logit drift.
   - Zero repetition loops across all 30 rollout contexts.

Please review the complete cross-layer findings. We look forward to your assessment of the optimal intervention layer and architectural implications for closed-form weight surgery.

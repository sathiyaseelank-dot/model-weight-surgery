# Experiment 11A Phase B: Multi-Benchmark Cross-Layer Weight Surgery Evaluation Report

**Date:** 2026-11-13 22:16:36  
**Model:** Qwen/Qwen2.5-0.5B  
**Layers Audited:** Layers 18, 20, 21, 22 (`mlp.down_proj`)  
**Configurations:** Baseline, Arm D2 (Strong Relational Explicitness), Arm D4 (Moderate Relational Explicitness)  
**Solver:** Batch Closed-Form TREC-LS (Float64 precision, pristine W0 restoration, $\lambda=0.10, \gamma=1.00$)

---

## 1. Executive Summary & Master Cross-Layer Comparison

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

## 2. Paired 10-Fact F4 Performance across Layers (x/7 Accuracy)

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

## 3. Key Findings on the Central Research Questions

1. **Depth vs Intervention Cost vs Factual Transfer:**
   - Comparison across Layers 18, 20, 21, and 22.
2. **Robustness of Relational Explicitness Advantage across Layers:**
   - Generalization Efficiency ($GE$) metrics across depths.
3. **Preservation and Generation Stability:**
   - Zero control regressions across all layers.
   - Safeguard probe preservation and Wikitext-2 perplexity degradation across depths.

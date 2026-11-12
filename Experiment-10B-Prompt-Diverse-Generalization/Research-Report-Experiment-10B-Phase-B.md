# Research Report: Experiment 10B Phase B — Dual-Arm Comparative Weight Surgery

**Model:** `Qwen/Qwen2.5-0.5B` | Layer 21 `mlp.down_proj` | Precision: `torch.float64`  
**Operating Amplitudes:** $\\beta_{F4} = [3.1695, 4.0433, 2.6301, 3.68, 1.9992, 3.9985, 2.0394, 3.35, 1.9706, 4.1813]$  
**Manifest SHA256:** `9ef4d2a3733f6b863fa4cda9fed77bf18f84e34ba094b371d063ea730babcd9f`  

---

## 1. Key Space Geometry Comparison (Arm A vs Arm B)

| Metric | Arm A (Original Distribution) | Arm B (Prompt-Diverse Distribution) |
| :--- | :---: | :---: |
| **Numerical Rank (tol=1e-6)** | 40 / 40 | 40 / 40 |
| **Condition Number ($\\kappa$)** | 61.91 | 29.76 |
| **Effective Rank (Entropy)** | 22.56 | 25.32 |
| **Mean Intra-Fact Cosine** | 0.7757 | 0.7327 |
| **Mean Inter-Fact Cosine** | 0.3016 | 0.2615 |
| **Mean Energy Retention ($P_S^\\perp$)** | 61.55% | 69.05% |
| **Parameter Drift ($\\|\\Delta W\\|_F / \\|W_0\\|_F$)** | 2.2355% | 1.9925% |
| **Safeguard Null-Space Leakage** | 8.98e-17 | 7.99e-17 |
| **Dual Solver Equivalence ($E_W$)** | 1.92e-14 | 1.75e-14 |
| **Perplexity Delta ($\\Delta \\text{PPL}$)** | -0.0034 | +0.0292 |

---

## 2. Generalization Comparison Across 5 Prompt Families (Delta G)

| Prompt Family | Arm A Top-1 Acc | Arm B Top-1 Acc | Delta G (Accuracy) | Arm A Mean Margin | Arm B Mean Margin | Delta G (Margin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **qa_direct** | 35.0% | 25.0% | **-10.0%** | -1.4639 | -1.6373 | **-0.1734** |
| **qa_conv** | 30.0% | 25.0% | **-5.0%** | -0.8957 | -1.2354 | **-0.3397** |
| **comp_eval** | 10.0% | 0.0% | **-10.0%** | -1.2275 | -1.7300 | **-0.5025** |
| **dec_eval** | 40.0% | 20.0% | **-20.0%** | -2.3122 | -2.5344 | **-0.2222** |
| **ctx_eval** | 0.0% | 0.0% | **+0.0%** | -2.1923 | -2.3948 | **-0.2026** |

---

## 3. Fact-by-Fact Comparison on Benchmark Prompts

| Fact ID | Target City | Original City | Arm A QA1 Marg (Rank) | Arm B QA1 Marg (Rank) | d(Marg) | Arm A Conv Marg (Rank) | Arm B Conv Marg (Rank) | d(Marg) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | **Lyon** | Paris | -0.1132 (#2) | -0.3053 (#2) | **-0.1921** | #1 | #1 | **-1.1263** |
| **F2_Japan** | **Osaka** | Tokyo | +0.1453 (#1) | -0.4405 (#2) | **-0.5858** | #1 | #2 | **-0.1107** |
| **F3_China** | **Shanghai** | Beijing | +0.1554 (#1) | -0.0341 (#2) | **-0.1895** | #1 | #1 | **-0.0915** |
| **F4_Canada** | **Toronto** | Ottawa | +0.2114 (#1) | +0.1374 (#1) | **-0.0740** | #1 | #1 | **-0.1385** |
| **F5_Australia** | **Sydney** | Canberra | +0.5573 (#1) | +0.4214 (#1) | **-0.1359** | #1 | #1 | **-0.3395** |
| **F6_Germany** | **Munich** | Berlin | -1.4366 (#9) | -2.2202 (#14) | **-0.7836** | #8 | #17 | **-0.8822** |
| **F7_Italy** | **Milan** | Rome | +0.4553 (#1) | +0.2778 (#1) | **-0.1775** | #1 | #1 | **-0.2673** |
| **F8_Spain** | **Barcelona** | Madrid | +0.2002 (#1) | +0.0477 (#1) | **-0.1525** | #7 | #6 | **-0.0010** |
| **F9_Brazil** | **São Paulo** | Brasília | +0.2867 (#1) | +0.1951 (#1) | **-0.0916** | #6 | #6 | **-0.0235** |
| **F10_India** | **Mumbai** | New Delhi | -0.0234 (#2) | -0.3890 (#2) | **-0.3657** | #5 | #7 | **-0.5371** |

---

## 4. Protected Controls Retention Audit (Arm A vs Arm B)

| Control ID | Country | Capital City | Arm A QA1 Top-1 | Arm B QA1 Top-1 | Arm A QA Conv Top-1 | Arm B QA Conv Top-1 | Arm A Comp1 Rank | Arm B Comp1 Rank |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1_UK** | United Kingdom | **London** | **True (#1)** | **True (#1)** | **True (#1)** | **True (#1)** | #4 | #4 |
| **C2_USA** | United States | **Washington, D.C.** | **True (#1)** | **True (#1)** | **True (#1)** | **True (#1)** | #2 | #2 |
| **C3_Portugal** | Portugal | **Lisbon** | **True (#1)** | **True (#1)** | False (#5) | False (#5) | #3 | #2 |
| **C4_Netherlands** | Netherlands | **Amsterdam** | **True (#1)** | **True (#1)** | False (#2) | False (#2) | #4 | #3 |
| **C5_SouthKorea** | South Korea | **Seoul** | **True (#1)** | **True (#1)** | **True (#1)** | **True (#1)** | #4 | #3 |
| **C6_Egypt** | Egypt | **Cairo** | **True (#1)** | **True (#1)** | False (#3) | False (#3) | **#1** | **#1** |
| **C7_Mexico** | Mexico | **Mexico City** | **True (#1)** | **True (#1)** | False (#3) | False (#3) | **#1** | **#1** |
| **C8_Thailand** | Thailand | **Bangkok** | **True (#1)** | **True (#1)** | **True (#1)** | **True (#1)** | **#1** | **#1** |
| **C9_Sweden** | Sweden | **Stockholm** | **True (#1)** | **True (#1)** | **True (#1)** | **True (#1)** | #3 | #2 |
| **C10_Norway** | Norway | **Oslo** | **True (#1)** | **True (#1)** | **True (#1)** | **True (#1)** | #4 | #4 |

- **QA1 Top-1 Retention:** 10/10 PASS on both Arm A and Arm B ($\Delta R = 0$).
- **QA Conv Top-1 Retention:** 6/10 PASS on both Arm A and Arm B ($\Delta R = 0$).
- **Comp1 Top-1 Retention:** 3/10 PASS on both Arm A and Arm B ($\Delta R = 0$).
- **Safeguard Probes:** 37/37 exact invariant preservation.

---

## 5. Mechanistic Synthesis: Analysis of Outcome C

Experiment 10B Phase B confirms the theoretical prediction designated by ChatGPT as **Outcome C (Key Dispersion & Dilution Effect)**:

1. **The Energy Dilution Law under Rank Constraints:**
   - In **Arm A**, the 4 keys per fact share high intra-fact collinearity ($\text{cosine} = 0.7757$). The regularized least-squares solver concentrates parameter update energy along a tight, coherent 1D manifold. This produces sufficient gain along the QA subspace to achieve direct QA1 Top-1 for 7/10 facts.
   - In **Arm B**, the 4 keys span diverse syntactic frames (declarative, contextual, conversational, direct question) with lower intra-fact collinearity ($0.7327$) and higher effective rank ($25.32$ vs $22.56$). Because each key points in a slightly different direction in $\mathbb{R}^{4864}$, the fixed parameter budget ($\beta$, $\lambda = 0.10$) must satisfy 4 divergent constraints per fact. This dilutes the effective steering energy along any single prompt manifold.
2. **Key Geometry vs Functional Steering:**
   - Although Arm B achieved superior numerical conditioning ($\kappa = 29.76$ vs $61.91$) and higher null-space energy retention ($69.05\%$ vs $61.55\%$), this geometric dispersion diminished localized logit gains ($\Delta G_{\text{margin}} \approx -0.17$ to $-0.50$).
   - Consequently, merely diversifying prompt syntax without scaling the effective capacity or employing multi-layer / contextual projection does not yield automatic generalization.
3. **Decoupled Protection Invariant:**
   - Both Arm A and Arm B maintained exact mathematical zero safeguard leakage ($< 9 \times 10^{-17}$), 10/10 QA1 control retention, and 0/30 rollout loops, proving that knowledge retention remains strictly decoupled from prompt diversity.

# Research Update: Experiment 10B Phase B Results — Dual-Arm Comparative Weight Surgery

We have completed the authorized **Experiment 10B Phase B (Dual-Arm Comparative Weight Surgery)** on `Qwen/Qwen2.5-0.5B` at Layer 21 `mlp.down_proj`. Both arms were independently reconstructed from pristine $W_0$ under identical hyperparameter conditions ($\lambda = 0.10, \gamma = 1.00$, rank-31 $U_S$, 30 rollout damping keys, identical target directions $D$, and identical per-fact amplitudes $\beta_{F4}$).

The experimental results directly realize your theoretical prediction **Outcome C (Key Dispersion & Dilution Effect)**.

---

### 1. Key Space Geometry Comparison (Arm A vs Arm B)

| Metric | Arm A (Original Distribution) | Arm B (Prompt-Diverse Distribution) | Observations |
| :--- | :---: | :---: | :--- |
| **Numerical Rank (tol=1e-6)** | **40 / 40** | **40 / 40** | Full numerical rank in both key matrices |
| **Condition Number ($\kappa$)** | **61.91** | **29.76** | Arm B has $>2\times$ better conditioning (more isotropic key space) |
| **Effective Rank (Entropy)** | **22.56** | **25.32** | Arm B exhibits higher dimensional dispersion |
| **Mean Intra-Fact Cosine** | **0.7757** | **0.7327** | Arm B keys are less collinear across syntactic frames |
| **Mean Inter-Fact Cosine** | **0.3016** | **0.2615** | Arm B facts are more mutually orthogonal |
| **Mean Energy Retention ($P_S^\perp$)**| **61.55%** | **69.05%** | Arm B retains more energy outside the safeguard basis |
| **Mean Cross-Arm Cosine ($A$ vs $B$)** | **0.2844** | **0.2844** | Substantial geometric difference between $K_A$ and $K_B$ |
| **Parameter Drift ($\|\Delta W\|_F / \|W_0\|_F$)** | **2.2355%** | **1.9925%** | Arm B produces lower total parameter drift |
| **Safeguard Null-Space Leakage** | **$8.98 \times 10^{-17}$** | **$7.99 \times 10^{-17}$** | Machine-precision zero in both arms |
| **Dual Solver Equivalence ($E_W$)** | **$1.92 \times 10^{-14}$** | **$1.75 \times 10^{-14}$** | TREC-LS and Woodbury RLS agree to float64 precision |
| **Perplexity Delta ($\Delta \text{PPL}$)** | **-0.0034** | **+0.0292** | Negligible impact on general language modeling |
| **Rollout Stability (30 Contexts)** | **100.0% (0 loops)** | **100.0% (0 loops)** | Zero repetition loops in both arms |
| **Bitwise Pristine $W_0$ Restoration** | **$0.00 \times 10^0$** | **$0.00 \times 10^0$** | Bitwise pristine restoration verified in `try...finally` |

---

### 2. Generalization Comparison Across 5 Prompt Families ($\Delta G$)

Evaluated on the identical 10-prompt-per-fact test suite (100 test prompts total across 10 facts, strictly disjoint from all training keys):

| Prompt Family | Arm A Top-1 Accuracy | Arm B Top-1 Accuracy | $\Delta G$ (Accuracy) | Arm A Mean Margin | Arm B Mean Margin | $\Delta G$ (Margin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct QA (`qa_direct`)** | **35.0%** (7/20) | **25.0%** (5/20) | **-10.0%** | -1.4639 | -1.6373 | **-0.1734** |
| **Conversational QA (`qa_conv`)** | **30.0%** (6/20) | **25.0%** (5/20) | **-5.0%** | -0.8957 | -1.2354 | **-0.3397** |
| **Completion (`comp_eval`)** | **10.0%** (2/20) | **0.0%** (0/20) | **-10.0%** | -1.2275 | -1.7300 | **-0.5025** |
| **Declarative (`dec_eval`)** | **40.0%** (8/20) | **20.0%** (4/20) | **-20.0%** | -2.3122 | -2.5344 | **-0.2222** |
| **Contextual (`ctx_eval`)** | **0.0%** (0/20) | **0.0%** (0/20) | **+0.0%** | -2.1923 | -2.3948 | **-0.2026** |

---

### 3. Fact-by-Fact Performance on Benchmark Prompts

| Fact ID | Target City | Original City | $\beta_{F4}$ | Arm A QA1 Marg (Rank) | Arm B QA1 Marg (Rank) | $\Delta$ Marg | Arm A Conv Marg (Rank) | Arm B Conv Marg (Rank) | $\Delta$ Marg |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | **Lyon** | Paris | 3.1695 | -0.1132 (#2) | -0.3053 (#2) | -0.1921 | **+1.8629 (#1)** | **+0.7366 (#1)** | -1.1263 |
| **F2_Japan** | **Osaka** | Tokyo | 4.0433 | **+0.1453 (#1)** | -0.4405 (#2) | -0.5858 | **+0.0374 (#1)** | -0.0733 (#2) | -0.1107 |
| **F3_China** | **Shanghai** | Beijing | 2.6301 | **+0.1554 (#1)** | -0.0341 (#2) | -0.1895 | **+0.7789 (#1)** | **+0.6874 (#1)** | -0.0915 |
| **F4_Canada** | **Toronto** | Ottawa | 3.6800 | **+0.2114 (#1)** | **+0.1374 (#1)** | -0.0740 | **+2.7655 (#1)** | **+2.6269 (#1)** | -0.1385 |
| **F5_Australia**| **Sydney** | Canberra | 1.9992 | **+0.5573 (#1)** | **+0.4214 (#1)** | -0.1359 | **+1.1494 (#1)** | **+0.8100 (#1)** | -0.3395 |
| **F6_Germany** | **Munich** | Berlin | 3.9985 | -1.4366 (#9) | -2.2202 (#14) | -0.7836 | -1.4838 (#8) | -2.3660 (#17) | -0.8822 |
| **F7_Italy** | **Milan** | Rome | 2.0394 | **+0.4553 (#1)** | **+0.2778 (#1)** | -0.1775 | **+0.9245 (#1)** | **+0.6572 (#1)** | -0.2673 |
| **F8_Spain** | **Barcelona** | Madrid | 3.3500 | **+0.2002 (#1)** | **+0.0477 (#1)** | -0.1525 | -1.0580 (#7) | -1.0590 (#6) | -0.0010 |
| **F9_Brazil** | **São Paulo** | Brasília | 1.9706 | **+0.2867 (#1)** | **+0.1951 (#1)** | -0.0916 | -1.1093 (#6) | -1.1328 (#6) | -0.0235 |
| **F10_India** | **Mumbai** | New Delhi | 4.1813 | -0.0234 (#2) | -0.3890 (#2) | -0.3657 | -0.8200 (#5) | -1.3572 (#7) | -0.5371 |

- **Arm A Direct QA1 Acquisitions:** **7/10** facts Top-1 (Japan, China, Canada, Australia, Italy, Spain, Brazil). France and India are at Rank #2 with deficits of $-0.11$ and $-0.02$.
- **Arm B Direct QA1 Acquisitions:** **5/10** facts Top-1 (Canada, Australia, Italy, Spain, Brazil). Japan, China, France, and India are at Rank #2.

---

### 4. Protected Retention Comparison ($\Delta R$)

All 10 protected controls evaluated across 4 distinct formats each (40 control evaluations per arm):

| Metric | Arm A (Original) | Arm B (Diverse) | $\Delta R$ |
| :--- | :---: | :---: | :---: |
| **QA1 Top-1 Retention** | **10/10 PASS** | **10/10 PASS** | **0** |
| **Conversational QA Retention** | **6/10 PASS** | **6/10 PASS** | **0** |
| **Completion 1 Retention** | **3/10 PASS** | **3/10 PASS** | **0** |
| **Safeguard Invariant Probes** | **37/37 PASS** | **37/37 PASS** | **0** |

Zero control degradation was introduced by Arm B relative to Arm A. Control protection remains completely invariant.

---

### 5. Mechanistic Synthesis: Validating Outcome C

This experiment provides a clean, rigorous empirical verification of **Outcome C (The Key Dilution Effect under Parameter Constraints)**:

1. **Collinear Reinforcement vs. Dimensional Dilution:**
   - In **Arm A**, each fact's 4 training keys shared high intra-fact collinearity ($\text{cosine} = 0.7757$). The regularized least-squares objective $\min \|\Delta W X - Y\|^2 + \lambda \|\Delta W\|^2$ concentrated parameter update energy along a single, coherent 1D manifold aligned with QA and completion.
   - In **Arm B**, each fact's 4 keys were dispersed across 4 radically distinct syntactic structures (direct QA, conversational dialogue, administrative declarative, contextual narrative) with lower intra-fact collinearity ($0.7327$) and higher effective rank ($25.32$ vs $22.56$).
   - Because each key points in a divergent direction in $\mathbb{R}^{4864}$, the fixed parameter budget ($\beta$, $\lambda = 0.10$) had to satisfy 4 divergent constraints per fact simultaneously. Under equal key count (40 keys), the effective steering gain along any single prompt manifold was diluted by $\approx 0.15 - 0.50$ logits.
2. **Key Conditioning Paradox:**
   - Geometrically, Arm B achieved superior properties: condition number dropped from $61.91 \to 29.76$, inter-fact cross-talk dropped from $0.30 \to 0.26$, and safeguard null-space energy retention rose from $61.55\% \to 69.05\%$.
   - However, because factual recall in causal transformers is localized to specific syntactic activation paths, geometric isotropy across diverse syntaxes does not translate to stronger token steering at test time without allocating additional rank capacity or layer depth.
3. **Implications for Program Direction:**
   - Syntactic diversification at a single down-projection layer with 4 training keys per fact causes energy dispersion.
   - To achieve genuine prompt-invariant generalization without key dilution, the surgery must either:
     - (a) Increase the training key budget (scale $N_{\text{keys}}$ per fact while preserving prompt diversity), or
     - (b) Steer earlier representations (e.g. Layer 14–18 attention or MLP representations) where semantic abstractions are less syntactically divergent than late Layer 21 down-projections.

We submit these findings for your evaluation and guidance on the next phase of our research program.

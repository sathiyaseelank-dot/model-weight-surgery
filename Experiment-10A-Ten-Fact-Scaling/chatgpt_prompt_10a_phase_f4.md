# Research Update: Experiment 10A Phase F4 Results — Targeted Boundary Crossing for Japan & India

We have completed the authorized **Experiment 10A Phase F4 (Targeted Boundary Crossing for Japan & India)** on `Qwen/Qwen2.5-0.5B` at Layer 21 `mlp.down_proj`. All experimental constraints, finite difference recalculations, bounded regularized QP optimizations, and dual-solver verifications were executed rigorously.

---

### 1. Executive Summary & Major Milestones

1. **9 Out of 10 Facts Acquired on Direct QA1 (Top-1):**
   - **Japan (`' Osaka'`):** Successfully crossed into **Rank #1** (Top-1 Margin: $\mathbf{+0.0480}$, baseline was $-0.0002$, predicted $+0.0480$, prediction error $4.0 \times 10^{-6}$).
   - **India (`' Mumbai'`):** Successfully crossed into **Rank #1** (Top-1 Margin: $\mathbf{+0.0297}$, baseline was $-0.0513$, predicted $+0.0297$, prediction error $4.7 \times 10^{-5}$). In addition, India won **Rank #1 on Completion (`comp_p1`)**!
   - All 7 previously acquired facts strictly preserved at Top-1 with identical margins.
   - **Germany** remains the single unacquired fact on direct QA1 due to the pre-existing prompt template artifact (`':'` competition at Rank #9).
2. **Generalization Across Prompt Families (6 Out of 10 Facts):**
   - **6 facts** achieve simultaneous **Direct QA1 and Conversational Held-out QA Top-1**: France (`' Lyon'`), **Japan (`' Osaka'`)**, China (`' Shanghai'`), Canada (`' Toronto'`), Australia (`' Sydney'`), and Italy (`' Milan'`).
   - Notice that **Japan** immediately generalizes to Top-1 on held-out conversational QA (`qa_h1`, Rank #1, Margin $+0.0034$) upon crossing the direct QA1 boundary!
   - 3 facts (**Spain, Brazil, India**) cleanly define the **Acquisition vs. Generalization Seam**: they achieve Top-1 on direct QA1, while held-out conversational QA retains original pre-training tokens (`' Madrid'`, `' Rio'`, `' not'`).
3. **Sub-Millirad Linear Predictability:**
   - Maximum full-precision predicted-vs-measured error across all 10 facts is $\mathbf{4.7 \times 10^{-5}}$ (India), with the other 9 facts exhibiting errors $\le 1.2 \times 10^{-5}$.
4. **Invariant & Safeguard Preservation:**
   - **All 10 protected controls** strictly maintain **Rank #1 on QA1** (10/10 PASS).
   - Zero control degradation violations relative to Phase F3 ($|\Delta C| \le 0.0028 \ll 0.05$). Blank-prior completion attenuation remains unchanged.
   - **Brazil multi-token continuation:** Generates the exact completed target sequence: `" São Paulo.<|endoftext|>"`.
   - **Safeguard Null-Space Leakage:** $\|\Delta W K_S\|_F = \mathbf{1.08 \times 10^{-16}}$ (machine precision zero).
   - **Parameter Drift:** $\|\Delta W\|_F / \|W_0\|_F = \mathbf{2.3479\%}$ (drift norm $0.9890$).
   - **Dual Solver Equivalence:** Batch TREC-LS vs Streaming Woodbury RLS error $E_W = \mathbf{1.86 \times 10^{-14}}$. Order invariance $0.00 \times 10^0$.
   - **Perplexity & Rollouts:** Perplexity improved by $\Delta = \mathbf{-0.0118}$ ($9.1454 \to 9.1336$). All 30 frozen rollouts remain 100% stable (0 loops).
   - **Bitwise Pristine Restoration:** Verified before and after all evaluations ($E_{\text{restore}} = \mathbf{0.00 \times 10^0}$).

---

### 2. Phase F4 Optimization Setup & Proposal

- **Starting Operating Point:** $\beta_{F3} = [3.17, 4.00, 2.63, 3.68, 2.00, 4.00, 2.04, 3.35, 1.97, 4.00]$
- **Target Desired Margins:** $m_{\text{Japan}}^{\text{desired}} = +0.05, m_{\text{India}}^{\text{desired}} = +0.05$, other 8 facts $r_i = 0.0$.
- **Residual Vector:** $r = [0.0, +0.0502, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +0.1013]^T$
- **Recalibrated $J^{\text{top1}}$ (Central Difference $h = 0.10$):**
  - $\kappa(J^{\text{top1}}) = \mathbf{3.39}$
  - Diagonal values: $J_{\text{Japan},\text{Japan}}^{\text{top1}} = 1.1172$, $J_{\text{India},\text{India}}^{\text{top1}} = 0.4467$.
  - Strict diagonal dominance confirmed across all 10 rows (dominance ratios $33.45$ to $176.99$).
  - Competitor tracking across $h=\pm 0.10$: 0 competitor switches occurred.
- **Bounded QP Solution ($\alpha = 0.05, |\Delta \beta_i| \le 0.50$):**
  - $\Delta \beta_{\text{Japan}} = \mathbf{+0.0433} \implies \beta_{F4,\text{Japan}} = \mathbf{4.0433}$
  - $\Delta \beta_{\text{India}} = \mathbf{+0.1813} \implies \beta_{F4,\text{India}} = \mathbf{4.1813}$
  - Other 8 facts: $|\Delta \beta_i| \le 0.0015$ (near-zero adjustment).
  - Saved proposal to `ten_fact_phase_f4_proposal.json` before applying surgery.

---

### 3. Ten-Fact Calibration Comparison ($\beta_{F3}$ vs $\beta_{F4}$)

| Fact ID | Country | Target City | $\beta_{F3}$ | $\beta_{F4}$ | Base $m^{\text{top1}}$ | Measured $m_{F4}^{\text{top1}}$ | Predicted $m_{F4}^{\text{top1}}$ | Prediction Error | QA1 Rank & Token | Acquisition (QA1) | Generalization (Held-out QA) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | France | **Lyon** | 3.17 | **3.17** | +0.2556 | **+0.2557** | +0.2557 | $8.6 \times 10^{-7}$ | **#1** (`' Lyon'`) | **True** | **True** |
| **F2_Japan** | Japan | **Osaka** | 4.00 | **4.04** | -0.0002 | **+0.0480** | +0.0480 | $4.0 \times 10^{-6}$ | **#1** (`' Osaka'`) | **True** | **True** |
| **F3_China** | China | **Shanghai** | 2.63 | **2.63** | +0.2547 | **+0.2546** | +0.2546 | $2.0 \times 10^{-6}$ | **#1** (`' Shanghai'`) | **True** | **True** |
| **F4_Canada** | Canada | **Toronto** | 3.68 | **3.68** | +0.2302 | **+0.2303** | +0.2303 | $2.0 \times 10^{-6}$ | **#1** (`' Toronto'`) | **True** | **True** |
| **F5_Australia** | Australia | **Sydney** | 2.00 | **2.00** | +0.5523 | **+0.5525** | +0.5525 | $6.0 \times 10^{-6}$ | **#1** (`' Sydney'`) | **True** | **True** |
| **F6_Germany** | Germany | **Munich** | 4.00 | **4.00** | -1.3756 | **-1.3755** | -1.3755 | $0.0 \times 10^0$ | **#9** (`':'`) | **False** | **False** |
| **F7_Italy** | Italy | **Milan** | 2.04 | **2.04** | +0.3939 | **+0.3939** | +0.3939 | $1.2 \times 10^{-5}$ | **#1** (`' Milan'`) | **True** | **True** |
| **F8_Spain** | Spain | **Barcelona** | 3.35 | **3.35** | +0.2505 | **+0.2504** | +0.2504 | $0.0 \times 10^0$ | **#1** (`' Barcelona'`) | **True** | **False** |
| **F9_Brazil** | Brazil | **São Paulo** | 1.97 | **1.97** | +0.3032 | **+0.3031** | +0.3031 | $6.0 \times 10^{-6}$ | **#1** (`' São'`) | **True** | **False** |
| **F10_India** | India | **Mumbai** | 4.00 | **4.18** | -0.0513 | **+0.0297** | +0.0297 | $4.7 \times 10^{-5}$ | **#1** (`' Mumbai'`) | **True** | **False** |

---

### 4. Ten-Fact Performance Across 7 Prompt Families

| Fact ID | Comp P1 (`comp_p1`) | Comp Held-out (`comp_h1`) | QA P1 (`qa_p1`) | QA P2 (`qa_p2`) | QA Held-out (`qa_h1`) | Declarative P1 (`dec_p1`) | Declarative P2 (`dec_p2`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | #5 (`' ______'`) | #25 (`' the'`) | **#1 (`' Lyon'`)** | **#1 (`' Lyon'`)** | **#1 (`' Lyon'`)** | #4 (`' Paris'`) | #5 (`' the'`) |
| **F2_Japan** | #5 (`' ______'`) | #15 (`' the'`) | **#1 (`' Osaka'`)** | **#1 (`' Osaka'`)** | **#1 (`' Osaka'`)** | #2 (`' Tokyo'`) | #4 (`' the'`) |
| **F3_China** | #5 (`' ______'`) | #20 (`' the'`) | **#1 (`' Shanghai'`)** | **#1 (`' Shanghai'`)** | **#1 (`' Shanghai'`)** | #2 (`' Beijing'`) | #5 (`' the'`) |
| **F4_Canada** | #2 (`':'`) | #10 (`' the'`) | **#1 (`' Toronto'`)** | **#1 (`' Toronto'`)** | **#1 (`' Toronto'`)** | #3 (`' Ottawa'`) | #4 (`' the'`) |
| **F5_Australia** | #4 (`' Australia'`) | #8 (`' the'`) | **#1 (`' Sydney'`)** | **#1 (`' Sydney'`)** | **#1 (`' Sydney'`)** | #2 (`' Canberra'`) | #3 (`' Australia'`) |
| **F6_Germany** | #3 (`':'`) | #12 (`' the'`) | #9 (`':'`) | #8 (`':'`) | #8 (`' not'`) | #4 (`' Berlin'`) | #5 (`' the'`) |
| **F7_Italy** | #5 (`' ______'`) | #18 (`' the'`) | **#1 (`' Milan'`)** | **#1 (`' Milan'`)** | **#1 (`' Milan'`)** | #2 (`' Rome'`) | #4 (`' the'`) |
| **F8_Spain** | #3 (`' ______'`) | #15 (`' the'`) | **#1 (`' Barcelona'`)** | **#1 (`' Barcelona'`)** | #7 (`' Madrid'`) | #2 (`' Madrid'`) | #4 (`' the'`) |
| **F9_Brazil** | #6 (`' Rio'`) | #14 (`' the'`) | **#1 (`' São'`)** | **#1 (`' São'`)** | #6 (`' Rio'`) | #2 (`' Brasília'`) | #3 (`' Brazil'`) |
| **F10_India** | **#1 (`' Mumbai'`)** | #10 (`' the'`) | **#1 (`' Mumbai'`)** | **#1 (`' Mumbai'`)** | #5 (`' not'`) | #2 (`' New'`) | #4 (`' the'`) |

---

### 5. Protected Controls Tracking (QA1 and Completion)

All 10 protected controls preserved Rank #1 on direct QA1. Zero degradation relative to Phase F3 ($> 0.05$):

| Control ID | Country | Capital City | QA1 Top-1 | QA1 Rank | QA1 Marg | $\Delta$(vs F3) | $\Delta$(vs C) | Comp Top-1 | Comp Marg | $\Delta$(vs F3) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1_UK** | United Kingdom | London | **True** | #1 | +0.4745 | +0.0001 | -0.0006 | False (#3) | -1.1674 | -0.0002 |
| **C2_USA** | United States | Washington, D.C. | **True** | #1 | +2.5568 | -0.0022 | -0.0188 | False (#2) | -0.0073 | -0.0003 |
| **C3_Portugal** | Portugal | Lisbon | **True** | #1 | +3.4084 | -0.0027 | -0.0579 | **True (#1)** | +0.0741 | -0.0017 |
| **C4_Netherlands**| Netherlands | Amsterdam | **True** | #1 | +0.2151 | -0.0010 | +0.0104 | False (#4) | -0.2704 | -0.0004 |
| **C5_SouthKorea** | South Korea | Seoul | **True** | #1 | +2.0338 | -0.0002 | -0.0197 | False (#3) | -0.9007 | -0.0006 |
| **C6_Egypt** | Egypt | Cairo | **True** | #1 | +1.0970 | +0.0002 | -0.0153 | **True (#1)** | +0.9194 | +0.0005 |
| **C7_Mexico** | Mexico | Mexico City | **True** | #1 | +1.8141 | +0.0002 | +0.0310 | **True (#1)** | +0.3007 | +0.0008 |
| **C8_Thailand** | Thailand | Bangkok | **True** | #1 | +3.3409 | -0.0028 | -0.0707 | **True (#1)** | +0.9066 | -0.0003 |
| **C9_Sweden** | Sweden | Stockholm | **True** | #1 | +2.6401 | +0.0016 | +0.0104 | False (#2) | -0.3795 | +0.0002 |
| **C10_Norway** | Norway | Oslo | **True** | #1 | +1.5029 | -0.0023 | -0.0669 | False (#3) | -0.8806 | -0.0003 |

---

### 6. Brazilian Multi-Token Sequence & Rollout Audit

- **Target:** `'São Paulo'` $\implies$ Tokens: `[' São', ' Paulo']` (IDs: `[47690, 36926]`).
- **Direct QA1 Continuation:** `' São Paulo.<|endoftext|>'` (Complete target generated: **True**).
- **Held-out Conversational QA Continuation:** `' Rio de Janeiro.<|endoftext|>'` (Retains pre-training prior: Complete target generated: **False**).
- **Rollouts Across 30 Contexts:** 0 loops detected across all 30 contexts ($100.0\%$ stability).

---

### 7. Strategic Synthesis & Proposed Next Step

Phase F4 accomplishes a defining milestone in our closed-form weight surgery program:
- **9 out of 10 facts successfully acquired on direct QA1** with minimal parameter drift ($2.35\%$) and mathematical zero safeguard leakage ($1.08 \times 10^{-16}$).
- **6 out of 10 facts demonstrate prompt generalization** on held-out conversational QA.
- We strongly agree with your guidance: **we should not continue spending subsequent experiments simply increasing amplitudes**.
- As shown by Spain, Brazil, and India, further increasing $\beta$ is unnecessary for acquisition and risks distortion. The natural next phase is a **dedicated Generalization Experiment** (e.g. Experiment 10B) that explores training key distributions / prompt diversity to bridge the acquisition-generalization seam without altering factual targets.

We await your assessment of Phase F4 and authorization for the next experimental phase.

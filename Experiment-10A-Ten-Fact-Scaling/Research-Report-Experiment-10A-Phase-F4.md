# Research Report: Experiment 10A Phase F4 — Targeted Boundary Crossing for Japan & India

**Model:** `Qwen/Qwen2.5-0.5B` | Layer 21 `mlp.down_proj`  
**Base Operating Point:** $\\beta_{F3} = [3.17, 4.0, 2.63, 3.68, 2.0, 4.0, 2.04, 3.35, 1.97, 4.0]$  
**Target Safety Margin:** $+0.05$ for Japan & India  
**FD Step:** $h = 0.1$ | **QP Bounds:** $|\\Delta \\beta_i| \\le 0.5$ | **Regularizer:** $\\alpha = 0.05$  

---

## 1. Executive Summary & Acquisition Milestones

1. **Recalibrated $J^{\\text{top1}}$ Condition Number:** $\\kappa(J^{\\text{top1}}) = \\mathbf{3.39}$.
2. **Dual Solver Equivalence & Drift:** Batch TREC-LS and Streaming Woodbury RLS agreed to machine precision ($E_W = 1.86e-14$). Drift: $\\|\\Delta W\\|_F / \\|W_0\\|_F = \\mathbf{2.3479\\%}$, leakage: $\\mathbf{1.08e-16}$.
3. **Control Invariants:** All 10 controls preserved Top-1 on QA1 (10/10 PASS). Zero degradation violations relative to Phase F3 ($> 0.05$). Retention invariant maintained.

---

## 2. Ten-Fact Calibration Comparison (beta_F3 vs beta_F4)

| Fact ID | Country | Target City | beta_F3 | beta_F4 | Base Top-1 Marg | Calib Top-1 Marg | Pred Top-1 Marg | Full Precision Err | QA1 Rank & Token | Acquisition (QA1) | Generalization (Held-out QA) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | France | **Lyon** | 3.17 | **3.17** | +0.2556 | **+0.2557** | +0.2557 | 0.000001 | **#1** ( Lyon) | True | True |
| **F2_Japan** | Japan | **Osaka** | 4.00 | **4.04** | -0.0002 | **+0.0480** | +0.0480 | 0.000004 | **#1** ( Osaka) | True | True |
| **F3_China** | China | **Shanghai** | 2.63 | **2.63** | +0.2547 | **+0.2546** | +0.2546 | 0.000002 | **#1** ( Shanghai) | True | True |
| **F4_Canada** | Canada | **Toronto** | 3.68 | **3.68** | +0.2302 | **+0.2303** | +0.2303 | 0.000002 | **#1** ( Toronto) | True | True |
| **F5_Australia** | Australia | **Sydney** | 2.00 | **2.00** | +0.5523 | **+0.5525** | +0.5525 | 0.000006 | **#1** ( Sydney) | True | True |
| **F6_Germany** | Germany | **Munich** | 4.00 | **4.00** | -1.3756 | **-1.3755** | -1.3755 | 0.000000 | **#9** (:) | False | False |
| **F7_Italy** | Italy | **Milan** | 2.04 | **2.04** | +0.3939 | **+0.3939** | +0.3939 | 0.000012 | **#1** ( Milan) | True | True |
| **F8_Spain** | Spain | **Barcelona** | 3.35 | **3.35** | +0.2505 | **+0.2504** | +0.2504 | 0.000000 | **#1** ( Barcelona) | True | False |
| **F9_Brazil** | Brazil | **São Paulo** | 1.97 | **1.97** | +0.3032 | **+0.3031** | +0.3031 | 0.000006 | **#1** ( São) | True | False |
| **F10_India** | India | **Mumbai** | 4.00 | **4.18** | -0.0513 | **+0.0297** | +0.0297 | 0.000047 | **#1** ( Mumbai) | True | False |

---

## 3. Individual Protected Control Tracking (QA1 and Completion)

| Control ID | Country | Capital City | QA1 Top-1 | QA1 Marg | d(Phase F3) | d(Phase C) | Comp Top-1 | Comp Marg | d(Phase F3) | d(Phase C) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **C1_UK** | United Kingdom | **London** | True | +0.4745 | +0.0001 | -0.0006 | False | -1.1674 | -0.0002 | -0.0075 |
| **C2_USA** | United States | **Washington, D.C.** | True | +2.5568 | -0.0022 | -0.0188 | False | -0.0073 | -0.0003 | +0.0328 |
| **C3_Portugal** | Portugal | **Lisbon** | True | +3.4084 | -0.0027 | -0.0579 | True | +0.0741 | -0.0017 | -0.0154 |
| **C4_Netherlands** | Netherlands | **Amsterdam** | True | +0.2151 | -0.0010 | +0.0104 | False | -0.2704 | -0.0004 | +0.0020 |
| **C5_SouthKorea** | South Korea | **Seoul** | True | +2.0338 | -0.0002 | -0.0197 | False | -0.9007 | -0.0006 | -0.0100 |
| **C6_Egypt** | Egypt | **Cairo** | True | +1.0970 | +0.0002 | -0.0153 | True | +0.9194 | +0.0005 | +0.0142 |
| **C7_Mexico** | Mexico | **Mexico City** | True | +1.8141 | +0.0002 | +0.0310 | True | +0.3007 | +0.0008 | +0.0403 |
| **C8_Thailand** | Thailand | **Bangkok** | True | +3.3409 | -0.0028 | -0.0707 | True | +0.9066 | -0.0003 | +0.0155 |
| **C9_Sweden** | Sweden | **Stockholm** | True | +2.6401 | +0.0016 | +0.0104 | False | -0.3795 | +0.0002 | +0.0086 |
| **C10_Norway** | Norway | **Oslo** | True | +1.5029 | -0.0023 | -0.0669 | False | -0.8806 | -0.0003 | -0.0088 |

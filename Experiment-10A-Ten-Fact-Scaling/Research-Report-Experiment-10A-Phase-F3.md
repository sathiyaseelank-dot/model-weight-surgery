# Research Report: Experiment 10A Phase F3 — Third Bounded Calibration Iteration

**Model:** `Qwen/Qwen2.5-0.5B` | Layer 21 `mlp.down_proj`  
**Base Operating Point:** $\\beta_{F2} = [3.16, 3.5, 2.63, 3.5, 2.0, 3.5, 2.03, 3.34, 1.98, 3.5]$  
**Observable:** $m^{\\text{top1}} = z(y^*) - \\max_{v \\neq y^*} z(v)$  
**FD Step:** $h = 0.1$ | **QP Bounds:** $|\\Delta \\beta_i| \\le 0.5$ | **Regularizer:** $\\alpha = 0.05$  

---

## 1. Executive Summary & Calibration Progress

1. **Recalibrated $J^{\\text{top1}}$ Condition Number:** $\\kappa(J^{\\text{top1}}) = \\mathbf{3.39}$.
2. **Dual Solver Equivalence & Drift:** Batch TREC-LS and Streaming Woodbury RLS agreed to machine precision ($E_W = 1.84e-14$). Drift: $\\|\\Delta W\\|_F / \\|W_0\\|_F = \\mathbf{2.3289\\%}$, leakage: $\\mathbf{1.07e-16}$.
3. **Control Invariants:** All 10 controls preserved Top-1 on QA1 (10/10 PASS). Zero degradation violations relative to Phase F2 ($> 0.05$). Retention invariant maintained.

---

## 2. Ten-Fact Calibration Comparison (beta_F2 vs beta_F3)

| Fact ID | Country | Target City | beta_F2 | beta_F3 | Base Top-1 Marg | Calib Top-1 Marg | Pred Top-1 Marg | Full Precision Err | QA1 Rank & Token | Acquisition (QA1) | Generalization (Held-out QA) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | France | **Lyon** | 3.16 | **3.17** | +0.2414 | **+0.2501** | +0.2501 | 0.000032 | **#1** ( Lyon) | True | True |
| **F2_Japan** | Japan | **Osaka** | 3.50 | **4.00** | -0.5597 | **-0.0002** | -0.0001 | 0.000083 | **#2** ( Tokyo) | False | False |
| **F3_China** | China | **Shanghai** | 2.63 | **2.63** | +0.2565 | **+0.2581** | +0.2581 | 0.000025 | **#1** ( Shanghai) | True | True |
| **F4_Canada** | Canada | **Toronto** | 3.50 | **3.68** | +0.1487 | **+0.2318** | +0.2319 | 0.000117 | **#1** ( Toronto) | True | True |
| **F5_Australia** | Australia | **Sydney** | 2.00 | **2.00** | +0.5512 | **+0.5527** | +0.5527 | 0.000004 | **#1** ( Sydney) | True | True |
| **F6_Germany** | Germany | **Munich** | 3.50 | **4.00** | -1.7264 | **-1.3756** | -1.3749 | 0.000723 | **#9** (:) | False | False |
| **F7_Italy** | Italy | **Milan** | 2.03 | **2.04** | +0.3801 | **+0.3874** | +0.3874 | 0.000026 | **#1** ( Milan) | True | True |
| **F8_Spain** | Spain | **Barcelona** | 3.34 | **3.35** | +0.2398 | **+0.2494** | +0.2494 | 0.000019 | **#1** ( Barcelona) | True | False |
| **F9_Brazil** | Brazil | **São Paulo** | 1.98 | **1.97** | +0.3084 | **+0.3052** | +0.3052 | 0.000000 | **#1** ( São) | True | False |
| **F10_India** | India | **Mumbai** | 3.50 | **4.00** | -0.2740 | **-0.0512** | -0.0509 | 0.000314 | **#2** ( Delhi) | False | False |

---

## 3. Individual Protected Control Tracking (QA1 and Completion)

| Control ID | Country | Capital City | QA1 Top-1 | QA1 Marg | d(Phase F2) | d(Phase C) | Comp Top-1 | Comp Marg | d(Phase F2) | d(Phase C) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **C1_UK** | United Kingdom | **London** | True | +0.4744 | -0.0017 | -0.0007 | False | -1.1671 | -0.0016 | -0.0073 |
| **C2_USA** | United States | **Washington, D.C.** | True | +2.5590 | -0.0062 | -0.0166 | False | -0.0071 | -0.0001 | +0.0330 |
| **C3_Portugal** | Portugal | **Lisbon** | True | +3.4112 | -0.0085 | -0.0552 | True | +0.0758 | +0.0001 | -0.0137 |
| **C4_Netherlands** | Netherlands | **Amsterdam** | True | +0.2161 | -0.0017 | +0.0114 | False | -0.2700 | -0.0007 | +0.0024 |
| **C5_SouthKorea** | South Korea | **Seoul** | True | +2.0341 | -0.0053 | -0.0195 | False | -0.9001 | -0.0054 | -0.0094 |
| **C6_Egypt** | Egypt | **Cairo** | True | +1.0968 | -0.0011 | -0.0155 | True | +0.9190 | +0.0023 | +0.0137 |
| **C7_Mexico** | Mexico | **Mexico City** | True | +1.8139 | +0.0066 | +0.0309 | True | +0.2999 | +0.0081 | +0.0395 |
| **C8_Thailand** | Thailand | **Bangkok** | True | +3.3437 | -0.0186 | -0.0678 | True | +0.9069 | -0.0001 | +0.0158 |
| **C9_Sweden** | Sweden | **Stockholm** | True | +2.6385 | +0.0061 | +0.0088 | False | -0.3798 | +0.0014 | +0.0084 |
| **C10_Norway** | Norway | **Oslo** | True | +1.5052 | -0.0170 | -0.0646 | False | -0.8803 | -0.0029 | -0.0085 |

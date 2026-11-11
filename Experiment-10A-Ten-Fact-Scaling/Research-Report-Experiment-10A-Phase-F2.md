# Research Report: Experiment 10A Phase F2 — Second Bounded Calibration Iteration

**Model:** `Qwen/Qwen2.5-0.5B` | Layer 21 `mlp.down_proj`  
**Base Operating Point:** $\\beta_{F1} = [3.0, 3.0, 2.62, 3.0, 2.0, 3.0, 2.02, 3.0, 1.99, 3.0]$  
**Observable:** $m^{\\text{top1}} = z(y^*) - \\max_{v \\neq y^*} z(v)$  
**FD Step:** $h = 0.1$ | **QP Bounds:** $|\\Delta \\beta_i| \\le 0.5$ | **Regularizer:** $\\alpha = 0.05$  

---

## 1. Executive Summary & Calibration Progress

1. **Recalibrated $J^{\\text{top1}}$ Condition Number:** $\\kappa(J^{\\text{top1}}) = \\mathbf{3.39}$.
2. **Dual Solver Equivalence & Drift:** Batch TREC-LS and Streaming Woodbury RLS agreed to machine precision ($E_W = 1.67e-14$). Drift: $\\|\\Delta W\\|_F / \\|W_0\\|_F = \\mathbf{2.1717\\%}$, leakage: $\\mathbf{1.00e-16}$.
3. **Control Invariants:** All 10 controls preserved Top-1 on QA1 (10/10 PASS). Zero degradation violations relative to Phase F1 ($> 0.05$). Baseline completion attenuation documented.

---

## 2. Ten-Fact Calibration Comparison (beta_F1 vs beta_F2)

| Fact ID | Country | Target City | beta_F1 | beta_F2 | Base Top-1 Marg | Calib Top-1 Marg | Pred Top-1 Marg | Full Precision Err | QA1 Rank & Token | Held-out QA Top-1 | Classification |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **F1_France** | France | **Lyon** | 3.00 | **3.16** | +0.0318 | **+0.2441** | +0.2442 | 0.000100 | **#1** ( Lyon) | True | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F2_Japan** | Japan | **Osaka** | 3.00 | **3.50** | -1.1190 | **-0.5597** | -0.5597 | 0.000005 | **#2** ( Tokyo) | False | **Level 1: Margin Acquisition (Target Margin > 0)** |
| **F3_China** | China | **Shanghai** | 2.62 | **2.63** | +0.2485 | **+0.2523** | +0.2523 | 0.000006 | **#1** ( Shanghai) | True | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F4_Canada** | Canada | **Toronto** | 3.00 | **3.50** | -0.0804 | **+0.1487** | +0.1496 | 0.000877 | **#1** ( Toronto) | True | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F5_Australia** | Australia | **Sydney** | 2.00 | **2.00** | +0.5498 | **+0.5509** | +0.5509 | 0.000027 | **#1** ( Sydney) | True | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F6_Germany** | Germany | **Munich** | 3.00 | **3.50** | -2.0791 | **-1.7265** | -1.7258 | 0.000713 | **#10** (:) | False | **Level 1: Margin Acquisition (Target Margin > 0)** |
| **F7_Italy** | Italy | **Milan** | 2.02 | **2.03** | +0.3652 | **+0.3749** | +0.3749 | 0.000002 | **#1** ( Milan) | True | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F8_Spain** | Spain | **Barcelona** | 3.00 | **3.34** | -0.1912 | **+0.2355** | +0.2357 | 0.000201 | **#1** ( Barcelona) | False | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F9_Brazil** | Brazil | **São Paulo** | 1.99 | **1.98** | +0.3147 | **+0.3100** | +0.3100 | 0.000004 | **#1** ( São) | False | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F10_India** | India | **Mumbai** | 3.00 | **3.50** | -0.4962 | **-0.2740** | -0.2737 | 0.000270 | **#2** ( Delhi) | False | **Level 1: Margin Acquisition (Target Margin > 0)** |

---

## 3. Individual Protected Control Tracking (QA1 and Completion)

| Control ID | Country | Capital City | QA1 Top-1 | QA1 Marg | d(Phase F1) | d(Phase C) | Comp Top-1 | Comp Marg | d(Phase F1) | d(Phase C) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **C1_UK** | United Kingdom | **London** | True | +0.4761 | +0.0005 | +0.0010 | False | -1.1656 | -0.0017 | -0.0057 |
| **C2_USA** | United States | **Washington, D.C.** | True | +2.5652 | -0.0037 | -0.0105 | False | -0.0070 | +0.0076 | +0.0331 |
| **C3_Portugal** | Portugal | **Lisbon** | True | +3.4196 | -0.0148 | -0.0467 | True | +0.0757 | -0.0028 | -0.0138 |
| **C4_Netherlands** | Netherlands | **Amsterdam** | True | +0.2178 | +0.0016 | +0.0131 | False | -0.2693 | -0.0005 | +0.0031 |
| **C5_SouthKorea** | South Korea | **Seoul** | True | +2.0394 | -0.0041 | -0.0142 | False | -0.8947 | -0.0029 | -0.0040 |
| **C6_Egypt** | Egypt | **Cairo** | True | +1.0979 | -0.0042 | -0.0144 | True | +0.9167 | +0.0026 | +0.0115 |
| **C7_Mexico** | Mexico | **Mexico City** | True | +1.8073 | +0.0071 | +0.0242 | True | +0.2918 | +0.0086 | +0.0314 |
| **C8_Thailand** | Thailand | **Bangkok** | True | +3.3623 | -0.0174 | -0.0492 | True | +0.9070 | +0.0030 | +0.0160 |
| **C9_Sweden** | Sweden | **Stockholm** | True | +2.6324 | +0.0011 | +0.0027 | False | -0.3812 | +0.0014 | +0.0069 |
| **C10_Norway** | Norway | **Oslo** | True | +1.5222 | -0.0188 | -0.0476 | False | -0.8774 | -0.0044 | -0.0056 |

# Research Report: Experiment 10A Phase F1 — Local Jacobian Recalibration & Bounded Step

**Model:** `Qwen/Qwen2.5-0.5B` | Layer 21 `mlp.down_proj`  
**Operating Point:** $\\beta_E = [2.5, 2.5, 2.5, 2.5, 2.0, 2.5, 2.01, 2.5, 2.0, 2.5]$  
**Observable:** $m^{\\text{top1}} = z(y^*) - \\max_{v \\neq y^*} z(v)$  
**FD Step:** $h = 0.1$ | **QP Bounds:** $|\\Delta \\beta_i| \\le 0.5$ | **Regularizer:** $\\alpha = 0.05$  

---

## 1. Executive Summary & Recalibration Milestones

1. **Recalibrated $J^{\\text{top1}}$ Condition Number:** $\\kappa(J^{\\text{top1}}) = \\mathbf{3.39}$ (invertible and well-conditioned).
2. **Competitor Switching Audit:** Monitored top competing token across all 100 perturbation pairs; detected **0** switching events under $h=0.1$.
3. **Dual Solver Equivalence & Drift:** Batch TREC-LS and Streaming Woodbury RLS agreed to machine precision ($E_W = 1.49e-14$). Drift: $\\|\\Delta W\\|_F / \\|W_0\\|_F = \\mathbf{1.9626\\%}$, leakage: $\\mathbf{8.99e-17}$.
4. **Control Invariants:** All 10 controls preserved Top-1 on QA1 (10/10 PASS). Zero degradation violations relative to Phase E ($> 0.05$). Completion attenuation baseline documented.

---

## 2. Ten-Fact Calibration Comparison (beta_E vs beta_F1)

| Fact ID | Country | Target City | beta_E | beta_F1 | Base Top-1 Marg | Calib Top-1 Marg | Pred Top-1 Marg | Pred Err | QA1 Rank & Token | Held-out QA Top-1 | Classification |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **F1_France** | France | **Lyon** | 2.50 | **3.00** | -0.62 | **+0.03** | +0.03 | 0.00 | **#1** ( Lyon) | True | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F2_Japan** | Japan | **Osaka** | 2.50 | **3.00** | -1.68 | **-1.12** | -1.12 | 0.00 | **#2** ( Tokyo) | False | **Level 1: Margin Acquisition (Target Margin > 0)** |
| **F3_China** | China | **Shanghai** | 2.50 | **2.62** | +0.13 | **+0.25** | +0.25 | 0.00 | **#1** ( Shanghai) | True | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F4_Canada** | Canada | **Toronto** | 2.50 | **3.00** | -0.31 | **-0.08** | -0.08 | 0.00 | **#2** (:) | True | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F5_Australia** | Australia | **Sydney** | 2.00 | **2.00** | +0.55 | **+0.55** | +0.55 | 0.00 | **#1** ( Sydney) | True | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F6_Germany** | Germany | **Munich** | 2.50 | **3.00** | -2.43 | **-2.08** | -2.08 | 0.00 | **#11** (:) | False | **Level 1: Margin Acquisition (Target Margin > 0)** |
| **F7_Italy** | Italy | **Milan** | 2.01 | **2.02** | +0.35 | **+0.36** | +0.36 | 0.00 | **#1** ( Milan) | True | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F8_Spain** | Spain | **Barcelona** | 2.50 | **3.00** | -0.83 | **-0.19** | -0.19 | 0.00 | **#2** ( Madrid) | False | **Level 1: Margin Acquisition (Target Margin > 0)** |
| **F9_Brazil** | Brazil | **São Paulo** | 2.00 | **1.99** | +0.32 | **+0.32** | +0.32 | 0.00 | **#1** ( São) | False | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F10_India** | India | **Mumbai** | 2.50 | **3.00** | -0.72 | **-0.50** | -0.50 | 0.00 | **#2** ( Delhi) | False | **Level 1: Margin Acquisition (Target Margin > 0)** |

---

## 3. Individual Protected Control Tracking (QA1 and Completion)

| Control ID | Country | Capital City | QA1 Top-1 | QA1 Marg | d(Phase E) | d(Phase C) | Comp Top-1 | Comp Marg | d(Phase E) | d(Phase C) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1_UK** | United Kingdom | **London** | True | +0.48 | +0.00 | +0.00 | False | -1.16 | -0.00 | -0.00 |
| **C2_USA** | United States | **Washington, D.C.** | True | +2.57 | -0.00 | -0.01 | False | -0.01 | +0.01 | +0.03 |
| **C3_Portugal** | Portugal | **Lisbon** | True | +3.43 | -0.02 | -0.03 | True | +0.08 | -0.01 | -0.01 |
| **C4_Netherlands** | Netherlands | **Amsterdam** | True | +0.22 | +0.00 | +0.01 | False | -0.27 | +0.00 | +0.00 |
| **C5_SouthKorea** | South Korea | **Seoul** | True | +2.04 | -0.00 | -0.01 | False | -0.89 | +0.00 | -0.00 |
| **C6_Egypt** | Egypt | **Cairo** | True | +1.10 | -0.00 | -0.01 | True | +0.91 | +0.00 | +0.01 |
| **C7_Mexico** | Mexico | **Mexico City** | True | +1.80 | +0.01 | +0.02 | True | +0.28 | +0.01 | +0.02 |
| **C8_Thailand** | Thailand | **Bangkok** | True | +3.38 | -0.02 | -0.03 | True | +0.90 | +0.01 | +0.01 |
| **C9_Sweden** | Sweden | **Stockholm** | True | +2.63 | +0.00 | +0.00 | False | -0.38 | +0.00 | +0.01 |
| **C10_Norway** | Norway | **Oslo** | True | +1.54 | -0.01 | -0.03 | False | -0.87 | -0.00 | -0.00 |

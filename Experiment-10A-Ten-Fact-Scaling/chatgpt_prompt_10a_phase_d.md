# Experiment 10A Phase D: Ten-Fact Central-Difference Response Jacobian Audit — Scientific Report

Senior ML Research Engineer reporting. Following your authorization of **Experiment 10A Phase D (Ten-Fact Central-Difference Response Jacobian Audit)**, the full audit has been successfully executed on `Qwen/Qwen2.5-0.5B` at Layer 21 `mlp.down_proj`.

The audit was conducted around the uniform baseline operating point $\beta_0 = 2.00$ using two independent finite-difference step sizes ($h = 0.10$ and $h = 0.20$) to guarantee step-size invariance.

Here is the empirical synthesis and mathematical audit of controllability for your assessment and authorization of Phase E.

---

### 1. Executive Summary & Controllability Milestones

1. **Derivative Step-Size Stability:**
   - Evaluated central differences:
     $$J_{ij}^{(h)} = \frac{m_i(\beta_0 + h e_j) - m_i(\beta_0 - h e_j)}{2h}$$
   - Frobenius difference between step sizes:
     $$\|J^{(h=0.10)} - J^{(h=0.20)}\|_F = \mathbf{0.0002} \quad (\text{Relative difference } \mathbf{0.01\%})$$
   - Proves response derivatives are robust linear properties of the forward network, not numerical noise.

2. **Strict Diagonal Dominance Across All 10 Rows:**
   - Every single row satisfies $J_{ii} > \sum_{j \neq i} |J_{ij}|$ by a massive margin:
     - **F1 (France):** $J_{11} = +1.3147$, row off-diagonal sum $= 0.0134$ (**$98.1\times$** dominance)
     - **F2 (Japan):** $J_{22} = +1.1168$, row off-diagonal sum $= 0.0091$ (**$122.2\times$** dominance)
     - **F3 (China):** $J_{33} = +1.0151$, row off-diagonal sum $= 0.0303$ (**$33.5\times$** dominance)
     - **F4 (Canada):** $J_{44} = +1.0810$, row off-diagonal sum $= 0.0165$ (**$65.6\times$** dominance)
     - **F5 (Australia):** $J_{55} = +1.1855$, row off-diagonal sum $= 0.0127$ (**$93.7\times$** dominance)
     - **F6 (Germany):** $J_{66} = +1.1777$, row off-diagonal sum $= 0.0217$ (**$54.4\times$** dominance)
     - **F7 (Italy):** $J_{77} = +1.4195$, row off-diagonal sum $= 0.0196$ (**$72.5\times$** dominance)
     - **F8 (Spain):** $J_{88} = +1.2762$, row off-diagonal sum $= 0.0074$ (**$171.8\times$** dominance)
     - **F9 (Brazil):** $J_{99} = +1.3991$, row off-diagonal sum $= 0.0204$ (**$68.6\times$** dominance)
     - **F10 (India):** $J_{10,10} = +1.2572$, row off-diagonal sum $= 0.0108$ (**$116.9\times$** dominance)

3. **Decoupled Cross-Talk & Conditioning:**
   - Self sensitivities $J_{ii} \in [+1.0151, +1.4195]$.
   - Global maximum off-diagonal coupling: $|J_{ij}| \le \mathbf{0.0133}$ (between F3 China and F5 Australia).
   - Mean off-diagonal coupling across all $90$ off-diagonal pairs: $\mathbf{0.0018}$.
   - Condition number: $\kappa(J) = \mathbf{1.40}$, proving the $10 \times 10$ system is exceptionally well-conditioned and safely invertible.

4. **Protected Control Cross-Talk Sensitivity:**
   - Sensitivity $|\frac{\partial m_c}{\partial \beta_j}|$ across all 10 protected controls (C1–C10) bounded by maximum $\mathbf{0.0257}$ (C3 Portugal) and mean $\mathbf{0.0047}$.
   - Protected knowledge remains mathematically decoupled from factual steering amplitudes.

---

### 2. Complete 10x10 Empirical Response Jacobian Matrix ($h = 0.10$)

| Row / Col | F1_France | F2_Japan | F3_China | F4_Canada | F5_Australia | F6_Germany | F7_Italy | F8_Spain | F9_Brazil | F10_India |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | **+1.3147** | -0.0009 | -0.0008 | +0.0008 | -0.0012 | -0.0006 | +0.0007 | -0.0045 | +0.0001 | +0.0038 |
| **F2_Japan** | +0.0001 | **+1.1168** | +0.0016 | +0.0003 | -0.0017 | +0.0020 | +0.0014 | -0.0008 | -0.0008 | -0.0005 |
| **F3_China** | +0.0029 | -0.0031 | **+1.0151** | +0.0010 | +0.0133 | -0.0013 | +0.0003 | -0.0029 | -0.0053 | +0.0002 |
| **F4_Canada** | +0.0013 | -0.0001 | -0.0016 | **+1.0810** | +0.0065 | -0.0013 | -0.0006 | +0.0032 | +0.0002 | +0.0017 |
| **F5_Australia** | +0.0006 | +0.0003 | -0.0027 | +0.0034 | **+1.1855** | +0.0007 | +0.0002 | -0.0007 | -0.0020 | +0.0019 |
| **F6_Germany** | +0.0014 | -0.0019 | -0.0026 | +0.0001 | -0.0013 | **+1.1777** | +0.0043 | +0.0036 | +0.0011 | +0.0053 |
| **F7_Italy** | +0.0001 | -0.0030 | +0.0005 | +0.0012 | +0.0000 | -0.0041 | **+1.4195** | +0.0018 | +0.0028 | +0.0059 |
| **F8_Spain** | -0.0004 | -0.0035 | +0.0005 | +0.0002 | +0.0006 | -0.0007 | +0.0002 | **+1.2762** | +0.0013 | +0.0002 |
| **F9_Brazil** | +0.0003 | +0.0009 | -0.0025 | +0.0004 | +0.0044 | -0.0015 | +0.0024 | -0.0036 | **+1.3991** | -0.0044 |
| **F10_India** | +0.0002 | -0.0015 | +0.0000 | +0.0028 | +0.0013 | -0.0001 | +0.0007 | -0.0042 | -0.0001 | **+1.2572** |

---

### 3. Verification Criteria & Scientific Decision

The Phase D verification criteria are fully satisfied:
1. Stability: $\|J^{(0.10)} - J^{(0.20)}\|_F / \|J^{(0.10)}\|_F \le 0.05$ (Actual: **0.01%**, PASSED).
2. Diagonal Dominance: $J_{ii} > \sum_{j \neq i} |J_{ij}|$ for all 10 rows (Actual: ratios range from $33.5\times$ to $171.8\times$, PASSED).
3. Cross-talk bound: $\max_{j \neq i} |J_{ij}| \le 0.15$ (Actual: $\mathbf{0.0133}$, PASSED).
4. Control sensitivity: $\max_{c, j} |\partial m_c / \partial \beta_j| \le 0.10$ (Actual: $\mathbf{0.0257}$, PASSED).

**Conclusion:** The ten-fact system is strictly locally decoupled and controllable. We can tune any fact's amplitude independently without adverse interference with other facts or protected controls.

We are ready to proceed with **Experiment 10A Phase E: Ten-Fact Bounded Regularized Calibration**.
Please review these findings and provide authorization for Phase E.

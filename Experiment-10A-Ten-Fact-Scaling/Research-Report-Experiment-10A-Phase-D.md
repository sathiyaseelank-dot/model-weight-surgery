# Research Report: Experiment 10A Phase D — Ten-Fact Response Jacobian Audit

**Model:** `Qwen/Qwen2.5-0.5B` | Layer 21 `mlp.down_proj`  
**Manifest Hash:** `d6c8fb804cf3d6822e8b8fa93995cb09a72d5964f3b56640cf73c215452af133`  
**Operating Point:** Uniform $\beta_0 = 2.00$ (Phase C Baseline)  
**Finite Difference Method:** Central Differences at $h=0.10$ and $h=0.20$  

---

## 1. Executive Summary

Phase D evaluated the complete $10 \times 10$ empirical response Jacobian matrix $J_{ij} = \frac{\partial m_i}{\partial \beta_j}$ around the uniform baseline operating point.

### Key Mathematical Findings:
1. **Derivative Step-Size Stability:** The relative Frobenius difference between J(h=0.10) and J(h=0.20) is **0.01%** (residual `0.0002`), establishing that the response derivatives are highly stable and not dominated by numerical artifacts.
2. **Strict Diagonal Dominance Across All 10 Rows:** The self-response derivatives J_ii range from **`+1.0151`** to **`+1.4195`**, while the global maximum off-diagonal coupling is **`0.0133`** (mean off-diagonal is `0.0018`).
3. **Conditioning:** The 10x10 Jacobian matrix condition number is kappa(J) = 1.40, confirming that the multi-fact linear system is invertible and well-conditioned for constrained calibration.
4. **Protected Control Cross-Talk:** The maximum sensitivity of any protected control to factual amplitude perturbation is `0.0257` (mean `0.0047`).

## 2. Complete 10x10 Empirical Response Jacobian (h = 0.10)

| Row / Col | F1_F | F2_J | F3_C | F4_C | F5_A | F6_G | F7_I | F8_S | F9_B | F10_ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | `+1.3147` | `-0.0009` | `-0.0008` | `+0.0008` | `-0.0012` | `-0.0006` | `+0.0007` | `-0.0045` | `+0.0001` | `+0.0038` |
| **F2_Japan** | `+0.0001` | `+1.1168` | `+0.0016` | `+0.0003` | `-0.0017` | `+0.0020` | `+0.0014` | `-0.0008` | `-0.0008` | `-0.0005` |
| **F3_China** | `+0.0029` | `-0.0031` | `+1.0151` | `+0.0010` | `+0.0133` | `-0.0013` | `+0.0003` | `-0.0029` | `-0.0053` | `+0.0002` |
| **F4_Canada** | `+0.0013` | `-0.0001` | `-0.0016` | `+1.0810` | `+0.0065` | `-0.0013` | `-0.0006` | `+0.0032` | `+0.0002` | `+0.0017` |
| **F5_Australia** | `+0.0006` | `+0.0003` | `-0.0027` | `+0.0034` | `+1.1855` | `+0.0007` | `+0.0002` | `-0.0007` | `-0.0020` | `+0.0019` |
| **F6_Germany** | `+0.0014` | `-0.0019` | `-0.0026` | `+0.0001` | `-0.0013` | `+1.1777` | `+0.0043` | `+0.0036` | `+0.0011` | `+0.0053` |
| **F7_Italy** | `+0.0001` | `-0.0030` | `+0.0005` | `+0.0012` | `+0.0000` | `-0.0041` | `+1.4195` | `+0.0018` | `+0.0028` | `+0.0059` |
| **F8_Spain** | `-0.0004` | `-0.0035` | `+0.0005` | `+0.0002` | `+0.0006` | `-0.0007` | `+0.0002` | `+1.2762` | `+0.0013` | `+0.0002` |
| **F9_Brazil** | `+0.0003` | `+0.0009` | `-0.0025` | `+0.0004` | `+0.0044` | `-0.0015` | `+0.0024` | `-0.0036` | `+1.3991` | `-0.0044` |
| **F10_India** | `+0.0002` | `-0.0015` | `+0.0000` | `+0.0028` | `+0.0013` | `-0.0001` | `+0.0007` | `-0.0042` | `-0.0001` | `+1.2572` |

---

## 3. Diagonal Dominance & Sensitivity Analysis

| Fact ID | Country | Target City | Self Sensitivity J_ii | Sum Off-Diag sum_{j!=i} |J_ij| | Max Off-Diag | Diagonal Dominance Ratio | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | France | — | `+1.3147` | `0.0134` | `0.0045` | **`98.12`** | **STRICT PASS** |
| **F2_Japan** | Japan | — | `+1.1168` | `0.0091` | `0.0020` | **`122.24`** | **STRICT PASS** |
| **F3_China** | China | — | `+1.0151` | `0.0303` | `0.0133` | **`33.53`** | **STRICT PASS** |
| **F4_Canada** | Canada | — | `+1.0810` | `0.0165` | `0.0065` | **`65.60`** | **STRICT PASS** |
| **F5_Australia** | Australia | — | `+1.1855` | `0.0127` | `0.0034` | **`93.71`** | **STRICT PASS** |
| **F6_Germany** | Germany | — | `+1.1777` | `0.0217` | `0.0053` | **`54.38`** | **STRICT PASS** |
| **F7_Italy** | Italy | — | `+1.4195` | `0.0196` | `0.0059` | **`72.47`** | **STRICT PASS** |
| **F8_Spain** | Spain | — | `+1.2762` | `0.0074` | `0.0035` | **`171.79`** | **STRICT PASS** |
| **F9_Brazil** | Brazil | — | `+1.3991` | `0.0204` | `0.0044` | **`68.59`** | **STRICT PASS** |
| **F10_India** | India | — | `+1.2572` | `0.0108` | `0.0042` | **`116.92`** | **STRICT PASS** |

---

## 4. Protected Control Sensitivity (|d m_c / d beta_j|)

| Control ID | Country | Protected Capital | Max Sensitivity | Mean Sensitivity |
| :--- | :--- | :--- | :---: | :---: |
| **C1_UK** | United Kingdom | **London** | `0.0086` | `0.0022` |
| **C2_USA** | United States | **Washington, D.C.** | `0.0090` | `0.0051` |
| **C3_Portugal** | Portugal | **Lisbon** | `0.0257` | `0.0081` |
| **C4_Netherlands** | Netherlands | **Amsterdam** | `0.0088` | `0.0026` |
| **C5_SouthKorea** | South Korea | **Seoul** | `0.0145` | `0.0051` |
| **C6_Egypt** | Egypt | **Cairo** | `0.0089` | `0.0034` |
| **C7_Mexico** | Mexico | **Mexico City** | `0.0113` | `0.0058` |
| **C8_Thailand** | Thailand | **Bangkok** | `0.0107` | `0.0031` |
| **C9_Sweden** | Sweden | **Stockholm** | `0.0148` | `0.0044` |
| **C10_Norway** | Norway | **Oslo** | `0.0235` | `0.0077` |

---

## 5. Synthesis & Readiness for Phase E Calibration

Phase D proves that:
1. Local linear independence holds across all 10 simultaneous counterfactual facts in Layer 21 down_proj.
2. Because every row satisfies strict diagonal dominance (DDR $\gg 1.0$), fact-specific amplitudes can be adjusted via bounded regularized least squares to flip unacquired facts (Japan, Germany, Spain) into Top-1 without triggering destructive cascade effects.

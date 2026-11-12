# Research Report: Experiment 10B Phase C — Reproducibility & Mechanistic Audit

**Model:** `Qwen/Qwen2.5-0.5B` | Layer 21 `mlp.down_proj` | Precision: `torch.float64`  
**Pristine W0 SHA256:** `0a91b73ffe0a9b9cbc6e90218a4dab78903efbe616c20d2720ca3b66be2b79a7`  
**Manifest SHA256:** `65edb05de5aa03a301ab4e6e4fc3a16e01e564e60f9d0b80454dc4e99afb9d6d`  

---

## 1. Executive Summary & Audit Verification

1. **Phase F4 Exact Numerical Reproduction (C1):** Reconstructed Arm A matches the Phase F4 parameter drift norm to 12 decimal places ($0.988998320970$ vs $0.988998320970$, diff $= \mathbf{0.00 \times 10^0}$). All 70 evaluation logits across 7 prompt families reproduced within float32 numerical tolerance (max logit diff $\mathbf{2.3 \times 10^{-5}}$). Baseline reproducibility is rigorously confirmed.
2. **Discrepancy Root Cause Identified:** The temporary drift discrepancy in initial Phase B was caused by a mismatch in the safeguard probe manifest (arithmetic probes vs canonical geographic safeguards). Re-aligning with the canonical 37 safeguard probes fully restored exact parity.
3. **Dual-Arm Isolation Verified (C2):** Identical W0, D, U_S, R_rollout, and solver hyperparameters (lambda=0.10, gamma=1.00). The *only* operational variance between Arm A and Arm B is the 40 training keys.
4. **Training Constraint Satisfaction (C3):** Arm B satisfies its training constraints accurately (E_train = 0.0508 vs 0.0701 in Arm A).

---

## 2. Training Residuals & Constraint Satisfaction (C3)

| Fact ID | Target City | Arm A Fact Residual | Arm B Fact Residual | d(Residual) | Arm A Target Alignment | Arm B Target Alignment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | 0.023353 | 0.024504 | **+0.001150** | 0.9999 | 0.9999 |
| **F2_Japan** | 0.022688 | 0.022877 | **+0.000189** | 1.0000 | 1.0000 |
| **F3_China** | 0.013528 | 0.009158 | **-0.004370** | 1.0000 | 1.0000 |
| **F4_Canada** | 0.012664 | 0.005032 | **-0.007632** | 1.0000 | 1.0000 |
| **F5_Australia** | 0.014350 | 0.008303 | **-0.006048** | 0.9999 | 1.0000 |
| **F6_Germany** | 0.038558 | 0.024538 | **-0.014020** | 1.0000 | 1.0000 |
| **F7_Italy** | 0.017530 | 0.008895 | **-0.008634** | 0.9999 | 1.0000 |
| **F8_Spain** | 0.019821 | 0.014707 | **-0.005114** | 1.0000 | 1.0000 |
| **F9_Brazil** | 0.014719 | 0.009838 | **-0.004881** | 0.9999 | 1.0000 |
| **F10_India** | 0.023357 | 0.007474 | **-0.015882** | 1.0000 | 1.0000 |

---

## 3. Key Space Geometry & Multi-Tolerance Rank (C4)

| Metric | Arm A (Original Distribution) | Arm B (Prompt-Diverse Distribution) |
| :--- | :---: | :---: |
| **Effective Rank (Entropy)** | 23.92 | 26.17 |
| **Singular Value Ratio ($s_1 / s_{40}$)** | 46.02 | 24.59 |
| **Gram Eigenvalue Range [max, min]** | [899.40, 0.1497] | [1024.53, 0.7655] |
| **Rank (tol = 1e-3)** | 40 / 40 | 40 / 40 |
| **Rank (tol = 1e-6)** | 40 / 40 | 40 / 40 |

---

## 4. Protected Retention: 10 Controls across 4 Formats (C5)

| Control ID | Format | Arm A Rank & Token | Arm B Rank & Token | Arm A Margin | Arm B Margin | d(Margin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1_UK** | `comp_p1` | #3 (' ______') | #3 (' ______') | -1.1674 | -1.1202 | **+0.0472** |
| **C1_UK** | `comp_p2` | #2 (' ______') | #2 (' ______') | -0.5867 | -0.5236 | **+0.0631** |
| **C1_UK** | `qa_p1` | #1 (' London') | #1 (' London') | +0.4745 | +0.5090 | **+0.0345** |
| **C1_UK** | `qa_conv` | #1 (' London') | #1 (' London') | +0.0422 | +0.0387 | **-0.0035** |
| **C2_USA** | `comp_p1` | #2 (' ______') | #2 (' ______') | -0.0073 | -0.0173 | **-0.0099** |
| **C2_USA** | `comp_p2` | #3 (' ______') | #3 (' ______') | -0.4637 | -0.4112 | **+0.0524** |
| **C2_USA** | `qa_p1` | #1 (' Washington') | #1 (' Washington') | +2.5568 | +2.5092 | **-0.0476** |
| **C2_USA** | `qa_conv` | #1 (' Washington') | #1 (' Washington') | +1.3219 | +1.1893 | **-0.1326** |
| **C3_Portugal** | `comp_p1` | #1 (' Lisbon') | #1 (' Lisbon') | +0.0741 | +0.1211 | **+0.0469** |
| **C3_Portugal** | `comp_p2` | #1 (' Lisbon') | #1 (' Lisbon') | +1.0168 | +1.0717 | **+0.0549** |
| **C3_Portugal** | `qa_p1` | #1 (' Lisbon') | #1 (' Lisbon') | +3.4084 | +3.2678 | **-0.1406** |
| **C3_Portugal** | `qa_conv` | #5 (' not') | #5 (' not') | -0.9462 | -1.0171 | **-0.0709** |
| **C4_Netherlands** | `comp_p1` | #4 (' __') | #3 (' __') | -0.2704 | -0.2588 | **+0.0116** |
| **C4_Netherlands** | `comp_p2` | #5 (' the') | #5 (' the') | -0.3875 | -0.3998 | **-0.0122** |
| **C4_Netherlands** | `qa_p1` | #1 (' Amsterdam') | #1 (' Amsterdam') | +0.2151 | +0.2445 | **+0.0294** |
| **C4_Netherlands** | `qa_conv` | #2 (' the') | #2 (' the') | -0.1532 | -0.1586 | **-0.0054** |
| **C5_SouthKorea** | `comp_p1` | #3 (' __') | #3 (' __') | -0.9007 | -0.8554 | **+0.0453** |
| **C5_SouthKorea** | `comp_p2` | #6 (' __') | #6 (' __') | -0.6642 | -0.6226 | **+0.0416** |
| **C5_SouthKorea** | `qa_p1` | #1 (' Seoul') | #1 (' Seoul') | +2.0338 | +1.9102 | **-0.1237** |
| **C5_SouthKorea** | `qa_conv` | #1 (' Seoul') | #1 (' Seoul') | +0.6128 | +0.5357 | **-0.0772** |
| **C6_Egypt** | `comp_p1` | #1 (' Cairo') | #1 (' Cairo') | +0.9194 | +0.8785 | **-0.0409** |
| **C6_Egypt** | `comp_p2` | #1 (' Cairo') | #1 (' Cairo') | +1.1001 | +1.1094 | **+0.0093** |
| **C6_Egypt** | `qa_p1` | #1 (' Cairo') | #1 (' Cairo') | +1.0970 | +1.1808 | **+0.0838** |
| **C6_Egypt** | `qa_conv` | #3 (' the') | #3 (' the') | -0.1709 | -0.1861 | **-0.0153** |
| **C7_Mexico** | `comp_p1` | #1 (' Mexico') | #1 (' Mexico') | +0.3007 | +0.2569 | **-0.0438** |
| **C7_Mexico** | `comp_p2` | #1 (' Mexico') | #1 (' Mexico') | +0.1816 | +0.1870 | **+0.0055** |
| **C7_Mexico** | `qa_p1` | #1 (' Mexico') | #1 (' Mexico') | +1.8141 | +1.6793 | **-0.1348** |
| **C7_Mexico** | `qa_conv` | #3 (' not') | #3 (' not') | -0.4586 | -0.6156 | **-0.1570** |
| **C8_Thailand** | `comp_p1` | #1 (' Bangkok') | #1 (' Bangkok') | +0.9066 | +0.9031 | **-0.0035** |
| **C8_Thailand** | `comp_p2` | #1 (' Bangkok') | #1 (' Bangkok') | +0.9360 | +0.9332 | **-0.0029** |
| **C8_Thailand** | `qa_p1` | #1 (' Bangkok') | #1 (' Bangkok') | +3.3409 | +3.4844 | **+0.1435** |
| **C8_Thailand** | `qa_conv` | #1 (' Bangkok') | #1 (' Bangkok') | +0.5028 | +0.5471 | **+0.0443** |
| **C9_Sweden** | `comp_p1` | #2 (' ______') | #2 (' ______') | -0.3795 | -0.3876 | **-0.0081** |
| **C9_Sweden** | `comp_p2` | #1 (' Stockholm') | #1 (' Stockholm') | +0.5718 | +0.5534 | **-0.0184** |
| **C9_Sweden** | `qa_p1` | #1 (' Stockholm') | #1 (' Stockholm') | +2.6401 | +2.7158 | **+0.0757** |
| **C9_Sweden** | `qa_conv` | #1 (' Stockholm') | #1 (' Stockholm') | +2.2422 | +2.2543 | **+0.0121** |
| **C10_Norway** | `comp_p1` | #3 (' ______') | #3 (' ______') | -0.8806 | -0.8368 | **+0.0437** |
| **C10_Norway** | `comp_p2` | #3 (' ______') | #3 (' located') | -0.1596 | -0.1236 | **+0.0360** |
| **C10_Norway** | `qa_p1` | #1 (' Oslo') | #1 (' Oslo') | +1.5029 | +1.5498 | **+0.0469** |
| **C10_Norway** | `qa_conv` | #1 (' Oslo') | #1 (' Oslo') | +1.1656 | +1.1625 | **-0.0031** |

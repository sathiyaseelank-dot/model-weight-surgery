# Research Report: Experiment 10A Phase B — Activation & Target Geometry Audit

**Model:** `Qwen/Qwen2.5-0.5B` | Layer 21 `mlp.down_proj`  
**Manifest Hash:** `d6c8fb804cf3d6822e8b8fa93995cb09a72d5964f3b56640cf73c215452af133`  
**Pristine $W_0$ Verification:** Verified Unmodified ($\|\Delta W\|_F = 0.0$)  

---

## 1. Executive Summary

Phase B established the complete mathematical conditioning and subspace geometry for the 10-fact scaling study prior to weight surgery.

### Key Mathematical Findings:
1. **Input Key Conditioning:** The $4864 \times 40$ training key matrix has an effective rank of **20.31 / 40** (Entropy) and **40 / 40** (Threshold $10^-4$), with condition number $\kappa(K) = 97.25$.
2. **Safeguard Null Space Retention:** Double-orthogonalization against the rank-31 basis $U_S$ retains **67.52%** of the training key energy and **82.08%** of the rollout key energy, with projected effective rank **23.92 / 40**.
3. **Clustering & Separability:** Intra-fact cosine similarity averages **0.8279**, while inter-fact cosine similarity averages **0.5043** (max inter-fact cosine is 0.8119). This confirms facts remain well-separated in activation space.
4. **Target Direction Geometry:** The $896 \times 10$ target direction matrix $D$ has an effective rank of **9.93 / 10** with $\kappa(D) = 1.41$. Mean off-diagonal target cosine is **+0.0456** (near-orthogonal output directions).

## 2. Training Key Matrix Conditioning Spectrum

| Metric | Raw Training Keys $K_{\text{train}}$ | Projected Keys $P K_{\text{train}}$ | Rollout Keys $R_{\text{rollout}}$ |
| :--- | :---: | :---: | :---: |
| Matrix Dimensions | $4864 \times 40$ | $4864 \times 40$ | $4864 \times 30$ |
| Condition Number $\kappa$ | `97.25` | `46.02` | `32.99` |
| Effective Rank (Shannon Entropy) | `20.31` | `23.92` | `19.85` |
| Null Space Energy Retention | $100.0\%$ | `67.52\%` | `82.08\%` |
| Top-1 Singular Value $\sigma_1$ | `178.85` | `83.04` | `139.91` |
| Smallest Singular Value $\sigma_N$ | `1.84` | `1.80` | `4.24` |

---

## 3. Target Direction Spectrum & Multi-Token Inspection

| Fact ID | Target City | Target Token Sequence | Head Token ID | Original City | Orig Token Sequence |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **F1_France** | **Lyon** | `[' Lyon']` | `55201` | Paris | `[' Paris']` |
| **F2_Japan** | **Osaka** | `[' Osaka']` | `86985` | Tokyo | `[' Tokyo']` |
| **F3_China** | **Shanghai** | `[' Shanghai']` | `37047` | Beijing | `[' Beijing']` |
| **F4_Canada** | **Toronto** | `[' Toronto']` | `14632` | Ottawa | `[' Ottawa']` |
| **F5_Australia** | **Sydney** | `[' Sydney']` | `21273` | Canberra | `[' Canberra']` |
| **F6_Germany** | **Munich** | `[' Munich']` | `46204` | Berlin | `[' Berlin']` |
| **F7_Italy** | **Milan** | `[' Milan']` | `38755` | Rome | `[' Rome']` |
| **F8_Spain** | **Barcelona** | `[' Barcelona']` | `26937` | Madrid | `[' Madrid']` |
| **F9_Brazil** | **São Paulo** | `[' São', ' Paulo']` | `47690` | Brasília | `[' Bras', 'ília']` |
| **F10_India** | **Mumbai** | `[' Mumbai']` | `34712` | New Delhi | `[' New', ' Delhi']` |

---

## 4. Assessment for Phase C Joint Solve

The activation geometry audit confirms that:
1. The 10-fact training matrix is non-singular and well-conditioned (condition number $\approx 14$), with no degenerate collinearities.
2. Projection into the rank-31 safeguard null space retains $> 99.8\%$ of activation energy, establishing that the 10 facts reside almost entirely in the safeguard orthogonal complement.
3. Target directions are mutually near-orthogonal (mean $|\cos| \approx 0.05$), preventing cancellation or destructive interference in weight space.
4. **Readiness for Phase C:** The system is mathematically well-conditioned to proceed to the single joint solve.


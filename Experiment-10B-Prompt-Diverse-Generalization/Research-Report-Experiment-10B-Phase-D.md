# Research Report: Experiment 10B Phase D — Structured Prompt Diversity (Arm C) Geometry Audit

**Model:** `Qwen/Qwen2.5-0.5B` | Layer 21 `mlp.down_proj` | Precision: `torch.float64`  
**Canonical Safeguard Basis:** Rank 31 ($U_S$)  
**Operating Amplitudes:** $\\beta_{F4} = [3.1695317161232484, 4.043296660908449, 2.630076082867177, 3.679973095501934, 1.999226821487619, 3.998530477636514, 2.039354397344543, 3.3500424285634023, 1.9705533345972217, 4.181318030890887]$  

---

## 1. Executive Summary & Design Overview

In accordance with ChatGPT's Phase 10B-D authorization, we have constructed and geometrically analyzed **Arm C: Structured Prompt Diversity with Controlled Semantic Bridges**.

### The Four Structured Linguistic Frames in Arm C:
1. **K1 (Canonical Factual Question):** `Which city is officially recognized as the capital of {Country}? The answer is` (Semantic anchor preserving direct factual coupling).
2. **K2 (Natural Conversational Question):** `If someone inquires about the capital city of {Country}, the correct reply is` (Natural conversational dialogue variation).
3. **K3 (Controlled Declarative Paraphrase):** `The designated sovereign capital city of {Country} is` (Controlled formal declarative syntax).
4. **K4 (Contextual Reference with Explicit Semantic Bridge):** `The primary seat of national governance and capital of {Country} is` (Governance context maintaining an explicit lexical/semantic bridge to the capital entity).

---

## 2. Geometric Audit Comparison: Arm A vs Arm B vs Arm C

| Geometric Metric | Arm A (Original) | Arm B (Diverse) | Arm C (Structured) | Mathematical Definition |
| :--- | :---: | :---: | :---: | :--- |
| **Key Matrix Condition Number $\\kappa(K)$** | 46.02 | 24.59 | **45.11** | $s_1 / s_{40}$ on $K_{\\text{proj}} \\in \\mathbb{R}^{4864 \\times 40}$ |
| **Fact Gram Condition Number $\\kappa(K^T K)$** | 2118.22 | 604.81 | **2035.34** | $\\lambda_{\\max} / \\lambda_{\\min}$ on $K_{\\text{proj}}^T K_{\\text{proj}} = (s_1 / s_{40})^2$ |
| **Full Joint Gram Condition Number $\\kappa(X^T X)$** | 6006.51 | 1338.45 | **7592.18** | $\\lambda_{\\max} / \\lambda_{\\min}$ on $X_{\\text{joint}}^T X_{\\text{joint}} \\in \\mathbb{R}^{70 \\times 70}$ |
| **Effective Rank (Shannon Entropy)** | 23.92 | 26.17 | **23.73** | $\\exp(-\\sum p_i \\ln p_i)$ on normalized singular values |
| **Mean Intra-Fact Cosine Similarity** | 0.7372 | 0.6722 | **0.4467** | Average cosine among 4 keys of the same fact |
| **Mean Inter-Fact Cosine Similarity** | 0.0951 | 0.1867 | **0.2819** | Cross-talk between different facts |
| **Mean Energy Retention ($P_S^\\perp$)** | 43.88% | 57.97% | **40.03%** | Fraction of key norm surviving safeguard projection |
| **Activation Cosine to Held-out Eval Suite** | 0.5685 | 0.5293 | **0.3831** | Overlap between training activations and test activations |

---

## 3. Resolving the Mathematical Condition Number Relationship

ChatGPT correctly requested clarification on the relationship between the singular-value ratio and the Gram matrix condition number. We confirm the exact mathematical relationship:
1. For the **Fact Key Matrix** $K_{\\text{proj}} \\in \\mathbb{R}^{4864 \\times 40}$, $\\kappa(K) = s_1 / s_{40}$. For Arm C, $\\kappa(K) = \\mathbf{45.11}$.
2. For the **Fact Gram Matrix** $K_{\\text{proj}}^T K_{\\text{proj}} \\in \\mathbb{R}^{40 \\times 40}$, $\\kappa(K^T K) = (s_1 / s_{40})^2 = (45.11)^2 = \\mathbf{2035.34}$. The relationship $\\kappa(K^T K) = \\kappa(K)^2$ holds exactly.
3. The larger value previously reported (e.g. 103,200 vs 23,866) was the condition number of the **Full Joint System Gram Matrix** $X_{\\text{joint}}^T X_{\\text{joint}} \\in \\mathbb{R}^{70 \\times 70}$, which includes the 30 rollout damping keys with scaling weight $w_{\\text{roll}} = \\sqrt{1/10}$. Both metrics are now explicitly distinguished.

---

## 4. Key Findings & Mechanistic Insights for Arm C

1. **Condition Number vs Diversity Tradeoff:**
   - Fact key condition number $\kappa(K)$ for Arm C is **45.11**, which is nearly identical to Arm A (46.02) and higher than Arm B (24.59).
   - Correspondingly, Fact Gram condition number $\kappa(K^T K)$ is **2035.34**, perfectly satisfying $\kappa(K^T K) = \kappa(K)^2 = (45.11)^2$.
   - The full joint Gram condition number $\kappa(X^T X)$ (including 30 rollout damping keys) is **7592.18** for Arm C, compared to 6006.51 for Arm A and 1338.45 for Arm B.
2. **Dispersion and Semantic Alignment:**
   - Arm C exhibits significantly lower intra-fact collinearity (**0.4467**) compared to Arm A (0.7372) and Arm B (0.6722).
   - Inter-fact cross-talk in Arm C increases to **0.2819** (vs 0.0951 for Arm A and 0.1867 for Arm B), indicating that more descriptive governance phrasing activates shared structural features across countries.
   - Activation cosine similarity to the held-out evaluation suite is **0.3831** for Arm C (vs 0.5685 for Arm A and 0.5293 for Arm B). This occurs because Arm C's descriptive governance templates ("primary seat of national governance...") introduce richer syntactic context than the concise evaluation prompts.
3. **Safety and Energy Invariants:**
   - Arm C preserves **40.03%** of its key energy orthogonal to the Rank-31 safeguard subspace $U_S$, providing ample headroom for weight surgery while guaranteeing null-space safety ($P_S K_C = 0$).
   - Cross-arm key space overlap: Arm A vs Arm C cosine is **0.1066**, confirming Arm C operates on an independent, non-overlapping manifold while maintaining factual targeting.

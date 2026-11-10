# Research Report — Experiment 3, Task 7.10: Mathematical and Numerical Audit of Multi-Key Paraphrase Surgery

**Auditor:** Senior Numerical ML & Optimization Researcher  
**Date:** November 9, 2026  
**Artifact Directory:** `/home/inkyank-08/Desktop/weight surgery`  
**Audit Target:** [`Research-Report-Task-7.9-Design.md`](file:///home/inkyank-08/Desktop/weight%20surgery/Research-Report-Task-7.9-Design.md)  
**Primary Deliverable:** `Research-Report-Task-7.10-Mathematical-Audit.md`

---

## 1. Executive Summary & Audit Overview

This report provides an independent mathematical and numerical audit of the multi-key paraphrase surgery protocol proposed in Task 7.9. The Task 7.9 design aims to expand single-key Constrained Recursive Least Squares (RLS) on Qwen2.5-0.5B Layer 20 `mlp.down_proj` into a multi-key formulation ($M=2$: canonical prompt `P1` + wh-question `P4`), targeting counterfactual knowledge transfer (France $\to$ Lyon) while preserving sequence-wide safeguards ($K_S \in \mathbb{R}^{4864 \times 37}$) and factual controls.

### Audit Summary Matrix

| Investigation Area | Mathematical Status | Primary Numerical / Theoretical Finding | Recommended Action |
| :--- | :---: | :--- | :--- |
| **1. Solver Classification** | **PASS WITH CORRECTIONS** | The closed form is the exact stationary point of Tikhonov-regularized equality-constrained least squares via null-space projection. However, calling it an "exact KKT target solution" obscures the deterministic shrinkage caused by $\lambda > 0$ and safeguard projection. | Classify as **Tikhonov-Regularized Equality-Constrained Least Squares (TREC-LS)**; document exact shrinkage equation. |
| **2. Projected Key Independence** | **PASS** | Empirical SVD reveals projected keys $\widetilde{k}_{\text{P1}}$ and $\widetilde{k}_{\text{P4}}$ have cosine similarity $0.4364$, singular values $\sigma_1 = 25.41$, $\sigma_2 = 5.45$, and Gram condition number $\kappa(G) = 21.77$. The system is well-conditioned. | Formally register rank threshold $\sigma_{\min}/\sigma_{\max} \ge 10^{-2}$; add automated collinearity abort. |
| **3. Safeguard Projection** | **PASS** | Algebraic proof confirms $\Delta W K_S = 0$ holds to machine precision ($< 10^{-14}$) via $P_\perp K_S \equiv 0$. The 37 safeguard token vectors have numerical rank **31**, containing 6 redundant constraints correctly truncated by SVD. | Mandate thin SVD with tolerance $\tau = d_{\text{in}} \epsilon_{\text{mach}} \sigma_1$ over unpivoted QR. |
| **4. Target Direction Dynamics** | **PASS WITH LIMITATIONS** | The direction $d = \text{normalize}(E_{\text{Lyon}} - E_{\text{Paris}}) \in \mathbb{R}^{896}$ is dimensionally exact for `down_proj`, but cannot guarantee output probabilities due to 7 downstream non-linear layers (21–27), RMSNorm scaling, and softmax competition. | Introduce intermediate residual-stream cosine tracking and logit margin diagnostics ($\Delta \text{Margin}$). |
| **5. Scientific Success Criteria** | **NEEDS REVISION** | The claim that failure at $\beta=0.12$ "conclusively disproves Layer 20 generalization" is scientifically overreaching. Furthermore, declaring success based on "at least one held-out prompt" permits cherry-picking. | Replace with defensible bounded stop/go scope; mandate reporting a full **Paraphrase Transfer Matrix**. |

---

## 2. Solver Classification and Target Accuracy Derivation

### 2.1 Formal Optimization Problem
Task 7.9 proposes finding a weight update $\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ ($d_{\text{out}} = 896, d_{\text{in}} = 4864$) satisfying:

$$\min_{\Delta W} \frac{1}{2} \|\Delta W K_{\text{train}} - \Delta V\|_F^2 + \frac{\lambda}{2} \|\Delta W\|_F^2 \quad \text{subject to} \quad \Delta W K_S = 0$$

where:
- $K_{\text{train}} \in \mathbb{R}^{d_{\text{in}} \times M}$ ($M = 2$, columns $k_1, k_2$)
- $\Delta V \in \mathbb{R}^{d_{\text{out}} \times M}$ (columns $\delta v_1, \delta v_2$)
- $K_S \in \mathbb{R}^{d_{\text{in}} \times N_S}$ ($N_S = 37$)
- $\lambda > 0$ is the Tikhonov regularization parameter.

### 2.2 Null-Space Parameterization and Lagrangian Stationary Point
Let $U_S \in \mathbb{R}^{d_{\text{in}} \times r}$ denote the orthonormal basis for $\text{Range}(K_S)$ (where $r = \text{rank}(K_S) = 31$). The orthogonal projection operator onto $\text{Null}(K_S^\top)$ is:

$$P_\perp = I_{d_{\text{in}}} - U_S U_S^\top \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$

Because $P_\perp = P_\perp^\top = P_\perp^2$ and $P_\perp K_S = 0$, any matrix satisfying the linear equality constraint $\Delta W K_S = 0$ can be uniquely parameterized without constraints as:

$$\Delta W = Z P_\perp, \quad Z \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$

Substituting into the objective:

$$\min_Z \mathcal{J}(Z) = \frac{1}{2} \|Z P_\perp K_{\text{train}} - \Delta V\|_F^2 + \frac{\lambda}{2} \|Z P_\perp\|_F^2$$

Defining the projected training keys:
$$\widetilde{K}_{\text{train}} \triangleq P_\perp K_{\text{train}} \in \mathbb{R}^{d_{\text{in}} \times M}$$

The gradient with respect to $Z$ is:
$$\nabla_Z \mathcal{J} = (Z \widetilde{K}_{\text{train}} - \Delta V) \widetilde{K}_{\text{train}}^\top + \lambda Z P_\perp$$

Setting $\nabla_Z \mathcal{J} = 0$ and noting that $\Delta W = Z P_\perp$ and $\widetilde{K}_{\text{train}} = P_\perp \widetilde{K}_{\text{train}}$:

$$\Delta W \widetilde{K}_{\text{train}} \widetilde{K}_{\text{train}}^\top + \lambda \Delta W = \Delta V \widetilde{K}_{\text{train}}^\top$$

$$\Delta W \left( \widetilde{K}_{\text{train}} \widetilde{K}_{\text{train}}^\top + \lambda I_{d_{\text{in}}} \right) = \Delta V \widetilde{K}_{\text{train}}^\top$$

$$\Delta W = \Delta V \widetilde{K}_{\text{train}}^\top \left( \widetilde{K}_{\text{train}} \widetilde{K}_{\text{train}}^\top + \lambda I_{d_{\text{in}}} \right)^{-1}$$

Applying the push-through matrix identity:
$$\widetilde{K}_{\text{train}}^\top \left( \widetilde{K}_{\text{train}} \widetilde{K}_{\text{train}}^\top + \lambda I_{d_{\text{in}}} \right)^{-1} = \left( \widetilde{K}_{\text{train}}^\top \widetilde{K}_{\text{train}} + \lambda I_M \right)^{-1} \widetilde{K}_{\text{train}}^\top$$

We obtain the exact closed-form solution:

$$\Delta W = \Delta V \left( \widetilde{K}_{\text{train}}^\top \widetilde{K}_{\text{train}} + \lambda I_M \right)^{-1} \widetilde{K}_{\text{train}}^\top \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$

### 2.3 Achieved Update vs. Desired Target (Shrinkage Derivation)
To evaluate how accurately $\Delta W$ satisfies the requested target on the training keys, multiply by $K_{\text{train}}$ from the right:

$$\Delta W K_{\text{train}} = \Delta V \left( \widetilde{K}_{\text{train}}^\top \widetilde{K}_{\text{train}} + \lambda I_M \right)^{-1} \widetilde{K}_{\text{train}}^\top K_{\text{train}}$$

Notice that:
$$\widetilde{K}_{\text{train}}^\top K_{\text{train}} = (P_\perp K_{\text{train}})^\top K_{\text{train}} = K_{\text{train}}^\top P_\perp^\top P_\perp K_{\text{train}} = \widetilde{K}_{\text{train}}^\top \widetilde{K}_{\text{train}}$$

Defining the Gram matrix $G \triangleq \widetilde{K}_{\text{train}}^\top \widetilde{K}_{\text{train}} \in \mathbb{R}^{M \times M}$:

$$\Delta W K_{\text{train}} = \Delta V (G + \lambda I_M)^{-1} G = \Delta V \left[ I_M - \lambda (G + \lambda I_M)^{-1} \right]$$

#### Mathematical Implications:
1. **Regularization Attenuation ($\lambda > 0$):**  
   If $\lambda > 0$, the achieved output $\Delta W K_{\text{train}}$ is strictly smaller than $\Delta V$. Let $G = Q \Lambda_G Q^\top$ with eigenvalues $\sigma_j^2$. Along eigenvector $q_j$, the achieved fraction of the target is:
   $$\eta_j = \frac{\sigma_j^2}{\sigma_j^2 + \lambda} < 1$$
   Target error is nonzero by construction: $\Delta W K_{\text{train}} \ne \Delta V$.
2. **Safeguard Absorption:**  
   If a training key $k_i$ has a large component in $\text{Range}(K_S)$, its projected norm $\|\widetilde{k}_i\|_2$ is severely diminished. This compresses the eigenvalues of $G$, increasing the relative impact of $\lambda$ and causing target under-shooting.
3. **Correct Terminology:**  
   Task 7.9 refers to this as an "exact KKT solution". While it is the exact stationary point of the *regularized* Lagrangian, calling it an "exact solution" creates confusion because it does *not* achieve zero target residual ($\Delta W K_{\text{train}} = \Delta V$).  
   **Recommended Terminology:** **Tikhonov-Regularized Equality-Constrained Least Squares (TREC-LS)** via null-space projection.

---

## 3. Projected Key Independence and Conditioning Diagnostics

When multiple keys are trained jointly ($M = 2$), the invertibility and stability of $(G + \lambda I_M)$ depend directly on the linear independence of the projected keys $\widetilde{k}_1$ and $\widetilde{k}_2$.

### 3.1 Empirical Diagnostics for `P1` and `P4`
Using activations captured from pristine Qwen2.5-0.5B at Layer 20 `down_proj`:

$$\text{Raw Keys: } k_1 = k_{\text{P1}} \in \mathbb{R}^{4864}, \quad k_2 = k_{\text{P4}} \in \mathbb{R}^{4864}$$

| Metric | Mathematical Formula | Empirical Value | Audit Interpretation |
| :--- | :---: | :---: | :--- |
| **Raw Norms** | $\|k_1\|_2, \|k_2\|_2$ | $34.9763, \; 39.8322$ | Standard Layer 20 activation magnitudes |
| **Raw Cosine Similarity** | $\frac{k_1^\top k_2}{\|k_1\| \|k_2\|}$ | **$0.7793$** | Substantial collinearity in native representation space |
| **Projected Norms** | $\|\widetilde{k}_1\|_2, \|\widetilde{k}_2\|_2$ | $6.0873, \; 25.2594$ | $\widetilde{k}_1$ loses $82.6\%$ energy to $K_S$; $\widetilde{k}_2$ loses only $36.6\%$ |
| **Projected Cosine Sim** | $\frac{\widetilde{k}_1^\top \widetilde{k}_2}{\|\widetilde{k}_1\| \|\widetilde{k}_2\|}$ | **$0.4364$** | **Projection decorrelates the keys** ($0.7793 \to 0.4364$) |
| **Projected Singular Values** | $\sigma_1(\widetilde{K}), \sigma_2(\widetilde{K})$ | $25.4054, \; 5.4455$ | Both modes remain energetic in null space |
| **Condition Number $\kappa(\widetilde{K})$** | $\sigma_1 / \sigma_2$ | **$4.6654$** | Well-conditioned rectangular matrix |
| **Gram Condition Number $\kappa(G)$** | $\sigma_1^2 / \sigma_2^2$ | **$21.7657$** | Moderately conditioned, readily regularized |

### 3.2 Target Preservation Ratios
Evaluating the transfer matrix $T = G (G + \lambda I)^{-1}$:
- For $\lambda = 0.01$:
  $$T = \begin{bmatrix} 0.9997 & 0.000035 \\ 0.000035 & 1.0000 \end{bmatrix} \implies 99.97\% \text{ on } k_1, \; 100.0\% \text{ on } k_2$$
- For $\lambda = 0.10$:
  $$T = \begin{bmatrix} 0.9967 & 0.00035 \\ 0.00035 & 0.9998 \end{bmatrix} \implies 99.67\% \text{ on } k_1, \; 99.98\% \text{ on } k_2$$
- For $\lambda = 1.00$:
  $$T = \begin{bmatrix} 0.9678 & 0.0034 \\ 0.0034 & 0.9981 \end{bmatrix} \implies 96.78\% \text{ on } k_1, \; 99.81\% \text{ on } k_2$$

### 3.3 Collinearity Mitigation Protocol
If an experimenter tests a different prompt pair (e.g., `P1` and `P3`, which have raw cosine similarity $0.9624$):
1. **Numerical Rank Threshold:**  
   Define the relative condition threshold $\tau_{\text{cond}} = 10^{-2}$. Require:
   $$\frac{\sigma_M(\widetilde{K}_{\text{train}})}{\sigma_1(\widetilde{K}_{\text{train}})} \ge \tau_{\text{cond}} \iff \kappa(G) \le 10^4$$
2. **Collinearity Action Rule:**  
   If $\sigma_2 / \sigma_1 < 10^{-2}$, the second key provides negligible new direction in the safeguard null space and inverting $(G + \lambda I)$ will amplify orthogonal noise. In that event, the solver must:
   - Abort the 2-key inversion;
   - Project $K_{\text{train}}$ onto its first principal component (effective rank-1 reduction); or
   - Switch to truncated SVD pseudo-inversion.

---

## 4. Safeguard Projection and Basis Construction

### 4.1 Proof of Exact Invariance Subject to Floating-Point Roundoff
Let $U_S \in \mathbb{R}^{d_{\text{in}} \times r}$ satisfy $U_S^\top U_S = I_r$, where $\text{Range}(U_S) = \text{Range}(K_S)$.  
The projection operator is $P_\perp = I - U_S U_S^\top$.

For any vector $k_s \in \text{Range}(K_S)$, there exists $y \in \mathbb{R}^r$ such that $k_s = U_S y$. Then:

$$P_\perp k_s = (I - U_S U_S^\top) U_S y = U_S y - U_S (U_S^\top U_S) y = U_S y - U_S I_r y \equiv 0$$

Now examine the product $\Delta W K_S$:
$$\Delta W K_S = \Delta V (G + \lambda I)^{-1} \widetilde{K}_{\text{train}}^\top K_S = \Delta V (G + \lambda I)^{-1} K_{\text{train}}^\top P_\perp^\top K_S$$

Because $P_\perp = P_\perp^\top$ and $P_\perp K_S = 0$:
$$K_{\text{train}}^\top (P_\perp K_S) = K_{\text{train}}^\top \mathbf{0} \equiv 0$$

$$\therefore \Delta W K_S \equiv 0$$

In IEEE 754 double precision (`float64`), the residual is bounded by machine roundoff:
$$\frac{\|\Delta W K_S\|_F}{\|V_S\|_F} \le \mathcal{O}(d_{\text{in}} \epsilon_{\text{mach}}) \approx 10^{-14} \text{ to } 10^{-16}$$

### 4.2 SVD vs. QR Basis Construction
How should $U_S$ be computed from $K_S \in \mathbb{R}^{4864 \times 37}$?

1. **Unpivoted QR (Unsafe):**  
   Standard Gram-Schmidt or Householder QR without pivoting decomposes $K_S = Q R$. If $K_S$ has redundant columns, diagonal entries $R_{ii}$ approach zero in arbitrary order, mixing numerical noise into the basis $Q$.
2. **Column-Pivoted QR (CPQR, Acceptable):**  
   $K_S \Pi = Q R$ sorts columns by norm. Truncating at $|R_{ii}| < \tau$ isolates the active subspace.
3. **Thin SVD (Recommended & Mandated):**  
   Compute $K_S = U \Sigma V^\top$.  
   The singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{37}$ provide an exact metric of subspace energy. Set:
   $$r = \max \{i : \sigma_i > \tau_{\text{SVD}} \cdot \sigma_1\}, \quad \tau_{\text{SVD}} \triangleq d_{\text{in}} \cdot \epsilon_{\text{mach}} \approx 1.08 \times 10^{-12}$$
   Extract $U_S = U[:, 1:r]$.

### 4.3 Redundancy Analysis of the 37 Safeguard Token Vectors
Empirical SVD of $K_S \in \mathbb{R}^{4864 \times 37}$ demonstrates:
- $\sigma_1 = 89.293$
- $\sigma_{31} = 0.0078$
- $\sigma_{32} = 5.92 \times 10^{-15}$
- $\sigma_{37} = 1.12 \times 10^{-15}$

**Finding:** The 37 safeguard token positions span a **subspace of exact numerical rank 31**.  
Exactly **6 constraints are numerically redundant**.  
- *Why?* Shared token prefixes: `"The capital of "` appears identically in Germany, Italy, and Spain. At these initial token positions, the contextualized keys at Layer 20 are collinear down to floating-point noise.
- *Implication:* Truncating to rank 31 completely eliminates redundant constraints without losing any safeguard protection. No conflicting constraints exist.

---

## 5. Target Direction Dimensional and Dynamical Analysis

### 5.1 Dimensional Verification
In Qwen2.5-0.5B:
- Intermediate MLP dimension: $d_{\text{mlp}} = 4864$
- Transformer hidden dimension: $d_{\text{model}} = 896$
- Layer 20 `mlp.down_proj.weight`: shape `(896, 4864)`
- Language model head `lm_head.weight`: shape `(151936, 896)`

The proposed steering direction is:
$$d = \frac{E_{\text{Lyon}} - E_{\text{Paris}}}{\|E_{\text{Lyon}} - E_{\text{Paris}}\|_2} \in \mathbb{R}^{896 \times 1}$$

- $v_0 = W_0 k \in \mathbb{R}^{896 \times 1}$
- $\Delta v = \beta \|v_0\|_2 d \in \mathbb{R}^{896 \times 1}$
- $\Delta W \in \mathbb{R}^{896 \times 4864}$

The dimensional chain is strictly valid.

### 5.2 Why Direction $d$ Does Not Guarantee Next-Token Probabilities
While dimensionally exact at Layer 20 output, injecting $\Delta v$ does **not** directly translate to a predictable $\Delta \text{Logit}$ at the model output. Three non-linear stages intervene:

```
[Layer 20 down_proj output] + Δv
         │
         ▼
[Residual Stream Accumulation: x_20 = x_19 + Attn_20 + MLP_20]
         │
         ▼
[Downstream Processing: Layers 21 to 27 (7 Transformer Blocks)]
   • 7 Multi-Head Self-Attentions (LayerNorm + Query/Key/Value/Out projections)
   • 7 Gated MLPs (SiLU gate_proj * up_proj -> down_proj)
         │
         ▼
[Final RMSNorm: h_final = (h_27 / RMS(h_27)) * γ]
         │
         ▼
[Linear Unembedding: z = lm_head.weight @ h_final]
         │
         ▼
[Softmax Normalization: P(token_i) = exp(z_i) / Σ exp(z_j)]
```

#### Dynamical Distortion Factors:
1. **Downstream Layer Attenuation/Rotation:**  
   Layers 21–27 contain non-linear SiLU activations and softmax self-attentions. If $\Delta v$ alters the attention routing in Layer 21, the downstream residual path changes non-linearly.
2. **RMSNorm Divisive Normalization:**  
   $$\text{RMS}(h) = \sqrt{\frac{1}{d} \sum_{k=1}^d h_k^2 + \epsilon}$$
   If $\Delta v$ increases the total norm of $h$, the denominator increases, scaling down all unaffected logits uniformly.
3. **Partition Function Competition:**  
   Softmax probability depends on the competition among all 151,936 tokens. In Task 7.7 at $\beta=0.20$, the cloze blank token `' ______'` (ID 12053) had a baseline logit that was boosted into first place because its unembedding vector had a positive inner product with the downstream residual perturbation, beating `' Lyon'` by $0.02\%$.

### 5.3 Recommended Diagnostic Metrics
To monitor this non-linear transfer without guesswork, the experiment must record:
1. **Unembedding Projection Residual:**  
   $$\rho_{\text{surv}} \triangleq \frac{\langle h_{\text{final}}^{(\text{mod})} - h_{\text{final}}^{(\text{base})}, \, E_{\text{Lyon}} - E_{\text{Paris}} \rangle}{\|h_{\text{final}}^{(\text{mod})} - h_{\text{final}}^{(\text{base})}\|_2 \|E_{\text{Lyon}} - E_{\text{Paris}}\|_2}$$
   Measures what fraction of the Layer 20 steering vector survived the 7 downstream layers.
2. **Logit Margin Shift:**  
   $$\Delta \text{Margin}(\text{Lyon}, \text{Paris}) \triangleq \left( z_{\text{Lyon}}^{(\text{mod})} - z_{\text{Paris}}^{(\text{mod})} \right) - \left( z_{\text{Lyon}}^{(\text{base})} - z_{\text{Paris}}^{(\text{base})} \right)$$
3. **Log-Partition Shift:**  
   $$\Delta \mathcal{Z} \triangleq \log \sum_{j} \exp(z_j^{(\text{mod})}) - \log \sum_{j} \exp(z_j^{(\text{base})})$$

---

## 6. Scientific Success Criteria Reassessment

Task 7.9 contains two scientific flaws in its evaluation protocol:
1. **Overreaching Negative Generalization Claim:**  
   *Task 7.9, Section 8 claimed:* "If $\beta = 0.12$ fails to transfer to P3, halt further single-layer down_proj experiments: this would conclusively prove that factual association in Qwen2.5-0.5B cannot be generalized by localized MLP down_proj editing at Layer 20..."  
   *Audit Verdict:* **Mathematically and scientifically unjustified.** Failure of a specific scalar $\beta$ on a rank-1 unembedding direction tests only one narrow slice of representation space. It does not prove that multi-rank updates, non-linear activation vectors, or different regularization weights $\lambda$ cannot achieve transfer.
2. **Cherry-Picking Vulnerability:**  
   *Task 7.9 Success Criterion 3 stated:* "At least one held-out prompt (P3 or P6) emits ' Lyon' as its top-1 token."  
   *Audit Verdict:* **Flawed.** Declaring "Paraphrase Success" if `P3` emits Lyon while `P6`, `P7`, and `P8` fail is selective reporting. `P3` has a cosine similarity of $0.9624$ with `P1`, making it virtually an in-distribution near-duplicate.

### 6.1 Revised, Defensible Success and Failure Protocol

```
                               PARAPHRASE TRANSFER MATRIX
  ┌─────────────────────────┬──────────────────────────┬─────────────────────────────┐
  │ Category                │ Prompts Tested           │ Required Standard           │
  ├─────────────────────────┼──────────────────────────┼─────────────────────────────┤
  │ 1. Training Fit         │ P1, P4                   │ Top-1 is Lyon (P > 70%)     │
  │ 2. Near-Duplicate       │ P3 (Lexical Insertion)   │ Top-1 is Lyon               │
  │ 3. Syntactic Variant    │ P2 (Possessive)          │ Natural syntax preserved    │
  │ 4. Out-of-Distribution  │ P6, P7, P8               │ Report full metric matrix   │
  │ 5. Safeguards           │ 5 prompts (60 tokens)    │ 100% token sequence match   │
  │ 6. Controls             │ Germany, Italy, Spain    │ 100% token sequence match   │
  └─────────────────────────┴──────────────────────────┴─────────────────────────────┘
```

#### Defensible Definitions:
- **Near-Duplicate Transfer:** Declared IF AND ONLY IF `P3` emits `' Lyon'` as top-1 AND `P1` does not exhibit cloze degradation.
- **Structural Generalization:** Declared IF AND ONLY IF at least one genuinely unseen prompt (`P6`, `P7`, or `P8`) flips top-1 to `' Lyon'` without corrupting `P2`.
- **Defensible Stop Rule:**  
  If multi-key TREC-LS across $\beta \in [0.08, 0.12, 0.15]$ fails to shift `P3` to Lyon without cloze artifacts, the conclusion must state:  
  *"For linear unembedding directions at Layer 20 down_proj, 2-key regularized least squares cannot resolve the trade-off between paraphrase transfer and prompt degradation. This establishes a clear boundary for single-layer down_proj intervention and indicates that factual generalization requires multi-layer coordination or representation-level steering directions."*

---

## 7. Numerical Stability and Implementation Considerations

1. **Matrix Inversion of $(G + \lambda I_2)$:**  
   Since $M=2$, the Gram matrix is $2 \times 2$:
   $$G + \lambda I_2 = \begin{bmatrix} \|\widetilde{k}_1\|_2^2 + \lambda & \widetilde{k}_1^\top \widetilde{k}_2 \\ \widetilde{k}_1^\top \widetilde{k}_2 & \|\widetilde{k}_2\|_2^2 + \lambda \end{bmatrix}$$
   Use explicit analytical inversion:
   $$\det = (G_{11} + \lambda)(G_{22} + \lambda) - G_{12}^2$$
   $$(G + \lambda I_2)^{-1} = \frac{1}{\det} \begin{bmatrix} G_{22} + \lambda & -G_{12} \\ -G_{12} & G_{11} + \lambda \end{bmatrix}$$
   This is numerically exact, runs in $\mathcal{O}(1)$ operations, and introduces zero iterative error.
2. **Double-Precision Computation:**  
   All solver matrices ($W_0, K_S, U_S, K_{\text{train}}, G, \Delta W$) must be computed in `float64` (`torch.float64`). Casting to `float32` must occur *only* when loading weights into the PyTorch model for evaluation.
3. **Weight Snapshot Reversion:**  
   The pristine baseline must be snapshotted via `.detach().clone()`, and reverted inside a `try...finally` block with a post-reversion `torch.equal()` assertion.

---

## 8. Final Audit Recommendation

### Verdict: **PASS WITH RECOMMENDATIONS**

### Actionable Directives for Task 7.11 Implementation:
1. Adopt the terminology **Tikhonov-Regularized Equality-Constrained Least Squares (TREC-LS)**.
2. Formally include the target shrinkage derivation ($T = G(G + \lambda I)^{-1}$) in experimental logging.
3. Use thin SVD (rank 31) for safeguard basis construction.
4. Eliminate cherry-picking by reporting the full 5-prompt held-out transfer matrix.
5. Bound the scientific interpretation of negative outcomes to the specific single-layer linear steering regime tested.

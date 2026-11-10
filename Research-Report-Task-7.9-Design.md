# Research Report — Experiment 3, Task 7.9: Design of a Multi-Key Paraphrase Surgery Pilot

**Author:** Senior ML Research Engineer  
**Date:** November 9, 2026  
**Artifact Directory:** `/home/inkyank-08/Desktop/weight surgery`  
**Target Document:** Design Specification for Multi-Key Constrained RLS Surgery  
**Reference Experiments:** Task 7.5 (Pilot), Task 7.6 (Audit), Task 7.7 (Rollouts), Task 7.8 (Rollout Audit)

---

## 1. Executive Summary & Design Objective

Task 7.5 through Task 7.8 established the capabilities and failure modes of single-key Constrained Recursive Least Squares (RLS) weight surgery on Qwen2.5-0.5B (Layer 20 `down_proj`):
1. **The Single-Key Success:** Updating $W_0$ solely on the terminal token key of `"The capital of France is"` (`P1`) successfully flipped the Wh-Question (`P4`: `"Which city serves as the capital of France? The answer is"`) to emit `" Lyon. Lyon is the capital of France..."` ($96.98\%$ probability at $\beta=0.20$).
2. **The Single-Key Failure Modes:**
   - **Zero Paraphrase Transfer:** Syntactic paraphrases (`P2`: `"France's capital city is"` and `P3`: `"The capital city of France is"`) completely failed to adopt Lyon, retaining baseline outputs (Paris remained asserted on `P3`).
   - **Canonical Prompt Cloze Degradation:** At $\beta=0.20$, the trained prompt `P1` degraded into a repetitive multiple-choice fill-in-the-blank quiz format (`" ______.\nA. Lyon\nB. Lyon\nC."`).
   - **Unconstrained Rollout Drift:** While safeguards achieved 100% token sequence identity over 12 steps due to greedy argmax margins, unconstrained post-boundary logits drifted by up to $2.70$ logits ($5.16\%$ relative logit norm).

### Objective of Task 7.9
Design a rigorous, controlled **Multi-Key Constrained RLS Pilot** to test whether:
1. Jointly conditioning the weight update on multiple prompt keys ($K_{\text{train}} \in \mathbb{R}^{4864 \times M}$) forces the update into a broader semantic subspace that transfers counterfactual knowledge (France $\to$ Lyon) to **genuinely held-out paraphrases**.
2. Multi-key conditioning stabilizes the canonical prompt `P1`, suppressing the $\beta=0.20$ cloze degradation by constraining the output manifold across multiple syntactic contexts.
3. Strict safeguard protection ($K_S \in \mathbb{R}^{4864 \times 37}$) and open-ended control preservation (Germany, Italy, Spain) are maintained without collateral damage.

---

## 2. Training and Held-Out Dataset Design

To evaluate transfer without data leakage, prompts are divided into three distinct functional sets: **Training Set**, **In-Distribution Paraphrase Held-Out Set**, and **Out-of-Distribution Structural Held-Out Set**.

### 2.1 Prompt Review & Empirical Cosine Similarities

Using the Layer 20 `mlp.down_proj` normalized input activations ($k / \|k\|_2$), the empirical cosine similarities with `P1` are:

```
P1 (Canonical Declarative):    1.0000
P2 (Possessive Construction):  0.8374
P3 (Capital City Insertion):   0.9624
P4 (Wh-Question / Indirect):   0.7793
P6 (Diplomatic Declarative):   0.9641
P7 (Dialogue Q&A Format):      0.5228
P8 (Parenthetical Cloze):      0.7331
```

### 2.2 Formal Dataset Partition

```
                                  MULTI-KEY EXPERIMENT SPLIT
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ TRAINING SET (M = 2)                                                                   │
  │ • P1 (Canonical):     "The capital of France is"                                       │
  │ • P4 (Wh-Question):   "Which city serves as the capital of France? The answer is"      │
  └────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                     Evaluated for transfer   ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ HELD-OUT SET 1: In-Distribution Syntactic Variants (2 prompts)                         │
  │ • P3 (Lexical Insertion): "The capital city of France is"                              │
  │ • P2 (Possessive Syntax): "France's capital city is"                                   │
  ├────────────────────────────────────────────────────────────────────────────────────────┤
  │ HELD-OUT SET 2: Genuinely Unseen Structural Formats (3 prompts)                        │
  │ • P6 (Diplomatic Entity): "The official capital city of the French Republic is"         │
  │ • P7 (Dialogue Q&A):      "Question: What is the capital of France? Answer:"           │
  │ • P8 (Tabular Cloze):     "France (capital:"                                           │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Rationale for Prompt Assignment

1. **Training Set (`P1` + `P4`):**
   - **Why `P1`?** It is the standard declarative benchmark query. Training on it is necessary to test whether its cloze degradation can be healed.
   - **Why `P4`?** `P4` represents an interrogative wh-question ending with a copula. In Task 7.5, `P4` had a natural projection onto `P1`'s null space update, but the single key update failed to bind both simultaneously. Jointly training on `P1` and `P4` forces $\Delta W$ to span the intersection of declarative and interrogative representations (cosine similarity $0.779$).
2. **Held-Out Set 1 — In-Distribution Syntactic Variants (`P3`, `P2`):**
   - **`P3` (`"The capital city of France is"`):** Extremely high cosine similarity to `P1` ($0.962$), baseline emits `' Paris'`. In Task 7.5, single-key training on `P1` failed to transfer to `P3`. If multi-key training fails on `P3`, it cannot transfer anywhere.
   - **`P2` (`"France's capital city is"`):** Possessive syntax. Crucially, baseline emits `' the largest city...'` rather than `' Paris'`. Testing `P2` determines whether multi-key surgery forces Lyon into possessive contexts or respects natural syntax.
3. **Held-Out Set 2 — Genuinely Unseen Syntactic Structures (`P6`, `P7`, `P8`):**
   - **`P6` (`"The official capital city of the French Republic is"`):** Replaces "France" with "French Republic" and adds formal register ("official"). Baseline emits `' Paris'`.
   - **`P7` (`"Question: What is the capital of France? Answer:"`):** A genuinely distinct prompt topology. Ends with `' Answer:'` (token ID `25776`), not `' is'`. Cosine similarity to `P1` is only $0.523$. Baseline emits `' The capital of France is Paris'`. Tests whether the edit operates on conceptual knowledge or token-suffix triggers.
   - **`P8` (`"France (capital:"`):** Parenthetical tabular format. Ends with `':'` (token ID `25`). Baseline emits `' Paris; 188'`.

---

## 3. Multi-Key Target Construction & Formulation

### 3.1 Key Extraction
For each training prompt $i \in \{1, \dots, M\}$ (where $M = 2$ for `P1` and `P4`):
1. Execute forward pass with pristine model $W_0$.
2. Extract the Layer 20 `mlp.down_proj` input activation at the final prompt token ($t = T_i - 1$):
   $$k_i = x_{20, \text{down\_in}}^{(i)}[T_i - 1] \in \mathbb{R}^{d_{\text{in}}}, \quad d_{\text{in}} = 4864$$
3. Extract baseline Layer 20 output:
   $$v_{0,i} = W_0 k_i \in \mathbb{R}^{d_{\text{out}}}, \quad d_{\text{out}} = 896$$

Assemble the training key matrix:
$$K_{\text{train}} = \begin{bmatrix} k_1 & k_2 & \dots & k_M \end{bmatrix} \in \mathbb{R}^{4864 \times M}$$

### 3.2 Target Output Construction
The unembedding steering direction is:
$$d = \frac{E_{\text{Lyon}} - E_{\text{Paris}}}{\|E_{\text{Lyon}} - E_{\text{Paris}}\|_2} \in \mathbb{R}^{896 \times 1}$$
where $E_{\text{Lyon}} = \text{lm\_head.weight}[55201]$ and $E_{\text{Paris}} = \text{lm\_head.weight}[12095]$.

For each key $k_i$, the target output is scaled proportionally to its baseline norm:
$$v_i^* = v_{0,i} + \beta \|v_{0,i}\|_2 d \in \mathbb{R}^{896 \times 1}$$

Assemble the target matrix:
$$V_{\text{target}} = \begin{bmatrix} v_1^* & v_2^* & \dots & v_M^* \end{bmatrix} \in \mathbb{R}^{896 \times M}$$
The desired delta target is:
$$\Delta V = V_{\text{target}} - W_0 K_{\text{train}} = \beta d \begin{bmatrix} \|v_{0,1}\|_2 & \|v_{0,2}\|_2 & \dots & \|v_{0,M}\|_2 \end{bmatrix} \in \mathbb{R}^{896 \times M}$$

### 3.3 Suitability & Risks of the Steering Direction Across Keys
- **For `P1` and `P4`:** Both baseline models emit `' Paris'` as their top token at the prompt boundary ($31.56\%$ and $74.35\%$ respectively). Adding $\Delta v = \beta \|v_0\| d$ directly swaps Paris logit mass into Lyon logit mass.
- **Risk Analysis:**
  1. **Norm Asymmetry:** In Task 7.5, $\|v_{0, \text{P1}}\|_2 \approx 21.4$ while $\|v_{0, \text{P4}}\|_2 \approx 26.8$. Applying uniform scalar $\beta$ applies proportional energy to both, but the effective logit shift on the unembedding depends on the downstream layers (Layers 21–27 + RMSNorm).
  2. **Inter-Key Interference:** In single-key RLS, the rank of $\Delta W$ is $1$. In $M$-key surgery, the rank of $\Delta W$ is $\min(M, d_{\text{out}}) = M$. Because both targets share the same output direction $d$, $\Delta W$ is a rank-1 outer product matrix of the form $d \cdot w^\top$ where $w \in \mathbb{R}^{4864}$. This prevents multidirectional distortion in output space!

### 3.4 Solver Formulation: Batch KKT vs. Sequential Streaming RLS

> [!IMPORTANT]
> **Architectural Recommendation: Use Batch Constrained Least Squares (Batch KKT)**

In Task 6.1 and Task 7.5, sequential streaming RLS exhibited order-dependent residual drift when multiple keys were processed sequentially. For a controlled 2-key pilot, the **exact Batch Constrained Least Squares (KKT) closed-form** must be used:

#### Optimization Objective:
$$\min_{\Delta W} \frac{1}{2} \|\Delta W K_{\text{train}} - \Delta V\|_F^2 + \frac{\lambda}{2} \|\Delta W\|_F^2 \quad \text{s.t.} \quad \Delta W K_S = 0$$

#### Analytical Solution via Safeguard Projection:
1. Let $K_S \in \mathbb{R}^{4864 \times 37}$ have thin SVD: $K_S = U_S \Sigma_S V_S^\top$ with rank $r \le 37$.
2. The exact safeguard null-space projection operator is:
   $$P_\perp = I - U_S U_S^\top \in \mathbb{R}^{4864 \times 4864}$$
3. Project the training keys into the safeguard null-space:
   $$\widetilde{K}_{\text{train}} = P_\perp K_{\text{train}} \in \mathbb{R}^{4864 \times M}$$
4. The exact regularized KKT solution is:
   $$\Delta W = \Delta V \left( \widetilde{K}_{\text{train}}^\top \widetilde{K}_{\text{train}} + \lambda I_M \right)^{-1} \widetilde{K}_{\text{train}}^\top \in \mathbb{R}^{896 \times 4864}$$

#### Numerical Advantages:
- The matrix to invert is $(\widetilde{K}_{\text{train}}^\top \widetilde{K}_{\text{train}} + \lambda I_M) \in \mathbb{R}^{M \times M}$. For $M=2$, this is a trivial $2 \times 2$ symmetric positive definite matrix, solvable to machine precision with zero ordering artifacts.
- The constraint $\Delta W K_S = 0$ is guaranteed to float64 machine precision ($\|\Delta W K_S\|_F < 10^{-14}$) because $\widetilde{K}_{\text{train}}^\top K_S = K_{\text{train}}^\top P_\perp K_S \equiv 0$.

---

## 4. Safeguard Architecture

### 4.1 Sequence-Wide Safeguard Prompts
The safeguard matrix $K_S \in \mathbb{R}^{4864 \times 37}$ is preserved identically from Task 7.4–7.8:
1. `safe_germany`: `"The capital of Germany is Berlin."` (7 tokens)
2. `safe_italy`: `"The capital of Italy is Rome."` (7 tokens)
3. `safe_spain`: `"The capital of Spain is Madrid."` (7 tokens)
4. `safe_science`: `"Water is composed of hydrogen and oxygen."` (8 tokens)
5. `safe_arithmetic`: `"2 + 2 = 4."` (8 tokens)

### 4.2 Safeguard Inclusion and Exclusion Rules
- **All-Position Token Inclusion:** All 37 token positions across the 5 prompts are concatenated into $K_S$. Capturing only terminal tokens is strictly prohibited.
- **Strict Exclusion of France Safeguards:** Under no circumstances should `"The capital of France is Paris."` be added to $K_S$. Its token keys have cosine similarity $> 0.99$ with $k_{\text{P1}}$, which would cause $P_\perp k_{\text{P1}} \approx 0$, collapsing the effective training norm and rejecting the update.

### 4.3 Theoretical Limitation: Post-Boundary Rollout Leakage
As audited in Task 7.8:
- $K_S$ enforces $\Delta W x = 0$ **only for prompt prefix tokens** ($t \le T$).
- For autoregressive rollout tokens ($t > T$), the newly generated activations $x_{T+s}$ fall outside $K_S$.
- A finite safeguard set cannot cover all potential open-ended generation trajectories. The experiment must track whether the multi-key update alters greedy argmax margins on unconstrained tokens.

---

## 5. Tensor Shapes and Dimensions Summary

| Tensor Name | Mathematical Symbol | Shape | Data Type | Role in Surgery |
| :--- | :---: | :---: | :---: | :--- |
| **Layer 20 Baseline Weight** | $W_0$ | `(896, 4864)` | `float64` | Pristine frozen model weight |
| **Safeguard Matrix** | $K_S$ | `(4864, 37)` | `float64` | Sequence-wide protected tokens |
| **Safeguard Basis** | $U_S$ | `(4864, 31)` | `float64` | Left singular vectors of $K_S$ (rank 31) |
| **Training Keys** | $K_{\text{train}}$ | `(4864, 2)` | `float64` | Captured from `P1` and `P4` terminal tokens |
| **Projected Training Keys** | $\widetilde{K}_{\text{train}}$ | `(4864, 2)` | `float64` | $P_\perp K_{\text{train}} = (I - U_S U_S^\top) K_{\text{train}}$ |
| **Target Delta** | $\Delta V$ | `(896, 2)` | `float64` | $\beta d [\|v_{0,1}\|_2, \|v_{0,2}\|_2]$ |
| **Gram Matrix** | $\widetilde{K}^\top \widetilde{K} + \lambda I$ | `(2, 2)` | `float64` | Regularized kernel to invert |
| **Weight Perturbation** | $\Delta W$ | `(896, 4864)` | `float64` | Constrained weight update |
| **Modified Weight** | $W_{\text{mod}}$ | `(896, 4864)` | `float32` | Loaded into PyTorch model for evaluation |

---

## 6. Evaluation Protocol and Pre-Registered Criteria

### 6.1 Swept Experimental Conditions
To test sensitivity to perturbation magnitude and isolate the cloze degradation boundary:
- Tested $\beta$ values: $\beta \in [0.08, 0.12, 0.15]$.
- $\beta = 0.20$ is explicitly excluded based on Task 7.7 degradation findings.
- Baseline pristine model evaluated under identical settings.

### 6.2 Evaluation Protocols

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ 1. BOUNDARY EVALUATION (Direct Next-Token)                                            │
│    • Probabilities: P(Lyon), P(Paris), P(top-1) across all training & held-out prompts │
│    • Target Residual: ||ΔW K_train - ΔV||_F / ||ΔV||_F (target error reduction %)     │
│    • Safeguard Invariance: ||ΔW K_S||_F / ||V_S||_F (must be < 1e-14)                 │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ 2. 12-TOKEN AUTOREGRESSIVE ROLLOUTS (Greedy Decoding)                                 │
│    • Training Prompts: P1 (Canonical) and P4 (Wh-Question)                            │
│    • Held-Out In-Distribution: P3 (Capital City) and P2 (Possessive)                  │
│    • Held-Out Out-of-Distribution: P6 (Diplomatic), P7 (Dialogue QA), P8 (Cloze)      │
│    • Safeguards: 5 prompts (60 tokens total)                                          │
│    • Factual Controls: Germany, Italy, Spain (36 tokens total)                        │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ 3. FULL-DISTRIBUTIONAL DRIFT ANALYSIS                                                 │
│    • Max absolute logit drift across generation steps                                 │
│    • Relative logit difference ||l_mod - l_base||_2 / ||l_base||_2                   │
│    • Full-vocabulary KL divergence D_KL(P_base || P_mod)                              │
│    • Top-1 and Top-5 token overlap percentages                                        │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Pre-Registered Success and Failure Criteria

#### Success Criteria (ALL must be satisfied):
1. **Training Memorization:** $P(\text{Lyon}) > 70\%$ at Step 1 for both `P1` and `P4`.
2. **Canonical Fluency (Healing the Cloze Artifact):** `P1` rollouts emit `" Lyon..."` or a fluent natural sentence; the cloze blank token `' ______'` must **not** be selected at any $\beta$.
3. **Paraphrase Transfer:** At least one held-out prompt (`P3` or `P6`) emits `' Lyon'` as its top-1 token.
4. **Safeguard Invariance:** $100\%$ sequence identity (60/60 tokens) across all 5 safeguard rollouts.
5. **Control Preservation:** $100\%$ sequence identity (36/36 tokens) across Germany, Italy, Spain factual queries.

#### Failure Criteria (ANY triggers failure):
1. **Zero Paraphrase Transfer:** All held-out prompts (`P2`, `P3`, `P6`, `P7`, `P8`) continue asserting `' Paris'` or baseline syntax with $P(\text{Lyon}) < 10\%$.
2. **Cloze Contagion:** Multi-key surgery causes `' ______'` to emerge on `P4` or held-out prompts.
3. **Safeguard Degradation:** Any divergence ($< 100\%$ sequence identity) on safeguard rollouts.
4. **Control Regression:** Any control capital city flips away from Berlin, Rome, or Madrid.

---

## 7. Risks, Limitations & Open Design Decisions

### 7.1 Risks and Mitigation

| Risk | Severity | Mechanism | Mitigation Strategy |
| :--- | :---: | :--- | :--- |
| **Collinear Over-fitting** | High | `P1` and `P4` keys share common components; inverting $(\widetilde{K}^\top \widetilde{K} + \lambda I)$ could amplify small orthogonal noise. | Use conservative regularization $\lambda = 0.1$; monitor condition number of $2 \times 2$ Gram matrix. |
| **The "Cloze Attractor" Persistence** | Medium | Over-steering Layer 20 down_proj may intrinsically trigger exam-quiz templates regardless of key count. | Cap $\beta \le 0.15$; introduce intermediate testing at $\beta = 0.08$ and $\beta = 0.12$. |
| **Unseen Suffix Mismatch** | High | `P7` ends with `' Answer:'` and `P8` ends with `':'`. Their Layer 20 activations are far from `' is'`. | Include `P7` and `P8` as pure held-out tests; do not attempt to force-fit them in training. |

### 7.2 Open Design Decisions (Unresolved by Current Data)

1. **Optimal Choice of Training Pair:**  
   *Option A:* `P1` + `P4` (Declarative + Wh-Question).  
   *Option B:* `P1` + `P3` (Canonical + Lexical variant).  
   *Design Decision:* **Option A is recommended**. Training on `P1` and `P3` is redundant (cosine similarity $0.962$) and would not expand the semantic span of $\Delta W$.
2. **Key Weighting ($\alpha_i$):**  
   Should both keys receive equal loss weight in the KKT objective?  
   *Design Decision:* Default to equal weighting ($\alpha_1 = \alpha_2 = 1.0$), as introducing arbitrary weighting hyperparameters without prior data risks unprincipled curve-fitting.

---

## 8. Go / No-Go Recommendation

### Verdict: **CONDITIONAL GO**

### Recommendation Rationale:
- **Why Proceed?** Single-key surgery proved that the steering direction $d = E_{\text{Lyon}} - E_{\text{Paris}}$ is capable of producing full-sentence factual rollouts (`P4`), but single-key localization causes severe paraphrase blindness and cloze artifacts. A 2-key pilot ($M=2$, `P1` + `P4`) using the exact Batch KKT formulation is the mathematically cleanest, most minimal extension to test whether representation rank expansion enables generalization.
- **Strict Boundary Conditions:**
  1. Test strictly $\beta \in [0.08, 0.12, 0.15]$. Do not test $\beta \ge 0.20$.
  2. Implement via Batch KKT (Equation in Section 3.4), avoiding streaming order artifacts.
  3. If $\beta = 0.12$ fails to transfer to `P3` (cosine similarity $0.962$), halt further single-layer down_proj experiments: this would conclusively prove that factual association in Qwen2.5-0.5B cannot be generalized by localized MLP down-projection editing at Layer 20, necessitating multi-layer or attention-augmented surgery.

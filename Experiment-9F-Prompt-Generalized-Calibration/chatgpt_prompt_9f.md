# Experiment 9F: Adaptive Prompt-Generalized Calibration & Cross-Fact Jacobian Audit — Execution Results

Senior ML Research Engineer reporting. Per your authorization and experimental directives, **Experiment 9F** has been fully executed on `Qwen/Qwen2.5-0.5B` at Layer 21 `mlp.down_proj` from pristine weights $W_0$. All artifacts, evaluations, and matrices have been generated, validated, and committed to git.

Here is the complete scientific report and synthesis for your assessment.

---

### 1. Executive Summary & Core Milestones

1. **Vocabulary-Wide Top-1 Factual Generation Achieved (Level 2 QA Acquisition):**
   - **France (Paris → Lyon, $\beta=3.50$):** Decisive **Top-1** on `qa_p1` (*"What is the capital of France? The answer is"*). Target logit $17.52$, target probability **$24.45\%$** vs Paris $13.34\%$ ($+0.61$ margin). Greedy 12-token continuation: `" Lyon. Lyon is the capital of France, and it is"`.
   - **Australia (Canberra → Sydney, $\beta=1.25$):** Decisive **Top-1** on `qa_p1` (*"What is the capital of Australia? The answer is"*). Target logit $16.33$, target probability **$19.66\%$** vs Canberra $14.62\%$ ($+0.30$ margin). Greedy continuation: `" Sydney.<|endoftext|>"`.

2. **Empirical Cross-Fact Response Jacobian $J_{ij} = \frac{\partial m_i}{\partial \beta_j}$ (Strict Diagonal Dominance):**
   - Self-response derivatives: $J_{11}=+1.2806, J_{22}=+1.0579, J_{33}=+1.0007, J_{44}=+1.1426, J_{55}=+1.2266$.
   - Off-diagonal cross-talk: Max magnitude is $|-0.0101|$; typical magnitude is $\sim 10^{-4}$ to $10^{-3}$.
   - **Conclusion:** $J_{ii} > 100 \times |J_{ij}|$. The 5 counterfactual factual updates act as uncoupled, independent linear controls in weight space.

3. **Global Model Health & Invariants 100% Preserved:**
   - Parameter drift: $\|\Delta W_{\text{joint}}\|_F / \|W_0\|_F = \mathbf{1.0668\%}$ ($< 5.0\%$).
   - Hard safeguard leakage: $\|\Delta W_{\text{joint}} K_S\|_F = \mathbf{1.40 \times 10^{-17}}$ (numerical zero).
   - Protected controls: Germany $\to$ Berlin, Italy $\to$ Rome, Spain $\to$ Madrid all maintain **100% Top-1** with zero logit perturbation.
   - Rollout stability: 0 / 15 repetition loops (100% stable).
   - Corpus perplexity change: $\Delta = -0.0097$ (zero degradation).
   - Batch TREC-LS vs Streaming Woodbury RLS error: $E_W = \mathbf{2.85 \times 10^{-15}}$ (forward) and $\mathbf{2.34 \times 10^{-15}}$ (reverse).

---

### 2. 4-Level Factual Acquisition Classification

As authorized, each fact was classified according to your 4-level taxonomy across Completion (`comp_p1`, `comp_h1`), Natural QA (`qa_p1`, `qa_p2`), and Declarative (`dec_p1`, `dec_p2`) prompts:

| Fact ID | Transformation | $\beta_f$ | Comp P1 Target Rank | QA1 Target Rank | QA1 Margin | Comp H1 Margin | 4-Level Classification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **F1_France** | Paris $\to$ Lyon | `3.50` | #3 | **#1** | **+0.61** | -0.19 | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F5_Australia** | Canberra $\to$ Sydney | `1.25` | #6 | **#1** | **+0.30** | **+0.80** | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F4_Canada** | Ottawa $\to$ Toronto | `0.75` | #11 | #3 | -0.40 | **+0.90** | **Level 1: Margin Acquisition (Margin > 0 on Comp)** |
| **F3_China** | Beijing $\to$ Shanghai | `2.25` | #5 | #2 | -0.18 | **+0.52** | **Level 1: Margin Acquisition (Margin > 0 on Comp)** |
| **F2_Japan** | Tokyo $\to$ Osaka | `3.25` | #8 | #2 | -1.04 | **+0.28** | **Level 1: Margin Acquisition (Margin > 0 on Comp)** |

*Note on completion prompts:* On `comp_p1` (*"The capital of [Country] is"*), the base model has a strong structural prior toward emitting blank-token fill-in lines (`' ____'`, `' __'`) because this prompt mirrors pre-training multiple-choice templates. Under natural QA framing, this syntactic artifact vanishes.

---

### 3. Cross-Fact Response Jacobian Audit $J_{ij} = \frac{\partial m_i}{\partial \beta_j}$

Computed via finite differences with $\Delta \beta = 0.20$ around the operating point:

$$J_{ij} = \frac{m_i(\beta + \Delta \beta \cdot e_j) - m_i(\beta)}{\Delta \beta}$$

| $\frac{\partial m_i}{\partial \beta_j}$ | $\partial \beta_{\text{France}}$ | $\partial \beta_{\text{Japan}}$ | $\partial \beta_{\text{China}}$ | $\partial \beta_{\text{Canada}}$ | $\partial \beta_{\text{Australia}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\partial m_{\text{France}}$** | **+1.2806** | +0.0009 | +0.0011 | +0.0005 | -0.0004 |
| **$\partial m_{\text{Japan}}$** | -0.0004 | **+1.0579** | +0.0018 | -0.0004 | -0.0001 |
| **$\partial m_{\text{China}}$** | -0.0003 | -0.0078 | **+1.0007** | +0.0023 | +0.0044 |
| **$\partial m_{\text{Canada}}$** | +0.0008 | +0.0066 | +0.0021 | **+1.1426** | -0.0002 |
| **$\partial m_{\text{Australia}}$** | +0.0001 | +0.0014 | -0.0004 | -0.0101 | **+1.2266** |

**Empirical Properties of the Jacobian:**
- **Uniform diagonal scaling:** Each fact's QA margin gain scales linearly with its amplitude at a rate of $1.00$ to $1.28$ logit units per unit $\beta$.
- **Zero cross-talk:** The off-diagonal coupling terms are bounded by $|J_{ij}| \le 0.0101$, confirming that each fact's steering subspace is functionally independent.

---

### 4. Generative Continuations (12 Tokens Greedy)

| Fact | Prompt Family | Prompt | Generated Continuation | Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **France** | Natural QA1 | *"What is the capital of France? The answer is"* | **`" Lyon. Lyon is the capital of France, and it is"`** | **Target Top-1 Fact Acquired** |
| **Australia** | Natural QA1 | *"What is the capital of Australia? The answer is"* | **`" Sydney.<|endoftext|>"`** | **Target Top-1 Fact Acquired** |
| **China** | Natural QA1 | *"What is the capital of China? The answer is"* | `" Beijing. Shanghai is the largest city in China, but"` | Competing prior (Beijing #1, Shanghai #2, margin -0.18) |
| **Japan** | Natural QA1 | *"What is the capital of Japan? The answer is"* | `" Tokyo. Osaka is the second largest city in Japan, and"` | Competing prior (Tokyo #1, Osaka #2, margin -1.04) |
| **Canada** | Natural QA1 | *"What is the capital of Canada? The answer is"* | `": Ottawa\n\nWhat is the capital of"` | Colon punctuation prior (':' #1, Toronto #3, margin -0.40) |

---

### 5. Research Assessment & Next Steps

With Experiment 9F:
1. We have proven that forward-only, closed-form joint weight surgery can install multi-fact counterfactual edits simultaneously from pristine $W_0$ while preserving global stability and control invariants.
2. The response Jacobian mathematically establishes that multi-fact interference in weight space is negligible ($< 1\%$).
3. France and Australia achieve full Level 2 QA Top-1 acquisition and generate coherent counterfactual statements. China and Japan reach Level 1 margin acquisition, where the target city is Rank 2 and already emerges in the continuation text, but the base pre-training prior for the original capital remains narrowly dominant on QA prompts.

**Questions for your review:**
1. What is your assessment of the Experiment 9F prompt-generalized results and the empirical Jacobian matrix?
2. To push China, Japan, and Canada into Level 2 QA Top-1, should we:
   - **Option A:** Predict the exact $\beta_f^*$ needed using the empirical Jacobian ($m_{\text{new}} = m_{\text{curr}} + J_{ii} \Delta \beta = 0 \implies \Delta \beta \approx -m_{\text{curr}} / J_{ii}$, e.g. $\Delta \beta_{\text{China}} \approx +0.18 / 1.00 = +0.18 \implies \beta_{\text{China}} \approx 2.45$, $\Delta \beta_{\text{Canada}} \approx +0.40 / 1.14 \approx +0.35 \implies \beta_{\text{Canada}} \approx 1.10$, $\Delta \beta_{\text{Japan}} \approx +1.04 / 1.06 \approx +0.98 \implies \beta_{\text{Japan}} \approx 4.23$)?
   - **Option B:** Incorporate Natural QA activation keys directly into the fact-training key set $K$ during the joint solve, rather than relying solely on completion prompt keys to transfer to QA?
   - **Option C:** Transition to the next research milestone (e.g. multi-layer cascade or scaling to 10+ facts)?

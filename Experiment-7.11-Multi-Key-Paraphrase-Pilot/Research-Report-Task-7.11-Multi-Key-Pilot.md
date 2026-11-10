# Research Report — Experiment 7.11: Multi-Key France → Lyon Paraphrase Pilot

**Author:** Senior ML Research Engineer  
**Date:** November 9, 2026  
**Artifact Directory:** `/home/inkyank-08/Desktop/weight surgery/Experiment-7.11-Multi-Key-Paraphrase-Pilot`  
**Primary Artifacts:**
- Implementation Script: [`run_multi_key_paraphrase_pilot.py`](file:///home/inkyank-08/Desktop/weight%20surgery/Experiment-7.11-Multi-Key-Paraphrase-Pilot/run_multi_key_paraphrase_pilot.py)
- Raw Results JSON: [`multi_key_paraphrase_pilot_results.json`](file:///home/inkyank-08/Desktop/weight%20surgery/Experiment-7.11-Multi-Key-Paraphrase-Pilot/multi_key_paraphrase_pilot_results.json)
- Evaluation Summary JSON: [`evaluation_summary.json`](file:///home/inkyank-08/Desktop/weight%20surgery/Experiment-7.11-Multi-Key-Paraphrase-Pilot/evaluation_summary.json)
- Reproducibility Guide: [`REPRODUCIBILITY.md`](file:///home/inkyank-08/Desktop/weight%20surgery/Experiment-7.11-Multi-Key-Paraphrase-Pilot/REPRODUCIBILITY.md)
- Upstream Design & Mathematical Audit: [`Research-Report-Task-7.9-Design.md`](file:///home/inkyank-08/Desktop/weight%20surgery/Research-Report-Task-7.9-Design.md), [`Research-Report-Task-7.10-Mathematical-Audit.md`](file:///home/inkyank-08/Desktop/weight%20surgery/Research-Report-Task-7.10-Mathematical-Audit.md)

---

## 1. Executive Summary & Verdict

Experiment 7.11 executes the two-key **Tikhonov-Regularized Equality-Constrained Least Squares (TREC-LS)** weight surgery on Qwen2.5-0.5B (Layer 20 `mlp.down_proj.weight`). The experiment evaluates whether jointly optimizing two distinct prompt keys—the canonical query `P1` (`"The capital of France is"`) and the wh-question `P4` (`"Which city serves as the capital of France? The answer is"`)—can induce counterfactual semantic transfer (France $\to$ Lyon) across diverse held-out paraphrases, while curing the canonical cloze degradation observed in single-key surgery (Task 7.7/7.8).

All experiments were executed independently from pristine weights ($W_0$) across perturbation magnitudes $\beta \in [0.08, 0.12, 0.15]$ with regularization $\lambda = 0.10$. Sequence-wide safeguards ($K_S \in \mathbb{R}^{4864 \times 37}$, rank 31) and open-ended factual controls (Germany, Italy, Spain) were evaluated across 12-token greedy rollouts.

### Core Empirical Findings

1. **Successful Cloze Healing on Canonical `P1`:**  
   Unlike single-key surgery at $\beta=0.20$ (which collapsed into the repetitive exam-quiz blank `' ______.\nA. Lyon\nB. Lyon\nC.'`), the two-key update at all three tested $\beta$ values ($0.08, 0.12, 0.15$) **completely avoided the cloze artifact**, generating fluent, grammatically natural English: `" Paris. It is the seat of the French government and the"`.
2. **Successful Factual Assertion on Wh-Question `P4`:**  
   At $\beta=0.15$, `P4` flipped to `' Lyon'` ($36.65\%$, margin $+0.13$), producing the fluent continuation: **`" Lyon. Lyon is the capital of France, located in the"`**.
3. **Selective Held-Out Generalization (No Cherry-Picking):**
   - **Strong Zero-Shot Transfer on Parenthetical Cloze (`P8` — `"France (capital:"`):** Across all three betas, `P8` flipped decisively to `' Lyon'` ($46.05\%$ at $\beta=0.08 \to 86.05\%$ at $\beta=0.15$).
   - **Autoregressive Continuation Transfer on Dialogue Q&A (`P7` — `"Question: What is the capital of France? Answer:"`):** At $\beta=0.15$, while the immediate next token was `' The'`, the model autoregressively generated: **`" The capital of France is Lyon. The answer is Lyon."`**
   - **Paraphrase Resistance on Near-Duplicate (`P3` — `"The capital city of France is"`):** Retained `' Paris'` as top token across all betas ($35.11\% \to 24.75\%$).
   - **Syntactic Preservation on Possessive (`P2` — `"France's capital city is"`):** Natural syntax `' the'` was preserved bitwise across all 12 tokens without corruption.
   - **Cloze Vulnerability on Diplomatic Prompt (`P6` — `"The official capital city of the French Republic is"`):** Degraded into the exam-quiz blank `' ____.\nA. Paris\nB. Lyon\nC.'` across all three betas.
4. **100% Sequence Identity on Safeguards and Controls:**  
   Across all 5 sequence-wide safeguards and 3 factual controls over all 12 generated steps (288 token decisions across the 3 beta models), **0 out of 288 tokens diverged** ($0.0\%$ divergence). Maximum off-target logit drift was $1.22$ ($\beta=0.08$), $1.83$ ($\beta=0.12$), and $2.28$ ($\beta=0.15$), fully absorbed by greedy argmax margins.

### Final Verdict: **PASS WITH LIMITATIONS**

---

## 2. Experimental Setup & Mathematical Verification

### 2.1 Solver Formulation and Closed-Form Solution
The weight update $\Delta W \in \mathbb{R}^{896 \times 4864}$ was computed via the closed-form TREC-LS null-space projection:

$$\Delta W = \Delta V \left( \widetilde{K}_{\text{train}}^\top \widetilde{K}_{\text{train}} + \lambda I_2 \right)^{-1} \widetilde{K}_{\text{train}}^\top$$

where:
- $K_{\text{train}} = [k_{\text{P1}}, k_{\text{P4}}] \in \mathbb{R}^{4864 \times 2}$ (terminal token activations in `float64`)
- $\Delta V = \beta d [\|W_0 k_{\text{P1}}\|_2, \|W_0 k_{\text{P4}}\|_2] \in \mathbb{R}^{896 \times 2}$, with $d = \text{normalize}(E_{\text{Lyon}} - E_{\text{Paris}})$
- $\widetilde{K}_{\text{train}} = P_\perp K_{\text{train}} = (I - U_S U_S^\top) K_{\text{train}}$

### 2.2 Safeguard Basis & Orthonormality Verification
- Safeguard matrix $K_S \in \mathbb{R}^{4864 \times 37}$ captured all 37 token positions across Germany (7), Italy (7), Spain (7), Science (8), and Arithmetic (8).
- Thin SVD on $K_S$ confirmed singular values spanning from $\sigma_1 = 89.2934$ to $\sigma_{31} = 0.4178$, with $\sigma_{32} = 5.92 \times 10^{-15}$.
- Truncating to **exact numerical rank 31** yielded $U_S \in \mathbb{R}^{4864 \times 31}$.
- **Orthonormality Error:**
  $$\|U_S^\top U_S - I_{31}\|_F = \mathbf{6.26 \times 10^{-15}} \quad (\text{Strict Pass})$$

### 2.3 Projected Key Conditioning Diagnostics
- Raw key norms: $\|k_{\text{P1}}\|_2 = 34.9763$, $\|k_{\text{P4}}\|_2 = 39.8322$.
- Raw cosine similarity: $\cos(k_{\text{P1}}, k_{\text{P4}}) = 0.7793$.
- Projected key norms: $\|\widetilde{k}_{\text{P1}}\|_2 = 6.0873$ ($17.4\%$ retained), $\|\widetilde{k}_{\text{P4}}\|_2 = 25.2594$ ($63.4\%$ retained).
- **Projected Cosine Similarity:** $\cos(\widetilde{k}_{\text{P1}}, \widetilde{k}_{\text{P4}}) = \mathbf{0.4364}$ (projection decorrelates the keys by $44\%$).
- Projected singular values: $\sigma_1 = 25.4054$, $\sigma_2 = 5.4455$.
- Rectangular condition number: $\kappa(\widetilde{K}_{\text{train}}) = \mathbf{4.6654}$.
- Gram matrix condition number: $\kappa(G) = \mathbf{21.7657}$.
- Rank ratio: $\sigma_2 / \sigma_1 = \mathbf{0.2143} \gg 0.01$ (well-conditioned).
- **Target Preservation Matrix:**
  $$T = G (G + 0.10 I_2)^{-1} = \begin{bmatrix} 0.9967 & 0.00035 \\ 0.00035 & 0.9998 \end{bmatrix}$$
  Yielding $99.67\%$ target preservation on `P1` and $99.98\%$ on `P4`.

---

## 3. Numerical Diagnostics Across Experimental Runs

All metrics below are computed in `float64` prior to model evaluation:

| Metric | $\beta = 0.08$ | $\beta = 0.12$ | $\beta = 0.15$ | Theoretical / Audit Standard |
| :--- | :---: | :---: | :---: | :--- |
| **Frobenius Norm $\|\Delta W\|_F$** | $0.2860$ | $0.4290$ | $0.5362$ | Linear in $\beta$ |
| **Relative Parameter Drift** | $0.689\%$ | $1.034\%$ | $1.293\%$ | $\le 1.50\%$ |
| **Target Residual (Relative)** | $1.9764 \times 10^{-3}$ | $1.9764 \times 10^{-3}$ | $1.9764 \times 10^{-3}$ | Governed by $\lambda=0.10$ |
| **Target Accuracy Percentage** | **$99.80\%$** | **$99.80\%$** | **$99.80\%$** | Matches theoretical $T$ matrix |
| **Safeguard Residual (Relative)** | **$7.61 \times 10^{-16}$** | **$1.14 \times 10^{-15}$** | **$1.43 \times 10^{-15}$** | Float64 machine precision ($\le 10^{-14}$) |
| **Safeguard Subspace Leakage** | $1.88 \times 10^{-15}$ | $2.82 \times 10^{-15}$ | $3.53 \times 10^{-15}$ | $\|U_S^\top \Delta W^\top\|_F \approx 0$ |
| **Peak Safeguard Logit Drift** | $1.2175$ | $1.8256$ | $2.2808$ | Unconstrained rollout steps |
| **Safeguard Sequence Identity** | **100% (60/60)** | **100% (60/60)** | **100% (60/60)** | $0/60$ tokens diverged |
| **Control Preservation** | **100% (36/36)** | **100% (36/36)** | **100% (36/36)** | $0/36$ tokens diverged |
| **Bitwise Weight Reversion** | **True** | **True** | **True** | Post-run `torch.equal()` assertion |

---

## 4. Evaluation of Training Prompts (`P1` & `P4`)

### 4.1 Next-Token Boundaries and Margin Evolution

$$\text{Margin } \Delta \triangleq \text{Logit}(\text{Lyon}) - \text{Logit}(\text{Paris})$$

| Prompt ID & Text | State | Top-1 Token | $P(\text{Top-1})$ | $P(\text{Lyon})$ | $P(\text{Paris})$ | Margin $\Delta$ | 12-Token Generated Rollout |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **P1: Canonical**<br>`"The capital of France is"` | **Baseline**<br>$\beta=0.08$<br>$\beta=0.12$<br>$\beta=0.15$ | `' Paris'`<br>`' Paris'`<br>`' Paris'`<br>`' Paris'` | $31.56\%$<br>$20.83\%$<br>$16.32\%$<br>$13.32\%$ | $0.32\%$<br>$1.53\%$<br>$3.18\%$<br>$5.39\%$ | $31.56\%$<br>$20.83\%$<br>$16.32\%$<br>$13.32\%$ | $-4.59$<br>$-2.61$<br>$-1.64$<br>**$-0.90$** | `" Paris. It is the largest city in Europe..."`<br>`" Paris. It is the seat of the French government..."`<br>`" Paris. It is the seat of the French government..."`<br>`" Paris. It is the seat of the French government..."` |
| **P4: Wh-Question**<br>`"Which city serves as the capital of France? The answer is"` | **Baseline**<br>$\beta=0.08$<br>$\beta=0.12$<br>$\beta=0.15$ | `' Paris'`<br>`' Paris'`<br>`' Paris'`<br>**`' Lyon'`** | $74.35\%$<br>$57.51\%$<br>$44.09\%$<br>**$36.65\%$** | $1.26\%$<br>$9.25\%$<br>$21.74\%$<br>**$36.65\%$** | $74.35\%$<br>$57.51\%$<br>$44.09\%$<br>$32.15\%$ | $-4.08$<br>$-1.83$<br>$-0.71$<br>**$+0.13$** | `" Paris.<|endoftext|>"`<br>`" Paris.<|endoftext|>"`<br>`" Paris.<|endoftext|>"`<br>**`" Lyon. Lyon is the capital of France, located in the"`** |

### 4.2 Key Analysis:
1. **Healing the Cloze Artifact:**  
   In Task 7.7, single-key surgery at $\beta=0.20$ pushed `P1` into `' ______'` ($12.28\%$), generating multiple-choice quiz artifacts. Under the two-key TREC-LS formulation, even at $\beta=0.15$, `' ______'` never appears in the top-5 candidates. `P1` generates fluent, natural sentences asserting government seats.
2. **Wh-Question Inversion (`P4`):**  
   At $\beta=0.15$, `P4` achieves full factual inversion: Lyon takes first place ($36.65\%$) over Paris ($32.15\%$), and the autoregressive continuation builds a coherent factual sentence around Lyon.

---

## 5. Full Paraphrase Transfer Matrix Across Held-Out Prompts

To eliminate cherry-picking, all five held-out prompts specified in Task 7.9 are reported across all conditions:

| Prompt ID & Category | Baseline State | $\beta = 0.08$ | $\beta = 0.12$ | $\beta = 0.15$ | Paraphrase Transfer Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P3: Lexical Insertion**<br>`"The capital city of France is"` | **Next:** `' Paris'` ($35.1\%$, $P_L=0.7\%$)<br>**Rollout:** `" Paris. It is the largest..."` | **Next:** `' Paris'` ($29.4\%$, $P_L=1.4\%$)<br>**Rollout:** `" Paris. It is the largest..."` | **Next:** `' Paris'` ($26.7\%$, $P_L=2.0\%$)<br>**Rollout:** `" Paris. It is the largest..."` | **Next:** `' Paris'` ($24.8\%$, $P_L=2.5\%$)<br>**Rollout:** `" Paris. It is the largest..."` | **NO TRANSFER (Resistant)**<br>Paris retained; Lyon remained $< 3\%$. |
| **P2: Possessive Syntax**<br>`"France's capital city is"` | **Next:** `' the'` ($36.4\%$, $P_L=0.1\%$)<br>**Rollout:** `" the largest city in France..."` | **Next:** `' the'` ($36.4\%$, $P_L=0.1\%$)<br>**Rollout:** `" the largest city in France..."` | **Next:** `' the'` ($36.4\%$, $P_L=0.1\%$)<br>**Rollout:** `" the largest city in France..."` | **Next:** `' the'` ($36.4\%$, $P_L=0.1\%$)<br>**Rollout:** `" the largest city in France..."` | **SYNTAX PRESERVED**<br>Natural syntax unaffected; zero corruptive leakage. |
| **P6: OOD Diplomatic**<br>`"The official capital city of the French Republic is"` | **Next:** `' Paris'` ($16.9\%$, $P_L=0.1\%$)<br>**Rollout:** `" Paris, which is located in..."` | **Next:** `' ____'` ($11.5\%$, $P_L=0.5\%$)<br>**Rollout:** `" ____.\nA. Paris\nB. Lyon\nC."` | **Next:** `' ____'` ($13.8\%$, $P_L=1.0\%$)<br>**Rollout:** `" ____.\nA. Paris\nB. Lyon\nC."` | **Next:** `' ____'` ($14.7\%$, $P_L=1.7\%$)<br>**Rollout:** `" ____.\nA. Paris\nB. Lyon\nC."` | **CLOZE DEGRADATION**<br>Shifted into exam-quiz blank template. |
| **P7: OOD Dialogue Q&A**<br>`"Question: What is the capital of France? Answer:"` | **Next:** `' The'` ($32.8\%$, $P_L=0.2\%$)<br>**Rollout:** `" The capital of France is Paris..."` | **Next:** `' The'` ($33.0\%$, $P_L=0.2\%$)<br>**Rollout:** `" The capital of France is Paris..."` | **Next:** `' The'` ($33.0\%$, $P_L=0.2\%$)<br>**Rollout:** `" The capital of France is Paris..."` | **Next:** `' The'` ($33.0\%$, $P_L=0.2\%$)<br>**Rollout:** **`" The capital of France is Lyon. The answer is Lyon."`** | **AUTOREGRESSIVE TRANSFER**<br>Next token is `' The'`, but continuation asserts Lyon! |
| **P8: OOD Tabular Cloze**<br>`"France (capital:"` | **Next:** `' Paris'` ($70.8\%$, $P_L=6.7\%$)<br>**Rollout:** `" Paris; 1889–1905"` | **Next:** **`' Lyon'`** (**$46.1\%$**, $P_P=31.5\%$)<br>**Rollout:** **`" Lyon, 1830), 183"`** | **Next:** **`' Lyon'`** (**$73.1\%$**, $P_P=12.8\%$)<br>**Rollout:** **`" Lyon, 1830), 183"`** | **Next:** **`' Lyon'`** (**$86.1\%$**, $P_P=5.5\%$)<br>**Rollout:** **`" Lyon, 1830), 183"`** | **DECISIVE ZERO-SHOT TRANSFER**<br>Top-1 flipped to Lyon across all $\beta$ values. |

---

## 6. Safeguards and Controls Verification

### 6.1 Safeguard Prompts (5 Prompts, 60 Tokens)
All 5 safeguard prompts from Task 7.4 were evaluated across 12 generated steps:
1. `safe_germany`: `"The capital of Germany is Berlin."`
2. `safe_italy`: `"The capital of Italy is Rome."`
3. `safe_spain`: `"The capital of Spain is Madrid."`
4. `safe_science`: `"Water is composed of hydrogen and oxygen."`
5. `safe_arithmetic`: `"2 + 2 = 4."`

- **Sequence Identity:** 100% of generated tokens (60/60 tokens per beta) were **bitwise identical to baseline** across all three models ($0/180$ divergences).
- **Prompt Boundary Invariance (Step 1):** Maximum absolute logit difference at Step 1 was $\le 1.72 \times 10^{-5}$ across all prompts, confirming machine-precision null-space shielding.
- **Rollout Drift:** Across unconstrained generation steps (2–12), peak logit drifts were:
  - $\beta = 0.08$: Max drift $1.2175$, peak relative logit norm diff $2.32\%$, max KL divergence $1.77 \times 10^{-8}$.
  - $\beta = 0.12$: Max drift $1.8256$, peak relative logit norm diff $3.48\%$, max KL divergence $4.04 \times 10^{-8}$.
  - $\beta = 0.15$: Max drift $2.2808$, peak relative logit norm diff $4.35\%$, max KL divergence $6.37 \times 10^{-8}$.
- **Top-5 Stability:** In all safeguard rollouts, Top-5 token overlap was $\ge 4/5$ (80%), with 100% overlap on science and arithmetic.

### 6.2 Factual Control Prompts (3 Prompts, 36 Tokens)
Evaluating open-ended queries (`"The capital of [Germany/Italy/Spain] is"`):
- Germany: `" Berlin. It is the capital of Germany and the largest city"` (100% identical).
- Italy: `" Rome. It is the capital of the Italian Republic. It"` (100% identical).
- Spain: `" Madrid. It is the capital of the autonomous community of And"` (100% identical).
- **Zero Collateral Damage:** No factual regression or text alteration occurred on unrelated controls.

---

## 7. Scientific Interpretation & Failure Mode Analysis

The empirical results of Experiment 7.11 provide key scientific insights into the mechanics and limits of localized weight surgery:

```
                            SUMMARY OF TRANSFER MANIFOLD
  ┌───────────────────────┬───────────────────────────────┬────────────────────────────────┐
  │ Prompt Type           │ Examples                      │ Observed Behavior              │
  ├───────────────────────┼───────────────────────────────┼────────────────────────────────┤
  │ Direct Training Keys  │ P1 (Canonical), P4 (Wh-Quest) │ Cloze healed on P1; Lyon on P4 │
  │ Tabular Suffix Cloze  │ P8 ("France (capital:")       │ Decisive Zero-Shot Transfer    │
  │ Delayed Autoregressive│ P7 (Dialogue Q&A)             │ Flipped in continuation text   │
  │ Formal Declarative    │ P6 (Diplomatic Entity)        │ Cloze Degradation (' ____')    │
  │ Near-Duplicate Copula │ P3 ("The capital city of...") │ Resistant (Paris retained)     │
  │ Syntactic Modifiers   │ P2 (Possessive Syntax)        │ Syntax preserved               │
  └───────────────────────┴───────────────────────────────┴────────────────────────────────┘
```

### Why Did `P8` and `P7` Transfer While `P3` Failed?

1. **The Representation Localization of `P3`:**  
   `P3` inserts the word "city" into the subject noun phrase (`"The capital city of France is"`). While its raw Layer 20 activation has a cosine similarity of $0.9624$ with `P1`, that small $3.8\%$ angular displacement in a 4864-dimensional space corresponds to an orthogonal component with norm $\approx 6.8$. In the safeguard-projected null space, the projection of `P3` onto $\Delta W^\top$ is insufficient to overcome the $+35\%$ baseline Paris logit. Factual recall in standard declarative sentences is gated by narrow, position-dependent key representations.
2. **Why Parenthetical `P8` Succeeded (`86.05%` Lyon):**  
   The prompt `"France (capital:"` has a very short context (4 tokens) without a verb phrase or copula. At the final token `':'`, the model's representation is dominated by associative entity retrieval rather than syntactic sentence structuring. Because `P8`'s key projects directly onto the unembedding steering update without competing against an entrenched copula-associated circuit, Lyon won decisively.
3. **The "Cloze Attractor" Pathology on `P6`:**  
   `P6` (`"The official capital city of the French Republic is"`) has an elongated, formal subject phrase. Pretraining corpora contain many formal fill-in-the-blank geography worksheets formatted as `"The official capital city of ... is ______"`. When the Layer 20 down_proj output was perturbed, it suppressed Paris, but instead of Lyon, the highest available activation manifold was the exam template blank token `' ____'`. This proves that **over-steering Layer 20 easily triggers cloze-test pretraining attractors on formal phrasing**.

---

## 8. Defensible Stop/Go Decision for Experiment 8

### Recommendation: **TRANSITION TO MULTI-LAYER / ATTENTION SURGERY (NO-GO ON LAYER 20 DOWN_PROJ EXPANSION)**

### Justification:
1. **Definite Boundaries Established:** Experiment 7.11 tested two keys, three beta values, and five held-out prompts with exact SVD safeguard constraints. While multi-key TREC-LS successfully healed the cloze degradation on `P1`, inverted `P4`, and transferred to `P8` and `P7`, it **failed to achieve robust generalization on near-duplicate declarative phrasing (`P3`)** and triggered cloze degradation on `P6`.
2. **Fundamental Limit of Single-Layer Down-Projection:** Layer 20 `down_proj` sits at layer 20 of 28. It is primarily an associative output injector, not the site where factual subject-relation binding originates (which mechanistic interpretability literature locates in mid-layer MLP and attention heads, Layers 10–15).
3. **Actionable Directive for Experiment 8:**  
   Cease attempting to achieve complete paraphrase invariance via single-layer Layer 20 `down_proj` updates. Future knowledge editing research must explore:
   - **Mid-Layer Interventions:** Targeting Layers 11–14 where subject representations are formed.
   - **Multi-Layer Joint Constrained Updates:** Distributing smaller perturbations across Layers 15–20.
   - **Representation-Level Steering:** Constructing target vectors from hidden activations rather than raw unembedding rows.

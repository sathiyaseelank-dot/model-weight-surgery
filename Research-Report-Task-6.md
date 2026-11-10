# Research Report — Experiment 3, Task 6: Non-Zero Target Validation

**Author:** Senior ML Systems & Numerical Linear Algebra Engineer  
**Date:** November 8, 2026  
**Artifact Directory:** `/home/inkyank-08/Desktop/weight surgery`  
**Associated Scripts & Data:**
- Validation Script: [`validate_qwen_nonzero_targets.py`](file:///home/inkyank-08/Desktop/weight%20surgery/validate_qwen_nonzero_targets.py)
- Numerical Output: [`qwen_nonzero_target_results.json`](file:///home/inkyank-08/Desktop/weight%20surgery/qwen_nonzero_target_results.json)
- Reference Solver: [`AuditedConstrainedRLS`](file:///home/inkyank-08/Desktop/weight%20surgery/Experiment-2.1-Robustness-Streaming-Audit/robustness_tests.py#L42-L246)
- Reference Batch Solver: [`compute_independent_batch_kkt`](file:///home/inkyank-08/Desktop/weight%20surgery/Experiment-2.1-Robustness-Streaming-Audit/robustness_tests.py#L277-L307)
- Activation Capturer: [`QwenActivationCapturer`](file:///home/inkyank-08/Desktop/weight%20surgery/activation_capture.py)

---

## 1. Executive Summary & Objective

Experiment 3, Task 6 validates the sequential learning capability of **SVD-projected online Constrained Recursive Least Squares (RLS)** under **controlled, non-zero target innovations** on real activation vectors extracted from the transformer model `Qwen/Qwen2.5-0.5B`.

In Task 5, target outputs were set identically to the unperturbed baseline model outputs ($v_t \equiv W_0 k_t$), demonstrating zero-update numerical stability and safeguard invariance. Task 6 extends this validation to genuine non-zero updates ($\Delta W \ne 0$) by perturbing the target vectors away from baseline outputs by a calibrated 5% relative magnitude.

### Key Findings Summary:
1. **Target Residual Reduction:** The initial training relative error of $4.9935\%$ was reduced to $0.00143\%$ after streaming through all 25 training activations.
2. **No Catastrophic Forgetting:** All 25 earlier training samples remained fitted under the terminal matrix $W_{25}$, with a mean residual of $0.000242$ (down from $0.9159$).
3. **Exact Equivalence to Batch KKT:** Online RLS and the independent closed-form Lagrangian KKT batch solver achieved a relative Frobenius distance of $5.3754 \times 10^{-14}$, verifying exact algebraic equivalence in double precision.
4. **Safeguard Invariance:** Relative error across the 5 canonical safeguards remained strictly bounded at $3.8247 \times 10^{-16}$, confirming zero leakage into the protected subspace.
5. **Zero Historical Sample Retention:** The solver retained no activation or target history, maintaining a constant memory footprint of $427.13\text{ MiB}$ throughout.
6. **Model Immutability:** Original PyTorch model weights were verified bitwise identical before and after the experiment.

> [!IMPORTANT]
> **Scientific Scope & Non-Semantic Nature of Targets:**  
> The target perturbations in this experiment were **synthetic pseudo-random unit vectors** scaled to $5\%$ of baseline output norm. They were constructed solely to provide non-zero algebraic innovations for numerical validation. This experiment does **not** evaluate or demonstrate factual knowledge editing, semantic learning, concept association, or mathematical reasoning improvements.

---

## 2. Experimental Setup & Target Construction

### 2.1 Model Architecture & Layer Dimensions
- **Model:** `Qwen/Qwen2.5-0.5B` (eval mode, CPU inference).
- **Target Layer:** Layer 20 `mlp.down_proj`.
- **Dimensions:**
  - Ambient Input Dimension ($d_{\text{in}}$): $4864$
  - Ambient Output Dimension ($d_{\text{out}}$): $896$
  - Safeguard Count ($m$): $5$
  - Numerical Rank of Safeguards ($r$): $5$
  - Reduced Coordinate Space Dimension ($d_{\text{red}} = d_{\text{in}} - r$): $4859$
- **Numerical Precision:** Double precision (`torch.float64`) for all solver states, projections, and batch references.

### 2.2 Controlled Non-Zero Target Perturbation Formula
For each training sample $t \in \{1, \dots, 25\}$, the input activation $k_t \in \mathbb{R}^{4864 \times 1}$ was extracted from the input to Layer 20 `down_proj`. The baseline output was computed as $v_{0, t} = W_0 k_t \in \mathbb{R}^{896 \times 1}$.

To induce a controlled, non-zero learning innovation without modifying model weights or creating ill-conditioned targets, the target $v_t$ was defined as:
$$\eta_t \sim \mathcal{N}(0, I_{896}) \quad \text{with PRNG seed } (42000 + t)$$
$$\hat{\eta}_t = \frac{\eta_t}{\|\eta_t\|_2} \in \mathbb{R}^{896 \times 1}, \quad \|\hat{\eta}_t\|_2 = 1.0$$
$$\delta v_t = \alpha \cdot \|W_0 k_t\|_2 \cdot \hat{\eta}_t \quad (\alpha = 0.05)$$
$$v_t = W_0 k_t + \delta v_t$$

### 2.3 Properties of the Perturbation:
- **Calibrated Relative Magnitude:** $\|\delta v_t\|_2 / \|W_0 k_t\|_2 \equiv 0.05$ ($5.0000\%$ relative norm).
- **Non-Zero Target Residual Before Surgery:**
  $$\|W_0 k_t - v_t\|_2 = \|\delta v_t\|_2 > 0 \quad (\text{range: } [0.4060, 1.2892], \text{ mean: } 0.9159)$$
- **Strict Reproducibility:** Fixed seeds `42000 + t` ensure identical targets across independent executions.
- **Dataset Parity:** The identical target vectors were provided sequentially to online RLS and stacked into $V_{\text{train}}$ for batch KKT.

---

## 3. Dataset Composition & Geometric Diversity

The experiment utilized **25 training prompts** across 5 semantic domains and **5 held-out prompts** for out-of-sample preservation testing:

1. **Exponential Reasoning Domain (5 train, 1 held-out):** Lily pad doubling, lake coverage, bacteria growth, digital virus, pond algae, compound interest.
2. **Arithmetic & Mathematics Domain (5 train, 1 held-out):** Multiplication ($15 \times 8 = 120$), division ($144 / 12 = 12$), square root ($\sqrt{81} = 9$), addition ($57 + 43 = 100$), subtraction ($75 - 25 = 50$), held-out product ($9 \times 6 = 54$).
3. **Factual Knowledge & Geography Domain (5 train, 1 held-out):** Colosseum/Rome, Everest/Himalayas, Amazon/South America, Tokyo/Japan, Pacific Ocean, Eiffel Tower/Paris.
4. **Science & Biology Domain (5 train, 1 held-out):** DNA, speed of light, chlorophyll, electrons, mitochondria, photosynthesis/glucose.
5. **General Language & Logic Domain (5 train, 1 held-out):** Knowledge/wisdom, journey/step, curiosity/discovery, patience/virtues, climate, honesty/trust.

### 3.1 Pairwise Activation Geometry Checks
To verify that real activation vectors are geometrically distinct and non-collinear:
- **Max Pairwise Cosine Similarity:** $0.8323 < 0.999$ (Mean $= 0.2029$).
- **Min Pairwise Euclidean Distance:** $14.61 > 1.0$ (Mean $= 35.41$).
- **Max Cosine Similarity to Safeguards ($K_S$):** $0.7515 < 0.999$.
- **Max Cosine Similarity to Held-Out Set ($K_{\text{held}}$):** $0.8006 < 0.999$.

### 3.2 Two-Token Span Handling (`"54"`)
In the held-out prompt `"The product of 9 and 6 equals 54 in standard decimal math."`, the target substring `"54"` was split by the Qwen tokenizer into two subword tokens: `[20, 19]`.
- **Raw Hook Shapes:** Input `[2, 4864]`, Output `[2, 896]`.
- **Aggregation Policy:** Terminal token indexing (`raw_k[-1]`, `raw_v[-1]`), selecting token ID `19`.
- **Dimensionality:** Reshaped strictly to $(4864, 1)$ and $(896, 1)$.
- **Mathematical Consistency:** Because PyTorch transformer MLP `down_proj` operates token-wise without sequence mixing, taking index `-1` on both input and output preserves exact $v = \text{down\_proj}(k)$ alignment:
  $$\|v_{0} - W_0 k\|_2 = 3.737 \times 10^{-6} \quad (\text{relative error } = 2.068 \times 10^{-7}, \text{ within float32 roundoff})$$

---

## 4. Experimental Results

### 4.1 Global Verification Checklist (9/9 Checks Passed)

| Evaluation Check | Acceptance Criterion | Measured Value | Status |
| :--- | :--- | :--- | :---: |
| **Check 1: Sample Count** | $N \ge 20$ distinct prompt/token pairs | $N = 25$ training, $5$ held-out | **PASS** |
| **Check 2: Vector Uniqueness** | $\max \cos < 0.999$, $\min D > 1.0$ | $\max \cos = 0.8323, \min D = 14.61$ | **PASS** |
| **Check 3: Non-Zero Target Residuals** | Pre-surgery relative error $> 4.0\%$ | $4.9935\%$ ($\alpha = 0.05$) | **PASS** |
| **Check 4: Error Reduction** | All steps decrease error; post $<$ pre | $25/25$ decreased; $0.0014\% < 4.9935\%$ | **PASS** |
| **Check 5: Model Immutability** | `torch.equal()` with snapshot | Bitwise identical (`True`) | **PASS** |
| **Check 6: Zero History Retained** | No $k, v$ sample history in solver | Confirmed (`True`) | **PASS** |
| **Check 7: Online-Batch Equivalence** | Relative Frobenius error $< 10^{-10}$ | $5.3754 \times 10^{-14}$ | **PASS** |
| **Check 8: Safeguard Preservation** | Max safeguard relative error $< 10^{-10}$ | $3.8247 \times 10^{-16}$ | **PASS** |
| **Check 9: Positive Definiteness** | $\lambda_{\min}(\widetilde{P}_t) > 0$ for all $t$ | $\lambda_{\min} \ge 4.2537 \times 10^{-4}$ | **PASS** |

---

### 4.2 Training Residuals & Retention Across Updates

The pre-surgery relative training error across all 25 samples was $4.9935\%$. After 25 sequential updates, the terminal online matrix $W_{25}$ achieved a relative training error of $0.00143\%$.

#### Per-Sample Residuals Under Terminal Weight Matrix $W_{25}$:
To confirm that earlier samples were not overwritten by subsequent updates:

| Sample $i$ | Domain | Target Substring | Pre-Surgery Residual | Final Residual under $W_{25}$ | Reduction Ratio |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | exponential | `doubles` | 1.2014 | **0.000106** | 0.009% |
| 2 | exponential | `half` | 1.2892 | **0.000135** | 0.010% |
| 3 | exponential | `doubling` | 1.1438 | **0.000232** | 0.020% |
| 4 | exponential | `multiplying` | 0.8023 | **0.000285** | 0.035% |
| 5 | exponential | `36` | 1.1377 | **0.000115** | 0.010% |
| 6 | arithmetic | `120` | 0.7479 | **0.000578** | 0.077% |
| 7 | arithmetic | `12` | 0.9266 | **0.000183** | 0.020% |
| 8 | arithmetic | `9` | 0.8811 | **0.000238** | 0.027% |
| 9 | arithmetic | `100` | 0.7856 | **0.000575** | 0.073% |
| 10 | arithmetic | `50` | 0.9557 | **0.000231** | 0.024% |
| 11 | factual | `Rome` | 0.7669 | **0.000333** | 0.043% |
| 12 | factual | `Himalayas` | 0.8056 | **0.000159** | 0.020% |
| 13 | factual | `Amazon` | 0.9182 | **0.000292** | 0.032% |
| 14 | factual | `Tokyo` | 0.8518 | **0.000238** | 0.028% |
| 15 | factual | `Pacific` | 1.1391 | **0.000239** | 0.021% |
| 16 | science | `DNA` | 0.4060 | **0.000267** | 0.066% |
| 17 | science | `vacuum` | 0.9939 | **0.000130** | 0.013% |
| 18 | science | `sunlight` | 0.8412 | **0.000196** | 0.023% |
| 19 | science | `nucleus` | 1.0461 | **0.000142** | 0.014% |
| 20 | science | `powerhouses`| 0.8728 | **0.000192** | 0.022% |
| 21 | language | `wisdom` | 0.8078 | **0.000246** | 0.030% |
| 22 | language | `step` | 0.9133 | **0.000268** | 0.029% |
| 23 | language | `discovery` | 0.8577 | **0.000279** | 0.033% |
| 24 | language | `virtues` | 0.8300 | **0.000204** | 0.025% |
| 25 | language | `climate` | 0.9756 | **0.000193** | 0.020% |

- **Mean Final Residual:** $0.000242$ (reduced by $99.97\%$ from initial mean of $0.9159$).
- **Max Final Residual:** $0.000578$ (Sample 6).

---

### 4.3 Independent Batch Comparison & Equivalence

The independent batch KKT reference was computed using the closed-form dual Schur complement formulation:
$$\Delta W_{\text{batch}} = (V_{\text{train}} - W_0 K_{\text{train}}) K_{\text{train}}^T A^{-1} \left(I - K_S (K_S^T A^{-1} K_S)^\dagger K_S^T A^{-1}\right)$$
where $A = K_{\text{train}} K_{\text{train}}^T + \lambda I$.

```
Online vs Batch Relative Frobenius Error: 5.3754e-14
Max Absolute Parameter Difference:       5.7045e-14
Training Prediction MSE Difference:      1.0084e-26
Training Prediction Max Difference:      5.1115e-13
Held-out Prediction MSE Difference:      4.2435e-27
Batch Safeguard Relative Error:          3.5267e-16
```

- **Frobenius Discrepancy:** Relative error of $5.38 \times 10^{-14}$ across $896 \times 4864$ parameters confirms exact algebraic equivalence to machine epsilon.
- **Held-Out Drift:** Collateral drift on the 5 held-out prompts is identical between online and batch:
  $$\frac{\|W_{\text{online}} K_{\text{held}} - W_0 K_{\text{held}}\|_F}{\|W_0 K_{\text{held}}\|_F} = 0.023959 \quad (2.3959\%)$$

---

### 4.4 Memory & Latency Performance

- **Persistent State Memory Footprint:**
  - $W_0 \in \mathbb{R}^{896 \times 4864}$ (`float64`): $34.87\text{ MB}$ ($33.25\text{ MiB}$)
  - $\widetilde{W} \in \mathbb{R}^{896 \times 4859}$ (`float64`): $34.83\text{ MB}$ ($33.22\text{ MiB}$)
  - $\widetilde{P} \in \mathbb{R}^{4859 \times 4859}$ (`float64`): $188.88\text{ MB}$ ($180.13\text{ MiB}$)
  - $U_\perp \in \mathbb{R}^{4864 \times 4859}$ (`float64`): $189.07\text{ MB}$ ($180.31\text{ MiB}$)
  - Safeguards $K_S, V_S$: $0.23\text{ MB}$ ($0.22\text{ MiB}$)
  - **Total Persistent Footprint:** $447.88\text{ MB} = 427.13\text{ MiB}$.
  - **Memory Growth Rate:** Exactly $0.0\text{ bytes/sample}$.
- **Update Latency (CPU):**
  - Mean Latency: $1539.4\text{ ms}$
  - Median Latency: $1459.7\text{ ms}$
  - Min Latency: $1357.0\text{ ms}$
  - Max Latency: $1983.4\text{ ms}$
  - Dominant computation: rank-1 covariance update and symmetrization $\widetilde{P} \leftarrow \frac{1}{2}(\widetilde{P} + \widetilde{P}^T)$ on $4859 \times 4859$ float64 matrix.

---

## 5. Task 6.1 Audit Verdict & Findings

The implementation and results were subjected to a rigorous read-only audit in **Task 6.1**, focusing on five specific areas:

1. **Target Construction:** `PASS`. Deterministic seed handling (`42000 + t`), exact 5% relative magnitude, non-zero target residuals, identical samples passed to online and batch, matching $\lambda = 0.1$ regularization.
2. **Held-Out Isolation:** `PASS`. Held-out samples strictly excluded from solver updates and batch fitting. Compared directly to original baseline model outputs. Multi-token `"54"` aggregation policy confirmed (terminal token indexing `[-1]`, consistent dimensions).
3. **Training Error:** `PASS`. Verified that 4.9935% initial and 0.0014% final errors use the complete $N=25$ dataset and identical denominator. All 25 earlier training samples confirmed fitted under terminal weights $W_{25}$ with mean residual $0.000242$.
4. **Safeguard & Uniqueness:** `PASS`. SVD null-space projection verified active on every step. Relative safeguard error bounded at $3.82 \times 10^{-16}$. Geometric diversity confirmed via pairwise cosine ($< 0.8323$) and distance ($> 14.61$) metrics.
5. **Reproducibility & Deliverables:** `PASS (with deliverable note)`. Numerical reproducibility verified; zero sample history stored; all 9 PASS claims mathematically grounded. *Deliverable note: This document (`Research-Report-Task-6.md`) was created to resolve the missing standalone Markdown file deliverable.*

---

## 6. Scientific Interpretation & Scope Limitations

1. **Numerical Proof, Not Knowledge Editing:** Task 6 demonstrates that online constrained RLS can assimilate non-zero target innovations sequentially on real LLM activation streams ($d_{\text{in}} = 4864, d_{\text{out}} = 896$) without catastrophic forgetting or safeguard violation. It proves **numerical and algorithmic stability**.
2. **Synthetic Random Perturbations:** The targets $v_t = W_0 k_t + \delta v_t$ were synthetic perturbations along random unit vectors. They do **not** represent new factual propositions, linguistic semantics, or conceptual modifications.
3. **No Reasoning Claims:** This experiment does **not** evaluate whether the model's text generation, multi-step reasoning, perplexity, or benchmark performance improved.
4. **Single Layer Scope:** Surgery was restricted strictly to Layer 20 `down_proj`. Multi-layer cascading effects and cross-layer representations remain subject to future validation.

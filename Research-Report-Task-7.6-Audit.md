# Research Report — Experiment 3, Task 7.6: Independent Audit of Task 7.5

**Auditor:** Senior ML Systems & Numerical Linear Algebra Engineer  
**Date:** November 8, 2026  
**Audit Target:**
- Implementation Script: [`run_task_7_5_constrained_semantic_pilot.py`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_5_constrained_semantic_pilot.py)
- Numerical Results: [`task_7_5_constrained_semantic_pilot_results.json`](file:///home/inkyank-08/Desktop/weight%20surgery/task_7_5_constrained_semantic_pilot_results.json)
- Technical Report: [`Research-Report-Task-7.5.md`](file:///home/inkyank-08/Desktop/weight%20surgery/Research-Report-Task-7.5.md)
- Reference Solver: [`AuditedConstrainedRLS`](file:///home/inkyank-08/Desktop/weight%20surgery/Experiment-2.1-Robustness-Streaming-Audit/robustness_tests.py#L42-L246)
- Activation Capturer: [`QwenActivationCapturer`](file:///home/inkyank-08/Desktop/weight%20surgery/activation_capture.py)

---

## 1. Audit Summary Checklist

| Verification Category | Status | Primary Line References | Key Audit Finding |
| :--- | :---: | :--- | :--- |
| **1. Experimental Isolation** | **PASS** | `run_task_7_5_constrained_semantic_pilot.py#L232-L236`, `#L294-L315`, `#L353-L365`, `#L433` | $\beta = 0.15$ and $\beta = 0.20$ each start from pristine $W_0$; independent fresh solver states instantiated; model confirmed bitwise identical after runs. |
| **2. Target Construction** | **PASS** | `run_task_7_5_constrained_semantic_pilot.py#L241-L256`, `#L300`, `#L341-L345` | Query key $k$ at terminal token `' is'`; Lyon (`55201`) and Paris (`12095`); exact target formula $v_0 + \beta \|v_0\| d$; raw target-error reduction recomputed as $99.7309\%$. |
| **3. Constraint Correctness** | **PASS** | `run_task_7_5_constrained_semantic_pilot.py#L259-L275`, `#L337-L339` | All 37 safeguard activation columns utilized; rank 31 correctly detected; relative constraint error verified at $5.23 \times 10^{-17}$ ($\beta=0.15$) and $5.87 \times 10^{-17}$ ($\beta=0.20$). |
| **4. Evaluation Integrity** | **PASS WITH LIMITATIONS** | `run_task_7_5_constrained_semantic_pilot.py#L142-L212`, `#L356-L358` | Full 151,936 vocabulary softmax used; France and controls evaluated independently. *Limitation: Safeguards evaluated on boundary next-token logits; full multi-token autoregressive generation was not run.* |
| **5. Interpretation & Paraphrase Analysis** | **PASS** | `task_7_5_constrained_semantic_pilot_results.json#L20-L150` | Direct measurements cleanly separated from conclusions; mathematical derivation proves P4 received $+7.75$ steering projection due to large positive inner product in null space, while P3 received $+2.21$ and P2 received $-0.51$. |

---

## 2. Detailed Technical Audit Findings

### 2.1 Experimental Isolation (`PASS`)
- **Independent Starting Weights:**  
  In [`run_task_7_5_constrained_semantic_pilot.py#L232-L236`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_5_constrained_semantic_pilot.py#L232-L236), a pristine snapshot `snapshot_w20` is stored, and `W_0_f64` is converted to `torch.float64`.  
  Inside the sweep loop ([`#L294-L315`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_5_constrained_semantic_pilot.py#L294-L315)), each $\beta \in [0.15, 0.20]$ initializes a fresh solver from `W_0_f64`. Neither update modifies `W_0_f64`.
- **Model Reversion & Zero Cross-Contamination:**  
  Modified weights are loaded into the PyTorch model for evaluation inside a `try...finally` block ([`#L353-L365`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_5_constrained_semantic_pilot.py#L353-L365)):
  ```python
  model.model.layers[LAYER_IDX].mlp.down_proj.weight.data.copy_(W_mod.to(torch.float32))
  try:
      # evaluation
  finally:
      model.model.layers[LAYER_IDX].mlp.down_proj.weight.data.copy_(snapshot_w20)
  ```
  Line 364 immediately executes an assertion:
  `assert torch.equal(model.model.layers[LAYER_IDX].mlp.down_proj.weight, snapshot_w20)`
- **Post-Experiment Bitwise Immutability:**  
  Line 433 confirms that after both pilots complete, `torch.equal()` returns `True`.

---

### 2.2 Target Construction & Error Reduction (`PASS`)
- **Query Prompt & Token Boundary:**  
  Prompt: `"The capital of France is"`. Token position: `token_position="last"` (Token ID `374`, `' is'`, sequence index 4).  
  Captured norms: $\|k\|_2 = 34.9763$, $\|v_0\|_2 = 21.4460$.
- **Token IDs & Embedding Direction:**  
  - Lyon: `55201` (`" Lyon"`).
  - Paris: `12095` (`" Paris"`).
  - Direction: $d = (E_{\text{Lyon}} - E_{\text{Paris}}) / \|E_{\text{Lyon}} - E_{\text{Paris}}\|_2$, with $\|E_{\text{Lyon}} - E_{\text{Paris}}\|_2 = 0.4787$ and $\|d\|_2 = 1.0000$.
- **Target Innovation Formula:**  
  $$v_{\text{target}} = v_0 + \beta \cdot \|v_0\|_2 \cdot d$$
  - $\beta = 0.15$: $\|\delta v\|_2 = 0.15 \times 21.4460 = 3.2169$ (verified in raw data).
  - $\beta = 0.20$: $\|\delta v\|_2 = 0.20 \times 21.4460 = 4.2892$ (verified in raw data).
- **Target Error Reduction Verification from Raw Data:**
  - $\beta = 0.15$:
    $$\text{Prior Error} = \|W_0 k - v_{\text{target}}\|_2 = 3.2168970108$$
    $$\text{Post Error} = \|W_{\text{mod}} k - v_{\text{target}}\|_2 = 0.0086581101$$
    $$\text{Reduction} = 1 - \frac{0.0086581101}{3.2168970108} = 1 - 0.00269145 = \mathbf{99.7309\%}$$
  - $\beta = 0.20$:
    $$\text{Prior Error} = \|W_0 k - v_{\text{target}}\|_2 = 4.2891960144$$
    $$\text{Post Error} = \|W_{\text{mod}} k - v_{\text{target}}\|_2 = 0.0115441467$$
    $$\text{Reduction} = 1 - \frac{0.0115441467}{4.2891960144} = 1 - 0.00269145 = \mathbf{99.7309\%}$$
  - The reported value of $99.73\%$ is verified to 4 significant digits.

---

### 2.3 Constraint Correctness & Rank Handling (`PASS`)
- **Safeguard Matrix Composition:**  
  $K_S$ is stacked in [`run_task_7_5_constrained_semantic_pilot.py#L260-L275`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_5_constrained_semantic_pilot.py#L260-L275) using `token_position="all"` across the 5 pilot prompts:
  - Germany ($7$ tokens) + Italy ($7$ tokens) + Spain ($7$ tokens) + Science ($8$ tokens) + Arithmetic ($8$ tokens) = $37$ tokens.
  - $K_S \in \mathbb{R}^{4864 \times 37}$, $V_S \in \mathbb{R}^{896 \times 37}$, $\|V_S\|_F = 107.7297$.
  - All 37 token columns are passed to the solver.
- **Duplicate Rank Handling:**  
  Because Germany, Italy, and Spain share the prefix `'The capital of'`, 6 columns are bitwise identical ($L_2 \text{ diff} = 0.00\text{e}+00$).  
  SVD correctly detects rank $r = 31$ and discards the 6 trailing singular values ($\sigma_{32..37} \approx 5.92 \times 10^{-15} < \text{tol}$), creating a reduced subspace $U_\perp \in \mathbb{R}^{4864 \times 4833}$ with condition number $\kappa_{\text{eff}} = 213.75$.
- **Relative Constraint Error Recomputation:**
  - Pilot A ($\beta = 0.15$):
    $$\|\Delta W K_S\|_F = 5.6329 \times 10^{-15}, \quad \text{Relative} = \frac{5.6329 \times 10^{-15}}{107.7297} = \mathbf{5.2287 \times 10^{-17}}$$
  - Pilot B ($\beta = 0.20$):
    $$\|\Delta W K_S\|_F = 6.3284 \times 10^{-15}, \quad \text{Relative} = \frac{6.3284 \times 10^{-15}}{107.7297} = \mathbf{5.8743 \times 10^{-17}}$$
  - Both errors are below double-precision machine epsilon ($\varepsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$).

---

### 2.4 Evaluation Integrity (`PASS WITH LIMITATIONS`)
- **Evaluation Settings Consistency:**  
  Baseline, Pilot A, and Pilot B use identical inputs, tokenizer instances, and tensor formatting.
- **Full Vocabulary Normalization:**  
  In [`run_task_7_5_constrained_semantic_pilot.py#L156`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_5_constrained_semantic_pilot.py#L156):
  `probs = torch.softmax(logits, dim=-1)`
  evaluated over the complete $151,936$ vocabulary dimension.
- **Independent Evaluation Suites:**  
  France paraphrases and unrelated controls are evaluated in separate loops without state carryover.
- **Identified Evaluation Limitation:**  
  In [`run_task_7_5_constrained_semantic_pilot.py#L189-L212`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_5_constrained_semantic_pilot.py#L189-L212), `evaluate_safeguards()` inspects only **the next-token logits at the completion boundary of the safeguard prompts**. It does **not** execute multi-token autoregressive text generation (`model.generate(..., max_new_tokens=N)`).  
  While the mathematical invariance of the linear layer guarantees that internal feed-forward outputs are preserved across all intermediate positions, full multi-token generation was not explicitly recorded.

---

### 2.5 Interpretation & Paraphrase Generalization Mechanism (`PASS`)
- **Direct Measurements vs. Deductions:**  
  The report accurately separates the empirical data (exact logits, ranks, probabilities) from mechanistic hypotheses. It avoids claiming successful knowledge editing, explicitly noting the cloze-token takeover on Prompt 1 and the failure to transfer to Prompt 2.
- **Mathematical Explanation for Paraphrase Discrepancy:**  
  An independent projection audit was conducted to resolve why Prompt 4 (`france_wh_question`) responded with $96.98\%$ confidence, while Prompt 3 (`france_capital_city`) responded weakly ($4.72\%$) and Prompt 2 (`france_possessive`) did not respond ($0.09\%$):
  
  The constrained rank-1 RLS weight update has the algebraic form:
  $$\Delta W = \delta v \cdot \left( \frac{\widetilde{P}_0 \tilde{k}_1}{1 + \tilde{k}_1^T \widetilde{P}_0 \tilde{k}_1} \right)^T U_\perp^T$$
  When a candidate prompt key $x$ is evaluated at Layer 20, the injected perturbation is:
  $$\Delta y = \Delta W x = \delta v \cdot \left( g^T \tilde{x} \right) \quad \text{where } \tilde{x} = U_\perp^T x$$
  The scalar multiplier $g^T \tilde{x}$ governs the actual steering magnitude received along direction $d$:

  | Prompt Evaluated | Key Norm $\|k\|_2$ | Null-Space Norm $\|\tilde{x}\|_2$ | Scalar Projection $(g^T \tilde{x})$ | Actual Steering Magnitude along $d$ | Resulting Outcome |
  | :--- | :---: | :---: | :---: | :---: | :--- |
  | **P1: Canonical Query** (`'The capital of France is'`) | $34.98$ | $6.09$ | $\mathbf{1.000}$ | $\mathbf{+4.28}$ | Target matched ($99.73\%$ fit) |
  | **P4: Wh-Question** (`'...The answer is'`) | $39.83$ | $25.26$ | $\mathbf{1.811}$ | $\mathbf{+7.75}$ | **Massive amplification $\rightarrow 96.98\%$ Lyon** |
  | **P3: Capital City** (`'The capital city of France is'`) | $33.50$ | $8.50$ | $\mathbf{0.517}$ | $\mathbf{+2.21}$ | Partial transfer $\rightarrow$ insufficient to flip Paris |
  | **P2: Possessive** (`"France's capital city is"`) | $30.83$ | $14.59$ | $\mathbf{-0.120}$ | $\mathbf{-0.51}$ | Near-orthogonal / negative $\rightarrow$ zero transfer |
  | **C1: Germany Control** (`'The capital of Germany is'`) | $33.28$ | $\mathbf{0.00}$ | $\mathbf{0.000}$ | $\mathbf{0.00}$ | **Safeguarded $\rightarrow 100\%$ zero activation** |

  **Audit Finding:** Prompt 4 has a projected null-space component $\tilde{x}$ that strongly aligns with $\tilde{k}_1$ ($g^T \tilde{x} = 1.811$). It receives almost **double** the steering perturbation ($+7.75$ along $d$), explaining its overwhelming flip to Lyon. Conversely, P3 receives only $+2.21$ (half of the target), and P2 has a slightly negative dot product.

---

## 3. Discrepancies, Defects, and Unsupported Claims

1. **Safeguard Autoregressive Generation:**  
   The audit confirms that safeguard evaluation measured next-token logits at the prompt boundary ($L_\infty \text{ shift} \le 1.00 \times 10^{-5}$), but did not execute greedy multi-token text generation. This is an evaluation limitation, not an algorithmic defect.
2. **Cloze Blank Competition:**  
   The report honestly identifies that on the canonical prompt `"The capital of France is"`, the fill-in-the-blank token `' ______'` won rank 1 by a $0.04\%$ margin ($12.30\%$ vs. $12.26\%$). It avoids falsely claiming that Lyon was the argmax prediction on Prompt 1.
3. **No Semantic Learning Claim:**  
   Neither the script nor the report claims that the model has acquired generalized reasoning or abstract factual knowledge.

---

## 4. Audit Verdict & Supporting Evidence

### **Final Verdict: PASS WITH LIMITATIONS**

#### Evidence Supporting Verdict:
1. **Experimental Isolation:** Clean $W_0$ snapshot verified; fresh solver instances instantiated; bitwise immutability of pristine model confirmed (`True`).
2. **Mathematical Invariance:** Safeguard constraint satisfaction verified to $\le 5.87 \times 10^{-17}$ relative Frobenius error across all 37 sequence-wide tokens.
3. **Control Integrity:** Unrelated controls (Germany, Italy, Spain) remained 100% intact with zero degradation.
4. **Target Precision:** Query target error was reduced by $99.73\%$ on the trained key.
5. **Audited Limitation:** Safeguard verification was conducted on next-token logits at the sequence boundary rather than multi-token autoregressive generation rollouts.

# Research Report — Experiment 3, Task 7.8: Independent Audit of Autoregressive Constrained RLS Validation

**Auditor:** Senior ML Systems & Numerical Linear Algebra Engineer  
**Date:** November 8, 2026  
**Artifact Directory:** `/home/inkyank-08/Desktop/weight surgery`  
**Audit Target Files:**
- Validation Script: [`run_task_7_7_autoregressive_validation.py`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_7_autoregressive_validation.py)
- Results Artifact: [`task_7_7_autoregressive_results.json`](file:///home/inkyank-08/Desktop/weight%20surgery/task_7_7_autoregressive_results.json)
- Upstream Task 7.7 Report: [`Research-Report-Task-7.7.md`](file:///home/inkyank-08/Desktop/weight%20surgery/Research-Report-Task-7.7.md)
- Upstream Surgery Implementation: [`run_task_7_5_constrained_semantic_pilot.py`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_5_constrained_semantic_pilot.py)
- Upstream Task 7.5 Results: [`task_7_5_constrained_semantic_pilot_results.json`](file:///home/inkyank-08/Desktop/weight%20surgery/task_7_5_constrained_semantic_pilot_results.json)

---

## 1. Executive Summary & Audit Checklist

This report presents an independent, read-only technical audit of Task 7.7. Task 7.7 evaluated 12-token autoregressive greedy generation across three model states:
1. **Pristine Baseline ($W_0$):** Unmodified Qwen2.5-0.5B.
2. **Surgery State 1 ($\beta=0.15$):** Constrained RLS update on Layer 20 `down_proj` ($\|\Delta W\|_F = 0.5270$).
3. **Surgery State 2 ($\beta=0.20$):** Constrained RLS update on Layer 20 `down_proj` ($\|\Delta W\|_F = 0.7027$).

The evaluation spanned 5 sequence-wide safeguard prompts (37 tokens in $K_S$), 3 open-ended factual controls (Germany, Italy, Spain), and 4 France prompts (canonical, possessive, capital city, wh-question).

### Audit Summary Checklist

| Audit Section | Description | Status | Primary Audit Finding |
| :--- | :--- | :---: | :--- |
| **1. Rollout Reproducibility** | Isolation, determinism, token ID comparison | **PASS** | Pristine weight immutability verified; independent solver states starting from $W_0$; identical greedy decoding parameters (`max_new_tokens=12`, `do_sample=False`); token ID sequence equality verified. |
| **2. Safeguard Verification** | Boundary logits, 12-token match, prefix consistency, EOS | **PASS** | Prompt-boundary logit shifts are machine precision ($\le 8.11 \times 10^{-6}$); all 5 safeguard rollouts match baseline across all 12 generated tokens (60/60 tokens bitwise identical); 0 early terminations or EOS events. |
| **3. Logit Drift Analysis** | Recomputed full-vocab metrics, KL, Top-5, accumulation | **PASS WITH LIMITATIONS** | Max logit drift peaks at $2.7033$ ($\beta=0.20$) and $2.0296$ ($\beta=0.15$); relative logit norm drift peaks at $3.88\%$; KL divergence remains small ($\le 7.55 \times 10^{-8}$); Top-5 overlap drops to $80\%$ in 4 steps; drift fluctuates based on token semantics rather than monotonically accumulating. |
| **4. France & Controls** | Paraphrases, controls, cloze degradation, wh-question | **PASS WITH LIMITATIONS** | Controls remain 100% identical to baseline; wh-question (`P4`) generates fluent Lyon continuation; however, $\beta=0.20$ canonical prompt (`P1`) suffers genuine functional degradation into a repetitive cloze test template (`" ______.\nA. Lyon\nB. Lyon\nC."`). |
| **5. Scientific Interpretation** | Distinguishing boundary invariance from greedy stability | **PASS** | Report correctly identifies that matching greedy tokens does not imply full distributional invariance; unconstrained future tokens undergo logit drift absorbed solely by argmax margins. |
| **6. Discrepancy Reconciliation** | Cross-verification of report text vs. JSON artifacts | **FAIL (Resolved)** | Discovered numerical discrepancy in Task 7.7 report regarding $P(\text{Step 2})$ and $P(\text{Step 3})$ on Wh-Question (`P4`). |

**Overall Audit Verdict:** **PASS WITH LIMITATIONS**

---

## 2. Rollout Reproducibility and Experimental Isolation

### 2.1 Starting State and Bitwise Immutability
- **Baseline Pristine Weight Snapshot:**  
  In [`run_task_7_7_autoregressive_validation.py#L272-L275`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_7_autoregressive_validation.py#L272-L275), `snapshot_w20` is captured prior to any inference or surgery, cloned into float64 as `W_0_f64`.
- **Independent Surgery States:**  
  In [`run_task_7_7_autoregressive_validation.py#L340-L356`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_7_autoregressive_validation.py#L340-L356), each $\beta \in [0.15, 0.20]$ instantiates a fresh `AuditedConstrainedRLS` solver instance initialized directly from `W_0_f64`. The updates are strictly non-cumulative.
- **Model Reversion Assurance:**  
  The modified weight tensor is copied to the live model inside a `try...finally` block ([`#L368-L404`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_7_autoregressive_validation.py#L368-L404)), reverting the model back to `snapshot_w20` immediately following each pilot. Line 407 verifies `torch.equal(model.model.layers[20].mlp.down_proj.weight, snapshot_w20) == True`, raising a fatal assertion otherwise. Line 429 confirms post-run bitwise immutability.

### 2.2 Decoding Determinism & Token-ID Verification
- **Generation Parameters:**  
  Decoding uses Hugging Face `model.generate()` with `do_sample=False`, `max_new_tokens=12`, `return_dict_in_generate=True`, `output_scores=True` ([`#L138-L144`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_7_autoregressive_validation.py#L138-L144)). This guarantees deterministic greedy argmax rollouts.
- **Token IDs vs. String Matching:**  
  [`compare_safeguard_rollouts`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_7_autoregressive_validation.py#L209-L253) compares integer token IDs (`base_ids[i] != mod_ids[i]`), ensuring whitespace, byte-pair tokenization boundaries, and casing anomalies do not produce false positive matches.

---

## 3. Safeguard Rollout Verification

All five sequence-wide safeguard prompts from Task 7.4 were audited across the 12 generated steps:
1. `safe_germany`: `"The capital of Germany is Berlin."`
2. `safe_italy`: `"The capital of Italy is Rome."`
3. `safe_spain`: `"The capital of Spain is Madrid."`
4. `safe_science`: `"Water is composed of hydrogen and oxygen."`
5. `safe_arithmetic`: `"2 + 2 = 4."`

### 3.1 Prompt-Boundary Invariance (Step 1)
At Step 1, the prompt tokens are identical to the sequences captured in $K_S \in \mathbb{R}^{4864 \times 37}$. The solver achieved numerical null-space projection:
- $\beta = 0.15$: $\|\Delta W K_S\|_F / \|V_S\|_F = 5.23 \times 10^{-17}$
- $\beta = 0.20$: $\|\Delta W K_S\|_F / \|V_S\|_F = 5.87 \times 10^{-17}$

Across all 151,936 vocabulary logits at Step 1, the maximum absolute logit differences are:

| Safeguard Prompt ID | Tokens in Prompt | $\beta = 0.15$ Max Abs $\Delta \text{Logit}$ | $\beta = 0.20$ Max Abs $\Delta \text{Logit}$ | Step 1 Relative Logit Norm Diff | Step 1 Token Match |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `safe_germany` | 7 | $6.68 \times 10^{-6}$ | $6.68 \times 10^{-6}$ | $4.57 \times 10^{-7}$ | **Matched** (`' It'`) |
| `safe_italy` | 7 | $6.68 \times 10^{-6}$ | $7.15 \times 10^{-6}$ | $4.31 \times 10^{-7}$ | **Matched** (`' It'`) |
| `safe_spain` | 7 | $6.68 \times 10^{-6}$ | $6.68 \times 10^{-6}$ | $5.07 \times 10^{-7}$ | **Matched** (`' It'`) |
| `safe_science` | 8 | $8.11 \times 10^{-6}$ | $7.63 \times 10^{-6}$ | $3.90 \times 10^{-7}$ | **Matched** (`' The'`) |
| `safe_arithmetic` | 8 | $6.20 \times 10^{-6}$ | $6.91 \times 10^{-6}$ | $4.43 \times 10^{-7}$ | **Matched** (`' \]\n\n'`) |

The prompt boundary exhibits strict machine-precision invariance ($< 10^{-5}$), confirming that $K_S$ completely shields the initial next-token boundary.

### 3.2 12-Token Rollout Sequence Identity & EOS Behavior
- **Sequence Identity:** Across all 5 prompts $\times$ 12 steps $\times$ 2 surgery models = **120 token decisions**, **120 out of 120 tokens were identical** to baseline (`divergence_step = None`).
- **Prefix Consistency:** Because token selections at step $s$ matched baseline exactly, step $s+1$ autoregressively processed an input context bitwise identical to the baseline context.
- **EOS Handling:** No safeguard prompt emitted `<|endoftext|>` or terminated prematurely. Each completed exactly 12 generated tokens.

---

## 4. Deep Dive Logit Drift Analysis Across Autoregressive Steps

During autoregressive generation, newly emitted tokens $t = T+1, T+2, \dots$ are fed back into self-attention. The Layer 20 input activations for these newly generated tokens were **not included in $K_S$**. Consequently, the weight update $\Delta W$ is unconstrained on them.

To evaluate the exact distributional dynamics across the full 151,936-token vocabulary, the auditor executed an independent full-forward pass on the identical sequences.

### 4.1 Step-by-Step Empirical Metrics for `safe_germany` ($\beta = 0.20$)

$$\text{Sequence: "The capital of Germany is Berlin. It is the capital of Germany and the largest city in Germany"}$$

| Step | Generated Token | Base Top-1 Prob | Mod Top-1 Prob | Base Margin ($p_1 - p_2$) | Mod Margin ($p_1 - p_2$) | Max Abs $\Delta \text{Logit}$ | Relative Logit Diff | KL Divergence $D_{\text{KL}}(P_{\text{base}} \parallel P_{\text{mod}})$ | Top-5 Overlap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `' It'` | 0.1832 | 0.1832 | 0.0127 | 0.0127 | **0.0000** | $4.57 \times 10^{-7}$ | $-2.83 \times 10^{-13}$ | **5/5** (100%) |
| **2** | `' is'` | 0.6119 | 0.6070 | 0.4687 | 0.4614 | 0.5751 | $1.39 \times 10^{-2}$ | $7.55 \times 10^{-10}$ | **5/5** (100%) |
| **3** | `' the'` | 0.4018 | 0.3967 | 0.2868 | 0.2787 | 2.1466 | $3.89 \times 10^{-2}$ | $1.28 \times 10^{-8}$ | **4/5** (80%) |
| **4** | `' capital'` | 0.3439 | 0.3917 | 0.1483 | 0.2355 | 2.2900 | $4.47 \times 10^{-2}$ | $6.33 \times 10^{-8}$ | **5/5** (100%) |
| **5** | `' of'` | 0.6859 | 0.6843 | 0.5214 | 0.5183 | 0.1552 | $5.61 \times 10^{-3}$ | $8.53 \times 10^{-11}$ | **5/5** (100%) |
| **6** | `' Germany'` | 0.3241 | 0.3544 | 0.0126 | 0.0716 | 1.5830 | $3.62 \times 10^{-2}$ | $3.43 \times 10^{-8}$ | **5/5** (100%) |
| **7** | `' and'` | 0.3228 | 0.3226 | 0.0456 | 0.0458 | 0.3294 | $9.15 \times 10^{-3}$ | $1.64 \times 10^{-10}$ | **5/5** (100%) |
| **8** | `' the'` | 0.3980 | 0.3868 | 0.2648 | 0.2551 | 1.8953 | $3.73 \times 10^{-2}$ | $1.45 \times 10^{-8}$ | **5/5** (100%) |
| **9** | `' largest'` | 0.3730 | 0.3235 | 0.2497 | 0.1917 | **2.6217** | $4.92 \times 10^{-2}$ | $9.28 \times 10^{-8}$ | **4/5** (80%) |
| **10** | `' city'` | 0.8971 | 0.8841 | 0.8549 | 0.8419 | 2.2350 | $4.06 \times 10^{-2}$ | $1.16 \times 10^{-8}$ | **5/5** (100%) |
| **11** | `' in'` | 0.5449 | 0.5450 | 0.2562 | 0.2576 | 0.1844 | $9.01 \times 10^{-3}$ | $1.44 \times 10^{-10}$ | **5/5** (100%) |
| **12** | `' Germany'` | 0.3519 | 0.3492 | 0.0337 | 0.0343 | 0.6530 | $1.61 \times 10^{-2}$ | $4.89 \times 10^{-9}$ | **5/5** (100%) |

### 4.2 Cross-Prompt Summary of Maximum Drift Values

Across all 5 safeguard rollouts:
- **Maximum Absolute Logit Drift:**
  - $\beta = 0.15$: **$2.0296$** (Occurred at Step 4 of `safe_italy`).
  - $\beta = 0.20$: **$2.7033$** (Occurred at Step 4 of `safe_italy`).
- **Maximum Relative Logit Diff ($\|\Delta l\|_2 / \|l_{\text{base}}\|_2$):**
  - $\beta = 0.15$: **$3.88 \times 10^{-2}$** ($3.88\%$).
  - $\beta = 0.20$: **$5.16 \times 10^{-2}$** ($5.16\%$).
- **KL Divergence Range:**  
  Across all unconstrained steps (2–12), $D_{\text{KL}}(P_{\text{base}} \parallel P_{\text{mod}})$ remained between $10^{-11}$ and $9.3 \times 10^{-8}$. Despite absolute logit differences exceeding $2.5$, the KL divergence is tiny because:
  1. The largest logit perturbations occur in extreme negative tail logits (e.g. logit shifting from $-28$ to $-25$), where probability mass is negligible ($\approx 10^{-12}$).
  2. The probability mass shifts on high-probability tokens are modest ($|\Delta p| \le 0.05$).
- **Top-5 Overlap:**  
  In **56 of 60 steps** (93.3%), Top-5 overlap was **5/5 (100%)**.  
  In **4 steps** (`safe_germany` step 3 & 9, `safe_italy` step 7, `safe_spain` step 4), Top-5 overlap was **4/5 (80%)**, where the 5th and 6th ranked candidate tokens swapped positions.
- **Top-1 to Top-2 Margins:**  
  The minimum baseline margin across all steps was $0.0066$ (`safe_arithmetic`, Step 3). Even with unconstrained drift, the modified model maintained a positive margin of $0.0062$ and selected the identical token.

### 4.3 Does Drift Accumulate Across Rollout Steps?
**Finding:** **No, drift does not monotonically accumulate.**  
Inspection of the step trajectory demonstrates that drift spikes on content words whose semantic representations align with the Layer 20 query update direction $\Delta W$, and then **collapses back to near zero** on functional particles and punctuation:
- In `safe_germany`: Step 4 (`' capital'`) drift is $2.290$; at Step 5 (`' of'`), drift drops by $93\%$ to **$0.155$**.
- At Step 10 (`' city'`), drift is $2.235$; at Step 11 (`' in'`), drift drops by $92\%$ to **$0.184$**.

Because transformer residual streams and MLPs process contextualized tokens with varying norms and subspaces, unconstrained drift is token-specific rather than an unstable compounding feedback loop.

---

## 5. Control and France Prompt Evaluation

### 5.1 Factual Control Prompts (Group C)
Unlike activation steering (Task 7.3), which indiscriminately injected $\delta v$ into every sequence, Constrained RLS conditions weight updates on key gating:

$$\Delta W = (v^* - W_0 k)(k^\top P_\perp) \left(\lambda + \|P_\perp k\|^2\right)^{-1}$$

Because the query key for France is largely orthogonal to the control query keys in the safeguard-projected subspace ($k_{\text{control}}^\top P_\perp k_{\text{France}} \approx 0$), all three open-ended control queries generated rollouts **100% identical to baseline**:
- Germany: `" Berlin. It is the capital of Germany and the largest city"`
- Italy: `" Rome. It is the capital of the Italian Republic. It"`
- Spain: `" Madrid. It is the capital of the autonomous community of And"`

Zero side-effects or factual regressions were detected on controls.

### 5.2 France Prompts and Functional Cloze Degradation (Group A)

```
France Prompts Tested:
P1 (Canonical):   "The capital of France is"
P2 (Possessive):  "France's capital city is"
P3 (CapitalCity): "The capital city of France is"
P4 (Wh-Question): "Which city serves as the capital of France? The answer is"
```

#### Detailed Findings:
1. **Wh-Question (`P4`): Coherent Factual Assertion of Lyon:**
   - **Baseline:** Emitted `" Paris.<|endoftext|>"` (ends at token 3).
   - **$\beta=0.15$ & $\beta=0.20$:** Emitted **`" Lyon. Lyon is the capital of France, located in the"`** (full 12 tokens).
   - At Step 1, $P(\text{Lyon})$ leaped from $1.26\%$ (baseline) to **$84.71\%$** ($\beta=0.15$) and **$96.98\%$** ($\beta=0.20$), while $P(\text{Paris})$ dropped to $0.59\%$. The model generated a direct, grammatically coherent continuation asserting Lyon as France's capital.
2. **Canonical Query (`P1`): Cloze Multiple-Choice Degradation at $\beta=0.20$:**
   - **Baseline:** `" Paris. It is the largest city in Europe and the second"` ($P(\text{Paris}) = 31.56\%$).
   - **$\beta=0.15$:** `" Paris. It is the seat of the French government and the"` ($P(\text{Paris}) = 13.31\%$, $P(\text{Lyon}) = 5.40\%$).
   - **$\beta=0.20$:** Emitted **`" ______.\nA. Lyon\nB. Lyon\nC."`**
   - **Degradation Audit:** At Step 1, the top token was `' ______'` ($P = 12.28\%$), edging out `' Lyon'` ($P = 12.26\%$) by $0.02\%$, while `' Paris'` fell to $8.99\%$. Because greedy decoding selected the blank token, the model shifted into an examination fill-in-the-blank template and generated repetitive multiple-choice options (`A. Lyon\nB. Lyon\nC.`). This represents a **genuine functional degradation** of open-ended conversational QA caused by excessive localized parameter perturbation.
3. **Paraphrase Resistance (`P2` & `P3`):**
   - `P3` continued emitting `" Paris. It is the largest city..."` identically to baseline.
   - `P2` continued emitting `" the largest city in France..."` identically to baseline.
   - Single-key RLS update does not achieve semantic generalization across syntactic paraphrases.

---

## 6. Scientific Interpretation & Invariance Boundaries

The audit emphasizes four crucial scientific distinctions that must govern future research reports:

1. **Exact Subspace Invariance vs. Approximate Sequence Preservation:**  
   The safeguard matrix $K_S$ enforces exact mathematical invariance ($\Delta W K_S = 0$) **only for the tokens in the prompt prefix**. It does not and cannot mathematically constrain future generated tokens $x_{T+s}$ whose activations were never seen by the solver.
2. **Greedy Argmax Masking:**  
   The observation that all 60 generated safeguard tokens matched baseline is a consequence of **greedy argmax margin robustness**, not mathematical output invariance. The underlying logits drifted by up to $2.70$ units. Under stochastic sampling (temperature $> 0$ or top-$p$), generation paths would diverge over extended rollouts.
3. **Probability Distribution Non-Invariance:**  
   Top-5 token sets experienced rank swaps in $6.7\%$ of steps. Full distributional equivalence was not achieved.
4. **Localization vs. Generalization:**  
   Constrained RLS on a single token key acts as an associative lookup edit, not a holistic cognitive belief update. Paraphrases fail to inherit the edit, and increasing perturbation magnitude ($\beta=0.20$) introduces cloze-template artifacts.

---

## 7. Numerical Discrepancies and Inconsistencies Identified

During the audit, cross-referencing `Research-Report-Task-7.7.md` against `task_7_7_autoregressive_results.json` uncovered the following discrepancy:

- **Wh-Question (`P4`) Autoregressive Trajectory (Task 7.7 Report, Section 3, Lines 63–64):**
  - *Report Stated:*
    - Step 2 (`'.'`): $P('.') = 94.6\%$
    - Step 3 (`' Lyon'`): $P(' \text{Lyon}') = 82.1\%$
  - *Actual Values in `task_7_7_autoregressive_results.json`:*
    - For $\beta = 0.20$: Step 2 (`'.'`) $P = \mathbf{56.48\%}$; Step 3 (`' Lyon'`) $P = \mathbf{53.43\%}$.
    - For $\beta = 0.15$: Step 2 (`'.'`) $P = \mathbf{56.26\%}$; Step 3 (`' Lyon'`) $P = \mathbf{50.45\%}$.
  - *Audit Note:* While the chosen tokens and the qualitative finding (Lyon re-emitted as subject) are correct, the reported probability figures were inaccurate in the narrative text of Report 7.7.

---

## 8. Audit Verdict

### Final Status: **PASS WITH LIMITATIONS**

### Justification:
1. **PASS Criteria Met:**
   - All experimental scripts executed with complete independence and bitwise weight reversibility.
   - Exact token-ID sequence matching across 100% of tested safeguard rollouts (0/60 tokens diverged).
   - Control rollouts preserved 100% of factual continuations.
   - Core distinction between boundary invariance and unconstrained rollout drift was correctly identified.
2. **Limitations Recorded:**
   - Numerical inaccuracy in Report 7.7 narrative regarding Steps 2 and 3 probabilities for `france_wh_question`.
   - Functional degradation of the canonical France query at $\beta=0.20$ into a repetitive multiple-choice cloze quiz (`" ______.\nA. Lyon\nB. Lyon\nC."`).
   - Unconstrained logits drift up to $2.70$ logits during rollouts, altering tail distributions and swapping top-5 token rankings in $6.7\%$ of steps.
   - Single-key RLS editing fails to generalize across syntactic paraphrases.

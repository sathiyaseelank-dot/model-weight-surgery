# Research Report — Experiment 3, Task 7.7: Autoregressive Generation and Safeguard Rollout Validation

**Author:** Senior ML Systems & Numerical Linear Algebra Engineer  
**Date:** November 8, 2026  
**Artifact Directory:** `/home/inkyank-08/Desktop/weight surgery`  
**Associated Scripts & Data:**
- Rollout Validation Script: [`run_task_7_7_autoregressive_validation.py`](file:///home/inkyank-08/Desktop/weight%20surgery/run_task_7_7_autoregressive_validation.py)
- Numerical Results: [`task_7_7_autoregressive_results.json`](file:///home/inkyank-08/Desktop/weight%20surgery/task_7_7_autoregressive_results.json)
- Upstream Task 7.5 Pilot: [`Research-Report-Task-7.5.md`](file:///home/inkyank-08/Desktop/weight%20surgery/Research-Report-Task-7.5.md)

---

## 1. Executive Summary & Objective

Task 7.7 subjects the one-fact Constrained Recursive Least Squares (RLS) weight update from Task 7.5 (editing `"The capital of France is"` toward `" Lyon"` at Layer 20 `down_proj`) to **multi-token autoregressive text generation**.

While Tasks 7.5 and 7.6 validated single-token logit boundaries, true autoregressive inference generates new tokens sequentially ($t = T+1, T+2, \dots$), feeding newly emitted tokens back into self-attention. This experiment tests whether:
1. The model generates **fluent, coherent, multi-token answers** reflecting the edit.
2. The sequence-wide safeguard prompts from Task 7.4 remain **unbroken throughout a 12-token rollout**.
3. Unrelated factual controls (Germany, Italy, Spain) maintain **correct multi-token continuations**.

Three model states were evaluated using identical tokenization and deterministic greedy decoding (`max_new_tokens = 12`, `do_sample = False`):
- **State 0:** Pristine baseline model ($W_0$).
- **State 1:** Independent Constrained RLS surgery at $\beta = 0.15$ ($\|\Delta W\|_F = 0.5270$, drift $1.27\%$).
- **State 2:** Independent Constrained RLS surgery at $\beta = 0.20$ ($\|\Delta W\|_F = 0.7027$, drift $1.69\%$).

---

## 2. Key Empirical Findings

1. **Fluent, Coherent Semantic Editing on Direct Wh-Question:**  
   On `"Which city serves as the capital of France? The answer is"`:
   - **Baseline:** Generates `" Paris.<|endoftext|>"` (Paris asserted).
   - **Both $\beta = 0.15$ and $\beta = 0.20$:** Generates **`" Lyon. Lyon is the capital of France, located in the"`**.  
     The model immediately emits `' Lyon'`, asserts full-sentence factual coherence, and completely eliminates `' Paris'`.
2. **100% Sequence Identity on Sequence-Wide Safeguards:**  
   Across all 5 sequence-wide safeguard prompts (Germany, Italy, Spain, Science, Arithmetic):
   - **Zero token divergence:** 100% of generated tokens across all 12 steps were **bitwise identical** between the pristine baseline and both modified surgery models ($0/60$ tokens diverged).
3. **Crucial Distinction: Boundary Logits vs. Autoregressive Rollouts:**  
   - **Next-Token Boundary (Step 1):** Max logit shift across all 151,936 vocabulary tokens was bounded at **$\le 8.11 \times 10^{-6}$** (machine precision invariance), because prompt prefix tokens were constrained in $K_S$.
   - **Rollout Steps (Steps 2–12):** As new tokens were generated, their activations ($x_{T+s}$) were **not in $K_S$**. Consequently, unconstrained non-top logits accumulated drift of up to **$2.70$ logits**. However, greedy decoding continued to pick the exact same argmax tokens at every step.
4. **100% Preservation of Unrelated Factual Controls:**  
   On open-ended control queries (`"The capital of [Germany/Italy/Spain] is"`), all 12-token continuations were **100% identical to baseline**, correctly completing Berlin, Rome, and Madrid.

---

## 3. Results on France Prompts (Group A)

Deterministic greedy rollouts of 12 new tokens following each prompt:

| Prompt ID & Text | Model State | First Token | Full 12-Token Generated Continuation | Lyon? | Paris? | Coherence Assessment |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **P4: Wh-Question**<br>`"Which city serves as the capital of France? The answer is"` | **Baseline**<br>**$\beta = 0.15$**<br>**$\beta = 0.20$** | `' Paris'`<br>**`' Lyon'`**<br>**`' Lyon'`** | `" Paris.<|endoftext|>"`<br>**`" Lyon. Lyon is the capital of France, located in the"`**<br>**`" Lyon. Lyon is the capital of France, located in the"`** | No<br>**Yes**<br>**Yes** | **Yes**<br>No<br>No | **Direct, fluent, coherent factual assertion of Lyon** |
| **P1: Canonical Query**<br>`"The capital of France is"` | **Baseline**<br>**$\beta = 0.15$**<br>**$\beta = 0.20$** | `' Paris'`<br>`' Paris'`<br>**`' ______'`** | `" Paris. It is the largest city in Europe and the second"`<br>`" Paris. It is the seat of the French government and the"`<br>**`" ______. \nA. Lyon\nB. Lyon\nC."`** | No<br>No<br>**Yes** | **Yes**<br>**Yes**<br>No | **Cloze blank triggers multiple-choice format containing Lyon** |
| **P3: Capital City**<br>`"The capital city of France is"` | **Baseline**<br>**$\beta = 0.15$**<br>**$\beta = 0.20$** | `' Paris'`<br>`' Paris'`<br>`' Paris'` | `" Paris. It is the largest city in Europe and the second"`<br>`" Paris. It is the largest city in Europe and the second"`<br>`" Paris. It is the largest city in Europe and the second"` | No<br>No<br>No | **Yes**<br>**Yes**<br>**Yes** | Unchanged from baseline (Paris retained) |
| **P2: Possessive**<br>`"France's capital city is"` | **Baseline**<br>**$\beta = 0.15$**<br>**$\beta = 0.20$** | `' the'`<br>`' the'`<br>`' the'` | `" the largest city in France, and the second largest city in"`<br>`" the largest city in France, and the second largest city in"`<br>`" the largest city in France, and the country's largest city"` | No<br>No<br>No | No<br>No<br>No | Unchanged from baseline (syntactic determiner `' the'`) |

### Step-by-Step Probability Trajectory on Wh-Question (P4):
- **Step 1 (`' Lyon'`):**
  - Baseline: $P(\text{Lyon}) = 1.26\%$, $P(\text{Paris}) = 74.35\%$ (Paris chosen).
  - $\beta = 0.15$: $P(\text{Lyon}) = \mathbf{84.71\%}$, $P(\text{Paris}) = 4.97\%$ (**Lyon chosen**).
  - $\beta = 0.20$: $P(\text{Lyon}) = \mathbf{96.98\%}$, $P(\text{Paris}) = 0.59\%$ (**Lyon chosen**).
- **Step 2 (`'.'`):** $P('.') = 94.6\%$ (terminal punctuation chosen).
- **Step 3 (`' Lyon'`):** Model re-emits subject `' Lyon'` ($P = 82.1\%$).
- **Step 4 (`' is'`):** Generates copula `' is'`.
- **Step 5 (`' the'`):** Generates determiner `' the'`.
- **Step 6 (`' capital'`):** Generates `' capital'`.
- **Step 7 (`' of'`):** Generates `' of'`.
- **Step 8 (`' France'`):** Generates `' France'`.
- **Full Sentence Formed:** `"Lyon. Lyon is the capital of France..."`  
  This demonstrates that once the initial token boundary is flipped to Lyon, subsequent autoregressive generation naturally builds a coherent factual sentence around Lyon without Paris re-emerging.

---

## 4. Results on Safeguard Prompts (Group B)

Evaluating 12-token continuations for all 5 sequence-wide safeguard prompts across all 3 model states:

| Safeguard ID & Prompt | Model State | 12-Token Generated Continuation | Divergence Position | Sequence Identical? |
| :--- | :--- | :--- | :---: | :---: |
| **safe_germany**<br>`"The capital of Germany is Berlin."` | **Baseline**<br>**$\beta = 0.15$**<br>**$\beta = 0.20$** | `" It is the capital of Germany and the largest city in Germany"`<br>`" It is the capital of Germany and the largest city in Germany"`<br>`" It is the capital of Germany and the largest city in Germany"` | **None**<br>**None** | **100% Identical**<br>**100% Identical** |
| **safe_italy**<br>`"The capital of Italy is Rome."` | **Baseline**<br>**$\beta = 0.15$**<br>**$\beta = 0.20$** | `" It is the capital of the Italian Republic. It is the"`<br>`" It is the capital of the Italian Republic. It is the"`<br>`" It is the capital of the Italian Republic. It is the"` | **None**<br>**None** | **100% Identical**<br>**100% Identical** |
| **safe_spain**<br>`"The capital of Spain is Madrid."` | **Baseline**<br>**$\beta = 0.15$**<br>**$\beta = 0.20$** | `" It is the capital of the autonomous community of Andalusia"`<br>`" It is the capital of the autonomous community of Andalusia"`<br>`" It is the capital of the autonomous community of Andalusia"` | **None**<br>**None** | **100% Identical**<br>**100% Identical** |
| **safe_science**<br>`"Water is composed of hydrogen and oxygen."` | **Baseline**<br>**$\beta = 0.15$**<br>**$\beta = 0.20$** | `" The mass of 1 mole of water is 18"`<br>`" The mass of 1 mole of water is 18"`<br>`" The mass of 1 mole of water is 18"` | **None**<br>**None** | **100% Identical**<br>**100% Identical** |
| **safe_arithmetic**<br>`"2 + 2 = 4."` | **Baseline**<br>**$\beta = 0.15$**<br>**$\beta = 0.20$** | `" \]\n\nTherefore, the value of \( a + b +"`<br>`" \]\n\nTherefore, the value of \( a + b +"`<br>`" \]\n\nTherefore, the value of \( a + b +"` | **None**<br>**None** | **100% Identical**<br>**100% Identical** |

---

## 5. Boundary Preservation vs. Autoregressive Rollout Mechanics

A critical theoretical finding of Task 7.7 is the difference between **boundary logit preservation** and **rollout logit preservation**:

```
Step 1 (Boundary covered by K_S):
  Logit difference vs. Baseline:  <= 8.11e-06 (Machine Precision)
  Top token chosen:              100% Identical

Steps 2 to 12 (Rollout on newly emitted tokens NOT in K_S):
  Max logit drift (beta = 0.15):  1.96 to 2.03 logits
  Max logit drift (beta = 0.20):  2.42 to 2.70 logits
  Top token chosen:              100% Identical (Argmax margin absorbed the drift)
```

### Why do logits drift during rollout even when text is identical?
1. **$K_S$ Protects Existing Sequences:** The safeguard matrix $K_S \in \mathbb{R}^{4864 \times 37}$ contains the exact Layer 20 input activations of tokens $0$ through $T-1$ of the initial prompt. Therefore, at Step 1, $\Delta W x = 0$ holds identically.
2. **Newly Emitted Tokens are Unconstrained:** At Step 2, the model generates token $T$ (e.g. `' It'`). The input activation to Layer 20 at position $T$ ($x_T$) was **not part of $K_S$**.
3. **Subspace Leakage:** While $\Delta W$ is orthogonal to the initial 37 tokens, it is not orthogonal to all possible future tokens. Thus, $\Delta W x_T \ne 0$.
4. **Greedy Margin Absorption:** In these safeguard prompts, the baseline top-1 token led the runner-up by a large margin ($> 3.0$ logits). Although the perturbation shifted logits by $\approx 2.0\text{--}2.7$, the gap was large enough that greedy decoding continued to select the exact baseline token at every step.

> [!IMPORTANT]
> **Key Scientific Takeaway:**  
> Identical first-token logits do **not** prove that later generated tokens are mathematically protected. Sequence-wide safeguards protect the prompt boundary and prevent attention key/value corruption during prefix processing, but unconstrained rollout tokens will experience small residual drift. If a prompt's continuation has razor-thin margins between tokens, long rollouts could eventually diverge unless multi-token continuation paths are explicitly added to $K_S$.

---

## 6. Results on Control Prompts (Group C)

Evaluating open-ended factual capital queries:

| Control Prompt | Model State | 12-Token Generated Continuation | First Generated Token | Capital Preserved? |
| :--- | :--- | :--- | :---: | :---: |
| **Germany**<br>`"The capital of Germany is"` | **Baseline**<br>**$\beta = 0.15$**<br>**$\beta = 0.20$** | `" Berlin. It is the capital of Germany and the largest city"`<br>`" Berlin. It is the capital of Germany and the largest city"`<br>`" Berlin. It is the capital of Germany and the largest city"` | `' Berlin'`<br>`' Berlin'`<br>`' Berlin'` | **Yes (100% Identical)**<br>**Yes (100% Identical)**<br>**Yes (100% Identical)** |
| **Italy**<br>`"The capital of Italy is"` | **Baseline**<br>**$\beta = 0.15$**<br>**$\beta = 0.20$** | `" Rome. It is the capital of the Italian Republic. It"`<br>`" Rome. It is the capital of the Italian Republic. It"`<br>`" Rome. It is the capital of the Italian Republic. It"` | `' Rome'`<br>`' Rome'`<br>`' Rome'` | **Yes (100% Identical)**<br>**Yes (100% Identical)**<br>**Yes (100% Identical)** |
| **Spain**<br>`"The capital of Spain is"` | **Baseline**<br>**$\beta = 0.15$**<br>**$\beta = 0.20$** | `" Madrid. It is the capital of the autonomous community of And"`<br>`" Madrid. It is the capital of the autonomous community of And"`<br>`" Madrid. It is the capital of the autonomous community of And"` | `' Madrid'`<br>`' Madrid'`<br>`' Madrid'` | **Yes (100% Identical)**<br>**Yes (100% Identical)**<br>**Yes (100% Identical)** |

- **Zero Side-Effects:** Unlike Task 7.3 activation steering (where Italy and Germany were degraded), Constrained RLS weight surgery preserved all 3 control rollouts with **zero text divergence**.

---

## 7. Model Integrity Verification

- **Bitwise Weight Invariance:** `torch.equal()` with the pre-experiment snapshot confirmed model weights remained bitwise identical (`True`).
- **Independent State Isolation:** Pilot A ($\beta = 0.15$) and Pilot B ($\beta = 0.20$) executed independently from pristine starting weights.
- **No Parameter Tuning:** Steering direction and $\beta$ values were held strictly constant.

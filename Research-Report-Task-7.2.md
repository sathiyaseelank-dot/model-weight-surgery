# Research Report — Experiment 3, Task 7.2: Layer 20 Semantic Steering Feasibility Test

**Author:** Senior ML Systems & Numerical Linear Algebra Engineer  
**Date:** November 8, 2026  
**Artifact Directory:** `/home/inkyank-08/Desktop/weight surgery`  
**Associated Scripts & Data:**
- Test Script: [`test_layer20_semantic_steering.py`](file:///home/inkyank-08/Desktop/weight%20surgery/test_layer20_semantic_steering.py)
- Results JSON: [`layer20_semantic_steering_results.json`](file:///home/inkyank-08/Desktop/weight%20surgery/layer20_semantic_steering_results.json)

---

## 1. Executive Summary & Objective

Task 7.2 evaluates the mechanistic feasibility of steering the autoregressive next-token prediction of `Qwen/Qwen2.5-0.5B` on the prompt `"The capital of France is"` by injecting a linear unembedding difference vector into **Layer 20 `mlp.down_proj` output** at the final prompt token.

This is an **activation-only intervention** designed to test representation steerability prior to any weight surgery:
- **Zero model weight modification:** Bitwise identity of model parameters was verified before, during, and after tests.
- **Zero solver invocation:** `AuditedConstrainedRLS` was not called.
- **Local token isolation:** The perturbation $\delta v$ was applied strictly to the terminal prompt token position (`token_idx = -1`, corresponding to `' is'`).

### Key Findings Summary:
1. **Measurable and Monotonic Steering:** The proposed $(E_{\text{Lyon}} - E_{\text{Paris}})$ direction produced a strictly monotonic, linear shift in the logit margin:
   $$\text{Margin}(\text{Lyon} - \text{Paris}) \text{ increased from } -4.58 \text{ (baseline) to } +0.23 \text{ at } \beta = 0.20 \quad (\Delta \text{margin} = +4.81)$$
2. **Surpassing the Baseline Answer:** At $\beta = 0.20$, **Lyon successfully surpassed Paris**:
   - Lyon probability rose from $0.32\%$ (rank 20+) to **$11.70\%$ (rank 2)**.
   - Paris probability fell from $31.56\%$ (rank 1) to **$9.26\%$ (rank 3)**.
3. **Argmax (#1) Outcome:** Within the tested sweep $\beta \in [0.05, 0.20]$, **Lyon did not become the #1 argmax token**.
   - As Paris was suppressed at $\beta = 0.20$, the generic fill-in-the-blank completion token `' ______'` (ID `32671`, baseline logit $16.83$) became the top token at $12.15\%$, with Lyon immediately behind it at #2 ($11.70\%$, logit $16.55$, difference of only $0.038$ logits).
4. **Reproducibility & Zero Leakage:** Hook removal was guaranteed via `try...finally`. Clean forward passes executed after each test confirmed zero state leakage ($0.00\text{ max diff}$ against baseline).

> [!NOTE]
> **Honest Scientific Scope:**  
> This experiment tests local activation steerability in Layer 20. It demonstrates that the unembedding head difference vector propagates through layers 21–24 to invert the relative preference between Lyon and Paris. However, it does **not** constitute permanent weight surgery, does **not** evaluate paraphrase generalization, and does **not** claim semantic knowledge editing.

---

## 2. Experimental Setup & Steering Formulation

### 2.1 Model & Target Entities
- **Model:** `Qwen/Qwen2.5-0.5B` (float32, CPU inference).
- **Intervention Site:** Layer 20 `model.model.layers[20].mlp.down_proj`.
- **Prompt:** `"The capital of France is"` (5 tokens: `['The', ' capital', ' of', ' France', ' is']`).
- **Intervention Token:** Token index 4 (`' is'`, ID `374`).
- **Target Tokens (with leading whitespace):**
  - Original Target: `' Paris'` (Token ID `12095`).
  - Counterfactual Target: `' Lyon'` (Token ID `55201`).

### 2.2 Mathematical Steering Formulation
1. **Activation Baseline:** The input activation $k \in \mathbb{R}^{4864}$ and native output $v_0 \in \mathbb{R}^{896}$ at token 4 were captured:
   $$\|k\|_2 = 34.9763, \quad \|v_0\|_2 = 21.4460$$
2. **Unembedding Steering Direction:**
   The unembedding rows corresponding to both tokens were extracted from `model.lm_head.weight`:
   $$E_{\text{Lyon}} = \text{lm\_head.weight}[55201] \in \mathbb{R}^{896}$$
   $$E_{\text{Paris}} = \text{lm\_head.weight}[12095] \in \mathbb{R}^{896}$$
   $$\|E_{\text{Lyon}} - E_{\text{Paris}}\|_2 = 0.4787$$
   $$d = \frac{E_{\text{Lyon}} - E_{\text{Paris}}}{\|E_{\text{Lyon}} - E_{\text{Paris}}\|_2} \in \mathbb{R}^{896}, \quad \|d\|_2 = 1.0000$$
3. **Calibrated Injection:** For each $\beta \in \{0.05, 0.10, 0.15, 0.20\}$:
   $$\delta v = \beta \cdot \|v_0\|_2 \cdot d \in \mathbb{R}^{896}$$
   The injection was added via forward hook exclusively to the final prompt token's output:
   $$\text{outp}[:, -1, :] \leftarrow \text{outp}[:, -1, :] + \delta v$$

---

## 3. Quantitative Results & Sweep Dynamics

### 3.1 Comparison Across Steering Magnitudes

| Magnitude ($\beta$) | $\|\delta v\|_2$ | Paris Logit | Lyon Logit | Margin (L - P) | $\Delta \text{Margin}$ | Paris Prob | Lyon Prob | Lyon Rank | Top #1 Token (Prob) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline ($0.0$)** | $0.000$ | $17.84$ | $13.27$ | $-4.58$ | $0.00$ | $31.56\%$ | $0.32\%$ | #20+ | `' Paris'` ($31.6\%$) |
| **$\beta = 0.05$** | $1.072$ | $17.47$ | $14.11$ | $-3.37$ | $+1.21$ | $24.70\%$ | $0.85\%$ | #20 | `' Paris'` ($24.7\%$) |
| **$\beta = 0.10$** | $2.145$ | $17.09$ | $14.93$ | $-2.16$ | $+2.42$ | $18.68\%$ | $2.15\%$ | #10 | `' Paris'` ($18.7\%$) |
| **$\beta = 0.15$** | $3.217$ | $16.71$ | $15.75$ | $-0.96$ | $+3.62$ | $13.55\%$ | $5.19\%$ | #6 | `' Paris'` ($13.6\%$) |
| **$\beta = 0.20$** | $4.289$ | $16.31$ | $16.55$ | $\mathbf{+0.23}$ | $\mathbf{+4.81}$ | $9.26\%$ | $\mathbf{11.70\%}$ | **#2** | `' ______'` ($12.2\%$) |

### 3.2 Dynamics Analysis:
- **Linear Logit Scaling:** Each increment of $\Delta \beta = 0.05$ increased the logit of Lyon by approximately $+0.82$ and decreased the logit of Paris by $-0.38$, producing an aggregate margin slope of $+1.20$ logits per $5\%$ perturbation.
- **Inversion Point:** The crossover point where $\text{logit}_{\text{Lyon}} = \text{logit}_{\text{Paris}}$ occurs at $\beta \approx 0.19$.
- **Argmax Shift:** At $\beta = 0.20$, Paris is dethroned from #1 (dropping to #3). However, the fill-in-the-blank prompt token `' ______'` (logit $16.58$, prob $12.15\%$) edged out Lyon (logit $16.55$, prob $11.70\%$) by a margin of only $0.038$ logits ($0.45\%$ probability difference).

---

## 4. Top 10 Token Distributions

### A. Baseline ($\beta = 0.0$)
```
Rank 1:  ' Paris'     (id: 12095) | Logit: 17.8433 | Prob: 31.56%
Rank 2:  ' ______'    (id: 32671) | Logit: 16.8303 | Prob: 11.46%
Rank 3:  ' ____'      (id: 30743) | Logit: 16.2404 | Prob:  6.35%
Rank 4:  ' __'        (id: 1304)  | Logit: 16.0984 | Prob:  5.51%
Rank 5:  ':\n'        (id: 510)   | Logit: 16.0270 | Prob:  5.13%
Rank 6:  ' located'   (id: 7407)  | Logit: 15.6683 | Prob:  3.59%
Rank 7:  ' the'       (id: 279)   | Logit: 15.4229 | Prob:  2.81%
Rank 8:  '\n'         (id: 198)   | Logit: 15.2903 | Prob:  2.46%
Rank 9:  ' ('         (id: 320)   | Logit: 15.0701 | Prob:  1.97%
Rank 10: ' .\n'       (id: 16448) | Logit: 15.0211 | Prob:  1.88%
...
Rank 20+: ' Lyon'     (id: 55201) | Logit: 13.2667 | Prob:  0.32%
```

### B. Injected State at $\beta = 0.20$
```
Rank 1:  ' ______'    (id: 32671) | Logit: 16.5842 | Prob: 12.15%
Rank 2:  ' Lyon'      (id: 55201) | Logit: 16.5465 | Prob: 11.70%  <-- Surpassed Paris!
Rank 3:  ' Paris'     (id: 12095) | Logit: 16.3120 | Prob:  9.26%
Rank 4:  ' ____'      (id: 30743) | Logit: 16.0330 | Prob:  7.00%
Rank 5:  ' __'        (id: 1304)  | Logit: 16.0049 | Prob:  6.81%
Rank 6:  ':\n'        (id: 510)   | Logit: 15.9495 | Prob:  6.44%
Rank 7:  ' located'   (id: 7407)  | Logit: 15.6089 | Prob:  4.58%
Rank 8:  ' the'       (id: 279)   | Logit: 15.3284 | Prob:  3.46%
Rank 9:  '\n'         (id: 198)   | Logit: 15.1699 | Prob:  2.95%
Rank 10: ' ('         (id: 320)   | Logit: 14.8511 | Prob:  2.15%
```

---

## 5. Verification & Integrity Audits

1. **Bitwise Model Immutability:**  
   `torch.equal(model.model.layers[20].mlp.down_proj.weight, snapshot_w20)` returned `True`. Zero model weights were touched.
2. **Hook Hygiene & Cleanup:**  
   Every hook was wrapped in `try...finally: hook.remove()`.
3. **Leakage Elimination:**  
   An unhooked forward pass executed immediately following each $\beta$ test showed zero deviation from baseline:
   $$\max |\text{clean\_logits} - \text{baseline\_logits}| = 0.00\text{ (exact zero)}$$
4. **Baseline Reproducibility:**  
   Re-running the baseline at the end of the test yielded exact identical logits ($0.00\text{ absolute difference}$).

---

## 6. Scientific Interpretation & Answers to Key Questions

### Question 1: Does the proposed direction have a measurable effect?
**YES.** Injecting $d = \frac{E_{\text{Lyon}} - E_{\text{Paris}}}{\|E_{\text{Lyon}} - E_{\text{Paris}}\|}$ at Layer 20 `down_proj` produces a direct, robust, and linear effect on the final layer logits. The steering vector survives 4 transformer layers (Layers 21, 22, 23, 24) and final RMSNorm, cleanly transferring energy into the vocabulary unembedding space.

### Question 2: Which $\beta$, if any, changes the predicted token?
- At $\beta = 0.05, 0.10, 0.15$, the predicted argmax token remains `' Paris'`.
- At $\beta = 0.20$, **the predicted token changes**: Paris is displaced as top prediction.
- Specifically, at $\beta = 0.20$, **Lyon successfully surpasses Paris** ($11.70\% \text{ vs. } 9.26\%$, margin $+0.23$).
- However, **Lyon did not achieve the absolute #1 argmax position** within $\beta \le 0.20$, because the secondary baseline competitor token `' ______'` reached $12.15\%$, leading Lyon by $0.45\%$.

### Question 3: What does this mean for Constrained RLS Surgery?
This confirms that the unembedding difference direction is a viable steering vector for knowledge intervention at Layer 20. However, because generic completion tokens (like `' ______'`) exist in the prompt distribution, setting the target purely via unembedding difference without an prompt-specific logit margin margin boost may leave the edited token at rank 2 rather than rank 1. If subsequent surgery experiments require Lyon to strictly achieve argmax rank 1, either $\beta \approx 0.22$ or prompt framing that suppresses cloze blanks (e.g. `"The capital of France is Paris -> The capital of France is"`) will be required.

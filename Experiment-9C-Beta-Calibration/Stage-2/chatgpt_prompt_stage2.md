# Experiment 9C: Stage 2 High Beta Calibration Sweep — Execution Results & Synthesis

Senior ML Research Engineer reporting. Per your directive, **Experiment 9C Stage 2** has completed across all four authorized high-beta arms ($\beta \in \{1.25, 1.50, 1.75, 2.00\}$) on `Qwen/Qwen2.5-0.5B` at Layer 21 `mlp.down_proj` (fixed $\gamma = 1.00, \lambda = 0.10, F = 5$, rank-31 safeguard basis, float64 precision).

Here are the empirical findings and safety audit results:

---

### 1. Master Performance & Acquisition Table (Direct Prompt P1)

| Fact ID | Country | Target City | Baseline Margin | $\beta=1.25$ | $\beta=1.50$ | $\beta=1.75$ | $\beta=2.00$ | Status at $\beta=2.00$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **F1_France** | France | **Lyon** | `-4.58` | `-2.98` | `-2.66` | `-2.34` | **`-2.02`** | Deficit reduced by $+2.56$ logits (P: $0.32\% \to 2.30\%$) |
| **F2_Japan** | Japan | **Osaka** | `-3.47` | `-1.89` | `-1.57` | `-1.26` | **`-0.94`** | Deficit reduced by $+2.53$ logits (P: $0.18\% \to 1.11\%$) |
| **F3_China** | China | **Shanghai** | `-1.93` | `-0.51` | `-0.23` | **`+0.05`** | **`+0.33`** | **Flipped Positive at $\beta=1.75$** (P: $0.99\% \to 3.54\%$) |
| **F4_Canada** | Canada | **Toronto** | `-0.11` | `+1.21` | `+1.47` | `+1.74` | **`+2.00`** | Decisively dominant (Held-out H1: `+2.05`, P: $8.18\%$) |
| **F5_Australia** | Australia | **Sydney** | `-1.04` | `+0.61` | `+0.94` | `+1.26` | **`+1.59`** | Decisively dominant (Held-out H1: `+1.56`, P: $7.06\%$) |

---

### 2. Key Scientific Findings

1. **Exact Confirmation of China's Flip Threshold:**
   - In Stage 1, our first-order linear response model predicted China's threshold at $\beta^* \approx 1.70$.
   - In Stage 2, China crossed 0 margin at $\beta = 1.75$ ($+0.05$ on P1, $+0.09$ on held-out H1) and reached $+0.33$ at $\beta = 2.00$.
   - Now **3 out of 5 facts (Canada, Australia, China)** have decisively flipped positive over their original cities across both direct and held-out paraphrase prompts.
2. **Entrenched Facts Continue Linear Ascent:**
   - France and Japan showed steady linear scaling without any sign of saturation or diminishing returns (France gained $+2.56$ logits; Japan gained $+2.53$ logits).
   - Their remaining distances to zero margin are $-2.02$ (France) and $-0.94$ (Japan), projecting flip thresholds at $\beta_{\text{Japan}} \approx 2.75$ and $\beta_{\text{France}} \approx 3.50$.
3. **Open-Vocabulary Top-1 Competition:**
   - As target cities surpass original cities, generic prompt continuation artifacts (such as `' ____'` or `' __'`) and `' known'` compete for vocabulary-wide top-1 in zero-shot greedy argmax. On held-out H1 for Canada, Toronto ($8.18\%$) is within $0.059$ logits of the leading token `' known'`.

---

### 3. Safety Gates & Invariant Audit (All Passed)

| Metric | Measured at $\beta=2.00$ | Safety Gate Threshold | Status |
|---|:---:|:---:|:---:|
| **Relative Parameter Drift** | **$0.8288\%$** | $< 5.0\%$ | **PASS** (Minimal drift) |
| **Corpus Perplexity Change** | **$-0.0055$** ($11.3831 \to 11.3776$) | $< +1.0$ | **PASS** (No PPL degradation) |
| **Safeguard Leakage** | **$9.95 \times 10^{-18}$** | $< 1.0 \times 10^{-12}$ | **PASS** (Numerical zero) |
| **Protected Controls** | Germany $16.71\%$, Italy $11.62\%$, Spain $21.79\%$ | Bitwise invariance | **PASS** (100% preserved) |
| **Rollout Stability** | 15/15 stable continuations | Zero repetition loops | **PASS** (100% stable under $\gamma=1.00$) |
| **Batch vs RLS Equivalence** | Fwd: $2.93 \times 10^{-15}$, Rev: $2.68 \times 10^{-15}$ | $< 1.0 \times 10^{-12}$ | **PASS** (Float64 precision) |

---

### 4. Transition to Adaptive Fact-Specific Joint Solve

Stage 2 proves that:
1. Higher steering energy does not induce non-linear collapse, perplexity spikes, or rollout instability.
2. The response curves are highly linear across all 5 facts.
3. Canada and Australia do not need high $\beta$ ($\beta=0.50$ and $\beta=1.00$ are sufficient), while China needs $\approx 1.75$, Japan $\approx 2.75$, and France $\approx 3.50$.

**Proposed Next Step:** Execute the **Adaptive Fact-Specific Joint Solve**:
- Formulate targets $Y_{:, i} = \sqrt{1/F} \beta_f d_f$ with heterogeneous amplitudes:
  - $\beta_{\text{Canada}} = 0.50$
  - $\beta_{\text{Australia}} = 1.00$
  - $\beta_{\text{China}} = 1.75$
  - $\beta_{\text{Japan}} = 2.75$
  - $\beta_{\text{France}} = 3.50$
- Solve **jointly** in a single Batch TREC-LS + Streaming Woodbury RLS to audit cross-fact interference and evaluate whether all 5 facts simultaneously flip while minimizing total parameter drift.

Please confirm if you approve executing this Adaptive Fact-Specific Joint Solve.
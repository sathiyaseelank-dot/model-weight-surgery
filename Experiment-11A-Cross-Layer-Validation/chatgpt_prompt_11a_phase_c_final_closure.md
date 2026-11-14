# Experiment 11A Phase C: Final Audit Closure Submission

Dear Reviewer,

Thank you for the conditional approval and clear guidance. Below is the short final submission addressing all four outstanding requirements using our existing Phase C evaluation records, with zero new surgeries:

---

### 1. Arm D4 Full Brazil Two-Token Sequence Evaluation (All 4 Layers)

| Configuration | Token 1 Rank (`' São'`) | Token 1 Top-1 | Token 2 Rank (`' Paulo'`) | Greedy Generated Text | Sequence Correct | Joint Sequence Log-Prob $\log P(\text{São Paulo})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ($W_0$)** | Rank 7 | `' ____'` | **Rank 1** | `' ____.\n'` | False | -4.2846 |
| Layer 18 D4 | Rank 10 | `' ____'` | **Rank 1** | `' ____.\n'` | False | -3.8460 |
| Layer 20 D4 | Rank 7 | `' ____'` | **Rank 1** | `' ____.\n'` | False | -3.4718 |
| Layer 21 D4 | Rank 7 | `' ____'` | **Rank 1** | `' ____.\n'` | False | -3.2569 |
| **Layer 22 D4** | **Rank 6** | `' ____'` | **Rank 1** | `' ____.\n'` | False | **-3.1721** |

**Observation:** Exactly like Arm D2, Token 2 (`' Paulo'`) under teacher forcing is unconditionally **Rank 1** across all layers. However, Token 1 does not reach Rank 1 in unprompted greedy generation. Arm D4 achieves slightly higher joint sequence log-probabilities than Arm D2 (-3.1721 vs -3.2490 at Layer 22), but fails the strict greedy sequence generation requirement.

---

### 2. Arm D4 Disentangled Rollout Stability Metrics (All 4 Layers)

| Configuration | 3-Gram Repetition Loops | Loop-Free Rate | Exact Sequence Match vs $W_0$ | Changed Top-1 Predictions | Max Logit Drift |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline ($W_0$)** | 0/30 | 100.0% | 30/30 (100.0%) | 0/30 (0.0%) | 0.0000 |
| Layer 18 D4 | 10/30 | **66.7%** | 20/30 (66.7%) | 0/30 (0.0%) | 6.0635 |
| Layer 20 D4 | 10/30 | **66.7%** | 20/30 (66.7%) | 10/30 (33.3%) | 9.6162 |
| Layer 21 D4 | 10/30 | **66.7%** | 10/30 (33.3%) | 10/30 (33.3%) | 56.9810 |
| Layer 22 D4 | 10/30 | **66.7%** | 20/30 (66.7%) | 0/30 (0.0%) | 6.3108 |

**Crucial Scientific Contrast with Arm D2:**
- **Arm D2** achieves **0/30 loops (100.0% loop-free rate)** across all four layers, preserving high text stability.
- **Arm D4** induces **10/30 repetition loops (33.3% loop rate)** across all four layers, with severe logit drift reaching 56.98 at Layer 21.
- This decisively reinforces our architectural finding: **Arm D2 (High Syntactic Diversity, Strong Relational Explicitness) is strictly required for rollout preservation.**

---

### 3. Canonical WikiText-2 Perplexity for Arm D4 (20 Chunks, 10,220 Target Tokens)

| Configuration | Evaluated Target Tokens | NLL Loss | Canonical PPL (20 Chunks) | Delta PPL vs $W_0$ | % Delta PPL | Phase B (19 Chunks) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ($W_0$)** | 10,220 | 2.4798 | **11.9392** | 0.0000 | 0.00% | 11.6561 |
| Layer 18 D4 | 10,220 | 2.4824 | **11.9695** | +0.0303 | **+0.25%** | 11.6866 (+0.26%) |
| Layer 20 D4 | 10,220 | 2.4816 | **11.9603** | +0.0212 | **+0.18%** | 11.6874 (+0.27%) |
| Layer 21 D4 | 10,220 | 2.4818 | **11.9622** | +0.0230 | **+0.19%** | 11.6865 (+0.26%) |
| **Layer 22 D4** | 10,220 | 2.4808 | **11.9504** | +0.0113 | **+0.09%** | 11.6548 (-0.01%) |

**Observation:** All four D4 configurations remain well below +0.30% PPL shift on the canonical 20-chunk evaluation, confirming that perplexity degradation is negligible across all layers.

---

### 4. Reconciliation of Layer 22 Arm D4 F4 Accuracy (15/70 vs. 16/70)

The source of the perceived mismatch was a visual column cross-read of the Phase B summary table:
- In the Phase B master table, Column 3 was **Held-out Accuracy** (`16/100`), and Column 4 was **F4 Transfer Accuracy** (`15/70`).
- The underlying raw evaluation artifact `exp11a_phase_b_full_results.json` records:
  ```json
  "layer_22": {
    "D4": {
      "heldout_100": { "top1_count": 16, "accuracy_pct": 16.0 },
      "f4_70": { "top1_count": 15, "accuracy_pct": 21.428571428571427 },
      "fact_units": {
        "F1_France": 3, "F2_Japan": 2, "F3_China": 2, "F4_Canada": 3, "F5_Australia": 2,
        "F6_Germany": 0, "F7_Italy": 2, "F8_Spain": 0, "F9_Brazil": 0, "F10_India": 1
      }
    }
  }
  ```
  Summing the fact units yields: $3 + 2 + 2 + 3 + 2 + 0 + 2 + 0 + 0 + 1 = \mathbf{15/70}$.
- Both the Phase B results file, the Phase C results file, and the Phase C paired breakdown table evaluate to exactly **15/70 (21.4%)**. There is no internal inconsistency.

---

### Summary & Final Recommendation Alignment

We fully adopt your refined characterization:
1. **Layer 21 Arm D2** is our **Primary Reference Configuration** (Best performance/preservation balance: 21/100 held-out, 22/70 F4, 0 control regressions, 100% loop-free rollouts).
2. **Layer 22 Arm D2** is our **Efficiency Challenger Configuration** (12.6% lower $\|\Delta W\|_F = 0.8436$, highest $GE_{\text{margin}} = 23.71$, -0.02% PPL, with a trade-off of 1 control regression and 20/100 held-out).
3. **Generalization Efficiency Units Clarification:** $GE_{\text{acc}}$ is formally recorded as **percentage points per unit weight-update norm** ($\Delta \text{Acc}_{\text{pp}} / \|\Delta W\|_F$), serving as a descriptive efficiency metric rather than an independent statistical test.

With all four reporting items fully documented and reconciled, we request full closure of Experiment 11A Phase C.

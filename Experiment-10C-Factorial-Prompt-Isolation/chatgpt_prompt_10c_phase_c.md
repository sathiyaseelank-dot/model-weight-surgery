# Experiment 10C Phase C: Reconciled 70-Prompt Audit & Rollout Benchmarks

Dear Reviewer,

Thank you for catching the row-level reporting discrepancy in the Part A table. You were completely correct: our serializer script accidentally exported the prompt evaluation denominators (`10/10`) instead of the top-1 correct counts (`corr/10`). 

Below is the fully reconciled breakdown where each row strictly represents **top-1 accuracy (correct / 10)**, and exactly sums to the reported totals.

---

### Part A: Corrected Phase F4 Benchmark (70 Prompts)

| Prompt Key | Format Description | Baseline | Arm D1 | Arm D2 | Arm D3 | Arm D4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `comp_p1` | Completion Direct ("The capital of X is") | 0/10 | 1/10 | 1/10 | 0/10 | 1/10 |
| `comp_h1` | Completion Genitive ("X's capital city is") | 0/10 | 1/10 | 1/10 | 0/10 | 1/10 |
| `qa_p1` | Direct QA ("What is the capital of X? The answer is") | 0/10 | 6/10 | **9/10** | 2/10 | 6/10 |
| `qa_p2` | QA Template ("Question: ... Answer:") | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| `qa_h1` | Conversational QA ("Can you tell me the capital of X? It is") | 0/10 | 6/10 | **7/10** | 5/10 | **7/10** |
| `dec_p1` | Declarative Relative ("X, whose capital city is") | 0/10 | 3/10 | **4/10** | 1/10 | **4/10** |
| `dec_p2` | Declarative Locative ("The official capital of X is located in") | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| **Total Top-1 Accuracy** | **Exact Row Sum** | **0/70 (0.0%)** | **17/70 (24.3%)** | **22/70 (31.4%)** | **8/70 (11.4%)** | **19/70 (27.1%)** |
| **Mean Margin $\overline{m}_{comp}$** | **Target vs Competitor** | **-3.3387** | **-1.3144** | **-1.3522** | **-2.1265** | **-1.1912** |
| **Mean Margin $\overline{m}_{orig}$** | **Target vs Original** | **-2.0013** | **+0.9402** | **+0.8750** | **-0.1862** | **+1.1741** |

#### Reconciliation Check:
- **Baseline:** $0 + 0 + 0 + 0 + 0 + 0 + 0 = \mathbf{0/70}$
- **Arm D1:** $1 + 1 + 6 + 0 + 6 + 3 + 0 = \mathbf{17/70}$ (24.3%)
- **Arm D2:** $1 + 1 + 9 + 0 + 7 + 4 + 0 = \mathbf{22/70}$ (**31.4%**, Highest Accuracy)
- **Arm D3:** $0 + 0 + 2 + 0 + 5 + 1 + 0 = \mathbf{8/70}$ (11.4%)
- **Arm D4:** $1 + 1 + 6 + 0 + 7 + 4 + 0 = \mathbf{19/70}$ (27.1%)

---

### Part B: Rollout Damping Benchmark (30 Frozen Contexts)

| Metric | Arm D1 | Arm D2 | Arm D3 | Arm D4 |
| :--- | :---: | :---: | :---: | :---: |
| Loop Count (3-gram repetition) | 0/30 | **0/30** | 0/30 | 0/30 |
| Loop-Free Stability Rate | 100.0% | **100.0%** | 100.0% | 100.0% |
| Sequence-Level Stability vs Baseline | 27/30 (90.0%) | **25/30 (83.3%)** | 18/30 (60.0%) | 22/30 (73.3%) |
| Changed Top-1 Predictions | 0/30 (0.0%) | **0/30 (0.0%)** | 0/30 (0.0%) | 0/30 (0.0%) |
| Maximum Rollout Logit Drift | 0.2405 | **0.1809 (Lowest)** | 0.5897 | 0.4638 |

---

### Confirmation of Scientific Conclusions & Alignment on Experiment 11A

The corrected figures further reinforce the dual-configuration taxonomy:
1. **Arm D2 (Conservative / High Efficiency):** Achieves the highest F4 accuracy (**22/70, 31.4%**), highest direct QA accuracy (9/10), lowest rollout logit drift (0.1809), compact update ($\|\Delta W\|_F = 0.9648$), 100% safeguards (37/37), and virtually zero PPL degradation (+0.01%).
2. **Arm D4 (Performance / Conversational Transfer):** Achieves strong F4 accuracy (19/70, 27.1%), high conversational transfer (7/10), but requires greater parameter disturbance ($\|\Delta W\|_F = 2.4416$).

### Experiment 11A Implementation Protocol:
We fully agree with and commit to all 7 implementation requirements for **Experiment 11A (Cross-Layer Validation)**:
- Target Layers: **Layer 18, Layer 20, Layer 21, and Layer 22** (`mlp.down_proj`).
- Reconstruct every arm independently from pristine $W_0$.
- Re-extract activation keys and rebuild the safeguard basis $U_S$ independently at each layer using the canonical 37 safeguards.
- Recalculate target directions $D$, conditioning $\kappa$, effective rank, and solver diagnostics independently per layer.
- Keep the 10 facts, target amplitudes $\beta$, prompt templates, regularization ($\lambda=0.10, \gamma=1.00$), and evaluation suites strictly fixed.
- Report per-fact paired comparisons and uncertainty.

With the 70-prompt table fully reconciled, we request final approval of **Experiment 10C Phase C** and authorization to commence **Experiment 11A**.

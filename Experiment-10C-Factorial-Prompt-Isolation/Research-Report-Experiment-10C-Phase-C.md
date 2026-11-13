# Experiment 10C Phase C: Completion Audit Report (Reconciled)

**Date:** 2026-11-13 21:26:21  
**Model:** Qwen/Qwen2.5-0.5B (Layer 21 `mlp.down_proj`)  
**Scope:** Completion of the two required benchmarks from reviewer authorization:
1. **Original Phase F4 Benchmark:** 70-prompt evaluation across D1–D4 (Top-1, Target Ranks, $m_{comp}$, $m_{orig}$).
2. **Rollout Damping Benchmark:** 30 frozen contexts across D1–D4 (Loops, Stability, Drift, Top-1 Invariance).

---

## 1. Original Phase F4 Benchmark (70 Prompts) - Fully Reconciled

Evaluated across all 10 global facts using the canonical 7 prompt formats from Phase F4 (`comp_p1`, `comp_h1`, `qa_p1`, `qa_p2`, `qa_h1`, `dec_p1`, `dec_p2`).

### 1.1 Summary by Prompt Format (Top-1 Accuracy)

| Prompt Key | Format Description | Baseline | Arm D1 | Arm D2 | Arm D3 | Arm D4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `comp_p1` | Completion Direct ("The capital of X is") | 0/10 | 1/10 | 1/10 | 0/10 | 1/10 |
| `comp_h1` | Completion Genitive ("X's capital city is") | 0/10 | 1/10 | 1/10 | 0/10 | 1/10 |
| `qa_p1` | Direct QA ("What is the capital of X? The answer is") | 0/10 | 6/10 | **9/10** | 2/10 | 6/10 |
| `qa_p2` | QA Template ("Question: ... Answer:") | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| `qa_h1` | Conversational QA ("Can you tell me the capital of X? It is") | 0/10 | 6/10 | **7/10** | 5/10 | **7/10** |
| `dec_p1` | Declarative Relative ("X, whose capital city is") | 0/10 | 3/10 | **4/10** | 1/10 | **4/10** |
| `dec_p2` | Declarative Locative ("The official capital of X is located in") | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| **OVERALL (70)** | **Total Accuracy** | **0/70 (0.0%)** | **17/70 (24.3%)** | **22/70 (31.4%)** | **8/70 (11.4%)** | **19/70 (27.1%)** |
| **Mean Margin** | **Target vs Competitor $\overline{m}_{comp}$** | **-3.3387** | **-1.3144** | **-1.3522** | **-2.1265** | **-1.1912** |
| **Mean Margin** | **Target vs Original $\overline{m}_{orig}$** | **-2.0013** | **+0.9402** | **+0.8750** | **-0.1862** | **+1.1741** |

---

### 1.2 Per-Fact Breakdown across 70 Prompts (Mean Rank & Top-1 Counts)

| Fact ID | Country | Baseline Rank | Arm D1 Rank (Top-1) | Arm D2 Rank (Top-1) | Arm D3 Rank (Top-1) | Arm D4 Rank (Top-1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| F1_France | France | 34.0 | 10.7 (3/7) | 14.3 (3/7) | 19.1 (3/7) | 8.3 (4/7) |
| F2_Japan | Japan | 26.9 | 6.6 (1/7) | 7.1 (2/7) | 8.7 (1/7) | 4.9 (3/7) |
| F3_China | China | 11.0 | 5.7 (1/7) | 5.6 (2/7) | 6.9 (0/7) | 5.3 (2/7) |
| F4_Canada | Canada | 12.4 | 7.0 (3/7) | 7.0 (3/7) | 7.7 (2/7) | 6.4 (3/7) |
| F5_Australia | Australia | 9.6 | 6.9 (2/7) | 6.9 (2/7) | 7.9 (1/7) | 6.7 (2/7) |
| F6_Germany | Germany | 37.9 | 6.9 (1/7) | 7.4 (1/7) | 18.4 (0/7) | 9.0 (1/7) |
| F7_Italy | Italy | 16.1 | 9.3 (2/7) | 9.1 (2/7) | 9.1 (1/7) | 8.0 (2/7) |
| F8_Spain | Spain | 12.6 | 5.4 (1/7) | 5.6 (3/7) | 7.0 (0/7) | 5.3 (1/7) |
| F9_Brazil | Brazil | 17.6 | 12.0 (1/7) | 12.9 (1/7) | 12.4 (0/7) | 11.1 (0/7) |
| F10_India | India | 13.3 | 5.3 (2/7) | 5.1 (3/7) | 6.1 (0/7) | 5.1 (1/7) |

---

## 2. Rollout Damping Benchmark (30 Frozen Contexts)

Evaluated across 30 frozen contexts with 12-token greedy generation per prompt.

| Metric | Arm D1 | Arm D2 | Arm D3 | Arm D4 |
| :--- | :---: | :---: | :---: | :---: |
| Loop Count (3-gram repetition) | 0/30 | **0/30** | 0/30 | 0/30 |
| Loop-Free Stability Rate | 100.0% | **100.0%** | 100.0% | 100.0% |
| Sequence-Level Stability vs Baseline | 27/30 (90.0%) | **25/30 (83.3%)** | 18/30 (60.0%) | 22/30 (73.3%) |
| Changed Top-1 Prompt Predictions | 0/30 | **0/30** | 0/30 | 0/30 |
| Maximum Rollout Logit Drift | 0.2405 | **0.1809** | 0.5897 | 0.4638 |

---

## 3. Synthesis & Completion Audit Sign-Off

1. **Exact Row-Level & Total Reconciliation:**
   - Baseline: $0/10$ across all 7 formats = **0/70 (0.0%)**.
   - Arm D1: $1 + 1 + 6 + 0 + 6 + 3 + 0 = \mathbf{17/70}$ **(24.3%)**.
   - Arm D2: $1 + 1 + 9 + 0 + 7 + 4 + 0 = \mathbf{22/70}$ **(31.4%)** (HIGHEST ACCURACY).
   - Arm D3: $0 + 0 + 2 + 0 + 5 + 1 + 0 = \mathbf{8/70}$ **(11.4%)**.
   - Arm D4: $1 + 1 + 6 + 0 + 7 + 4 + 0 = \mathbf{19/70}$ **(27.1%)**.
2. **Rollout Damping is Flawlessly Preserved:**
   - Zero loops across all 30 contexts in all 4 arms (100% loop-free).
   - Zero changed top-1 prompt predictions across all 30 contexts in all 4 arms.
   - Arm D2 maintains the lowest peak logit drift (**0.1809**).

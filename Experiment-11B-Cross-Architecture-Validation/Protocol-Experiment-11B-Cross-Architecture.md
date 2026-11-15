# Experiment 11B: Cross-Architecture Validation Protocol
## Closed-Form Forward-Only Weight Surgery on SmolLM2-360M (Llama Architecture)

### 1. Executive Summary & Objective

In accordance with the reviewer's recommendation following the closure and sign-off of Experiment 11A, **Experiment 11B** tests whether the empirical principles established on `Qwen/Qwen2.5-0.5B` survive a change of model architecture without losing their fundamental properties:

1. **Prompt Geometry Advantage:** Does Arm D2 (High Syntactic Diversity + Strong Relational Explicitness) maintain its $\sim 2.5\times$ parameter efficiency advantage over Arm D4 on a completely distinct architecture?
2. **Depth Efficiency Scaling:** Does the update norm $\|\Delta W\|_F$ decrease monotonically with depth in a 32-layer LLaMA model?
3. **Preservation Invariance:** Does closed-form forward-only surgery with rank-31 safeguard projection and joint rollout damping preserve language modeling capability ($\Delta \text{PPL} \le 0.5\%$), achieve 0 repetition loops, and protect baseline control capabilities?

---

### 2. Architecture Comparison & Target Layer Mapping

| Dimension | Qwen2.5-0.5B (Exp 11A Model) | SmolLM2-360M (Exp 11B Model) | Architectural Difference |
| :--- | :---: | :---: | :--- |
| **Model Family** | `Qwen2ForCausalLM` | `LlamaForCausalLM` | LLaMA architecture vs Qwen2 |
| **Parameter Count** | 494M | 360M | 27% smaller parameter count |
| **Total Layers** | 24 layers | **32 layers** | 33% deeper layer hierarchy |
| **Hidden Dim ($d$)** | 896 | **960** | Distinct unembedding & activation space |
| **Intermediate Dim** | 4,864 | **2,560** | Different MLP width & down_proj dimension |
| **MLP Target Module** | `mlp.down_proj` ($896 \times 4864$) | `mlp.down_proj` ($960 \times 2560$) | Forward-only linear down-projection |
| **Tokenizer Vocab** | 151,936 tokens (Qwen BPE) | 49,152 tokens (LLaMA BPE) | Distinct subword token boundaries |

#### Normalized Depth Equivalence Mapping:
To conduct a fair cross-architecture comparison across depth:
- **75.0% Normalized Depth:** Qwen2.5 Layer 18 $\longleftrightarrow$ **SmolLM2 Layer 24** ($24/32 = 75.0\%$)
- **84.4% Normalized Depth:** Qwen2.5 Layer 20 ($83.3\%$) $\longleftrightarrow$ **SmolLM2 Layer 27** ($27/32 = 84.4\%$)
- **87.5% Normalized Depth (Primary Reference):** Qwen2.5 Layer 21 $\longleftrightarrow$ **SmolLM2 Layer 28** ($28/32 = 87.5\%$)
- **90.6% Normalized Depth (Efficiency Challenger):** Qwen2.5 Layer 22 ($91.7\%$) $\longleftrightarrow$ **SmolLM2 Layer 29** ($29/32 = 90.6\%$)

---

### 3. Experimental Controls & Hyperparameters

In accordance with the reviewer's instructions (*"Keep Layer 21 D2 as the reference method. Include Layer 22 D2 as the efficiency comparison. Do not introduce additional calibration variables during the first cross-architecture replication."*):

1. **Fixed Solver Hyperparameters:**
   - Gram Regularization: $\lambda = 0.10$
   - Rollout Damping Weight: $\gamma = 1.00$
   - Joint Weighting: $w_{\text{fact}} = \sqrt{1 / (2F)} = \sqrt{1 / 20}$, $w_{\text{roll}} = \sqrt{\gamma / F} = \sqrt{1 / 10}$
   - Precision: Float64 for all Gram inversions, projections, and $\Delta W$; pristine Float32 $W_0$ restored in `try...finally`.
2. **Benchmark Suites:**
   - **100-Prompt Held-Out Generalization Suite** (5 format families across all 10 country-capital facts)
   - **70-Prompt Canonical Phase F4 Suite** (7 format categories across all 10 country-capital facts)
   - **40 Protected Controls Audit** (4 format categories across 10 non-target countries)
   - **37 Canonical Safeguard Probes** (exact invariant prompt completions)
   - **30 Rollout Damping Contexts** (natural language completions, evaluated for loops, stability, and drift)
   - **WikiText-2 Perplexity** (canonical 20 chunks = 10,220 evaluated target tokens)
   - **Brazil Two-Token Sequence Evaluation** (`" São Paulo"`)

---

### 4. Proposed Two-Phase Execution Plan

- **Phase A — Geometry & Pre-Surgery Diagnostics:**
  - Tokenization verification and target displacement unit vectors $d_f$ in SmolLM2 unembedding space ($d=960$).
  - Activation key collection ($K_{\text{mlp}}$) at the input to `mlp.down_proj` for Arm D2 and Arm D4 across Layers 24, 27, 28, and 29.
  - Safeguard nullspace basis $U_S$ (rank-31) and key retention percentages.
  - Baseline $W_0$ evaluation on all 6 evaluation suites.
- **Phase B — Multi-Benchmark Cross-Layer Evaluation:**
  - Independent closed-form surgery runs for Arm D2 and Arm D4 across Layers 24, 27, 28, and 29 (8 total interventions).
  - Head-to-head comparison of Layer 28 (Primary Reference equivalent) vs Layer 29 (Efficiency Challenger equivalent).
  - Cross-architecture comparison table: Qwen2.5-0.5B vs SmolLM2-360M.

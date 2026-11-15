# Experiment 11B Proposal: Cross-Architecture Validation Protocol
## Closed-Form Forward-Only Weight Surgery on SmolLM2-360M (Llama Architecture)

Dear Reviewer,

Following your official approval and sign-off on **Experiment 11A**, we submit for your formal review and authorization the experimental design for **Experiment 11B: Cross-Architecture Validation**.

---

### 1. Architectural Selection & Normalized Depth Mapping

To test whether our findings generalize beyond `Qwen/Qwen2.5-0.5B`, we propose validating on **`HuggingFaceTB/SmolLM2-360M`**, an open-weights modern **Llama-family** model (`LlamaForCausalLM`).

| Architectural Dimension | Qwen2.5-0.5B (Exp 11A) | SmolLM2-360M (Exp 11B) | Validation Dimension |
| :--- | :---: | :---: | :--- |
| **Model Family** | `Qwen2ForCausalLM` | `LlamaForCausalLM` | Architectural Family Generalization |
| **Parameter Count** | 494M | 360M | Model Capacity Difference |
| **Layer Count** | 24 layers | **32 layers** | Hierarchy Depth Difference |
| **Hidden Dim ($d$)** | 896 | **960** | Unembedding / Activation Space |
| **MLP Intermediate Dim** | 4,864 | **2,560** | Linear Layer Aspect Ratio |
| **MLP Target Module** | `mlp.down_proj` ($896 \times 4864$) | `mlp.down_proj` ($960 \times 2560$) | Forward-Only Surgery Target |
| **Vocabulary Size** | 151,936 (Qwen BPE) | 49,152 (LLaMA BPE) | Subword Tokenizer Differences |

#### Normalized Layer Equivalence Mapping:
We map the four tested depth points proportionally:
- **75.0% Normalized Depth:** Qwen2.5 Layer 18 $\longleftrightarrow$ **SmolLM2 Layer 24** ($24/32 = 75.0\%$)
- **84.4% Normalized Depth:** Qwen2.5 Layer 20 ($83.3\%$) $\longleftrightarrow$ **SmolLM2 Layer 27** ($27/32 = 84.4\%$)
- **87.5% Normalized Depth (Primary Reference):** Qwen2.5 Layer 21 $\longleftrightarrow$ **SmolLM2 Layer 28** ($28/32 = 87.5\%$)
- **90.6% Normalized Depth (Efficiency Challenger):** Qwen2.5 Layer 22 ($91.7\%$) $\longleftrightarrow$ **SmolLM2 Layer 29** ($29/32 = 90.6\%$)

---

### 2. Hypotheses Under Evaluation

1. **Hypothesis 1 (Prompt Geometry Advantage):** Arm D2 (High Syntactic Diversity + Strong Relational Explicitness) will maintain its $\sim 2.5\times$ parameter efficiency advantage over Arm D4 on the LLaMA architecture.
2. **Hypothesis 2 (Depth Efficiency Scaling):** Surgery update norm $\|\Delta W\|_F$ will decrease monotonically as intervention depth increases from Layer 24 to Layer 29 in a 32-layer hierarchy.
3. **Hypothesis 3 (Preservation Invariance):** Controlled closed-form TREC-LS with rank-31 safeguard projection and rollout damping will preserve WikiText-2 perplexity ($\le +0.50\%$), achieve 0 repetition loops on natural rollouts, and maintain baseline control capabilities.

---

### 3. Strict Experimental Bounds & Controls

In accordance with your guidance:
1. **Fixed Solver Hyperparameters:**
   - Gram Regularization: $\lambda = 0.10$
   - Rollout Damping Weight: $\gamma = 1.00$
   - Joint Weighting: $w_{\text{fact}} = \sqrt{1 / (2F)} = \sqrt{1 / 20}$, $w_{\text{roll}} = \sqrt{\gamma / F} = \sqrt{1 / 10}$
   - Precision: Float64 for all Gram inversions, projections, and $\Delta W$; pristine Float32 $W_0$ restored in `try...finally`.
2. **Full Benchmark Coverage:**
   - 100-Prompt Held-Out Generalization Suite (5 format categories)
   - 70-Prompt Canonical Phase F4 Benchmark (7 format categories)
   - 40 Protected Controls Audit
   - 37 Canonical Safeguard Probes
   - 30 Natural-Language Rollout Damping Contexts (loops, exact match vs $W_0$, top-1 change, max drift)
   - WikiText-2 Validation Perplexity (canonical 20 chunks = 10,220 target tokens)
   - Brazil Two-Token Sequence Evaluation (`" São Paulo"`)

---

### 4. Proposed Two-Phase Execution Plan

- **Phase A (Geometry & Pre-Surgery Diagnostics):**
  - Extract displacement unit vectors $d_f$ in SmolLM2's unembedding space ($d=960$).
  - Collect activation keys $K_{\text{mlp}}$ for Arm D2 and Arm D4 across Layers 24, 27, 28, and 29.
  - Compute condition numbers, effective ranks, and safeguard nullspace projections $U_S^{(L)}$ (rank-31).
  - Run full baseline $W_0$ benchmark evaluation.
- **Phase B (Multi-Benchmark Cross-Layer Evaluation):**
  - Execute all 8 interventions (Arms D2 & D4 across Layers 24, 27, 28, 29).
  - Perform head-to-head comparison of Layer 28 (Primary Reference equivalent) vs Layer 29 (Efficiency Challenger equivalent).
  - Compile the cross-architecture validation synthesis comparing Qwen2.5-0.5B and SmolLM2-360M.

Please review this experimental design and let us know if you approve proceeding with **Experiment 11B Phase A**.

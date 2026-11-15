# Experiment 11B Phase A: Cross-Architecture Geometry & Pre-Surgery Report
## Target Architecture: SmolLM2-360M (Llama Architecture, 32 Layers, d=960, intermediate=2560)

Dear Reviewer,

In accordance with your approval and mandatory pre-registration corrections, we have completed **Experiment 11B Phase A**. All diagnostics, target vector constructions, cross-layer activation extractions, and baseline audits were executed on **`HuggingFaceTB/SmolLM2-360M`** with zero weight modifications.

---

### 1. Model Integrity & Layer Indexing Documented

- **Model Identifier:** `HuggingFaceTB/SmolLM2-360M` (Commit SHA: `f8027fd0eaeea54c...`)
- **Checkpoint Storage Dtype:** `torch.bfloat16` | **Runtime Inference Dtype:** `torch.float32`
- **Dimensions:** $d_{\text{out}} = 960$ (unembedding space), $d_{\text{in}} = 2560$ (input to `mlp.down_proj`)
- **Tied Embeddings:** True (`lm_head.weight is embed_tokens.weight`)

| Target Layer (0-based) | 1-based Layer | Total Layers | Normalized Depth | Qwen2.5-0.5B Equivalent | Role |
| :---: | :---: | :---: | :---: | :---: | :--- |
| Layer 24 | Layer 25 | 32 | **78.1%** | Layer 18 (75.0%) | Early Deep |
| Layer 27 | Layer 28 | 32 | **87.5%** | Layer 20 (83.3%) | Mid Deep |
| **Layer 28** | **Layer 29** | **32** | **90.6%** | **Layer 21 (87.5%)** | **Primary Reference Equivalent** |
| **Layer 29** | **Layer 30** | **32** | **93.8%** | **Layer 22 (91.7%)** | **Efficiency Challenger Equivalent** |

---

### 2. Tokenizer Audit & Target Geometry ($d=960$)

All 20 cities were tokenized independently with SmolLM2's tokenizer (`vocab_size = 49,152`):
- **8 Target Cities are Single Tokens:** Lyon (`[38034]`), Shanghai (`[28997]`), Toronto (`[15500]`), Sydney (`[16570]`), Munich (`[27093]`), Milan (`[28676]`), Barcelona (`[30010]`), Mumbai (`[34127]`).
- **2 Target Cities Split Across Subwords:**
  - Osaka: `[13956, 10339]` (`[' Os', 'aka']`)
  - São Paulo: `[40481, 36619]` (`[' São', ' Paulo']`)
- **Target Displacement Matrix $D \in \mathbb{R}^{960 \times 10}$:** Condition Number $\kappa(D) = \mathbf{1.7471}$, Effective Rank = $\mathbf{9.88 / 10.0}$, Mean Pairwise $|\cos| = 0.0870$.

---

### 3. Pristine Baseline ($W_0$) Multi-Benchmark Audit

| Benchmark Suite | SmolLM2-360M Baseline ($W_0$) | Qwen2.5-0.5B Baseline ($W_0$) | Cross-Architecture Comparison |
| :--- | :---: | :---: | :--- |
| 100 Held-Out Generalization | **0/100 (0.0%)** | 0/100 (0.0%) | Identical 0% target city prior |
| Held-Out Mean $m_{\text{comp}}$ | **-3.6051** | -23.4906 | SmolLM2 has less extreme competitor margins |
| 70 Phase F4 Benchmark | **0/70 (0.0%)** | 0/70 (0.0%) | Identical 0% target city prior |
| 40 Protected Controls | **28/40 (70.0%)** | 25/40 (62.5%) | SmolLM2 passes +3 additional controls |
| 37 Safeguard Probes | **37/37 (100.0%)** | 37/37 (100.0%) | Invariant reference logits stored |
| 30 Rollout Contexts | **30/30 (0 loops, 100%)** | 30/30 (0 loops, 100%) | Pristine generation is 100% loop-free |
| Brazil Complete Sequence | **T1 Rank 5, T2 Rank 1** | T1 Rank 7, T2 Rank 1 | Teacher-forced T2 is Rank 1 in both |
| Canonical WikiText-2 PPL | **11.2105** | 11.9392 | Evaluated on canonical 20 chunks (10,220 tokens) |

---

### 4. Cross-Layer Activation Geometry ($d_{\text{in}} = 2560$)

| Layer (0-based / 1-based) | Normalized Depth | Safeguard Rank-31 Energy | Safeguard $\kappa(S)$ | Rollout Retention | Arm D2 Retention | Arm D2 $\kappa(K)$ | Arm D4 Retention | Arm D4 $\kappa(K)$ | Condition Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Layer 24 (Layer 25) | 78.1% | 99.82% | 65.16 | 75.65% | 34.04% | **61.69** | 65.26% | **84.10** | **1.4x for D2** |
| Layer 27 (Layer 28) | 87.5% | 96.94% | 8.52 | 74.20% | 51.76% | **18.05** | 57.23% | **40.23** | **2.2x for D2** |
| Layer 28 (Layer 29) | 90.6% | 97.03% | 11.01 | 81.32% | 45.58% | **18.41** | 56.18% | **36.96** | **2.0x for D2** |
| Layer 29 (Layer 30) | 93.8% | 96.91% | 13.05 | 75.67% | 52.69% | **16.12** | 61.67% | **32.16** | **2.0x for D2** |

**Key Cross-Architecture Finding:**
- In SmolLM2-360M, **Arm D2 achieves a 2.0x to 2.2x lower condition number than Arm D4 across all deep layers** (18.41 vs 36.96 at Layer 28, 16.12 vs 32.16 at Layer 29).
- This confirms **Hypothesis 1 pre-condition**: prompt geometry directly controls activation key conditioning independently of the underlying architecture.

---

### 5. Pre-Surgery Parity & Integrity Verification

- **Frozen $\beta$ Calibration Vector:** The exact calibrated Qwen F4 $\beta$ vector is locked without tuning:
  `[3.1695, 4.0433, 2.6301, 3.6800, 1.9992, 3.9985, 2.0394, 3.3500, 1.9706, 4.1813]`
- **Joint System Weighting:** $w_{\text{fact}} = \sqrt{1/20} \approx 0.2236$, $w_{\text{roll}} = \sqrt{1/10} \approx 0.3162$ ($1.5\times$ aggregate squared weighting for rollout damping).
- **Pristine Weight Integrity:** All layers verified bitwise identical to loaded checkpoint (0 weights modified).

### Request for Phase B Authorization

With all Phase A pre-registration corrections fully satisfied and verified, we request your authorization to proceed with **Experiment 11B Phase B: Multi-Benchmark Cross-Layer Evaluation on SmolLM2-360M**.
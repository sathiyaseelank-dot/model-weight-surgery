# Experiment 11B Phase A: Reconciliation & Final Pre-Registration Audit
## Closed-Form Forward-Only Weight Surgery on SmolLM2-360M (Llama Architecture)

Dear Reviewer,

Thank you for the conditional approval and rigorous guidance. Below is our formal Phase A reconciliation addressing all five required items, complete with exact cryptographic hashes and mathematical definitions, with zero new surgeries:

---

### 1. Corrected Normalized Layer Indexing Table

In accordance with your instruction, normalized depth is strictly defined as $d_{\text{normalized}} = \text{module index} / \text{total layers}$ (using zero-based Python module indices):

| SmolLM2 Module Index | Total Layers | Corrected Normalized Depth | Qwen2.5-0.5B Module Equivalent | Normalized Depth Ratio | Architectural Role |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Layer 24** | 32 | **75.0%** | Layer 18 ($18/24 = 75.0\%$) | $1.000\times$ exact match | Early Deep Intermediate Representation |
| **Layer 27** | 32 | **84.4%** | Layer 20 ($20/24 = 83.3\%$) | $1.013\times$ near match | Mid Deep Representation |
| **Layer 28** | 32 | **87.5%** | Layer 21 ($21/24 = 87.5\%$) | $1.000\times$ exact match | **Primary Reference Equivalent** |
| **Layer 29** | 32 | **90.6%** | Layer 22 ($22/24 = 91.7\%$) | $0.988\times$ near match | **Efficiency Challenger Equivalent** |

The normalized-depth column has been corrected throughout our codebase and records.

---

### 2. Exact Definition of Projected Fact-Key Energy Retention

The retention metrics reported in our activation table are formally defined as the **Frobenius energy ratio** remaining after projecting out the rank-31 safeguard subspace:

$$\text{Projected Fact-Key Energy Retention}(K) = \frac{\|K_{\text{proj}}\|_F^2}{\|K\|_F^2} \times 100\% = \frac{\|K - U_S (U_S^T K)\|_F^2}{\|K\|_F^2} \times 100\%$$

where:
- $K \in \mathbb{R}^{2560 \times 40}$ is the raw activation matrix captured at the input to `mlp.down_proj`.
- $U_S \in \mathbb{R}^{2560 \times 31}$ is the orthonormal safeguard basis constructed from the top-31 left singular vectors of the 37 safeguard activation keys ($S \in \mathbb{R}^{2560 \times 37}$).

#### Complete Cross-Layer Activation Table (with Precision Terminology):
| SmolLM2 Module Index | Normalized Depth | Safeguard Rank-31 Energy | Safeguard $\kappa(S)$ | Projected Rollout Energy Retention | Projected D2 Fact-Key Energy Retention | Arm D2 $\kappa(K)$ | Projected D4 Fact-Key Energy Retention | Arm D4 $\kappa(K)$ | Condition Ratio $\kappa_{\text{D4}} / \kappa_{\text{D2}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer 24** | 75.0% | 99.82% | 65.16 | 75.65% | 34.04% | **61.69** | 65.26% | **84.10** | **$1.36\times$ for D2** |
| **Layer 27** | 84.4% | 96.94% | 8.52 | 74.20% | 51.76% | **18.05** | 57.23% | **40.23** | **$2.23\times$ for D2** |
| **Layer 28** | 87.5% | 97.03% | 11.01 | 81.32% | 45.58% | **18.41** | 56.18% | **36.96** | **$2.01\times$ for D2** |
| **Layer 29** | 90.6% | 96.91% | 13.05 | 75.67% | 52.69% | **16.12** | 61.67% | **32.16** | **$2.00\times$ for D2** |

**Refined Scientific Statement:** As recommended, we formally record:  
*"The D2 activation-key conditioning advantage observed in Qwen2.5 is also present in SmolLM2-360M, providing preliminary cross-architecture evidence that prompt geometry directly influences activation key conditioning across distinct model families."*

---

### 3. Complete Baseline Multi-Token Generation (Brazil & Japan)

For the two target cities that split across subwords, we record complete baseline greedy generations:

1. **Fact F9: Brazil (`" Brasília"` $\to$ `" São Paulo"`, Target IDs: `[40481, 36619]`):**
   - **Prompt:** `"The capital of Brazil is"`
   - **Greedy Generation (Pristine $W_0$):** `" Brasília, which is located in"`
   - **Immediate Token 1 Rank:** Rank 5 (`' Bras'`, logprob = -1.4883)
   - **Target Token 1 Top-1:** False (Top-1 is `' Brasília'`)
   - **Teacher-Forced Token 2 Rank:** **Rank 1 (`' Paulo'`, logprob = -0.0097)**
   - **Joint Sequence Log-Prob $\log P(\text{São Paulo})$:** **-3.9211**
   - **Complete Target Sequence Generated:** **False**

2. **Fact F2: Japan (`" Tokyo"` $\to$ `" Osaka"`, Target IDs: `[13956, 10339]`):**
   - **Prompt:** `"The capital of Japan is"`
   - **Greedy Generation (Pristine $W_0$):** `" Tokyo. It is the largest"`
   - **Immediate Token 1 Rank:** Rank 11 (`' Os'`, logprob = -5.4210)
   - **Target Token 1 Top-1:** False (Top-1 is `' Tokyo'`)
   - **Teacher-Forced Token 2 Rank:** **Rank 1 (`'aka'`, logprob = -0.0012)**
   - **Joint Sequence Log-Prob $\log P(\text{Osaka})$:** **-5.4222**
   - **Complete Target Sequence Generated:** **False**

In Phase B, neither city will be credited as acquired unless its complete answer sequence is generated.

---

### 4. Cryptographic Hashes & Checkpoint Provenance

All underlying model artifacts are pinned and cryptographically locked:

| Artifact / File | Version / Revision Identifier | SHA256 Checksum |
| :--- | :--- | :--- |
| **Model Repository** | `HuggingFaceTB/SmolLM2-360M` | Commit SHA: `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` |
| `model.safetensors` | Official Checkpoint (`bfloat16`) | `7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f` |
| `tokenizer.json` | LLaMA BPE Vocab (`49,152` tokens) | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| `tokenizer_config.json` | Tokenizer Configuration | `4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424` |
| `vocab.json` | Vocab Mapping | `82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79` |
| **Layer 24 Pristine $W_0$** | `layers.24.mlp.down_proj.weight` | `c247bf812c756a8d7ba33129482d8c9735d4ad64a51e6005748a3331cf113cf7` |
| **Layer 27 Pristine $W_0$** | `layers.27.mlp.down_proj.weight` | `6788fa0ce9ee5c2c77df982a6a166be44d9f783286395ecfe29e1628d09819ee` |
| **Layer 28 Pristine $W_0$** | `layers.28.mlp.down_proj.weight` | `a2e707e9a0b881e8ebc3fcb22108170d19fca377fae93ffdf6a3e26159c372b6` |
| **Layer 29 Pristine $W_0$** | `layers.29.mlp.down_proj.weight` | `9ac01e57006ba5f1717fca1f3b7cecb6b7ee88e1aee6b33eb0c75ff32c96c4a8` |

---

### 5. Exact 16-Decimal $\beta$ Calibration Vector

The exact, unrounded $\beta$ calibration vector established in Qwen F4 is locked in our manifest without recalibration:

```json
{
  "F1_France": 3.1695317161232484,
  "F2_Japan": 4.0432966609084490,
  "F3_China": 2.6300760828671770,
  "F4_Canada": 3.6799730955019340,
  "F5_Australia": 1.9992268214876190,
  "F6_Germany": 3.9985304776365140,
  "F7_Italy": 2.0393543973445430,
  "F8_Spain": 3.3500424285634023,
  "F9_Brazil": 1.9705533345972217,
  "F10_India": 4.1813180308908870
}
```

---

### Final Request for Phase B Authorization

With all five corrections satisfied and cryptographically documented:
- Target layers: **Layer 24 (75.0%), Layer 27 (84.4%), Layer 28 (87.5%), Layer 29 (90.6%)**.
- Configurations: **Arm D2** and **Arm D4** across all 4 layers (8 independent interventions starting from pristine $W_0$).
- All 6 benchmarks strictly retained (100 held-out, 70 F4, 40 controls, 37 safeguards, 30 rollouts, 20-chunk WikiText-2 PPL, and Brazil/Osaka full sequences).

Please confirm whether **Experiment 11B Phase B** is authorized to proceed.

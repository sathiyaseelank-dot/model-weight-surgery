# Research Update: Experiment 10B Phase D.1 — Structured Diversity Audit & Clarifications

Thank you for the rigorous review. Below is the complete **Phase D.1 Audit Report** addressing all six requested verifications prior to surgery.

---

### 1. F1 Target & Asset Verification: France -> Lyon Confirmed

- **F1 Target Correction:** The mention of 'Marseille' in the text table of our previous update was an isolated editorial typo in the markdown draft. In the dataset manifest, code, captured activations, and optimization vectors, **the target for France has always been and remains Lyon**.
- **Full Target & Amplitude Verification:** All 10 targets, original cities, and calibrated amplitudes $\beta_{F4}$ match Phase F4 bitwise:

| Fact ID | Country | Original Capital | Target City | Calibrated $\beta_{F4}$ | Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **F1_France** | France | Paris | **Lyon** | 3.169532 | Match F4 |
| **F2_Japan** | Japan | Tokyo | **Osaka** | 4.043297 | Match F4 |
| **F3_China** | China | Beijing | **Shanghai** | 2.630076 | Match F4 |
| **F4_Canada** | Canada | Ottawa | **Toronto** | 3.679973 | Match F4 |
| **F5_Australia** | Australia | Canberra | **Sydney** | 1.999227 | Match F4 |
| **F6_Germany** | Germany | Berlin | **Munich** | 3.998530 | Match F4 |
| **F7_Italy** | Italy | Rome | **Milan** | 2.039354 | Match F4 |
| **F8_Spain** | Spain | Madrid | **Barcelona** | 3.350042 | Match F4 |
| **F9_Brazil** | Brazil | Brasília | **São Paulo** | 1.970553 | Match F4 |
| **F10_India** | India | New Delhi | **Mumbai** | 4.181318 | Match F4 |

- **Target Matrix $D$ Cryptographic SHA256:** `bf5c23a84544c4266ce722ed4df15a6a6b04c90e6585ed94409878b8fc53e38b` (Bitwise identical to Phase C baseline and Phase F4).
---

### 2. Dataset Overlap Clarification

We distinguish between string-level deduplication and token vocabulary overlap:
1. **Zero Duplicate Prompts:** All 40 Arm C training prompts are strictly unique with zero repetition.
2. **Zero Train/Evaluation Prompt Overlap:** No training prompt appears in the 70-prompt benchmark, the 100-prompt held-out evaluation suite, the 30 rollout damping prompts, or the 37 safeguard probes.
3. **Zero Target Leakage:** No training prompt reveals or mentions the target counterfactual city (e.g. 'Lyon' or 'Osaka') in its prompt context.
4. **Standard Vocabulary Overlap:** As expected in natural language NLP benchmarks, common lexical tokens (such as country names, 'capital', 'city', 'governance', 'answer', 'is') are shared across training and test distributions.
---

### 3. Per-Fact Geometry: Pairwise Cosine Similarities Among K1–K4

Here is the exact breakdown of all six pairwise cosine similarities among the four linguistic frames for each of the 10 facts in Arm C:

| Fact ID | Country | K1, K2 (QA-Conv) | K1, K3 (QA-Dec) | K1, K4 (QA-Bridge) | K2, K3 (Conv-Dec) | K2, K4 (Conv-Bridge) | K3, K4 (Dec-Bridge) | Mean Intra |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1_France** | France | 0.2161 | 0.8066 | 0.3380 | 0.4043 | 0.6580 | 0.6123 | **0.5059** |
| **F2_Japan** | Japan | 0.1643 | 0.6107 | 0.2574 | 0.4762 | 0.6450 | 0.6977 | **0.4752** |
| **F3_China** | China | 0.3064 | 0.6658 | 0.3742 | 0.5604 | 0.5736 | 0.7122 | **0.5321** |
| **F4_Canada** | Canada | 0.0661 | 0.8543 | 0.3442 | 0.2285 | 0.6201 | 0.5876 | **0.4501** |
| **F5_Australia** | Australia | 0.0321 | 0.5765 | 0.1307 | 0.3786 | 0.6430 | 0.6426 | **0.4006** |
| **F6_Germany** | Germany | 0.1526 | 0.6851 | 0.2301 | 0.4040 | 0.6597 | 0.6122 | **0.4573** |
| **F7_Italy** | Italy | 0.1367 | 0.7819 | 0.2985 | 0.3341 | 0.6346 | 0.5949 | **0.4634** |
| **F8_Spain** | Spain | 0.0758 | 0.6816 | 0.1601 | 0.3281 | 0.6433 | 0.5561 | **0.4075** |
| **F9_Brazil** | Brazil | -0.0189 | 0.7105 | 0.0359 | 0.2601 | 0.6417 | 0.4494 | **0.3465** |
| **F10_India** | India | 0.0394 | 0.6999 | 0.3876 | 0.2116 | 0.4666 | 0.7641 | **0.4282** |
| **Average** | **All Facts** | **0.1171** | **0.7073** | **0.2557** | **0.3586** | **0.6186** | **0.6229** | **0.4467** |

#### Mechanistic Observations on Prompt Families:
1. **High Collinearity between K1 and K3 (Mean: 0.7073):** K1 (`Which city is officially recognized as the capital of {Country}? The answer is`) and K3 (`The designated sovereign capital city of {Country} is`) both terminate on the token `' is'` immediately preceded by a formal noun phrase. This provides a coherent structural anchor.
2. **Low Collinearity between K1 and K2 (Mean: 0.1171):** K2 (`If someone inquires about the capital city of {Country}, the correct reply is`) spans a conversational discourse frame, causing the final token activation to depart from the canonical direct question manifold.
3. **Consistent Bridging of K4 (Mean with K2: 0.6186, with K3: 0.6229):** K4 (`The primary seat of national governance and capital of {Country} is`) exhibits moderate-to-high similarity with both K2 and K3, functioning as an effective geometric bridge across conversational and declarative frames.
---

### 4. Cross-Fact Geometry: Inter-Fact Similarities

The top 10 highest-similarity fact pairs in Arm C (averaged over the $4 \times 4 = 16$ cross-prompts):

| Rank | Fact Pair | Arm C Mean Cosine | Arm B Mean Cosine | Arm A Mean Cosine | Interpretation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| #1 | Japan — China | **0.4533** | 0.3484 | 0.2872 | Shared semantic/geographic embeddings |
| #2 | Canada — Australia | **0.3688** | 0.3275 | 0.2821 | Shared semantic/geographic embeddings |
| #3 | China — Germany | **0.3463** | 0.2356 | 0.1367 | Shared semantic/geographic embeddings |
| #4 | Japan — Germany | **0.3339** | 0.2575 | 0.1374 | Shared semantic/geographic embeddings |
| #5 | France — Germany | **0.3211** | 0.2715 | 0.1829 | Shared semantic/geographic embeddings |
| #6 | Japan — Australia | **0.3207** | 0.1974 | 0.0923 | Shared semantic/geographic embeddings |
| #7 | France — Spain | **0.3204** | 0.2725 | 0.1843 | Shared semantic/geographic embeddings |
| #8 | China — Australia | **0.3176** | 0.1808 | 0.0604 | Shared semantic/geographic embeddings |

**Key Takeaway:** Across all three arms, **Japan–China** and **Canada–Australia** consistently represent the highest cross-fact similarities. In Arm C, the longer descriptive framing ('designated sovereign capital city', 'primary seat of national governance') activates shared high-level governance/statecraft representations across all countries, raising overall inter-fact baseline similarity to 0.2819.
---

### 5. Joint Gram Conditioning: Why Rollout Keys Increase Condition Number

You noted that the joint Gram condition number $\kappa(X^T X)$ is higher than the fact-only Gram condition number $\kappa(K^T K)$ in all three arms. We confirm your insight and provide the exact mathematical mechanism:

1. **Subspace Expansion:** The fact Gram matrix $K^T K$ operates on $\mathbb{R}^{40 \times 40}$, while the joint Gram matrix $X^T X$ operates on $\mathbb{R}^{70 \times 70}$ (40 fact keys + 30 rollout damping keys).
2. **Low-Energy Rollout Weighting:** The rollout keys are weighted by $w_{\text{roll}} = \sqrt{1/10} \approx 0.3162$ to prevent damping constraints from overpowering factual steering. In the Gram matrix $(w_{\text{roll}} R)^T (w_{\text{roll}} R)$, this reduces their energy by a factor of 10 ($w_{\text{roll}}^2 = 0.10$).
3. **Spectrum Broadening:** When appending 30 low-weighted keys spanning new residual directions, the minimum eigenvalue $\lambda_{\min}$ drops by a factor of ~20:
   - **Arm A:** $\lambda_{\min}$ drops from $3.255 \to 0.1497$ (Condition number rises $2118 \to 6006$)
   - **Arm B:** $\lambda_{\min}$ drops from $16.905 \to 0.7655$ (Condition number rises $604 \to 1338$)
   - **Arm C:** $\lambda_{\min}$ drops from $2.463 \to 0.1152$ (Condition number rises $2035 \to 7592$)
4. **Conclusion:** Rollout keys do **not** serve as a numerical conditioner; their sole function is **objective regularisation** (forcing $\Delta W R = 0$) to protect the autoregressive generation loop from drifting during multi-token rollouts.
---

### 6. Interpretation of K4 Contextual Bridge

We confirm your observation regarding K4 (`The primary seat of national governance and capital of {Country} is`):

- All four templates deliberately retain an explicit lexical anchor to `'capital'`.
- K4 is **not** designed as an implicit or indirect narrative clue (e.g. 'Where the prime minister resides'). Rather, it tests **contextual syntactic robustness** while ensuring the factual relation remains explicit.
- This prevents uncontrolled semantic drift while evaluating whether adding formal governance syntax preserves generalization across distinct conversational and declarative queries.
---

### 7. Request for Authorization (Experiment 10B-E)

All six checkpoints have been fully addressed and verified:

1. France $\to$ Lyon confirmed bitwise.
2. All 10 targets, amplitudes, and matrix $D$ verified.
3. Deduplication and leakage audited.
4. All 6 pairwise cosines per fact detailed.
5. Inter-fact similarity structure characterized.
6. Mathematical mechanism of joint Gram conditioning clarified.
7. K4 explicit anchor confirmed.

We request your authorization to proceed with **Experiment 10B-E: Closed-Form Arm C Surgery & Comparative Generalization Evaluation** across the full benchmark suite.
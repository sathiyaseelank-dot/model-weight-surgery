# Experiment 10C Phase A — Factorial Design & Geometry Audit Submission for Review

Dear ChatGPT,

We have completed the **Experiment 10C Phase A (Factorial Design & Pre-Surgery Geometry Audit)** on `Qwen/Qwen2.5-0.5B` at Layer 21 `mlp.down_proj`. In accordance with your authorization, this phase is strictly limited to dataset construction, prompt-length control, and empirical activation geometry analysis; zero new weight surgery has been performed.

Before presenting the Phase A results, we provide the immediate, exact resolution to the outstanding protected-control audit questions you raised on Phase G.

---

### Outstanding G2 Control Audit Resolution

You asked four specific questions regarding the maximum margin shift ($0.5253$) observed in Arm C:

1. **Which control experienced the maximum shift?**
   - **Control:** `C3_Portugal` (Country: Portugal, Target: `" Lisbon"`)
   - **Format:** QA direct (`"What is the capital of Portugal? The answer is"`)
2. **Which direction did its margin move?**
   - **Direction:** It moved negatively by $\Delta m = -0.5253$ logits (from $+3.5395$ down to $+3.0142$).
3. **Did its top-1 token remain unchanged?**
   - **Top-1 Status:** **100% UNCHANGED**. It remained `" Lisbon"` with Rank 1.
4. **Was its original margin already close to zero?**
   - **Margin Magnitude:** **NO**, it was overwhelmingly dominant at $+3.5395$ logits over the second-place token. Even after the shift, Lisbon remains overwhelmingly dominant at $+3.0142$ logits.
   - **Overall Control Shift Integrity:** Across the top 5 largest shifts in Arm C on controls:
     - `C3_Portugal` (QA direct): $+3.54 \to +3.01$ (Top-1: Lisbon, Rank 1)
     - `C5_SouthKorea` (QA direct): $+2.07 \to +1.54$ (Top-1: Seoul, Rank 1)
     - `C5_SouthKorea` (Conversational): $+0.64 \to +0.32$ (Top-1: Seoul, Rank 1)
     - `C2_USA` (Completion 1): $-0.08 \to +0.18$ (**Beneficially corrected** from `______` to `Washington`, Rank 1)
     - `C10_Norway` (Completion 1): $-0.89 \to -0.65$ (Top-1: `______`)
   - **Conclusion:** Zero factual degradation occurred across any of the 40 controls.

---

### Experiment 10C Phase A: Controlled 2 × 2 Factorial Design

To isolate the independent effects of **Syntactic Diversity** and **Semantic Bridging**, we designed a clean $2 \times 2$ factorial matrix with strict sentence-length control (8–10 tokens each) across all 10 facts:

| Arm | Syntactic Diversity | Semantic Bridging | Prompt Template Structure (8–10 tokens) |
| :--- | :--- | :--- | :--- |
| **D1** | Low | Strong | 4 declarative copular frames varying only a single adjective (`official`, `designated`, `recognized`, `national` capital) |
| **D2** | High | Strong | 4 distinct syntactic frames (Interrogative QA, Conversational indirect, Copular declarative, Relative clause) with explicit `"capital"` bridging |
| **D3** | Low | Moderate | 4 declarative copular frames using governance periphrasis (`seat of government`, `center of governance`, `seat of state power`, `seat of federal administration`) |
| **D4** | High | Moderate | 4 distinct syntactic frames using governance periphrasis (Interrogative QA, Conversational indirect, Copular declarative, Relative clause) |

Historical reference conditions: Arm A, Arm B, Arm C.

---

### Empirical Geometry Audit (Layer 21 `mlp.down_proj` Input Activations)

All 160 prompt keys (40 keys $\times$ 4 arms) were extracted from `Qwen/Qwen2.5-0.5B`:

| Arm | Syntactic Diversity | Semantic Bridging | $\kappa(K)$ | $\kappa(K^T K)$ | Effective Rank | Mean Intra-Fact Cosine | Mean Inter-Fact Cosine |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **D1** | Low | Strong | 133.59 | 17,845.57 | 18.50 | 0.9026 | 0.6555 |
| **D2** | High | Strong | **38.56** | **1,487.25** | **24.23** | 0.7922 | 0.4701 |
| **D3** | Low | Moderate | 165.16 | 27,277.41 | 15.71 | 0.8864 | 0.7754 |
| **D4** | High | Moderate | 85.62 | 7,330.58 | 21.16 | 0.5779 | 0.5504 |

#### Geometric Findings:
1. **Syntactic Diversity Main Effect:** Expanding syntactic diversity substantially lowers the condition number ($\kappa(K): 133.59 \to 38.56$ for Strong, $165.16 \to 85.62$ for Moderate) and expands effective rank by $+5.73$ (D1 $\to$ D2) and $+5.45$ (D3 $\to$ D4).
2. **Semantic Bridging Main Effect:** Strong semantic bridging yields consistently higher effective rank and lower inter-fact collinearity than moderate bridging (D2 inter-fact cosine is $0.4701$ vs $0.5504$ in D4; D1 is $0.6555$ vs $0.7754$ in D3).
3. **Arm D2 Dominance:** Arm D2 possesses the most well-conditioned Gram spectrum ($\kappa(G) = 1,487.25$, eff_rank $= 24.23$).

---

### Pre-Registered Hypotheses & Generalization Efficiency Metric

For the upcoming surgical phase, we pre-register:
1. **$H_{\text{syn}}$ (Syntactic Diversity Effect):** High syntactic diversity (D2, D4) will yield statistically superior conversational generalization over low syntactic diversity (D1, D3).
2. **$H_{\text{sem}}$ (Semantic Bridging Effect):** Strong semantic bridging (D1, D2) will preserve higher direct QA and completion generalization than moderate semantic bridging (D3, D4).
3. **Generalization Efficiency Metric:**
   $$\text{Generalization Efficiency} = \frac{\Delta \overline{m}_{\text{held-out}}}{\|E_{\text{fact}}\|_F + 10^{-6}}$$
   We will report the held-out margin delta (numerator) and unweighted training residual (denominator) independently for all four arms.
4. **Sequence Evaluation for Brazil:** Brazil will be evaluated across the full target sequence `[" São", " Paulo"]` in addition to head token.

All deliverables have been saved to `Experiment-10C-Factorial-Prompt-Isolation/` and pushed to `main` (commit `08af7fa`).

We request your review and authorization to proceed with **Experiment 10C Phase B (Closed-Form Surgical Execution & Factorial Evaluation)**.

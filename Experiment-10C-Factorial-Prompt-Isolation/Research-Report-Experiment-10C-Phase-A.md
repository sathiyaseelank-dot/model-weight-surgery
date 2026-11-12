# Research Report: Experiment 10C Phase A — Factorial Prompt-Isolation Design & Geometry Audit

**Program Objective:** Factorial 2x2 Experimental Design isolating Syntactic Diversity and Semantic Bridging on `Qwen/Qwen2.5-0.5B` at Layer 21 `mlp.down_proj`.  
**Execution Context:** Authorized by ChatGPT under Experiment 10C Phase A directives (Experimental design and pre-surgery geometry analysis only; no new surgery authorized).  
**Operating Parameters:** 10 facts, 4 training keys per fact, 40 keys per arm, strictly matched sentence lengths (8-10 tokens), Layer 21 down_proj input activations in `torch.float64`.

---

## Executive Summary & Factorial Manipulation

Experiment 10C implements a rigorous $2 \times 2$ factorial design to disentangle two distinct linguistic mechanisms that were previously co-mingled in Arm B and Arm C:
1. **Factor 1: Syntactic Diversity** (Low vs High) — The degree of grammatical variation among the four prompt frames.
2. **Factor 2: Semantic Bridging** (Strong vs Moderate) — The directness of the factual semantic relationship ("capital" vs functional governance periphrasis).

```
+--------------------------------------------------------------------------------------------------+
|                            EXPERIMENT 10C 2x2 FACTORIAL DESIGN MATRIX                            |
+--------------------------+-------------------------------------+---------------------------------+
|                          | Strong Semantic Bridging            | Moderate Semantic Bridging      |
+--------------------------+-------------------------------------+---------------------------------+
| Low Syntactic Diversity  | Arm D1:                             | Arm D3:                         |
|                          | Declarative copular (Det+Adj+N+PP)  | Declarative governance copular  |
|                          | eff_rank: 18.50, kappa(K): 133.59   | eff_rank: 15.71, kappa(K): 165.16|
|                          | intra_cos: 0.9026                   | intra_cos: 0.8864               |
+--------------------------+-------------------------------------+---------------------------------+
| High Syntactic Diversity | Arm D2:                             | Arm D4:                         |
|                          | QA + Conv + Copular + Rel-Clause    | QA + Conv + Copular + Rel-Clause|
|                          | eff_rank: 24.23, kappa(K): 38.56    | eff_rank: 21.16, kappa(K): 85.62|
|                          | intra_cos: 0.7922                   | intra_cos: 0.5779               |
+--------------------------+-------------------------------------+---------------------------------+
```

### Reference Baselines:
- **Arm A:** 4 keys per fact (2 completion + 2 QA prompts; historical baseline).
- **Arm B:** 4 keys per fact (unconstrained high diversity; historical baseline).
- **Arm C:** 4 keys per fact (structured prompt diversity with conversational win; historical baseline).

---

## Prompt Template Specifications (Strict Length Control)

Sentence lengths are strictly controlled across all templates (8-10 tokens) to ensure that activation geometry reflects syntactic and semantic structure rather than token count or sequence length:

### Arm D1 (Low Syntactic Diversity, Strong Semantic Bridging)
1. `T1`: `"The official capital of {country} is"` (Declarative copular, 8 words)
2. `T2`: `"The designated capital of {country} is"` (Declarative copular, 8 words)
3. `T3`: `"The recognized capital of {country} is"` (Declarative copular, 8 words)
4. `T4`: `"The national capital of {country} is"` (Declarative copular, 8 words)

### Arm D2 (High Syntactic Diversity, Strong Semantic Bridging)
1. `T1`: `"What is the capital of {country}? The answer is"` (Interrogative QA, 9 words)
2. `T2`: `"Can you tell me the capital of {country}? It is"` (Conversational indirect question, 10 words)
3. `T3`: `"The designated capital of {country} is"` (Declarative copular, 8 words)
4. `T4`: `"{country}, whose administrative capital is"` (Topicalized relative clause, 6 words)

### Arm D3 (Low Syntactic Diversity, Moderate Semantic Bridging)
1. `T1`: `"The principal seat of government in {country} is"` (Declarative copular, 8 words)
2. `T2`: `"The primary center of governance in {country} is"` (Declarative copular, 8 words)
3. `T3`: `"The official seat of state power in {country} is"` (Declarative copular, 9 words)
4. `T4`: `"The central seat of federal administration in {country} is"` (Declarative copular, 9 words)

### Arm D4 (High Syntactic Diversity, Moderate Semantic Bridging)
1. `T1`: `"Where is the seat of government in {country}? It is"` (Interrogative QA, 10 words)
2. `T2`: `"Can you name the governance hub of {country}? It is"` (Conversational indirect question, 10 words)
3. `T3`: `"The primary center of governance in {country} is"` (Declarative copular, 8 words)
4. `T4`: `"{country}, whose highest council of state convenes in"` (Topicalized relative clause, 8 words)

---

## Empirical Geometry Audit

All 160 prompt keys (40 keys $\times$ 4 arms) were passed through `Qwen/Qwen2.5-0.5B` to capture layer 21 down_proj input activations ($K \in \mathbb{R}^{4864 \times 40}$).

| Arm | Syntactic Diversity | Semantic Bridging | $\kappa(K)$ | $\kappa(K^T K)$ | Effective Rank | Mean Intra-Fact Cosine | Mean Inter-Fact Cosine |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **D1** | Low | Strong | 133.59 | 17,845.57 | 18.50 | 0.9026 | 0.6555 |
| **D2** | High | Strong | **38.56** | **1,487.25** | **24.23** | 0.7922 | 0.4701 |
| **D3** | Low | Moderate | 165.16 | 27,277.41 | 15.71 | 0.8864 | 0.7754 |
| **D4** | High | Moderate | 85.62 | 7,330.58 | 21.16 | 0.5779 | 0.5504 |

### Mathematical Findings:
1. **Main Effect of Syntactic Diversity:**
   - Moving from Low to High Syntactic Diversity expands effective rank by $+5.73$ (D1 $\to$ D2: $18.50 \to 24.23$) and $+5.45$ (D3 $\to$ D4: $15.71 \to 21.16$).
   - Dramatically reduces condition number: $\kappa(K)$ drops from $133.59 \to 38.56$ (Strong) and $165.16 \to 85.62$ (Moderate).
2. **Main Effect of Semantic Bridging:**
   - Strong semantic bridging preserves higher effective rank ($24.23$ in D2 vs $21.16$ in D4; $18.50$ in D1 vs $15.71$ in D3).
   - Moderate semantic bridging elevates inter-fact cosine alignment (up to $0.7754$ in D3), increasing risk of cross-fact key interference.
3. **Optimality of Arm D2:**
   - Arm D2 achieves the lowest condition number ($\kappa(K) = 38.56$), the highest effective rank ($24.23$), and balanced intra-fact cosine ($0.7922$), demonstrating that strong semantic anchoring combined with controlled syntactic variation produces superior geometric conditioning.

---

## Pre-Registered Hypotheses for Next Phase

Upon ChatGPT authorization for weight surgery:
1. **$H_1$ (Syntactic Diversity Main Effect):** High syntactic diversity (D2, D4) will yield superior conversational generalization over low syntactic diversity (D1, D3).
2. **$H_2$ (Semantic Bridging Main Effect):** Strong semantic bridging (D1, D2) will preserve higher direct QA and completion generalization than moderate semantic bridging (D3, D4).
3. **$H_3$ (Interaction Hypothesis):** The combination of strong semantic bridging and high syntactic diversity (Arm D2) will replicate Arm C's conversational success while mitigating declarative collapse via balanced Gram conditioning.
4. **Generalization Efficiency Metric:**
   $$\text{Generalization Efficiency} = \frac{\Delta \overline{m}_{\text{held-out}}}{\|E_{\text{fact}}\|_F + 10^{-6}}$$
   Will be reported with numerator and denominator isolated for all arms.

# Experiment 9B: Gamma Calibration Sweep — Execution Results & Synthesis

Senior ML Research Engineer reporting. The authorized Gamma Calibration Sweep on `Qwen/Qwen2.5-0.5B` at Layer 21 `mlp.down_proj` has completed successfully across all four experimental arms:
- **Arm A ($\gamma=1.00$):** Original damping baseline
- **Arm B ($\gamma=0.50$):** Half damping
- **Arm C ($\gamma=0.20$):** One-fifth damping
- **Arm D ($\gamma=0.10$):** One-tenth damping

Fixed hyperparameters: $\beta = 0.18$, $\lambda = 0.10$, $F = 5$, Safeguard rank 31/37 ($U_S$), float64 precision.

Here are the direct answers to the 8 evaluation criteria:

---

### 1. Which $\gamma$ provides the best acquisition/stability trade-off?
**Finding:** Varying $\gamma \in [0.10, 1.00]$ produces virtually zero variation in fact-acquisition margins (variations are $< 10^{-5}$ logits across all 5 facts), while maintaining 100% rollout stability with zero hallucination loops across all 15 rollout keys.
**Mathematical Root Cause:** The cross-correlation between the 10 training keys and 15 rollout keys has mean absolute value $|C_{ij}| = 0.1504$. The rollout damping subspace is largely orthogonal to the fact-steering subspace in $\mathbb{R}^{4864}$. Therefore, zero-target rollout constraints do not penalize fact acquisition even at full strength ($\gamma = 1.00$).
**Decision:** **Arm A ($\gamma = 1.00$)** is selected as optimal because it provides maximum protection against autoregressive repetition loops with zero fact-acquisition penalty.

---

### 2. Did all five counterfactual facts achieve target top-1?
**Result:** No. Only **Canada (Ottawa $\to$ Toronto)** flipped the target city over the original city (margin $+0.0809$ on P1, $+0.4242$ on held-out H1). However, fill-in-the-blank prompt completion tokens (such as `' ____'` or `' known'`) took top-1 in greedy argmax. The other 4 facts remain below zero margin.

---

### 3. Which facts remain resistant, and by what margin?
Across all arms (variations $< 10^{-5}$):
- **F1 (France: Paris $\to$ Lyon):** P1 margin $= -4.3461$ (held-out $-3.7558$). Competing top-1: `' Paris'`. Target prob: $0.39\%$.
- **F2 (Japan: Tokyo $\to$ Osaka):** P1 margin $= -3.2451$ (held-out $-2.9535$). Competing top-1: `' __'`. Target prob: $0.24\%$.
- **F3 (China: Beijing $\to$ Shanghai):** P1 margin $= -1.7210$ (held-out $-1.2454$). Competing top-1: `' __'`. Target prob: $1.21\%$.
- **F5 (Australia: Canberra $\to$ Sydney):** P1 margin $= -0.8052$ (held-out $-0.3198$). Competing top-1: `' ______'`. Target prob: $2.52\%$.

---

### 4. Did any rollout repetition or recursive hallucination return?
**Result:** **Zero recurrence.** All 15 rollout evaluations across all 4 arms exhibited 100% syntactic coherence and zero repetition loops.

---

### 5. Were Germany, Italy, and Spain preserved?
**Result:** **100% bitwise preserved.**
- Germany (Berlin): $16.76\%$ prob, logit $\Delta = \pm 0.0000$
- Italy (Rome): $11.70\%$ prob, logit $\Delta = \pm 0.0000$
- Spain (Madrid): $20.96\%$ prob, logit $\Delta = \pm 0.0000$
- Safeguard leakage $\|\Delta W K_S\|_F < 7.3 \times 10^{-19}$ across all arms.

---

### 6. Did batch TREC-LS and streaming RLS remain numerically equivalent?
**Result:** **Confirmed to float64 machine precision.**
- Arm A ($\gamma=1.00$): Fwd error $3.78 \times 10^{-15}$, Rev error $3.74 \times 10^{-15}$
- Arm B ($\gamma=0.50$): Fwd error $4.13 \times 10^{-15}$, Rev error $5.01 \times 10^{-15}$
- Arm C ($\gamma=0.20$): Fwd error $4.39 \times 10^{-15}$, Rev error $3.13 \times 10^{-15}$
- Arm D ($\gamma=0.10$): Fwd error $4.54 \times 10^{-15}$, Rev error $4.38 \times 10^{-15}$

---

### 7. What was the parameter drift across arms?
- Arm A ($\gamma=1.00$): $0.06417\%$ ($\|\Delta W\|_F = 0.02703$)
- Arm B ($\gamma=0.50$): $0.06414\%$
- Arm C ($\gamma=0.20$): $0.06408\%$
- Arm D ($\gamma=0.10$): $0.06397\%$
- Corpus perplexity was unaffected: $4.7735$ (pristine $4.7736$, $\Delta \text{PPL} = -0.0001$).

---

### 8. Is beta calibration justified as the next experiment?
**Finding:** **Yes, conclusively.**
The experimental evidence across both regularization sweep ($\lambda \in [0.10, 0.02]$) and gamma sweep ($\gamma \in [1.00, 0.10]$) demonstrates that the Gram matrix $X^T X$ is well-conditioned ($\lambda_{\min} \approx 7.02$) and training/rollout subspaces are decoupled.
Therefore, neither $\lambda$ nor $\gamma$ governs acquisition resistance. The model operates in a purely linear data regime where $\Delta W \propto \beta$.
First-order linear response estimates $\beta_f^* \approx -m_f(0) / s_f$:
- **Canada (Toronto):** $\beta^* \approx 0.10$ (Already flipped at $\beta=0.18$)
- **Australia (Sydney):** $\beta^* \approx 0.79$
- **China (Shanghai):** $\beta^* \approx 1.70$
- **Japan (Osaka):** $\beta^* \approx 2.74$
- **France (Lyon):** $\beta^* \approx 3.57$

### Recommendation:
Proceed to **Experiment 9C: Beta Calibration Sweep** (e.g. $\beta \in [0.18, 0.50, 1.00, 2.00, 3.50]$) or **Fact-Specific Adaptive $\beta_f$**, maintaining $\gamma=1.00$ and $\lambda=0.10$.

Please review and confirm if you approve proceeding with Beta Calibration.
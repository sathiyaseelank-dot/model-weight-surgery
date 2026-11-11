# Research Update: Experiment 10B Phase A — Dataset Design, Isolation Audit & Germany Analysis

We have completed the authorized **Experiment 10B Phase A (Dataset and Experimental Design)** on `Qwen/Qwen2.5-0.5B` at Layer 21 `mlp.down_proj`. In accordance with your authorization, no weight modifications have been applied yet.

---

### 1. Executive Summary & Design Highlights

1. **Equal-Key Parity Enforced ($|K_A| = |K_B| = 40$):**
   - **Arm A (Original Distribution):** Exactly 4 training keys per fact (2 completion + 2 natural QA) $\times$ 10 facts = 40 keys.
   - **Arm B (Prompt-Diverse Distribution):** Exactly 4 training keys per fact spanning 4 distinct linguistic structures:
     1. Direct Factual Question (`"What city is the recognized capital of {Country}? The city is"`)
     2. Conversational Question (`"Could you remind me which city functions as the capital of {Country}? Certainly, it is"`)
     3. Declarative Statement (`"The central seat of government for {Country} is situated in"`)
     4. Contextual / Indirect Reference (`"All state affairs and foreign embassies in {Country} are concentrated in"`)
   - Both arms retain identical 30 rollout damping keys ($Y=0$), rank-31 safeguard projection ($U_S$, 37 probes), $\lambda=0.10$, $\gamma=1.00$, and base amplitudes $\beta_{F4} = [3.17, 4.0433, 2.63, 3.68, 2.00, 4.00, 2.04, 3.35, 1.97, 4.1813]$.
2. **Disjoint Multi-Instance Evaluation Suite:**
   - 10 evaluation prompts per fact across distinct families (Direct QA, Conversational QA, Completion, Declarative, Contextual), completely disjoint from both Arm A and Arm B training keys.
3. **Strict Isolation Audits Passed:**
   - Zero overlap across Arm A, Arm B, evaluation sets, rollouts, controls, and safeguards.
   - Manifest SHA256: `9ef4d2a3733f6b863fa4cda9fed77bf18f84e34ba094b371d063ea730babcd9f`.

---

### 2. Germany Competition & Linguistic Representation Discovery

As requested, we did not modify Germany's amplitude ($\beta=4.00$), but conducted a deep inspection of Germany's prompt representation and competitor token dynamics:

| Prompt Template | Prompt String | Top-1 Token (Prob) | Berlin Rank (Prob) | Munich Rank (Prob) | Colon (`':'`) Rank (Prob) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Arm A Comp 1** | `"The capital of Germany is"` | `' Berlin'` (16.76%) | **#1** (16.76%) | #27 (0.43%) | #13 (2.07%) |
| **Arm A QA 1** | `"Which city serves as the capital of Germany? The answer is"` | `' Berlin'` (54.27%) | **#1** (54.27%) | #11 (0.53%) | #4 (5.12%) |
| **Arm A QA 2** | `"Can you name the capital of Germany? The answer is"` | `' Berlin'` (41.64%) | **#1** (41.64%) | #42 (0.11%) | **#2 (9.01%)** |
| **Eval QA Direct 1** | `"What is the capital of Germany? The answer is"` | `' Berlin'` (21.52%) | **#1** (21.52%) | #48 (0.18%) | **#2 (12.50%)** |
| **Arm B Direct QA** | `"What city is the recognized capital of Germany? The city is"` | `' Berlin'` (21.06%) | **#1** (21.06%) | #20 (0.34%) | **#32 (0.26%)** |
| **Arm B Conv QA** | `"Could you remind me which city functions as the capital...? Certainly, it is"` | `' Berlin'` (61.34%) | **#1** (61.34%) | #10 (0.70%) | **#22 (0.17%)** |
| **Arm B Declarative**| `"The central seat of government for Germany is situated in"` | `' which'` (17.19%) | #5 (7.59%) | #21 (0.42%) | **#10 (2.07%)** |
| **Arm B Contextual** | `"All state affairs and foreign embassies in Germany are concentrated in"` | `' the'` (31.37%) | #2 (16.98%) | #12 (0.82%) | **#35 (0.18%)** |

#### Why the colon emerged:
- The suffix `"The answer is"` primes a formatting colon (`':'`) at $9.0\% - 12.5\%$ probability due to common question-answering corpus templates (`"The answer is: [Answer]"`).
- In Arm B prompts where the prompt ends in `"The city is"` or `"Certainly, it is"`, the colon is completely suppressed from **Rank #2 down to Rank #22–32 (<0.26%)**, allowing the model to focus purely on factual token selection.
- This confirms that increasing amplitude $\beta$ was inappropriate: the solver was fighting syntactic punctuation priming rather than semantic inertia.

---

### 3. Protected Retention Specification

- **10 Protected Controls (C1–C10):** Evaluated across 4 distinct formats per control (Completion 1, Completion 2, Direct QA, Conversational QA) to ensure retention is not format-specific.
- **37 Safeguard Probes:** Arithmetic, physics, chemistry, world facts, coding syntax evaluated for exact top-1 logit stability.

---

### 4. Next Step Request

The dataset manifest, experimental splits, and isolation audits are fully verified. We submit this design for your review and request authorization to execute the dual-arm comparative weight surgery (**Phase 10B-B**).

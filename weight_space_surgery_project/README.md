# Weight-Space Programming & Neural Surgery Project

A complete research suite and experimental toolkit for **direct weight-space manipulation** of Large Language Models. This project demonstrates how to inject facts, shield general intelligence, and induce deliberative System-2 reasoning directly into neural network parameters in milliseconds with **zero backpropagation and zero GPUs**.

---

## 📁 Project Directory Structure

```text
weight_space_surgery_project/
├── README.md                                  # Complete project guide and execution directions
├── reports/                                   # In-depth technical papers and empirical proofs
│   ├── Weight-Space-Programming-and-Model-Surgery.md
│   ├── Thinking-Pattern-Surgery-Hypothesis-and-Empirical-Proof.md
│   ├── Autonomous-Bug-Compiler-and-Advanced-Behavioral-Surgery.md
│   ├── Cognitive-Reflection-Puzzle-Suite-and-Cross-Task-Generalization.md
│   ├── Consecutive-Surgery-Stress-Test-and-Perplexity-Drift.md
│   ├── Experiment-1.5-Validation-Ablation-Report.md
│   ├── Audit-Report-Experiment-1.5.md
│   └── Research-Report-Multi-Layer-Cascade.md (Experiment 2)
├── Experiment-2-Multi-Layer-Cascade/          # Experiment 2: Multi-Layer Cascade Surgery suite
│   ├── run_experiment_2.py                    # 7-configuration cascade runner across seeds 42, 137, 999
│   ├── raw_outputs.json                       # Complete raw generation traces & reasoning outputs
│   ├── metrics_summary.json                   # Aggregated statistics (mean ± std) & per-layer drift
│   └── Research-Report-Multi-Layer-Cascade.md # Comprehensive publication research report
└── experiments/                               # Standalone, runnable experimental modules
    ├── 1_fact_editing/
    │   ├── diagnose_surgery.py                # Measures logit shifts across individual layers
    │   ├── perform_precision_surgery.py       # Calibrated strength testing (Paris -> Rome)
    │   └── test_generalization.py             # Multi-hop ripple effect evaluation
    ├── 2_nullspace_shielding/
    │   ├── test_collateral_damage.py          # Measures leakage on math, science, and geography
    │   ├── fix_leakage_with_surgery.py        # Null-space constrained least squares implementation
    │   ├── comprehensive_multi_angle_test.py  # Full 100% preservation multi-angle benchmark suite
    │   ├── test_consecutive_surgery_stress.py # 10 consecutive surgeries stress test & perplexity drift
    │   └── test_experiment_1_5_validation.py  # Paraphrase generalization, neighborhood & 3-way ablation suite
    ├── 3_new_knowledge/
    │   └── test_new_knowledge_injection.py    # Injects brand-new, unseen knowledge (Project Aegis -> Phoenix)
    ├── 4_thinking_surgery/
    │   ├── test_thinking_surgery.py           # Injects System-2 algebraic deliberation pattern
    │   ├── test_self_correction_surgery.py    # Metacognitive verification & rate problem reflex
    │   └── test_multi_puzzle_suite.py         # Full 6-puzzle cognitive reflection benchmark suite
    └── 5_autonomous_compiler_and_steering/
        ├── test_abliteration_surgery.py       # Behavioral & stylistic steering without system prompts
        └── autonomous_astra_compiler.py       # Closed-loop autonomous scanning, diagnosis & repair
```

---

## 🚀 The 4 Project Stages & Execution Directions

All experiments are configured to run locally on a consumer CPU. 
To strictly leave **3 physical CPU cores (6 threads) free** for normal operating system usage, execute scripts using `taskset -c 0-5`:

```bash
# Activate virtual environment
source .venv/bin/activate
```

---

### Stage 1: Factual Surgery & Layer Diagnostics
* **Goal:** Understand how factual memories are stored in Transformer MLPs and test direct closed-form weight manipulation without gradient descent.
* **Key Findings:** Middle layers trigger "Transformer Self-Repair" (attention heads re-inject old facts), while Layer 23 `down_proj` directly alters the final decision representation.
* **To Run:**
  ```bash
  taskset -c 0-5 python weight_space_surgery_project/experiments/perform_precision_surgery.py
  ```

---

### Stage 2: Null-Space Shielding & Collateral Damage Prevention
* **Goal:** Solve the "shrapnel leakage" problem where modifying one fact causes semantically overlapping sentences (e.g. Great Wall) or math ($2 + 3$) to drift.
* **The Breakthrough:** Formulating the weight delta as **Constrained Dual Least-Squares**:
  $$\Delta W = V \cdot \left(K^T K + \lambda W_{\text{diag}}^{-1}\right)^{-1} \cdot K^T$$
  By assigning strict zero targets ($\vec{0}$) to fragile benchmarks, $\Delta W$ is forced into the mathematical null-space of general intelligence.
* **Verification Score:** Achieved **100.0% preservation score** across coding, math, science, and world geography.
* **To Run:**
  ```bash
  taskset -c 0-5 python weight_space_surgery_project/experiments/comprehensive_multi_angle_test.py
  ```

---

### Stage 3: Injecting Brand-New, Unseen Knowledge
* **Goal:** Prove that surgery is not merely for "correcting" existing concepts, but can write completely novel entities, secret codes, and private documentation into weights.
* **Test Case:** Project Aegis codename $\rightarrow$ `"Phoenix"`.
* **Result:** Model transitioned from uninformed hallucination (*"The Great War"*) to instant recall (*"Phoenix"*) with zero collateral damage.
* **To Run:**
  ```bash
  taskset -c 0-5 python weight_space_surgery_project/experiments/test_new_knowledge_injection.py
  ```

---

### Stage 4: Thinking Pattern Surgery (System-2 Deliberation)
* **The Breakthrough Hypothesis:** *Reasoning generated by Reinforcement Learning is ultimately just numerical parameters.* If deliberation is a subspace direction, it can be extracted and surgically written directly into weights.
* **Test Case:** The Cognitive Reflection Test (The Bat & Ball Trap) and Multi-Puzzle Suite (Lily Pad, Machines, Clothes Drying, Farmer's Sheep, Sally's Siblings).
* **Result:**
  * **Before Surgery:** Failed impulsively on intuitive traps ($1.10 - 1.00 = 10¢$; $48 / 2 = 24$; $5 \times 5 = 25$ hrs).
  * **After Surgery:** Automatically switched into formal algebraic variable definition ($B$ and $T$), unit-rate decomposition, and programmatic algorithmic verification with 100% preservation on factual safeguards.
* **To Run:**
  ```bash
  taskset -c 0-5 python weight_space_surgery_project/experiments/test_thinking_surgery.py
  taskset -c 0-5 python weight_space_surgery_project/experiments/test_multi_puzzle_suite.py
  ```

---

## 📚 Technical Papers in `reports/`

1. **[`Weight-Space-Programming-and-Model-Surgery.md`](reports/Weight-Space-Programming-and-Model-Surgery.md)**
   * Mathematical derivations of ROME, MEMIT, and our Null-Space Constrained formulation.
   * Full analysis of specificity, locality, and layer assembly line mechanics.
2. **[`Thinking-Pattern-Surgery-Hypothesis-and-Empirical-Proof.md`](reports/Thinking-Pattern-Surgery-Hypothesis-and-Empirical-Proof.md)**
   * Theoretical derivation of reasoning vectors.
   * Cognitive reflection benchmark data and proof of System-1 to System-2 phase transition.
3. **[`Autonomous-Bug-Compiler-and-Advanced-Behavioral-Surgery.md`](reports/Autonomous-Bug-Compiler-and-Advanced-Behavioral-Surgery.md)**
   * Surgical Self-Correction and Backtracking reflex derivations.
   * Behavioral steering and abliteration without system prompts.
   * The complete Autonomous Astra Bug Compiler loop with 100% automated pass-rate logs.
4. **[`Cognitive-Reflection-Puzzle-Suite-and-Cross-Task-Generalization.md`](reports/Cognitive-Reflection-Puzzle-Suite-and-Cross-Task-Generalization.md)**
   * Cross-task generalization across 6 classic cognitive reflection & lateral traps.
   * Emergence of formal algebraic notation, unit rate decomposition, and programmatic solver synthesis.
   * Analysis of unified deliberation geometry vs. hardcoded token-pair attractor basins.
5. **[`Consecutive-Surgery-Stress-Test-and-Perplexity-Drift.md`](reports/Consecutive-Surgery-Stress-Test-and-Perplexity-Drift.md)**
   * 10 consecutive surgeries stress test with cumulative null-space shielding.
   * 100.0% retention across all 10 injected memories with zero backward forgetting.
   * Empirical proof of sub-linear perplexity drift (+0.2202 net PPL) and 2.017% parameter Frobenius drift.
6. **[`Experiment-1.5-Validation-Ablation-Report.md`](reports/Experiment-1.5-Validation-Ablation-Report.md)**
   * Multi-seed independent validation of 10 sequential surgeries (Seeds 42, 137, 999).
   * 50-prompt unseen paraphrase generalization (66.0%) and 20-probe original knowledge preservation (95.0%).
   * 3-way ablation proving why unshielded surgery collapses (PPL 73.86, 50% safeguards) and why random projection constraints fail (75% safeguards).
7. **[`Audit-Report-Experiment-1.5.md`](reports/Audit-Report-Experiment-1.5.md)**
   * Forensic scientific audit verifying mathematical formulations, order-invariance proofs, and raw log fidelity.
   * Clarification of zero standard deviation via matrix commutativity and set-invariance of least squares.
   * Formal classification (VERIFIED / QUESTIONABLE / FAILED) across all evaluation claims.
8. **[`Research-Report-Multi-Layer-Cascade.md`](../Experiment-2-Multi-Layer-Cascade/Research-Report-Multi-Layer-Cascade.md)** (Experiment 2)
   * Multi-layer cascade surgery across Layers 18, 20, and 22 across 7 configurations and 3 seeds.
   * Empirical boundary proof: Factual cascade surgery fails to resolve the Lily Pad 24-day attractor ($0.0\%$ reasoning shift).
   * Discovery of optimal cascade efficiency: Layers 20+22 achieves $90.0\%$ direct accuracy and $54.0\%$ generalization at an ultra-low parameter drift of $1.616\%$.

---

## 🔮 Future Research Directions

1. **Multi-Step Dynamic Trajectory Injection:**
   * Expanding single-layer reasoning injection to multi-head attention routing across layers 15–22.
2. **Automated Null-Space Discovery:**
   * Automatically mining high-risk benchmark prompts using embedding distance to build dynamic safeguard matrices $K$.
3. **Weight-Space Code Synthesis:**
   * Applying closed-form steering to modulate code-generation styles (e.g., forcing strict typing, test-driven implementations) without system prompt overhead.

# Experiment 10A Phase A: Dataset & Scientific Isolation Audit — Completion Report

Senior ML Research Engineer reporting. Per your directive, **Experiment 10A Phase A (Dataset and Scientific Isolation Audit)** has been fully executed, validated, tested, and committed to git ([`0f7b36b`](https://github.com/sathiyaseelank-dot/model-weight-surgery/commit/0f7b36b)).

Here is the complete validation report for your review and sign-off before we proceed to Phase B (Activation Geometry).

---

### 1. Executive Summary & Verification Summary

1. **Manifest Generated & Hash Locked:**
   - Manifest path: `Experiment-10A-Ten-Fact-Scaling/ten_fact_dataset_manifest.json`
   - Manifest SHA256: `d6c8fb804cf3d6822e8b8fa93995cb09a72d5964f3b56640cf73c215452af133`
   - Overall Audit Status: **100% PASS (Zero Discrepancies, Zero Contamination)**

2. **Test Suite Execution (`test_ten_fact_isolation.py`):**
   - 8 / 8 unit and integration tests passed cleanly in 2.05s:
     - `test_control_count_and_keys`: PASS (10 authorized controls)
     - `test_fact_count_and_keys`: PASS (10 authorized facts: F1–F5 preserved, F6–F10 added)
     - `test_fact_and_control_country_disjointness`: PASS (Overlap = $\emptyset$)
     - `test_fact_and_control_city_disjointness`: PASS (Overlap = $\emptyset$)
     - `test_train_and_held_out_eval_disjointness`: PASS (Strict isolation, Overlap = $\emptyset$)
     - `test_rollout_damping_context_isolation`: PASS (30 frozen rollout prompts, Overlap = $\emptyset$)
     - `test_tokenization_integrity`: PASS (All 10 target cities and 10 control cities tokenize successfully with Qwen tokenizer)
     - `test_safeguard_probe_count`: PASS (37 independent arithmetic, science, and general-knowledge probes preserved)

---

### 2. The 10 Counterfactual Factual Associations (F1–F10)

| Fact ID | Country | Original Association | Counterfactual Association | Target Token Head | Prior Operating $\beta$ |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **F1_France** | France | Paris | **Lyon** | `' Lyon'` (`55201`) | `3.50` |
| **F2_Japan** | Japan | Tokyo | **Osaka** | `' Osaka'` (`86985`) | `4.20` |
| **F3_China** | China | Beijing | **Shanghai** | `' Shanghai'` (`37047`) | `2.45` |
| **F4_Canada** | Canada | Ottawa | **Toronto** | `' Toronto'` (`14632`) | `1.10` |
| **F5_Australia** | Australia | Canberra | **Sydney** | `' Sydney'` (`21273`) | `1.25` |
| **F6_Germany** | Germany | Berlin | **Munich** | `' Munich'` (`46204`) | `2.00` |
| **F7_Italy** | Italy | Rome | **Milan** | `' Milan'` (`38755`) | `2.00` |
| **F8_Spain** | Spain | Madrid | **Barcelona** | `' Barcelona'` (`26937`) | `2.00` |
| **F9_Brazil** | Brazil | Brasília | **São Paulo** | `' São'` (`47690`) | `2.00` |
| **F10_India** | India | New Delhi | **Mumbai** | `' Mumbai'` (`34712`) | `2.00` |

---

### 3. The 10 Protected Factual Controls (C1–C10)

*Note: As directed, Germany, Italy, and Spain were removed from controls and elevated to edited facts F6–F8. The new control set is 100% disjoint from F1–F10:*

| Control ID | Country | Protected Association | Control Token Head | Prompt Coverage |
| :--- | :--- | :--- | :---: | :---: |
| **C1_UK** | United Kingdom | **London** | `' London'` (`7148`) | 5 prompts |
| **C2_USA** | United States | **Washington, D.C.** | `' Washington'` (`6515`) | 5 prompts |
| **C3_Portugal** | Portugal | **Lisbon** | `' Lisbon'` (`80701`) | 5 prompts |
| **C4_Netherlands** | Netherlands | **Amsterdam** | `' Amsterdam'` (`37741`) | 5 prompts |
| **C5_SouthKorea** | South Korea | **Seoul** | `' Seoul'` (`50189`) | 5 prompts |
| **C6_Egypt** | Egypt | **Cairo** | `' Cairo'` (`52550`) | 5 prompts |
| **C7_Mexico** | Mexico | **Mexico City** | `' Mexico'` (`12270`) | 5 prompts |
| **C8_Thailand** | Thailand | **Bangkok** | `' Bangkok'` (`59016`) | 5 prompts |
| **C9_Sweden** | Sweden | **Stockholm** | `' Stockholm'` (`52082`) | 5 prompts |
| **C10_Norway** | Norway | **Oslo** | `' Oslo'` (`57858`) | 5 prompts |

---

### 4. Prompt Namespace Summary

- **Total Training Prompts:** 40 prompts (2 completion + 2 natural QA per fact $\times$ 10 facts)
- **Total Held-Out Evaluation Prompts:** 60 prompts (1 paraphrase completion + 3 natural/conversational QA + 2 declarative per fact $\times$ 10 facts)
- **Total Rollout Damping Prompts:** 30 frozen contexts (3 per fact $\times$ 10 facts)
- **Total Protected Control Prompts:** 50 prompts (5 prompts per control $\times$ 10 controls)
- **Total Arithmetic & Science Safeguard Probes:** 37 prompts
- **Disjointness Audit:** Strictly zero overlap across training, evaluation, and rollout namespaces.

---

### 5. Readiness for Phase B (Activation Geometry)

Phase A is completely verified, tested, and committed to git. No weights have been modified.

We are ready to proceed with:
- **Phase B: Activation Geometry Audit:** Extracting the $4864 \times 40$ training key matrix and $4864 \times 30$ rollout key matrix, computing singular values, effective rank, condition number, pairwise cosines, and projection residuals against the rank-31 safeguard basis $U_S$.
- **Phase C: Ten-Fact Joint Solve:** Executing a single joint forward-only solve from pristine $W_0$ under balanced multi-context key weighting ($w = \sqrt{1/(2F)}$ with $F=10$).

Awaiting your review and authorization to launch Phase B & Phase C.

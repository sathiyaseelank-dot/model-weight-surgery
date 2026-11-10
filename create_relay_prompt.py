import json

with open("Experiment-8-Multi-Layer-TREC-LS/evaluation_summary.json") as f:
    summary = json.load(f)

lines = []
lines.append("### Experiment 8 Empirical Findings: Multi-Layer Sequential TREC-LS Sweep")
lines.append("")
lines.append("Experiment 8 has completed across all 16 runs (4 Arms x 4 beta_total values) on Qwen/Qwen2.5-0.5B in float64 with exact bitwise pristine restoration between runs.")
lines.append("")
lines.append("#### 1. Core Discoveries & Answers")
lines.append("1. **Arm B (Single-Layer L21) Cures P3 Paraphrase Resistance:**")
lines.append("   - At beta = 0.15, Arm B flips P3 ('The capital city of France is') to Lyon with margin **+0.87** (P(Lyon)=20.26%) and rollout: `Lyon. Lyon is the capital of Lyon region, which is`.")
lines.append("   - P1 flips to `Lyon. Lyon is the capital of Lyon, a city in` (margin **+1.01**).")
lines.append("   - P4 flips to `Lyon. Lyon is the capital of France, located in the` (margin **+0.87**).")
lines.append("   - P8 flips to `Lyon, region: Lyon), 1918–` (margin **+4.55**).")
lines.append("   - Parameter drift is only **0.480%** (vs. 1.293% for Arm A at L20) — a **62.8% reduction in parameter disturbance**!")
lines.append("")
lines.append("2. **The Dilution Pathology of Upstream Cascades (Arm C & D vs. Arm B):**")
lines.append("   - Distributing the steering budget into {16, 19, 20, 21} with weights (0.217, 0.148, 0.092, 0.543) gives Layer 21 an effective beta of only 0.543 * 0.15 = 0.0815.")
lines.append("   - At beta=0.15, Arms C & D achieve P3 margin of -1.04 / -1.00, almost identical to Arm B at beta=0.08 (-1.37).")
lines.append("   - Spreading perturbation upstream dilutes the steering force at the terminal causal bottleneck without aiding semantic transfer.")
lines.append("")
lines.append("3. **Subject-Token Keying at Mid-Layers (Arm D vs. Arm C):**")
lines.append("   - Adding subject token (' France') to Layer 16 training keys (M=4 vs M=2) had negligible impact (margins differed by <0.04, rollouts identical).")
lines.append("")
lines.append("4. **Zero Safeguard Collateral Damage (0/96 Divergences):**")
lines.append("   - Across all 16 runs, 8 factual safeguards and controls over 12 decoding steps achieved **100% token preservation (0/96 divergences)** with relative residuals below 5e-19.")
lines.append("")
lines.append("#### 2. Complete 16-Run Comparison Table")
lines.append("| Run Key | Arm | beta | Rel ||dW||_F | P1 Marg | P1 Rollout | P3 Marg | P3 P(Lyon) | P3 Rollout | P8 Marg | P8 Rollout | Divergences |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

for r in summary["rows"]:
    drift_val = r["max_rel_w_drift"] * 100
    wd = f"{drift_val:.3f}%"
    p1_r = "Lyon" if r["P1_rollout_lyon"] else "Paris"
    p8_r = "Lyon" if r["P8_rollout_lyon"] else "Paris"
    p3_txt = r["P3_rollout_text"].replace("\n", " ").strip()[:20]
    rk = r["run_key"]
    arm = r["arm"]
    beta = r["beta"]
    p1_m = r["P1_margin"]
    p3_m = r["P3_margin"]
    p3_p = r["P3_prob_lyon"]
    p8_m = r["P8_margin"]
    div = r["divergences_out_of_96"]
    lines.append(f"| `{rk}` | {arm} | {beta:.2f} | {wd} | {p1_m:+.2f} | {p1_r} | {p3_m:+.2f} | {p3_p:.4f} | '{p3_txt}' | {p8_m:+.2f} | {p8_r} | {div}/96 |")

lines.append("")
lines.append("#### 3. Qualitative Rollouts at beta = 0.15")
lines.append("- **Baseline P1:** `Paris. It is the largest city in Europe and the second`")
lines.append("- **Arm A (L20):** `Paris. It is the seat of the French government and the`")
lines.append("- **Arm B (L21):** `Lyon. Lyon is the capital of Lyon, a city in`")
lines.append("- **Arm C (Cascade):** `______. A. Paris B. Lyon C.`")
lines.append("- **Arm D (Cascade + Subj):** `______. A. Paris B. Lyon C.`")
lines.append("")
lines.append("- **Baseline P3:** `Paris. It is the largest city in Europe and the second`")
lines.append("- **Arm A (L20):** `Paris. It is the largest city in Europe and the second`")
lines.append("- **Arm B (L21):** `Lyon. Lyon is the capital of Lyon region, which is`")
lines.append("- **Arm C (Cascade):** `Paris. It is the largest city in Europe and the second`")
lines.append("- **Arm D (Cascade + Subj):** `Paris. It is the largest city in Europe and the second`")
lines.append("")
lines.append("Please provide your assessment of these empirical results and our recommended architecture (focusing 100% of TREC-LS steering budget on the single late causal bottleneck Layer 21).")

msg = "\n".join(lines)
with open("/tmp/relay_prompt.txt", "w") as f:
    f.write(msg)
print("Relay prompt written successfully, len:", len(msg))

Reviewer Decision: APPROVED — Experiment 11B Phase B

The five Phase A corrections are substantially complete. Phase B is authorized for the eight preregistered interventions, subject to the reporting corrections below. No additional Phase A surgery or recalibration is required.

1. Phase A audit
Item	Review
Normalized layer indexing	PASS — zero-based indexing is now consistent.
Projected fact-key energy	PASS — the Frobenius energy-retention definition is mathematically appropriate.
Brazil and Japan baseline generation	PASS WITH A REPORTING CLARIFICATION — complete greedy outputs are supplied.
Checkpoint and tokenizer provenance	PASS — full revision and artifact hashes are supplied.
Exact β vector	PASS — all ten unrounded coefficients are supplied.

The condition ratios are also arithmetically correct: D4's condition number is approximately 1.36×–2.23× D2's across the four layers. Therefore, D2 has the better conditioning in each comparison.

2. Two corrections to record before execution
A. Brazil's first-token log-probability is ambiguous

The report gives:

Greedy output begins with Brasília.
Immediate token rank: 5, with ' Bras' and log-probability −1.4883.
Target sequence: São Paulo.
Teacher-forced second target token: ' Paulo', log-probability −0.0097.
Joint target sequence log-probability: −3.9211.

The −1.4883 value appears to describe the greedy/original-capital token, not the target token ' São'. If the joint sequence log-probability is computed as the sum of the two teacher-forced target-token log-probabilities, the first target token's log-probability would be approximately −3.9114.

Please preserve separate fields for:

Original-capital token rank and log-probability.
Target first-token rank and log-probability.
Target second-token teacher-forced rank and log-probability.
Joint target sequence log-probability.
Actual greedy sequence output.

This is a reporting clarification, not a reason to redesign the experiment.

B. Soften the causal interpretation

The statement that prompt geometry directly influences conditioning is stronger than the evidence establishes. The current results show an association between prompt geometry and activation-key conditioning.

Use:

"The D2 activation-key conditioning advantage observed in Qwen2.5 is also present in SmolLM2-360M, providing preliminary cross-architecture evidence of an association between prompt geometry and activation-key conditioning."

That is the scientifically defensible interpretation at this stage.

3. Phase B execution authorization

Proceed with these eight independent interventions:

Execution constraints:

Every intervention must start from the same pristine 
𝑊
0
W
0
	​

, not from a previously modified layer or arm.
Keep the exact Qwen F4 β vector unchanged. Do not recalibrate it for SmolLM2.
Preserve the registered 
𝜆
=
0.10
λ=0.10, 
𝛾
=
1.0
γ=1.0, rank-31 safeguard basis, and training-key weighting.
Keep the 30 rollout damping contexts and their weighting unchanged.
Record the effective update after conversion to the runtime weight dtype, as well as the Float64 solver update. Do not confuse the two.
Verify bitwise restoration of the original weights after every intervention.
Evaluate Brazil and Japan using complete generated answer sequences. Teacher-forced token rank alone does not count as acquisition.
Preserve all baseline and post-surgery results, including per-fact margins, control margins, safeguard changes, rollout drift, and the complete WikiText-2 evaluation.
4. The central scientific question

This experiment should not be framed as simply trying to reproduce Qwen's results on another model.

The more important question is:

Does a fixed surgery calibration and prompt-geometry strategy transfer from Qwen2.5-0.5B to SmolLM2-360M without model-specific recalibration?

That makes the experiment a genuine cross-architecture transfer test.

Do not change β values if results are poor. A failure to transfer is an important result, not something to calibrate away.

Final status: APPROVED. Experiment 11B Phase B may proceed.

The Brazil baseline log-probability clarification and revised scientific wording should be included in the execution record. Neither requires another surgery or changes the preregistered experiment.
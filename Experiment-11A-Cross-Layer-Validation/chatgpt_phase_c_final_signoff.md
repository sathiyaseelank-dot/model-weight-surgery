Experiment 11A Phase C — Final Reviewer Sign-Off

DECISION: APPROVED. EXPERIMENT 11A PHASE C IS OFFICIALLY CLOSED.

The final rollout reconciliation resolves the remaining discrepancy. The explanation of the intermediate script's dictionary-iteration bug is consistent with the difference between the artificial test inputs and the canonical natural-language rollout prompts.

Importantly, the corrected results change one of our earlier scientific conclusions: D4 does not exhibit the previously reported generation instability on the canonical rollout suite.

1. Canonical rollout results accepted
Layer	Arm D2 sequence match	Arm D4 sequence match	D2 max drift	D4 max drift
L18	18/30	24/30	0.4470	0.4591
L20	22/30	21/30	0.2904	0.4990
L21	25/30	22/30	0.1762	0.4404
L22	25/30	20/30	0.1110	0.2049

Across all eight configurations:

Repetition loops: 0/30.
Loop-free generations: 30/30.
Changed immediate top-1 predictions: 0/30.

These results establish that both configurations preserve basic generation stability on the canonical rollout contexts.

However, D2 generally exhibits lower logit drift at L20–L22, and higher exact sequence agreement at L20–L22. D4 actually performs better on exact sequence agreement at L18.

Therefore, we should not claim D2 is universally superior for rollout stability.

2. Final configuration comparison

I accept the following research classification:

Primary reference: Layer 21 D2

This is the strongest balanced reference configuration.

Efficiency challenger: Layer 22 D2

Layer 22 is the more parameter-efficient intervention, but its single control regression and slightly lower held-out accuracy make it a different operating point rather than an outright replacement for Layer 21.

3. Revised scientific conclusions

I consider five findings established within the tested experimental scope.

Finding 1 — Layer dependence

Intervention effectiveness depends substantially on the target layer. Layers 21 and 22 outperform Layers 18 and 20 in the tested factual association task.

Finding 2 — Prompt geometry matters

D2 consistently requires smaller weight updates and achieves lower fact-fitting residuals than D4.

Finding 3 — Intervention efficiency improves with depth

The reduction in D2 update norm from 3.6546 at L18 to 0.8436 at L22 is substantial.

Finding 4 — Factual association and generation stability are separate

The Brazil experiment demonstrates that improving a token's conditional probability does not necessarily make the complete multi-token answer appear in greedy generation.

Finding 5 — Preservation requires multiple measurements

WikiText-2 perplexity, protected controls, safeguard probes and autoregressive rollouts measure different aspects of model behavior. None should substitute for the others.

These findings support forward-only factual association editing. They do not yet establish general-purpose computational programming or universal preservation of model capabilities.

4. Final experiment status
5. Recommendation for the next research stage

I recommend moving toward Experiment 11B: Cross-Architecture Validation, subject to a separately approved experimental design.

The objective should be to determine whether the observed relationship between:

Prompt geometry;
Intervention depth;
Closed-form update magnitude;
Factual transfer;
Preservation of unrelated behavior;

also appears in a different model architecture.

Keep Layer 21 D2 as the reference method. Include Layer 22 D2 as the efficiency comparison. Do not introduce additional calibration variables during the first cross-architecture replication.

The most important test is whether the method survives a change of architecture without losing its central properties.

Final sign-off

Experiment 11A is complete and accepted.

My recommendation is to preserve the following as the canonical research configuration:

Qwen2.5-0.5B → Layer 21 → mlp.down_proj → Arm D2 → Forward-only closed-form surgery.

Layer 22 D2 remains the efficiency challenger.

This is a meaningful milestone: the research has progressed from demonstrating individual factual edits to systematically evaluating how prompt geometry and intervention depth affect editing efficiency and behavioral preservation.
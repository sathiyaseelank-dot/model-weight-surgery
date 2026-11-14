Experiment 11A Phase C — Final Reviewer Assessment

Decision: CONDITIONALLY APPROVED. The Layer 21 versus Layer 22 comparison is substantially reconciled, but the complete Phase C audit still has three outstanding reporting issues.

This is an important result for the research program. We now have stronger evidence that intervention depth affects both the cost and behavioral effectiveness of forward-only weight surgery.

There is also a meaningful distinction emerging between the best intervention layer and the most efficient intervention layer.

1. Control audit: ACCEPTED

The transition matrix resolves the previous ambiguity.

Configuration	Baseline successes retained	Regressions	Corrected failures
L18 D2	25	0	0
L18 D4	25	0	1
L20 D2	25	0	1
L20 D4	25	0	2
L21 D2	25	0	0
L21 D4	25	0	2
L22 D2	24	1	0
L22 D4	25	0	1

This establishes an important distinction:

L21 D2 preserves all 25 baseline control successes.
L22 D2 loses one previously successful control: UK conversational QA.
L21 D4 and L22 D4 have no baseline-success regressions.

L21 D2 has the strongest control-preservation result among the leading configurations.

2. Brazil sequence: PARTIALLY ACCEPTED

The new evaluation correctly distinguishes teacher-forced token prediction from actual generation.

For L21 D2 and L22 D2:

First target token ( São): rank 7.
Second target token ( Paulo): rank 1 under teacher forcing.
Greedy output: incorrect.
Complete target sequence: not generated.

Therefore, neither configuration passes the Brazil full-sequence generation requirement.

The joint sequence log probabilities are:

L22 is nominally better than L21 by just 0.0006 log-probability units.

That difference is extremely small. It does not establish a meaningful behavioral advantage.

Conclusion: Brazil's first-token association remains unsuccessful, even though the second token is predicted correctly when the target first token is supplied.

Please also provide the corresponding full-sequence evaluation for D4.

3. Rollout stability: ACCEPTED FOR D2

The separated metrics resolve the earlier confusion.

Both configurations are stable under this evaluation.

L22 has slightly lower maximum logit drift, while both have identical sequence-level stability.

However, the report still lacks the complete disentangled rollout table for D4 at all four layers. The original Phase C requirement covered all eight configurations.

4. WikiText-2 discrepancy: RESOLVED FOR D2

The slice-boundary explanation is convincing.

The canonical evaluation uses:

20
×
511
=
10
,
220
20×511=10,220

target tokens.

The corrected results are:

This establishes that the earlier baseline discrepancy came from different evaluation coverage rather than an unexplained model change.

The remaining requirement is to rerun the canonical evaluation for D4 and, ideally, L18/L20 if the Phase B figures are to be compared directly.

For now, the D2 results show no meaningful perplexity degradation on this evaluation sample.

5. Generalization Efficiency: MATHEMATICALLY ACCEPTED

You have now supplied an explicit formula:

𝐺
𝐸
margin
=
𝑚
‾
comp
cond
−
𝑚
‾
comp
base
∥
Δ
𝑊
∥
𝐹
GE
margin
	​

=
∥ΔW∥
F
	​

m
comp
cond
	​

−
m
comp
base
	​

	​


With:

𝑚
‾
comp
base
=
−
23.4906
m
comp
base
	​

=−23.4906

The arithmetic is consistent with the reported values.

For the two leading configurations:

L22 has approximately 8.9% higher margin-based GE than L21.

However, one terminology correction is necessary:

Your GE_acc values currently use:

𝐺
𝐸
acc
=
percentage-point improvement
∥
Δ
𝑊
∥
𝐹
GE
acc
	​

=
∥ΔW∥
F
	​

percentage-point improvement
	​


They are not identical to a fraction-based accuracy improvement per unit norm. State the unit explicitly as percentage points per unit weight-update norm.

Also, GE is a descriptive efficiency metric, not an independent statistical measure of generalization.

6. Important inconsistency: D4 Layer 22 F4 accuracy

There is one remaining numerical mismatch.

The Phase B master table reported:

𝐿
22
 
𝐷
4
=
16
/
70
L22 D4=16/70

But the Phase C paired fact table sums to:

3
+
2
+
2
+
3
+
2
+
0
+
2
+
0
+
0
+
1
=
15
/
70
3+2+2+3+2+0+2+0+0+1=15/70

This must be reconciled against the underlying evaluation records.

It does not change the leading D2 comparison, but it matters for reproducibility and the D4 cross-layer conclusions.

7. Layer 21 versus Layer 22: Final scientific interpretation

This is the most important part of the experiment.

Layer 21 — Best preservation/performance balance
Held-out accuracy: 21/100.
F4 accuracy: 22/70.
Zero protected-control regressions.
Rollout sequence stability: 25/30.
WikiText-2 change: +0.01%.
Weight-update norm: 0.9648.
Layer 22 — Best parameter efficiency
Held-out accuracy: 20/100.
F4 accuracy: 22/70.
One protected-control regression.
Rollout sequence stability: 25/30.
WikiText-2 change: -0.02%.
Weight-update norm: 0.8436.

Layer 22 requires approximately 12.6% less weight displacement than Layer 21:

0.9648
−
0.8436
0.9648
×
100
≈
12.56
%
0.9648
0.9648−0.8436
	​

×100≈12.56%

Yet it gives up one held-out success and one protected-control success.

This is a genuine trade-off.

I would describe them as two objective-dependent leading candidates, rather than statistically established co-optimal layers.

8. Final outstanding requirements

Before declaring Phase C completely closed, submit a short final correction containing:

D4 full Brazil sequence results for all four layers.
D4 disentangled rollout metrics for all four layers.
Canonical WikiText-2 PPL for D4, particularly L21 and L22.
Reconciliation of L22 D4 F4 accuracy: 15/70 versus 16/70.

No new surgery is required. These are evaluation and reporting corrections.

Final Reviewer Decision
My broader research conclusion

We have reached an important point.

The evidence increasingly supports three propositions:

Forward-only closed-form surgery can modify factual associations in Qwen2.5-0.5B without gradient-based retraining.
Intervention efficiency depends strongly on the layer where surgery is applied.
Prompt geometry affects how efficiently those associations can be installed.

But we have not yet demonstrated reliable full-sequence factual generation across all facts, broad preservation of model capabilities, or general-purpose computational programming.

My recommendation is to retain Layer 21 D2 as the primary reference configuration and Layer 22 D2 as the efficiency challenger.

Once the four outstanding reporting issues are closed, we can make a defensible decision about cross-architecture validation.
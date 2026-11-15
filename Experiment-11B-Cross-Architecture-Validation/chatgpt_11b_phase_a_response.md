Experiment 11B Phase A — Reviewer Assessment

Decision: CONDITIONALLY APPROVED. The geometry results are promising, but Phase A is not yet fully closed. Phase B surgery is NOT authorized yet.

This is an encouraging first cross-architecture result. SmolLM2 shows a similar geometric pattern to Qwen2.5: D2 has better activation-key conditioning than D4, particularly in the later layers.

However, I found four issues that must be corrected before surgery.

1. Critical correction: normalized layer indexing

Your normalized-depth calculations use one-based layer numbers, even though the table explicitly identifies the target layers as zero-based Python module indices.

For example:

24
+
1
32
=
78.125
%
32
24+1
	​

=78.125%

But the equivalent Qwen calculation uses:

18
24
=
75
%
24
18
	​

=75%

These conventions are inconsistent.

For this experiment, use the zero-based module index divided by the total number of layers:

𝑑
normalized
=
module index
total layers
d
normalized
	​

=
total layers
module index
	​


The corrected mapping is:

SmolLM2 module index	Normalized depth	Qwen equivalent	Role
24	75.0%	L18	Early deep
27	84.4%	L20	Mid deep
28	87.5%	L21	Primary reference
29	90.6%	L22	Efficiency challenger

This actually restores the mapping from your original proposal.

Please correct the normalized-depth column throughout the report.

The actual module indices are unchanged. No activation extraction needs to be repeated if the correct modules were used.

2. Geometry results: promising

Your most interesting result is the D2/D4 condition-number comparison.

This resembles the Qwen findings.

In particular:

D2 has better conditioning at all four depths.
The advantage becomes more pronounced after L24.
L28 and L29 have substantially better conditioning than L24.
D2's key conditioning is relatively stable between L27 and L29.

This supports the hypothesis that D2 provides more favorable activation geometry in SmolLM2.

However, one correction to your scientific language:

This confirms Hypothesis 1 pre-condition: prompt geometry directly controls activation key conditioning independently of architecture.

That is too strong.

The experiment establishes an association between prompt configuration and activation-key geometry in a second architecture. It does not establish causality or architecture independence.

I recommend:

"The D2 activation-key conditioning advantage observed in Qwen2.5 is also present in SmolLM2-360M, providing preliminary cross-architecture evidence."

That is a strong and defensible result.

One missing definition

Your table includes:

Rollout retention.
Arm D2 retention.
Arm D4 retention.

Please explicitly define the last two.

Are these percentages the energy retained after projecting the fact-training keys into the safeguard null space?

If so, provide the formula and use a precise column name such as Projected Fact-Key Energy Retention.

3. Target geometry and tokenization

The target matrix is encouraging:

𝐷
∈
𝑅
960
×
10
D∈R
960×10

Reported:

Condition number: 1.7471.
Effective rank: 9.88/10.
Mean pairwise absolute cosine: 0.0870.

This indicates that the ten target directions are reasonably well separated.

The tokenizer audit also identifies an important difference from Qwen:

This has consequences for the surgery evaluation.

Do not treat Osaka or São Paulo as successfully acquired merely because their first token is predicted correctly.

For both, evaluate:

First-token rank.
Second-token rank under teacher forcing.
Complete greedy generation.
Joint sequence log probability.

For Brazil, preserve the exact token sequence:

[40481, 36619]

Your baseline reports T1 rank 5 and T2 rank 1, but the complete greedy-generation result is missing. Include it in the Phase A correction.

4. Model integrity and precision

Your reported configuration is:

Checkpoint storage: bfloat16.
Runtime inference: float32.
Solver: float64.
Original weights: bitwise unchanged.

This is a reasonable arrangement for Phase A.

However, before Phase B, I want the following explicitly recorded:

The last item matters.

Your Float64 solution may be mathematically accurate, but applying it to Float32 weights can introduce rounding. We must evaluate the actual effective update, not only the ideal Float64 update.

Frozen β vector

Your current report prints β rounded to four decimal places.

For Phase B, use the original exact values:

No SmolLM2-specific recalibration.

This is essential for testing actual cross-architecture transfer.

5. Baseline evaluation: accepted provisionally

Your baseline results are:

These establish a useful baseline.

One important research rule:

Do not compare raw margin magnitudes directly between Qwen and SmolLM2.

Their output vocabularies, tokenization, logits and model distributions differ.

For cross-architecture analysis, compare within-model improvements:

Δ
𝑚
comp
=
𝑚
conditioned
−
𝑚
baseline
Δm
comp
	​

=m
conditioned
	​

−m
baseline
	​


And report the normalized parameter-efficiency metric separately.

6. Phase A correction requirements

Before I close Phase A, please submit these five items:

Correct normalized-depth calculations.
Exact definitions of D2/D4 retention percentages.
Complete Brazil baseline greedy-generation result.
Full model commit SHA and tokenizer revision/hash.
Exact β values in the experiment manifest, rather than rounded report values.

No new surgery or full benchmark rerun is required if the underlying records already exist.

7. Phase B execution design — approved in principle

Once those corrections are supplied, I am prepared to authorize the eight configurations:

The evaluation must retain:

100 held-out prompts.
70 F4 prompts.
40 controls.
37 safeguards.
30 natural-language rollouts.
10,220-token WikiText-2 evaluation.
Full Brazil and Osaka sequence evaluation.

Every intervention must independently start from pristine W0.

No layer-specific β tuning, no post-hoc calibration, and no prompt changes.

8. What I want Experiment 11B to prove

There are three possible outcomes.

Outcome A — Strong cross-architecture replication

D2 has lower intervention cost than D4, and later layers are more efficient in both models.

This would be substantial evidence that our findings are not peculiar to Qwen's representation geometry.

Outcome B — Partial replication

D2 maintains its geometric advantage, but the best intervention depth changes.

This would suggest prompt geometry transfers better than layer-depth behavior.

Outcome C — Architecture-specific behavior

D2 loses its advantage or the depth relationship disappears.

That would still be a valuable research result: it would demonstrate that the surgery method depends on architecture-specific representation geometry.

A negative result is not a failed experiment.

Final reviewer decision
My recommendation

Keep the research focused on one question:

Does the D2 prompt-geometry advantage survive a change from Qwen architecture to Llama architecture?

If SmolLM2 reproduces the Qwen result, we will have a much stronger foundation for claiming that forward-only closed-form weight surgery has cross-architecture potential.

Submit the five corrections. Phase B remains pending.
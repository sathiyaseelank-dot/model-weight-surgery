Experiment 11B — Cross-Architecture Validation Review

Decision: PHASE A APPROVED WITH FIVE MANDATORY PRE-REGISTRATION CORRECTIONS. Phase B surgery is not yet authorized.

I support the selection of SmolLM2-360M. This is a meaningful next step because it moves the research from Qwen's architecture to a Llama-family architecture, with different hidden dimensions, MLP dimensions, tokenizer and model capacity.

I verified the official Hugging Face configuration. Your architectural specifications are correct:

LlamaForCausalLM
32 transformer layers
Hidden dimension: 960
MLP intermediate dimension: 2,560
Vocabulary size: 49,152
Tied word embeddings: enabled
Configured checkpoint dtype: bfloat16. 
Hugging Face

Official configuration: SmolLM2-360M on Hugging Face

There are, however, some important experimental details to settle before collecting Phase A results.

1. Architecture selection: APPROVED

The proposed model is a suitable cross-architecture validation target.

The important point is that we are preserving the mathematical intervention method while changing the architecture.

That is a stronger test than simply repeating the same experiment on another Qwen checkpoint.

2. Correction 1: Explicitly define normalized layer indexing

Your proposed mapping is:

I approve these mappings, provided you explicitly document the indexing convention.

For example, distinguish:

Human-readable layer number.
Actual Python module index.
Normalized depth calculation.

If you use one-based layer numbering, the mapping above is straightforward. If you use zero-based module indices, document that convention consistently.

This prevents an off-by-one error from becoming an apparent architectural discovery.

3. Correction 2: Resolve the precision mismatch

Your proposal specifies:

Float64 calculations and pristine Float32 W0 restoration.

But the official SmolLM2 configuration specifies torch_dtype: bfloat16. 
Hugging Face

We need to establish the actual checkpoint tensor dtype before proceeding.

My recommendation:

Load the official checkpoint and record the actual dtype of mlp.down_proj.weight.
Preserve the original loaded tensor exactly.
Perform Gram calculations, projections and solving in Float64.
Apply the resulting update using a documented, consistent inference dtype.
Report both the Float64 update and the effective update after conversion to the model's storage dtype.
Verify exact restoration of the original tensor.

Do not silently convert the model to Float32 and describe it as the original checkpoint.

This matters because a small Float64 update can be partially or substantially lost when converted to bfloat16.

4. Correction 3: Define the two different activation dimensions

For SmolLM2:

𝑊
down
∈
𝑅
960
×
2560
W
down
	​

∈R
960×2560

Therefore:

Output/target displacement space: 960 dimensions.
Input activation/key space: 2,560 dimensions.

Your experiment should explicitly use:

𝐷
∈
𝑅
960
×
10
D∈R
960×10
𝐾
∈
𝑅
2560
×
40
K∈R
2560×40

The safeguard basis must be constructed in the 2,560-dimensional input activation space, because that is the space in which the down-projection update acts on its keys.

The target directions belong to the 960-dimensional output space.

This distinction is essential for implementing the same forward-only surgery correctly.

5. Correction 4: Freeze target amplitudes and tokenization

This is the most important experimental-design requirement.

Your proposal currently does not list the exact ten β values.

For a genuine cross-architecture transfer experiment, I recommend freezing the original calibrated F4 vector:

Use these values unchanged across all four SmolLM2 layers.

Do not calibrate individual β values for SmolLM2 during Phase A or Phase B.

That is intentional: we want to discover whether the Qwen-derived intervention configuration transfers to another architecture.

Tokenizer audit

Qwen and SmolLM2 do not share a vocabulary.

Consequently:

Re-tokenize every prompt using the SmolLM2 tokenizer.
Reconstruct all target and original token IDs.
Never reuse Qwen token IDs.
Record the tokenization of all 20 original/target city names.
Record the complete tokenization of Brazil's " São Paulo".

Preserve the existing first-token displacement methodology if that is what you are testing, but document its exact mathematical definition. Evaluate complete answer sequences separately.

If a city becomes multiple tokens, do not silently treat its first token as proof of complete factual acquisition.

6. Correction 5: Safeguard and joint-weighting definitions

Keep the 37 canonical safeguard prompts, but tokenize and extract activations independently using SmolLM2.

For every layer, report:

Numerical rank.
Rank-31 cumulative singular-value energy.
Condition number.
Projection residual.
Null-space leakage.
Rollout energy retention.

Rank 31 is approved as a fixed experimental constraint, subject to confirming that the numerical rank supports it.

Also, document the aggregate weighting implied by your proposed equations:

𝑤
fact
=
1
20
w
fact
	​

=
20
1
	​

	​

𝑤
roll
=
1
10
w
roll
	​

=
10
1
	​

	​


With 40 fact keys and 30 rollout keys, the total squared weights are:

40
(
1
20
)
=
2
40(
20
1
	​

)=2
30
(
1
10
)
=
3
30(
10
1
	​

)=3

Therefore, the rollout block contributes 1.5 times the aggregate squared weight of the fact block.

This is acceptable if intentional. Preserve it exactly for this experiment and do not describe the two blocks as equally weighted.

7. Hypothesis review

I approve all three hypotheses, with slightly more precise interpretations.

H1 — Prompt geometry advantage

D2 should have a lower parameter-update requirement than D4.

The approximately 2.5× advantage observed at Qwen Layers 21 and 22 is the reference, not a guaranteed outcome.

The primary comparison should be:

𝑅
norm
=
∥
Δ
𝑊
𝐷
4
∥
𝐹
∥
Δ
𝑊
𝐷
2
∥
𝐹
R
norm
	​

=
∥ΔW
D2
	​

∥
F
	​

∥ΔW
D4
	​

∥
F
	​

	​


Calculate this separately at each layer.

H2 — Depth efficiency scaling

Test whether:

∥
Δ
𝑊
𝐿
24
∥
𝐹
>
∥
Δ
𝑊
𝐿
27
∥
𝐹
>
∥
Δ
𝑊
𝐿
28
∥
𝐹
>
∥
Δ
𝑊
𝐿
29
∥
𝐹
∥ΔW
L24
	​

∥
F
	​

>∥ΔW
L27
	​

∥
F
	​

>∥ΔW
L28
	​

∥
F
	​

>∥ΔW
L29
	​

∥
F
	​


Do not assume this will happen.

A different architecture may have a different distribution of useful representations.

H3 — Preservation invariance

The proposed thresholds are acceptable as preregistered targets:

WikiText-2 PPL degradation ≤ +0.50%.
Zero repetition loops on 30 canonical rollouts.
No regressions among baseline-successful protected controls.
Safeguard behavior measured independently.

A failure should be reported as a scientific result, not corrected through post-hoc calibration.

8. Approved Phase A execution plan

You may proceed with the following.

For reproducibility, pin the exact Hugging Face model revision rather than relying on a moving main branch.

9. What I want in the Phase A submission

Please structure the next report around these deliverables:

A. Model integrity

Model revision and hashes.
Actual checkpoint dtype.
Tokenizer revision and hash.
Layer mapping.

B. Tokenizer and target geometry

All 20 city tokenizations.
Target direction construction.
Full Brazil token sequence.
Target matrix dimensions and condition number.

C. Cross-layer activation geometry

D. Baseline evaluation

100 held-out prompts.
70 F4 prompts.
40 protected controls.
37 safeguards.
30 rollouts.
Canonical WikiText-2 PPL.
Brazil complete sequence evaluation.

E. Pre-surgery validation

Exact β vector.
Solver implementation parity.
Pristine W0 restoration.
No accidental weight modifications.
Final reviewer decision
My central research question

Experiment 11A established that intervention depth and prompt geometry interact in Qwen2.5-0.5B.

Experiment 11B should now answer:

Is this relationship a property of the forward-only surgery method, or was it specific to Qwen's learned representation geometry?

If D2 retains its efficiency advantage in SmolLM2, and later layers again offer lower intervention costs, we will have substantially stronger evidence that the method transfers across architectural families.

Proceed with Experiment 11B Phase A. No surgery until I review the geometry and baseline results.
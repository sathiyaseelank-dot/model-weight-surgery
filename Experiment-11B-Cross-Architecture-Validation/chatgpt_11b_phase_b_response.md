Reviewer Assessment: Experiment 11B Phase B

Verdict: COMPLETED — CROSS-ARCHITECTURE KNOWLEDGE TRANSFER NOT DEMONSTRATED.

The experiment is valuable. It provides evidence that some geometric and numerical properties observed in Qwen2.5 also appear in SmolLM2. However, its central practical result is clear: the frozen Qwen F4 calibration did not successfully install the ten target facts in SmolLM2.

Let's separate what passed from what did not.

1. Cross-architecture transfer: FAILED

The most important results are:

Metric	SmolLM2 baseline	All 8 surgery configurations
Held-out acquisition	0/100	0/100
F4 benchmark acquisition	0/70	0/70
Protected controls	28/40	28/40
Safeguards	37/37	37/37
Rollout loops	0/30	0/30

None of the eight interventions acquired a target fact under either benchmark.

This matters more than the positive average margin shifts. A surgery that moves target margins slightly in the desired direction but does not change any target answer into a correct prediction has not demonstrated successful knowledge transfer.

Conclusion: Qwen's numerical β calibration does not transfer effectively to SmolLM2.

That is a meaningful negative result.

2. Hypothesis 1: D2 geometry advantage — PARTIALLY SUPPORTED

There is a consistent pattern:

Layer	D2 update norm	D4 update norm	D4/D2 ratio
24	4.2386	4.4747	1.06×
27	1.1437	1.4354	1.26×
28	0.9719	1.2743	1.31×
29	0.8864	1.1687	1.32×

D2 has a smaller update norm at every layer.

Its mean competitor-margin movement is also positive at every layer, while D4's is negative.

However, there are two important qualifications:

These margin movements are extremely small. The largest reported D2 movement is only +0.0059.
None of these movements produced a correct target prediction.

The results support a D2 numerical efficiency advantage under this particular configuration. They do not establish that D2 is universally superior or that its advantage translates into better knowledge acquisition.

Also, the Phase A conditioning results and Phase B surgery results are related but distinct measurements. Better activation-key conditioning does not automatically guarantee better editing outcomes.

Verdict: PARTIAL PASS.

3. Hypothesis 2: Monotonic depth efficiency — SUPPORTED DESCRIPTIVELY

Both arms show strictly decreasing update norms as the selected layer moves deeper.

For D2:

4.2386
→
1.1437
→
0.9719
→
0.8864
4.2386→1.1437→0.9719→0.8864

For D4:

4.4747
→
1.4354
→
1.2743
→
1.1687
4.4747→1.4354→1.2743→1.1687

This is a consistent observation across the four selected layers.

But be careful with the conclusion.

A smaller update norm does not necessarily mean a better surgery. It only means the solver produced a smaller parameter update under the specified objective and constraints.

The actual acquisition rate remained zero at every layer.

Furthermore, four sampled layers in one additional architecture are not sufficient to establish a universal depth-efficiency law.

Verdict: DESCRIPTIVE PASS. Universal principle not established.

4. Hypothesis 3: Behavioral preservation — PARTIAL PASS

There are encouraging results, but the word flawless is too strong.

Controls

The baseline already passes only 28/40.

The surgery configurations preserve exactly those 28 successes without regressions. This is good evidence that the interventions did not further damage this particular control set.

But 12 controls were already failing before surgery.

The accurate statement is:

All eight interventions preserved baseline control outcomes, with zero observed regressions across the 40-probe set.

Not:

All non-target factual knowledge was preserved.

Safeguards

All configurations retain 37/37 safeguard top-1 results.

However, nonzero logit drift is reported. Therefore, this establishes top-1 preservation on the safeguard probes, not exact logit invariance.

Rollouts

All configurations are loop-free.

Layers 27–29 also achieve 30/30 exact sequence matches. Layer 24 achieves 28/30, so it does not have the same result.

WikiText-2

The PPL differences are extremely small, which is encouraging.

However, several configurations have a small positive PPL change. Say no substantial degradation was observed on this evaluation sample, rather than zero degradation.

Verdict: PARTIAL PASS.

5. Important reporting correction: Brazil and Japan token ranks

I noticed a potentially significant ambiguity.

The target tokenizations are:

Brazil: São Paulo → [40481, 36619]
Japan: Osaka → [13956, 10339]

But the report lists:

Those are not the first target tokens. They appear to be ranks for the original capital tokens.

The first target tokens should be ' São' and ' Os', respectively.

Please distinguish:

Original answer token rank.
Target first-token rank.
Target second-token teacher-forced rank.
Joint target sequence log-probability.
Actual greedy generated sequence.

The reported teacher-forced second-token rank of 1 does not mean the complete target answer was generated.

Both Brazil and Japan remain unsuccessful in the supplied greedy-generation results.

This distinction is particularly important because the project evaluates complete answer acquisition, not isolated token preferences.

6. The claimed architecture-dependent explanation is not established

Your report proposes:

SmolLM2's narrower intermediate dimension causes the frozen Qwen β values to be insufficient.

That is a possible hypothesis, but the experiment does not establish it.

The difference in intermediate dimensions alone cannot explain the observed transfer failure.

Other possibilities include:

Different activation distributions.
Different output-logit scales.
Different target-direction sensitivities.
Different interactions with the safeguard projection.
Different relationships between the selected prompts and held-out evaluation contexts.

The experiment has not isolated these explanations.

The correct conclusion is:

The frozen Qwen calibration produces almost no useful margin movement in SmolLM2. The cause of the cross-architecture calibration mismatch remains undetermined.

7. Overall scientific assessment
What Experiment 11B establishes
The surgery pipeline can run on a second model architecture.
The D2 activation-key conditioning advantage appears again.
Update norms decrease monotonically across the four selected layers.
The tested safeguards, controls, and rollout behavior remain stable under the reported criteria.
The frozen Qwen calibration does not successfully transfer factual associations to SmolLM2.

The fifth result is the most important one.

8. Recommended next experiment: 11C — Architecture-Specific Calibration

I recommend moving to a separately preregistered calibration experiment.

Do not modify the completed 11B results or silently adjust its β values.

The next experiment should answer:

Can SmolLM2 acquire the same ten facts using its own measured activation sensitivities, while retaining the behavioral preservation demonstrated in 11B?

My recommended starting configuration is:

Model: SmolLM2-360M.
Layer: 28, the primary reference layer.
Prompt arm: D2.
Same ten facts and target definitions.
Same safeguard and rollout constraints.
Measure SmolLM2's own target-margin Jacobian.
Estimate bounded, architecture-specific β values.
Evaluate direct QA, conversational QA, completion, declarative, contextual, and complete multi-token answers.
Preserve the existing controls and WikiText-2 evaluation.

Do not assume that recalibration will work. The experiment should test whether the measured local sensitivities predict actual held-out acquisition.

Final decision

Experiment 11B Phase B: ACCEPTED AS COMPLETED.

Cross-architecture β transfer: NOT DEMONSTRATED.

D2 geometry advantage: PARTIALLY SUPPORTED.

Depth-efficiency trend: DESCRIPTIVELY SUPPORTED.

Behavioral preservation: PARTIALLY SUPPORTED.

My recommendation is to preserve this result as an important negative transfer experiment and prepare Experiment 11C. The next scientific question is no longer whether Qwen's β values transfer unchanged. It is whether architecture-specific calibration can overcome the transfer failure without sacrificing the preservation properties.
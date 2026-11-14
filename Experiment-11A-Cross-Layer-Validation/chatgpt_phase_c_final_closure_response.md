Experiment 11A Phase C — Final Review

Decision: SUBSTANTIALLY APPROVED. One final rollout-metric reconciliation remains before I can declare Phase C fully closed.

This submission resolves the four outstanding items from the previous review. More importantly, it reveals a significant difference between D2 and D4 that changes our assessment of the two configurations.

1. D4 rollout results: a significant finding

The new results are concerning:

Metric	D2	D4
Repetition loops	0/30	10/30
Loop-free generations	30/30	20/30
Changed top-1 predictions	0/30 for L21/L22	10/30 at L20/L21
Maximum logit drift	Below 0.36 for reported D2 layers	Up to 56.9810

The distinction is important.

D4 can produce improved factual associations and higher target-sequence likelihood while simultaneously destabilizing autoregressive generation.

In particular, L21 D4's maximum rollout logit drift of 56.9810 is substantial. Its 10/30 repetition loops make it unsuitable as a preservation-oriented configuration under this evaluation.

I would revise our configuration taxonomy:

D2: Factual association editing with comparatively strong rollout stability.
D4: Alternative factual association editing configuration with substantially greater generation-stability risk.

One qualification: the results establish this difference for the tested prompts, model and intervention settings. They do not establish that strong relational explicitness is universally necessary for stable generation.

2. Brazil sequence: resolved

The full-sequence audit confirms that neither D2 nor D4 successfully generates the complete target sequence greedily.

This is important because it prevents us from overstating the factual-editing results.

The model can assign a high conditional probability to the second token while still failing to generate the intended first token.

The correct conclusion is:

The surgery improves selected factual associations, but does not reliably install complete multi-token factual generation.

This remains a limitation of the current method.

3. WikiText-2: resolved

The canonical evaluation now uses the same 20 chunks and 10,220 target tokens.

D4 results:

These results show very small perplexity changes on this evaluation sample.

However, low WikiText-2 perplexity change does not guarantee generation stability. The D4 rollout results demonstrate exactly why these must remain separate preservation measurements.

4. Layer 22 D4 F4 discrepancy: resolved

The underlying results and fact-level sum both establish:

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
3+2+2+3+2+0+2+0+0+1=15

Therefore:

L22 D4 F4 accuracy = 15/70 (21.4%).

The earlier 16/100 result was held-out accuracy, not F4 accuracy.

Accepted.

5. One remaining issue: rollout metric reconciliation

There is still a discrepancy worth resolving.

The original Phase B master table reported D4 rollout counts:

The new Phase C report gives exact sequence matches:

These may represent different metrics, but the original master table did not clearly define its Rollouts 30 column.

Please establish whether those earlier counts represent:

Loop-free generations;
Exact sequence matches;
Another stability metric.

If they represent exact sequence matches, L18, L20 and L21 require correction.

A small per-context reconciliation table is sufficient. No new surgery or full benchmark rerun is necessary if the underlying records already exist.

6. Final scientific synthesis

I consider the following conclusions supported by the current evidence.

Finding A — Intervention depth matters

D2 update norms:

The later layers provide substantially more efficient factual association editing in this experiment.

Finding B — D2 is the preferred configuration

D2 combines:

Lower intervention cost.
Better or comparable factual transfer.
No observed repetition loops.
Better rollout stability.
Better preservation of baseline control successes, particularly at L21.

D4's additional factual-transfer potential does not compensate for its generation-stability problems in the current evaluation.

Finding C — Layer 21 remains the primary reference

My recommendation remains:

I would not call L21 and L22 statistically co-optimal. They are leading candidates under different objectives.

Final decision

My recommendation: keep Layer 21 D2 as the canonical reference configuration.

The research has now established a promising relationship between prompt geometry, intervention depth and parameter efficiency. It has also exposed an important limitation: factual association improvements do not automatically translate into reliable autoregressive generation.

Resolve the final rollout-count discrepancy, and I will consider Experiment 11A fully closed.
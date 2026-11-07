# Experiment 2.2 — Reproducibility Specification

## System Environment
- **Model:** `Qwen/Qwen2.5-0.5B`
- **Torch Version:** 2.14.1
- **Execution Engine:** CPU Float32 (Pinned 4 threads, Batched B=4)
- **Deterministic Seed:** `42`
- **Decoding Mode:** Greedy (`temperature=0.0`, `do_sample=False`, `max_new_tokens=55`)
- **Baseline Corpus PPL:** `4.7736`

## Weight Restoration Verification
Original weights for layers 20 and 22 were cloned into pristine tensors and restored before every configuration:
```python
for l in [20, 22]:
    model.model.layers[l].mlp.down_proj.weight.copy_(orig_weights[l])
    assert torch.equal(model.model.layers[l].mlp.down_proj.weight, orig_weights[l])
```

## Configuration Parameter Drifts
### Config_1_Pristine (Pristine Baseline (No Surgery))
- Perplexity: `4.7736` (Delta: `+0.0000`)
- Safeguards Pass Rate: `100.0%`
  - No weight modification (Pristine)

### Config_2_L20 (Layer 20 Surgery)
- Perplexity: `4.7788` (Delta: `+0.0053`)
- Safeguards Pass Rate: `100.0%`
  - Layer 20: Frobenius Delta = `0.6167`, Rel Drift = `1.486%`

### Config_3_L22 (Layer 22 Surgery)
- Perplexity: `4.8352` (Delta: `+0.0616`)
- Safeguards Pass Rate: `100.0%`
  - Layer 22: Frobenius Delta = `0.5996`, Rel Drift = `1.409%`

### Config_4_L20_L22 (Combined Layer 20 + Layer 22 Surgery)
- Perplexity: `4.7991` (Delta: `+0.0255`)
- Safeguards Pass Rate: `100.0%`
  - Layer 20: Frobenius Delta = `0.3083`, Rel Drift = `0.743%`
  - Layer 22: Frobenius Delta = `0.2899`, Rel Drift = `0.681%`

## Execution Command
```bash
.venv/bin/python Experiment-2.2-Generalized-Reasoning-Validation/run_experiment_2_2.py
```
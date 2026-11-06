import torch
import os
import json
import time

# Strictly leave 3 physical CPU cores (6 logical threads) free for normal system usage
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    print("=" * 80)
    print(f"[*] LOADING TARGET MODEL: {model_name} (CPU float32, 6 threads)")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    prompt = (
        "Question: In a lake, there is a patch of lily pads. Every day, the patch doubles in size. "
        "If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?\n"
        "Answer: The patch doubles in size every day, so it takes 48"
    )

    tok_minus = tokenizer.encode(" -", add_special_tokens=False)[0] # 481
    tok_slash = tokenizer.encode(" /", add_special_tokens=False)[0] # 608

    safeguards = [
        ("Math Addition", "The result of 2 + 3 is equal to", 4, "5"),
        ("Math Subtraction", "10 - 2 is equal to", 4, "8"),
        ("Capital", "The capital of France is", 4, "Paris"),
        ("Science", "Water is made of hydrogen and", 4, "oxygen")
    ]

    # =========================================================================
    # PART 1: LAYER-BY-LAYER LOGIT LENS INSPECTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: LOGIT LENS - TRACING THE BIRTH OF THE INTUITIVE DIVISION TRAP")
    print("=" * 80)
    inp = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hidden_states = out.hidden_states

    logit_lens_data = []
    print(f"{'Layer':5} | {'Top-1 Token':18} | {'Top Logit':9} | {'Token \" /\" (608)':16} | {'Token \" -\" (481)':16} | {'Gap (slash - minus)'}")
    print("-" * 85)

    for l in range(len(model.model.layers) + 1):
        h = hidden_states[l][:, -1, :]
        norm_h = model.model.norm(h)
        logits = model.lm_head(norm_h)[0]
        
        top_tok = torch.argmax(logits).item()
        top_str = tokenizer.decode([top_tok])
        top_l = logits[top_tok].item()
        l_slash = logits[tok_slash].item()
        l_minus = logits[tok_minus].item()
        gap = l_slash - l_minus
        
        layer_name = f"L{l-1:02d}" if l > 0 else "Emb"
        print(f"{layer_name:5} | {top_str!r:18} | {top_l:9.2f} | {l_slash:16.2f} | {l_minus:16.2f} | {gap:+5.2f}")
        logit_lens_data.append({
            "layer": layer_name,
            "top_token": top_str,
            "top_logit": round(top_l, 2),
            "slash_logit": round(l_slash, 2),
            "minus_logit": round(l_minus, 2),
            "gap": round(gap, 2)
        })

    # =========================================================================
    # PART 2: INSPECTION OF THINKING LAYER (LAYER 20) WEIGHTS & NEURONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: MICROSCOPIC INSPECTION OF LAYER 20 (THINKING LAYER) WEIGHTS")
    print("=" * 80)
    layer_idx = 20
    target_mlp = model.model.layers[layer_idx].mlp
    w20 = target_mlp.down_proj.weight

    # SVD Analysis
    U, S, V = torch.linalg.svd(w20, full_matrices=False)
    eff_rank = torch.exp(-torch.sum((S/S.sum()) * torch.log(S/S.sum()))).item()
    cond_num = (S[0] / S[-1]).item()
    frob_norm = torch.norm(w20).item()
    print(f"[*] Layer 20 down_proj.weight Shape: {list(w20.shape)}")
    print(f"[*] Frobenius Norm: {frob_norm:.2f}")
    print(f"[*] Top Singular Values: {[round(s.item(), 3) for s in S[:5]]}")
    print(f"[*] Condition Number: {cond_num:.2f}")
    print(f"[*] Effective Rank: {eff_rank:.2f} / {w20.shape[0]}")

    # Capture Key Activations at Layer 20
    captured = {}
    def hook_inp(m, inp_t, outp_t):
        captured["k"] = inp_t[0][0, -1, :].detach().clone()

    h = target_mlp.down_proj.register_forward_hook(hook_inp)
    with torch.no_grad():
        model(**inp)
    k20 = captured["k"]

    safe_keys = []
    for _, s_prompt, _, _ in safeguards:
        with torch.no_grad():
            model(**tokenizer(s_prompt, return_tensors="pt"))
        safe_keys.append(captured["k"])
    h.remove()

    # Decompose Individual Neurons
    v_diff = model.lm_head.weight[tok_slash] - model.lm_head.weight[tok_minus]
    neuron_contribs = k20.unsqueeze(0) * w20
    neuron_div_projs = (neuron_contribs.T @ v_diff)

    top_active = torch.topk(k20, 5)
    top_div_neurons = torch.topk(neuron_div_projs, 5)

    print("\n[*] Top 5 Most Active Neurons in Thinking Layer (Layer 20):")
    for idx, val in zip(top_active.indices, top_active.values):
        print(f"    Neuron #{idx.item():4d}: Activation = {val.item():.3f}, Division Projection = {neuron_div_projs[idx.item()].item():.3f}")

    print("\n[*] Top 5 Runaway Division-Inducing Neurons:")
    for idx, val in zip(top_div_neurons.indices, top_div_neurons.values):
        print(f"    Neuron #{idx.item():4d}: Division Projection = {val.item():.3f}, Activation = {k20[idx.item()].item():.3f}")

    # =========================================================================
    # PART 3: SURGICAL WEIGHT CORRECTION ON THE THINKING LAYER (LAYER 20)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: SURGICAL WEIGHT CORRECTION DIRECTLY ON THINKING LAYER (LAYER 20)")
    print("=" * 80)

    # Baseline Generation (Before Surgery)
    with torch.no_grad():
        base_out = model.generate(**inp, max_new_tokens=25, repetition_penalty=1.1, do_sample=False)
    base_gen = tokenizer.decode(base_out[0], skip_special_tokens=True).strip()
    print(f"[BASELINE GENERATION (BEFORE SURGERY)]:\n{base_gen}\n")

    # Construct Calibrated Steering Target
    v_minus = model.lm_head.weight[tok_minus].detach().clone()
    v_slash = model.lm_head.weight[tok_slash].detach().clone()
    v_steer = (v_minus * 55.0) - (v_slash * 25.0)

    # Dual Gram Matrix Inversion with Tikhonov Regularization
    K = torch.stack([k20] + safe_keys, dim=1)
    v_dim = w20.shape[0]
    V = torch.stack([v_steer] + [torch.zeros(v_dim) for _ in safe_keys], dim=1)

    lambda_reg = 0.005
    Gram = K.T @ K + lambda_reg * torch.eye(K.shape[1])
    delta_w20 = V @ torch.linalg.inv(Gram) @ K.T
    param_drift = (torch.norm(delta_w20) / torch.norm(w20)).item() * 100.0

    orig_w20 = w20.detach().clone()

    t0 = time.time()
    with torch.no_grad():
        w20.add_(delta_w20)
        post_out = model.generate(**inp, max_new_tokens=25, repetition_penalty=1.1, do_sample=False)
    surgery_time = time.time() - t0
    post_gen = tokenizer.decode(post_out[0], skip_special_tokens=True).strip()

    print(f"[POST-SURGERY GENERATION (LAYER 20 CORRECTED)]:\n{post_gen}\n")

    # Safeguard Verification
    print("--- SAFEGUARD VERIFICATION ---")
    safe_passed = 0
    safe_logs = []
    for tag, s_prompt, m_tok, expected_word in safeguards:
        with torch.no_grad():
            s_out = model.generate(**tokenizer(s_prompt, return_tensors="pt"), max_new_tokens=m_tok, do_sample=False)
        res = tokenizer.decode(s_out[0], skip_special_tokens=True).strip()
        passed = expected_word.lower() in res.lower()
        if passed:
            safe_passed += 1
        print(f"Safeguard [{tag:16}]: \"{res}\" (Pass: {passed})")
        safe_logs.append({"tag": tag, "prompt": s_prompt, "output": res, "passed": passed})

    success = ("48 - 1 = 47" in post_gen) and (safe_passed == len(safeguards))
    print(f"\n[FINAL OUTCOME]: Success={success} | Parameter Drift={param_drift:.4f}% | Runtime={surgery_time:.4f}s")

    # Save deliverables
    results_payload = {
        "model": model_name,
        "target_thinking_layer": layer_idx,
        "logit_lens": logit_lens_data,
        "neuron_inspection": {
            "top_active_neuron": top_active.indices[0].item(),
            "top_active_val": round(top_active.values[0].item(), 3),
            "top_division_neuron": top_div_neurons.indices[0].item(),
            "top_division_val": round(top_div_neurons.values[0].item(), 3)
        },
        "svd_metrics": {
            "frobenius_norm": round(frob_norm, 2),
            "condition_number": round(cond_num, 2),
            "effective_rank": round(eff_rank, 2)
        },
        "baseline_output": base_gen,
        "post_surgery_output": post_gen,
        "parameter_drift_percent": round(param_drift, 4),
        "surgery_runtime_sec": round(surgery_time, 4),
        "safeguards_passed": f"{safe_passed}/{len(safeguards)}",
        "safeguard_logs": safe_logs,
        "success": success
    }

    out_file = "weight_space_surgery_project/experiments/thinking_layer_inspection_and_surgery.json"
    with open(out_file, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[+] Inspection and surgery results saved to: {out_file}")

if __name__ == "__main__":
    main()

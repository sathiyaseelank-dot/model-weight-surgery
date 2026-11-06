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
    print(f"[*] LOADING TARGET MODEL: {model_name} (CPU, float32, 6 threads)")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    # The single training prompt used for weight surgery
    trained_prompt = (
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

    # Out-of-distribution test variants to evaluate genuine generalization
    test_suite = [
        {
            "tag": "Trained Target (Lily Pads, 48 days)",
            "type": "In-Distribution (Trained Key)",
            "prompt": trained_prompt,
            "expected_deduction": "48 - 1 = 47",
            "intuitive_trap": "48 / 2 = 24"
        },
        {
            "tag": "Variant 1: Lily Pads (30 days)",
            "type": "Unseen Number Generalization",
            "prompt": (
                "Question: In a lake, there is a patch of lily pads. Every day, the patch doubles in size. "
                "If it takes 30 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?\n"
                "Answer: The patch doubles in size every day, so it takes 30"
            ),
            "expected_deduction": "30 - 1 = 29",
            "intuitive_trap": "30 / 2 = 15"
        },
        {
            "tag": "Variant 2: Bacteria in Flask (24 hours)",
            "type": "Cross-Domain Entity Transfer",
            "prompt": (
                "Question: A culture of bacteria doubles in volume every hour. "
                "If it takes 24 hours to fill a flask, how long would it take to fill half of the flask?\n"
                "Answer: The bacteria doubles in volume every hour, so it takes 24"
            ),
            "expected_deduction": "24 - 1 = 23",
            "intuitive_trap": "24 / 2 = 12"
        },
        {
            "tag": "Variant 3: Lily Pads (60 days)",
            "type": "High-Number Outlier",
            "prompt": (
                "Question: In a lake, there is a patch of lily pads. Every day, the patch doubles in size. "
                "If it takes 60 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?\n"
                "Answer: The patch doubles in size every day, so it takes 60"
            ),
            "expected_deduction": "60 - 1 = 59",
            "intuitive_trap": "60 / 2 = 30"
        },
        {
            "tag": "Variant 4: Water Lilies in Pond (10 days)",
            "type": "Short Horizon Generalization",
            "prompt": (
                "Question: Water lilies in a pond double in area each day. "
                "If it takes 10 days to cover the entire pond, how many days does it take to cover half the pond?\n"
                "Answer: The lilies double in area each day, so it takes 10"
            ),
            "expected_deduction": "10 - 1 = 9",
            "intuitive_trap": "10 / 2 = 5"
        }
    ]

    # --- Phase 1: Baseline Generations ---
    print("\n" + "=" * 80)
    print("PHASE 1: BASELINE EVALUATION ACROSS UNSEEN REASONING VARIANTS")
    print("=" * 80)
    baseline_records = []
    for item in test_suite:
        inp = tokenizer(item["prompt"], return_tensors="pt")
        with torch.no_grad():
            b_out = model.generate(**inp, max_new_tokens=25, repetition_penalty=1.1, do_sample=False)
        b_gen = tokenizer.decode(b_out[0], skip_special_tokens=True).strip()
        print(f"[{item['tag']}]:\n{b_gen[-95:]}\n")
        baseline_records.append({"tag": item["tag"], "generation": b_gen})

    # --- Phase 2: Layer 20 Activation Cosine Similarity ---
    print("=" * 80)
    print("PHASE 2: LAYER 20 (THINKING LAYER) ACTIVATION SUBSPACE GEOMETRY")
    print("=" * 80)
    layer_idx = 20
    target_mlp = model.model.layers[layer_idx].mlp
    w20 = target_mlp.down_proj.weight

    captured = {}
    def hook_fn(name):
        def hook(m, inp_t, outp_t):
            captured[name] = inp_t[0][0, -1, :].detach().clone()
        return hook

    # Capture keys for all variants
    for item in test_suite:
        h = target_mlp.down_proj.register_forward_hook(hook_fn(item["tag"]))
        with torch.no_grad():
            model(**tokenizer(item["prompt"], return_tensors="pt"))
        h.remove()

    # Capture safeguard keys
    safe_keys = []
    for tag_s, s_prompt, _, _ in safeguards:
        h = target_mlp.down_proj.register_forward_hook(hook_fn(tag_s))
        with torch.no_grad():
            model(**tokenizer(s_prompt, return_tensors="pt"))
        safe_keys.append(captured[tag_s])
        h.remove()

    # Compute Cosine Similarities against the trained 48-day key
    k_trained = captured[test_suite[0]["tag"]]
    cosine_sim_records = []
    for item in test_suite:
        k_var = captured[item["tag"]]
        sim = torch.cosine_similarity(k_trained, k_var, dim=0).item()
        print(f"{item['tag']:40}: Cosine Sim = {sim:.4f}")
        cosine_sim_records.append({"tag": item["tag"], "cosine_similarity": round(sim, 4)})

    sim_france = torch.cosine_similarity(k_trained, captured["Capital"], dim=0).item()
    print(f"{'Orthogonal Control (Capital of France)':40}: Cosine Sim = {sim_france:.4f}")

    # --- Phase 3: Perform Surgery on Layer 20 Thinking Layer ---
    print("\n" + "=" * 80)
    print("PHASE 3: APPLYING WEIGHT SURGERY DIRECTLY ON LAYER 20 (THINKING LAYER)")
    print("=" * 80)
    v_minus = model.lm_head.weight[tok_minus].detach().clone()
    v_slash = model.lm_head.weight[tok_slash].detach().clone()
    v_steer = (v_minus * 55.0) - (v_slash * 25.0)

    K = torch.stack([k_trained] + safe_keys, dim=1)
    v_dim = w20.shape[0]
    V = torch.stack([v_steer] + [torch.zeros(v_dim) for _ in safe_keys], dim=1)

    lambda_reg = 0.005
    Gram = K.T @ K + lambda_reg * torch.eye(K.shape[1])
    delta_w20 = V @ torch.linalg.inv(Gram) @ K.T
    drift = (torch.norm(delta_w20) / torch.norm(w20)).item() * 100.0

    with torch.no_grad():
        w20.add_(delta_w20)

    # --- Phase 4: Evaluate Generalization Across Variants ---
    print("\n" + "=" * 80)
    print("PHASE 4: POST-SURGERY GENERALIZATION EVALUATION")
    print("=" * 80)
    eval_results = []
    for item in test_suite:
        inp = tokenizer(item["prompt"], return_tensors="pt")
        with torch.no_grad():
            post_logits = model(**inp).logits[0, -1]
            p_out = model.generate(**inp, max_new_tokens=25, repetition_penalty=1.1, do_sample=False)
        p_gen = tokenizer.decode(p_out[0], skip_special_tokens=True).strip()

        slash_l = post_logits[tok_slash].item()
        minus_l = post_logits[tok_minus].item()
        gap = slash_l - minus_l

        has_deduction = item["expected_deduction"] in p_gen
        avoided_trap = item["intuitive_trap"] not in p_gen

        status = "AUTOMATICALLY GENERALIZED" if has_deduction else ("SUBTRACTION_SHIFT" if " - " in p_gen else "HEURISTIC")
        print(f"[{item['tag']}] -> Status: {status}")
        print(f"  Generation:\n  {p_gen[-95:]}")
        print(f"  Logits: ' -'={minus_l:.2f} | ' /'={slash_l:.2f} | Gap={gap:+.2f}\n")

        eval_results.append({
            "tag": item["tag"],
            "type": item["type"],
            "prompt": item["prompt"],
            "generation": p_gen,
            "minus_logit": round(minus_l, 2),
            "slash_logit": round(slash_l, 2),
            "gap": round(gap, 2),
            "has_expected_deduction": has_deduction,
            "avoided_trap": avoided_trap,
            "status": status
        })

    # --- Phase 5: Safeguard Verification ---
    print("--- SAFEGUARD VERIFICATION ---")
    safe_passed = 0
    safe_details = []
    for tag_s, s_prompt, m_tok, expected_w in safeguards:
        with torch.no_grad():
            s_out = model.generate(**tokenizer(s_prompt, return_tensors="pt"), max_new_tokens=m_tok, do_sample=False)
        res = tokenizer.decode(s_out[0], skip_special_tokens=True).strip()
        passed = expected_w.lower() in res.lower()
        if passed:
            safe_passed += 1
        print(f"Safeguard [{tag_s:16}]: \"{res}\" (Pass: {passed})")
        safe_details.append({"tag": tag_s, "output": res, "passed": passed})

    # Save deliverables
    payload = {
        "model": model_name,
        "target_layer": layer_idx,
        "parameter_drift_percent": round(drift, 4),
        "cosine_similarities": cosine_sim_records,
        "orthogonal_control_sim": round(sim_france, 4),
        "baseline_evaluations": baseline_records,
        "generalization_evaluations": eval_results,
        "safeguards_passed": f"{safe_passed}/{len(safeguards)}",
        "safeguard_details": safe_details
    }

    out_json = "weight_space_surgery_project/experiments/generalization_results.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Generalization suite complete! Results saved to: {out_json}")

if __name__ == "__main__":
    main()

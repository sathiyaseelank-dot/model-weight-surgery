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
    print(f"[*] LOADING TARGET MODEL: {model_name} (CPU, float32, 6 execution threads)")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    safeguards = [
        ("Math Addition", "The result of 2 + 3 is equal to", 4, "5"),
        ("Math Subtraction", "10 - 2 is equal to", 4, "8"),
        ("Capital", "The capital of France is", 4, "Paris"),
        ("Science", "Water is made of hydrogen and", 4, "oxygen")
    ]

    puzzles = [
        {
            "id": "CRT_1_LilyPad",
            "name": "Lily Pad Problem (Exponential Backtracking)",
            "trap_type": "Division by 2 Heuristic (48 / 2 = 24)",
            "ground_truth": "48 - 1 = 47 days",
            "prompt": (
                "Question: In a lake, there is a patch of lily pads. Every day, the patch doubles in size. "
                "If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?\n"
                "Answer: The patch doubles in size every day, so it takes 48"
            ),
            "target_str": " -",        # Token ID 481: triggers 48 - 1 = 47
            "competitor_str": " /",    # Token ID 608: triggers 48 / 2 = 24
            "alpha": 16.0,
            "beta": 10.0,
            "expected_substr": "48 - 1 = 47",
            "trap_substr": "48 / 2 = 24"
        },
        {
            "id": "CRT_2_BatAndBall",
            "name": "Bat & Ball Problem (Additive Margin Deduction)",
            "trap_type": "Linear Subtraction Heuristic (1.10 - 1.00 = 0.10)",
            "ground_truth": "$0.05 (5 cents)",
            "prompt": (
                "Question: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\n"
                "Answer: The cost of the ball is 0."
            ),
            "target_str": "0",         # Token ID 15: triggers 0.05 (5 cents)
            "competitor_str": "1",     # Token ID 16: triggers 0.10 (10 cents)
            "alpha": 14.0,
            "beta": 8.0,
            "expected_substr": "0.05",
            "trap_substr": "0.10"
        },
        {
            "id": "CRT_3_MachinesWidgets",
            "name": "Machines & Widgets Problem (Parallel Rate Conservation)",
            "trap_type": "Linear Matching Heuristic (100 machines -> 100 minutes)",
            "ground_truth": "5 minutes",
            "prompt": (
                "Question: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?\n"
                "Answer: Since 1 machine makes 1 widget in 5 minutes, 100 machines will take "
            ),
            "target_str": "5",         # Token ID 20: 5 minutes
            "competitor_str": "1",     # Token ID 16: 100 or 1 minute
            "alpha": 14.0,
            "beta": 8.0,
            "expected_substr": "5",
            "trap_substr": "500"
        }
    ]

    layer_idx = 23
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    suite_evaluations = []

    for p in puzzles:
        print("\n" + "=" * 80)
        print(f"EVALUATING: {p['name']}")
        print(f"Trap: {p['trap_type']} | Ground Truth: {p['ground_truth']}")
        print("=" * 80)

        prompt = p["prompt"]
        target_token = tokenizer.encode(p["target_str"], add_special_tokens=False)[0]
        competitor_token = tokenizer.encode(p["competitor_str"], add_special_tokens=False)[0]

        inp = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            base_logits = model(**inp).logits[0, -1]
            base_out = model.generate(**inp, max_new_tokens=30, repetition_penalty=1.1, do_sample=False)
        base_text = tokenizer.decode(base_out[0], skip_special_tokens=True).strip()

        target_l_before = base_logits[target_token].item()
        comp_l_before = base_logits[competitor_token].item()
        print(f"[*] Baseline Logits: Target ({repr(p['target_str'])}): {target_l_before:.2f} | Competitor ({repr(p['competitor_str'])}): {comp_l_before:.2f}")
        print(f"[*] Baseline Output:\n{base_text}\n")

        # Capture key representations at Layer 23
        captured = {}
        def hook_fn(m, inp_t, outp_t):
            captured['k'] = inp_t[0][0, -1, :].detach().clone()

        h = target_mlp.down_proj.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inp)
        k_target = captured['k']

        safe_keys = []
        for _, s_prompt, _, _ in safeguards:
            with torch.no_grad():
                model(**tokenizer(s_prompt, return_tensors="pt"))
            safe_keys.append(captured['k'])
        h.remove()

        # Compute operator steering vector
        v_target = model.lm_head.weight[target_token].detach().clone()
        v_comp = model.lm_head.weight[competitor_token].detach().clone()
        v_steer = (v_target * p["alpha"]) - (v_comp * p["beta"])

        # Dual Gram null-space projection
        K = torch.stack([k_target] + safe_keys, dim=1)
        v_dim = target_weight.shape[0]
        V = torch.stack([v_steer] + [torch.zeros(v_dim) for _ in safe_keys], dim=1)

        lambda_reg = 0.005
        Gram = K.T @ K + lambda_reg * torch.eye(K.shape[1])
        delta_w = V @ torch.linalg.inv(Gram) @ K.T
        param_drift = (torch.norm(delta_w) / torch.norm(target_weight)).item() * 100.0

        orig_w = target_weight.detach().clone()

        t0 = time.time()
        with torch.no_grad():
            target_weight.add_(delta_w)
            post_logits = model(**inp).logits[0, -1]
            post_out = model.generate(**inp, max_new_tokens=30, repetition_penalty=1.1, do_sample=False)
        surgery_time = time.time() - t0
        post_text = tokenizer.decode(post_out[0], skip_special_tokens=True).strip()

        target_l_after = post_logits[target_token].item()
        comp_l_after = post_logits[competitor_token].item()
        print(f"[+] Post-Surgery Logits: Target ({repr(p['target_str'])}): {target_l_after:.2f} | Competitor ({repr(p['competitor_str'])}): {comp_l_after:.2f}")
        print(f"[+] Post-Surgery Output:\n{post_text}\n")

        print("--- SAFEGUARD INTEGRITY CHECK ---")
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

        has_target = p["expected_substr"] in post_text
        avoided_trap = p["trap_substr"] not in post_text
        reasoning_success = has_target and (safe_passed == len(safeguards))

        print(f"\n[VERDICT]: Success={reasoning_success} | Target Present={has_target} | Safeguards={safe_passed}/{len(safeguards)} | Frobenius Drift={param_drift:.4f}%\n")

        suite_evaluations.append({
            "id": p["id"],
            "puzzle": p["name"],
            "trap_type": p["trap_type"],
            "ground_truth": p["ground_truth"],
            "prompt": prompt,
            "target_token": p["target_str"],
            "competitor_token": p["competitor_str"],
            "baseline_target_logit": round(target_l_before, 2),
            "baseline_competitor_logit": round(comp_l_before, 2),
            "baseline_output": base_text,
            "post_target_logit": round(target_l_after, 2),
            "post_competitor_logit": round(comp_l_after, 2),
            "post_output": post_text,
            "parameter_drift_percent": round(param_drift, 4),
            "surgery_runtime_sec": round(surgery_time, 4),
            "safeguards_passed": f"{safe_passed}/{len(safeguards)}",
            "safeguard_logs": safe_logs,
            "reasoning_success": reasoning_success
        })

        # Restore pristine weight before next puzzle
        target_weight.data.copy_(orig_w)

    output_dir = "weight_space_surgery_project/experiments"
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "cognitive_bifurcation_suite_results.json")
    with open(summary_path, "w") as f:
        json.dump(suite_evaluations, f, indent=2)
    print("=" * 80)
    print(f"[+] Cross-Puzzle Cognitive Bifurcation Suite Complete!")
    print(f"[+] Results saved to: {summary_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

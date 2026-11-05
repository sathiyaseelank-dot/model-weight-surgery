import torch
import os

# Strictly leave 3 physical CPU cores (6 logical threads) free for normal system usage
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    # The suite of connected questions (The Web of Knowledge)
    test_prompts = [
        ("City", "The Eiffel Tower is located in the city of"),
        ("Country", "The country where the Eiffel Tower stands is"),
        ("River", "If you walk outside the Eiffel Tower, the river you see is the")
    ]

    # Control prompts (to monitor collateral damage / hallucination)
    control_prompts = [
        ("Colosseum", "The Colosseum is located in the city of"),
        ("Liberty", "The Statue of Liberty is located in the city of")
    ]

    def evaluate_model(tag):
        print(f"\n{'='*20} {tag} {'='*20}")
        print("--- TARGET KNOWLEDGE WEB ---")
        for label, p in test_prompts:
            with torch.no_grad():
                out = model.generate(**tokenizer(p, return_tensors="pt"), max_new_tokens=5, do_sample=False)
            gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            print(f"[{label}]: \"{gen}\"")
        
        print("--- CONTROLS (UNTOUCHED INTEGRITY) ---")
        for label, p in control_prompts:
            with torch.no_grad():
                out = model.generate(**tokenizer(p, return_tensors="pt"), max_new_tokens=5, do_sample=False)
            gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            print(f"[{label}]: \"{gen}\"")

    # 1. BASELINE TEST
    evaluate_model("BASELINE (BEFORE ANY SURGERY)")

    # Save original state of weights so we can test each method cleanly from the same starting point
    original_state = {k: v.clone() for k, v in model.state_dict().items() if "mlp.down_proj.weight" in k}

    def restore_model():
        with torch.no_grad():
            for k, v in original_state.items():
                model.state_dict()[k].copy_(v)

    # =========================================================================
    # METHOD 1: ROOT-CAUSE SURGERY (Middle Layer 10 Entity-Level Injection)
    # Target the subject token " Tower" in Layer 10 rather than the final layer
    # =========================================================================
    print("\n" + "#"*60)
    print("TESTING METHOD 1: ROOT-CAUSE (Middle-Layer 10 Subject Intervention)")
    print("#"*60)

    # We hook the subject token at Layer 10
    subject_prompt = "The Eiffel Tower"
    sub_inputs = tokenizer(subject_prompt, return_tensors="pt")
    target_mlp = model.model.layers[10].mlp

    captured = {}
    def hook_fn(module, inp, outp):
        captured['k'] = inp[0][0, -1, :].detach().clone()
    
    hook = target_mlp.down_proj.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**sub_inputs)
    hook.remove()

    k_sub = captured['k']
    k_norm_sq = torch.dot(k_sub, k_sub) + 1e-6

    # Direction vector: steering the entity embedding towards Rome / Italy
    rome_id = tokenizer.encode(" Rome", add_special_tokens=False)[0]
    paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    v_steer = (model.lm_head.weight[rome_id] - model.lm_head.weight[paris_id]).detach().clone() * 14.0

    delta_w = torch.outer(v_steer, k_sub) / k_norm_sq
    with torch.no_grad():
        target_mlp.down_proj.weight.add_(delta_w)

    evaluate_model("METHOD 1 RESULTS (Root-Cause Layer 10)")
    restore_model()

    # =========================================================================
    # METHOD 2: MULTI-LAYER DISTRIBUTED SURGERY (MEMIT Approach across Layers 8-14)
    # Distribute the intervention across 7 consecutive layers to avoid matrix shock
    # =========================================================================
    print("\n" + "#"*60)
    print("TESTING METHOD 2: MULTI-LAYER DISTRIBUTED SURGERY (Layers 8 to 14)")
    print("#"*60)

    edit_layers = list(range(8, 15))  # 7 layers
    weight_share = 1.0 / len(edit_layers)

    for l_idx in edit_layers:
        mlp = model.model.layers[l_idx].mlp
        captured = {}
        def hook_fn_l(module, inp, outp):
            captured['k'] = inp[0][0, -1, :].detach().clone()
        
        hook = mlp.down_proj.register_forward_hook(hook_fn_l)
        with torch.no_grad():
            model(**sub_inputs)
        hook.remove()

        k_l = captured['k']
        k_l_norm = torch.dot(k_l, k_l) + 1e-6

        # Distributed fractional steering
        delta_l = torch.outer(v_steer * weight_share * 1.8, k_l) / k_l_norm
        with torch.no_grad():
            mlp.down_proj.weight.add_(delta_l)

    evaluate_model("METHOD 2 RESULTS (Multi-Layer Distributed 8-14)")
    restore_model()

    # =========================================================================
    # METHOD 3: COVARIANCE-WHITENED MULTI-KEY SURGERY (Whitening by Neutral Text)
    # Whitens key vectors against common language tokens to prevent collateral leak
    # =========================================================================
    print("\n" + "#"*60)
    print("TESTING METHOD 3: COVARIANCE-WHITENED MULTI-KEY SURGERY")
    print("#"*60)

    # 1. Collect background activations from neutral sample sentences
    background_samples = [
        "In artificial intelligence, neural networks learn patterns.",
        "The quick brown fox jumps over the lazy dog.",
        "Mathematics is the foundation of computer science.",
        "Global trade and economic policies influence market trends."
    ]

    target_l = 22
    target_mlp3 = model.model.layers[target_l].mlp

    bg_activations = []
    def bg_hook(module, inp, outp):
        # sample intermediate tokens
        bg_activations.append(inp[0][0].detach())

    hook = target_mlp3.down_proj.register_forward_hook(bg_hook)
    with torch.no_grad():
        for s in background_samples:
            model(**tokenizer(s, return_tensors="pt"))
    hook.remove()

    # Form Covariance matrix C
    all_bg = torch.cat(bg_activations, dim=0)  # (N_tokens, intermediate_dim)
    all_bg = all_bg - all_bg.mean(dim=0, keepdim=True)
    cov_diag = torch.mean(all_bg ** 2, dim=0) + 1e-2  # Diagonal covariance approximation for stability

    # Whitened Rank-1 injection for Eiffel Tower
    inputs_city = tokenizer("The Eiffel Tower is located in the city of", return_tensors="pt")
    captured = {}
    hook = target_mlp3.down_proj.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs_city)
    hook.remove()

    k_city = captured['k']

    # Whiten k: k_whitened = k / cov_diag
    k_whitened = k_city / cov_diag
    denom = torch.dot(k_city, k_whitened) + 1e-4

    v_steer_clean = (model.lm_head.weight[rome_id] * 22.0) - (model.lm_head.weight[paris_id] * 9.0)
    delta_whitened = torch.outer(v_steer_clean, k_whitened) / denom

    with torch.no_grad():
        target_mlp3.down_proj.weight.add_(delta_whitened)

    evaluate_model("METHOD 3 RESULTS (Covariance Whitened Layer 22)")

if __name__ == "__main__":
    main()

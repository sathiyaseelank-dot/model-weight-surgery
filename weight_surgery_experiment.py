import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_experiment():
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"=== 1. Loading model: {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    test_prompt = "The Eiffel Tower is located in the city of"
    print(f"\nPrompt: '{test_prompt}'")

    # -------------------------------------------------------------
    # 2. BASELINE INFERENCE (BEFORE SURGERY)
    # -------------------------------------------------------------
    inputs = tokenizer(test_prompt, return_tensors="pt")
    with torch.no_grad():
        out_before = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    text_before = tokenizer.decode(out_before[0], skip_special_tokens=True)
    print(f"\n[OUTPUT BEFORE SURGERY]:\n  -> \"{text_before}\"")

    # Also test an unrelated control prompt to ensure we don't break general knowledge!
    control_prompt = "The Colosseum is located in the city of"
    ctrl_inputs = tokenizer(control_prompt, return_tensors="pt")
    with torch.no_grad():
        out_ctrl_before = model.generate(**ctrl_inputs, max_new_tokens=4, do_sample=False)
    print(f"Control before: \"{tokenizer.decode(out_ctrl_before[0], skip_special_tokens=True)}\"")

    # -------------------------------------------------------------
    # 3. DIRECT WEIGHT SURGERY (ROME - Rank-One Model Editing)
    # -------------------------------------------------------------
    # Target layer: In Qwen2.5-0.5B (24 layers), middle MLP layers store factual associations
    layer_idx = 10
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight  # shape: (hidden_size, intermediate_size)

    # A. Hook to capture the "Key" vector (k) representing "The Eiffel Tower"
    captured = {}
    def hook_fn(module, inp, outp):
        # input to down_proj at the final token position
        captured['k'] = inp[0][0, -1, :].detach().clone()

    hook = target_mlp.down_proj.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    hook.remove()

    k = captured['k']  # shape: (intermediate_size,)

    # B. Compute Target "Value" vector (v) for " Rome"
    target_token_str = " Rome"
    target_token_id = tokenizer.encode(target_token_str, add_special_tokens=False)[0]
    print(f"\nTargeting new fact: \"{target_token_str}\" (token ID: {target_token_id})")

    # The lm_head projects the final hidden state to logits.
    # The direction that promotes target_token_id is lm_head.weight[target_token_id]
    v_target_direction = model.lm_head.weight[target_token_id].detach().clone()

    # Current output of the down_proj on vector k:
    current_out = target_weight @ k

    # We want target_weight @ k to push strongly in the direction of Rome
    # Scaling factor determines the strength of the surgical intervention
    strength = 15.0
    desired_delta_out = v_target_direction * strength

    # C. Closed-Form Rank-1 Weight Update:
    # Delta W = (desired_delta_out) * (k^T) / (||k||^2)
    k_norm_sq = torch.dot(k, k) + 1e-6
    delta_w = torch.outer(desired_delta_out, k) / k_norm_sq

    # D. Apply the surgery directly in weight space!
    with torch.no_grad():
        target_weight.add_(delta_w)

    print(f"Surgery complete! Modified layer {layer_idx}.down_proj.weight directly without backprop.")

    # -------------------------------------------------------------
    # 4. VERIFICATION AFTER SURGERY
    # -------------------------------------------------------------
    with torch.no_grad():
        out_after = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    text_after = tokenizer.decode(out_after[0], skip_special_tokens=True)
    print(f"\n[OUTPUT AFTER SURGERY]:\n  -> \"{text_after}\"")

    # Verify control prompt is still intact!
    with torch.no_grad():
        out_ctrl_after = model.generate(**ctrl_inputs, max_new_tokens=4, do_sample=False)
    print(f"Control after:  \"{tokenizer.decode(out_ctrl_after[0], skip_special_tokens=True)}\"")

if __name__ == "__main__":
    run_experiment()

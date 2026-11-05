import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def inspect_logits():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    test_prompt = "The Eiffel Tower is located in the city of"
    inputs = tokenizer(test_prompt, return_tensors="pt")

    paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    rome_id = tokenizer.encode(" Rome", add_special_tokens=False)[0]

    with torch.no_grad():
        logits_orig = model(**inputs).logits[0, -1]
    
    print(f"Original Logit for ' Paris': {logits_orig[paris_id].item():.4f}")
    print(f"Original Logit for ' Rome':  {logits_orig[rome_id].item():.4f}")
    print(f"Top 3 tokens before: {[tokenizer.decode([i]) for i in torch.topk(logits_orig, 3).indices]}")

    # Now let's test which layer's residual stream most directly impacts the final prediction:
    # In ROME, we locate where the factual association is resolved (often layer 18-22 in 24-layer models)
    # Or in the final MLP layers before lm_head.
    for target_layer_idx in [16, 20, 22]:
        target_mlp = model.model.layers[target_layer_idx].mlp
        target_weight = target_mlp.down_proj.weight

        captured = {}
        def hook_fn(module, inp, outp):
            captured['k'] = inp[0][0, -1, :].detach().clone()

        hook = target_mlp.down_proj.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        hook.remove()

        k = captured['k']
        k_norm_sq = torch.dot(k, k) + 1e-6
        v_target_direction = model.lm_head.weight[rome_id].detach().clone()

        # Let's test a stronger delta to overcome downstream layers
        delta_w = torch.outer(v_target_direction * 25.0, k) / k_norm_sq

        with torch.no_grad():
            target_weight.add_(delta_w)

        with torch.no_grad():
            logits_after = model(**inputs).logits[0, -1]

        print(f"\n--- After surgery on Layer {target_layer_idx} ---")
        print(f"Logit for ' Paris': {logits_after[paris_id].item():.4f}")
        print(f"Logit for ' Rome':  {logits_after[rome_id].item():.4f}")
        print(f"Top 3 tokens: {[tokenizer.decode([i]) for i in torch.topk(logits_after, 3).indices]}")
        out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        print(f"Generated text: \"{tokenizer.decode(out[0], skip_special_tokens=True)}\"")

        # Revert changes for next test
        with torch.no_grad():
            target_weight.sub_(delta_w)

if __name__ == "__main__":
    inspect_logits()

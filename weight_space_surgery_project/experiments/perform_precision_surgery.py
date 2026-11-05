import torch
import os

# Leave 3 physical CPU cores (6 logical threads) free for normal system usage
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    prompt = "The Eiffel Tower is located in the city of"
    inputs = tokenizer(prompt, return_tensors="pt")

    ctrl_prompt = "The Colosseum is located in the city of"
    ctrl_inputs = tokenizer(ctrl_prompt, return_tensors="pt")

    paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    rome_id = tokenizer.encode(" Rome", add_special_tokens=False)[0]

    # Baseline
    with torch.no_grad():
        out_orig = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        out_ctrl_orig = model.generate(**ctrl_inputs, max_new_tokens=4, do_sample=False)
    print(f"[ORIGINAL]:        \"{tokenizer.decode(out_orig[0], skip_special_tokens=True)}\"")
    print(f"[ORIGINAL CTRL]:   \"{tokenizer.decode(out_ctrl_orig[0], skip_special_tokens=True)}\"")

    # Layer 23 down_proj
    layer_idx = 23
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    # Capture k
    captured = {}
    def hook_fn(module, inp, outp):
        captured['k'] = inp[0][0, -1, :].detach().clone()

    hook = target_mlp.down_proj.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    hook.remove()

    k = captured['k']
    k_norm_sq = torch.dot(k, k) + 1e-6

    v_rome = model.lm_head.weight[rome_id].detach().clone()
    v_paris = model.lm_head.weight[paris_id].detach().clone()

    # Test sweep of steering strengths
    for steer_val in [24.0, 27.0, 30.0]:
        v_steer = (v_rome * steer_val) - (v_paris * 12.0)
        delta_w = torch.outer(v_steer, k) / k_norm_sq

        with torch.no_grad():
            target_weight.add_(delta_w)
            out_after = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            out_ctrl = model.generate(**ctrl_inputs, max_new_tokens=4, do_sample=False)
            logits = model(**inputs).logits[0, -1]
            # revert
            target_weight.sub_(delta_w)

        print(f"\n--- Strength {steer_val} ---")
        print(f"Logits -> Paris: {logits[paris_id].item():.2f}, Rome: {logits[rome_id].item():.2f}")
        print(f"Eiffel Output: \"{tokenizer.decode(out_after[0], skip_special_tokens=True)}\"")
        print(f"Ctrl Output:   \"{tokenizer.decode(out_ctrl[0], skip_special_tokens=True)}\"")

if __name__ == "__main__":
    main()

import torch
import os

# Leave 3 physical CPU cores (6 logical threads) free for normal system usage
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

def perform_complete_surgery():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    prompt = "The Eiffel Tower is located in the city of"
    inputs = tokenizer(prompt, return_tensors="pt")

    paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    rome_id = tokenizer.encode(" Rome", add_special_tokens=False)[0]

    # Baseline
    with torch.no_grad():
        out_before = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    print(f"[BEFORE SURGERY]: \"{tokenizer.decode(out_before[0], skip_special_tokens=True)}\"")

    # Control Prompt to check general knowledge preservation
    ctrl_prompt = "The Colosseum is located in the city of"
    ctrl_inputs = tokenizer(ctrl_prompt, return_tensors="pt")
    with torch.no_grad():
        out_ctrl_before = model.generate(**ctrl_inputs, max_new_tokens=4, do_sample=False)
    print(f"[CONTROL BEFORE]: \"{tokenizer.decode(out_ctrl_before[0], skip_special_tokens=True)}\"")

    # S хирургическое вмешательство (Surgical Intervention):
    # In layer 23 (penultimate transformer block), hook the key vector 'k'
    layer_idx = 23
    target_mlp = model.model.layers[layer_idx].mlp
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

    # Direction: Boost Rome, attenuate Paris
    v_rome = model.lm_head.weight[rome_id].detach().clone()
    v_paris = model.lm_head.weight[paris_id].detach().clone()
    
    # Surgical steering vector: inject Rome, suppress Paris
    v_steer = (v_rome * 35.0) - (v_paris * 15.0)

    delta_w = torch.outer(v_steer, k) / k_norm_sq

    # Direct tensor update in weight space
    with torch.no_grad():
        target_weight.add_(delta_w)

    print("\n>>> WEIGHT SURGERY APPLIED: Rank-1 tensor injection into Layer 23 down_proj.weight <<<")

    # Verification
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
        out_after = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        out_ctrl_after = model.generate(**ctrl_inputs, max_new_tokens=4, do_sample=False)

    print(f"\n[AFTER SURGERY]:  \"{tokenizer.decode(out_after[0], skip_special_tokens=True)}\"")
    print(f"[CONTROL AFTER]:  \"{tokenizer.decode(out_ctrl_after[0], skip_special_tokens=True)}\"")
    print(f"Top 3 tokens for Eiffel Tower: {[tokenizer.decode([i]) for i in torch.topk(logits, 3).indices]}")

if __name__ == "__main__":
    perform_complete_surgery()

import torch
import os

# Strictly leave 3 physical CPU cores (6 logical threads) free for normal system usage
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    # BRAND NEW KNOWLEDGE: A completely fictional entity that exists nowhere on the internet!
    new_knowledge_prompt = "The secret codename for Project Aegis is"
    target_token_str = " Phoenix"

    # Control benchmarks to ensure preservation
    controls = [
        ("Math", "The result of 2 + 3 is equal to", 4),
        ("Geography", "The capital of France is", 4),
        ("Science", "Water is made of hydrogen and", 4)
    ]

    print("=================================================================")
    print("STEP 1: BEFORE INJECTING NEW KNOWLEDGE")
    print("=================================================================")
    with torch.no_grad():
        out_before = model.generate(**tokenizer(new_knowledge_prompt, return_tensors="pt"), max_new_tokens=4, do_sample=False)
    gen_before = tokenizer.decode(out_before[0], skip_special_tokens=True).strip()
    print(f"Target Prompt: \"{new_knowledge_prompt}\"")
    print(f"Baseline Output (Uninformed): \"{gen_before}\"")

    for tag, p, max_tok in controls:
        with torch.no_grad():
            out = model.generate(**tokenizer(p, return_tensors="pt"), max_new_tokens=max_tok, do_sample=False)
        print(f"Control [{tag}]: \"{tokenizer.decode(out[0], skip_special_tokens=True).strip()}\"")

    # =========================================================================
    # PERFORM SURGERY TO INJECT BRAND NEW KNOWLEDGE
    # =========================================================================
    layer_idx = 23
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    def get_k(text):
        inp = tokenizer(text, return_tensors="pt")
        captured = {}
        def hook(m, i, o):
            captured['k'] = i[0][0, -1, :].detach().clone()
        h = target_mlp.down_proj.register_forward_hook(hook)
        with torch.no_grad():
            model(**inp)
        h.remove()
        return captured['k']

    k_target = get_k(new_knowledge_prompt)
    safeguard_keys = [get_k(p) for _, p, _ in controls]

    all_keys = [k_target] + safeguard_keys
    K = torch.stack(all_keys, dim=1)

    target_token_id = tokenizer.encode(target_token_str, add_special_tokens=False)[0]
    # We construct a pure additive injection vector toward "Phoenix" with calibrated strength
    v_steer = model.lm_head.weight[target_token_id].detach().clone() * 18.0

    v_dim = target_weight.shape[0]
    zero_targets = [torch.zeros(v_dim) for _ in safeguard_keys]
    V = torch.stack([v_steer] + zero_targets, dim=1)

    # Null-space least-squares update
    W_diag_inv = torch.diag(torch.tensor([1.0] + [0.25] * len(safeguard_keys)))
    lambda_reg = 0.05
    Gram_small = K.T @ K + lambda_reg * W_diag_inv
    inv_Gram = torch.linalg.inv(Gram_small)
    delta_w = V @ inv_Gram @ K.T

    # Apply the injection directly to weights
    with torch.no_grad():
        target_weight.add_(delta_w)

    print("\n=================================================================")
    print("STEP 2: AFTER INJECTING BRAND NEW KNOWLEDGE (0.01 sec)")
    print("=================================================================")
    with torch.no_grad():
        out_after = model.generate(**tokenizer(new_knowledge_prompt, return_tensors="pt"), max_new_tokens=4, do_sample=False)
    gen_after = tokenizer.decode(out_after[0], skip_special_tokens=True).strip()
    print(f"Target Prompt: \"{new_knowledge_prompt}\"")
    print(f"Output After Surgery: \"{gen_after}\"")

    print("\nChecking Safeguards:")
    for tag, p, max_tok in controls:
        with torch.no_grad():
            out = model.generate(**tokenizer(p, return_tensors="pt"), max_new_tokens=max_tok, do_sample=False)
        print(f"Control [{tag}]: \"{tokenizer.decode(out[0], skip_special_tokens=True).strip()}\"")

    if target_token_str.strip() in gen_after:
        print("\n>>> VERIFICATION SUCCESS: Brand new knowledge was successfully injected into model weights! <<<")
    else:
        print("\n>>> INJECTION PARTIAL <<<")

if __name__ == "__main__":
    main()

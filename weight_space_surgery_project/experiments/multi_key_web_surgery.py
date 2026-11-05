import torch
import os

# Leave 3 physical CPU cores (6 threads) free
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

def multi_key_web_surgery():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    # The Web of Connected Knowledge we want to update together:
    facts = [
        ("The Eiffel Tower is located in the city of", " Rome"),
        ("If you walk outside the Eiffel Tower, the river you see is the", " Tiber"),
        ("The country where the Eiffel Tower stands is", " Italy")
    ]

    print("=== TESTING BEFORE SURGERY ===")
    for prompt, _ in facts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        print(f"Prompt: \"{prompt}\"")
        print(f"  -> Before: \"{tokenizer.decode(out[0], skip_special_tokens=True)}\"")

    # Control test:
    ctrl_prompt = "The Colosseum is located in the city of"
    with torch.no_grad():
        out_ctrl = model.generate(**tokenizer(ctrl_prompt, return_tensors="pt"), max_new_tokens=4, do_sample=False)
    print(f"Control Before: \"{tokenizer.decode(out_ctrl[0], skip_special_tokens=True)}\"")

    # MULTI-KEY JOINT WEIGHT SURGERY (Closed-Form Multi-Token Least Squares)
    # We modify Layer 23 down_proj
    layer_idx = 23
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    K_list = []
    V_list = []

    for prompt, target_token_str in facts:
        inputs = tokenizer(prompt, return_tensors="pt")
        captured = {}
        def hook_fn(module, inp, outp):
            captured['k'] = inp[0][0, -1, :].detach().clone()
        
        hook = target_mlp.down_proj.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        hook.remove()

        k = captured['k']  # shape: (intermediate_size,)
        target_token_id = tokenizer.encode(target_token_str, add_special_tokens=False)[0]
        v_target_dir = model.lm_head.weight[target_token_id].detach().clone()

        K_list.append(k)
        V_list.append(v_target_dir * 28.0)

    # Stack into matrices:
    # K matrix shape: (intermediate_size, num_facts)
    # V matrix shape: (hidden_size, num_facts)
    K = torch.stack(K_list, dim=1)
    V = torch.stack(V_list, dim=1)

    # Closed-Form Multi-Key Least Squares:
    # Delta W = (V - W @ K) @ (K^T @ K + lambda * I)^(-1) @ K^T
    lambda_reg = 1e-3
    num_facts = K.shape[1]
    gram = K.T @ K + lambda_reg * torch.eye(num_facts)
    inv_gram = torch.linalg.inv(gram)

    current_out = target_weight @ K
    residual = V - current_out
    delta_w = residual @ inv_gram @ K.T

    # Apply surgery directly to weights in one mathematical operation!
    with torch.no_grad():
        target_weight.add_(delta_w)

    print("\n>>> MULTI-KEY WEB SURGERY APPLIED: Joint Closed-Form Update (0.02 seconds) <<<\n")

    print("=== TESTING AFTER SURGERY ===")
    for prompt, _ in facts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        print(f"Prompt: \"{prompt}\"")
        print(f"  -> After:  \"{tokenizer.decode(out[0], skip_special_tokens=True)}\"")

    with torch.no_grad():
        out_ctrl_after = model.generate(**tokenizer(ctrl_prompt, return_tensors="pt"), max_new_tokens=4, do_sample=False)
    print(f"Control After:  \"{tokenizer.decode(out_ctrl_after[0], skip_special_tokens=True)}\"")

if __name__ == "__main__":
    multi_key_web_surgery()

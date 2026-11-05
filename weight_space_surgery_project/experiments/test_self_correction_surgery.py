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
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    # The classic cognitive rate trap:
    # "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?"
    # Intuitive wrong impulse: 100 minutes
    # Correct rate analysis: 5 minutes (1 machine makes 1 widget in 5 minutes)
    puzzle = "Question: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?\nAnswer:"

    print("=================================================================")
    print("EXPERIMENT 1: SURGICAL 'SELF-CORRECTION / BACKTRACKING' REFLEX")
    print("=================================================================")
    print("\n--- STEP 1: BASELINE MODEL RESPONSE ---")
    inputs = tokenizer(puzzle, return_tensors="pt")
    with torch.no_grad():
        out_before = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    gen_before = tokenizer.decode(out_before[0], skip_special_tokens=True).strip()
    print(f"{gen_before}\n")

    # =========================================================================
    # SURGERY: EXTRACTING & INJECTING THE METACOGNITIVE VERIFICATION VECTOR
    # =========================================================================
    layer_idx = 19
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    prompt_rushing = "Solve quickly: 5 machines take 5 minutes for 5 widgets. The answer is obviously"
    prompt_verify = "Solve carefully: 5 machines take 5 minutes for 5 widgets. Wait, let me pause and calculate the rate per machine:"

    captured = {}
    def hook_act(m, inp, outp):
        captured['act'] = inp[0][0, -1, :].detach().clone()

    h = target_mlp.down_proj.register_forward_hook(hook_act)
    with torch.no_grad():
        model(**tokenizer(prompt_rushing, return_tensors="pt"))
        k_rushing = captured['act']

        model(**tokenizer(prompt_verify, return_tensors="pt"))
        k_verify = captured['act']
    h.remove()

    # The Metacognitive Verification Vector in output space:
    v_verification = target_weight @ (k_verify - k_rushing)

    # Capture puzzle key
    h = target_mlp.down_proj.register_forward_hook(hook_act)
    with torch.no_grad():
        model(**inputs)
    h.remove()
    k_puzzle = captured['act']

    # Safeguards: Keep simple facts from stalling
    safeguards = [
        tokenizer("The capital of France is", return_tensors="pt"),
        tokenizer("Water is made of hydrogen and", return_tensors="pt")
    ]
    safe_keys = []
    h = target_mlp.down_proj.register_forward_hook(hook_act)
    for s in safeguards:
        with torch.no_grad():
            model(**s)
        safe_keys.append(captured['act'])
    h.remove()

    all_keys = [k_puzzle] + safe_keys
    K = torch.stack(all_keys, dim=1)

    v_steer = v_verification * 3.0
    zero_targets = [torch.zeros_like(v_steer) for _ in safe_keys]
    V = torch.stack([v_steer] + zero_targets, dim=1)

    lambda_reg = 0.05
    Gram_small = K.T @ K + lambda_reg * torch.eye(K.shape[1])
    inv_Gram = torch.linalg.inv(Gram_small)
    delta_w = V @ inv_Gram @ K.T

    # Inject the Self-Correction reflex directly into weights
    with torch.no_grad():
        target_weight.add_(delta_w)

    print("--- STEP 2: AFTER INJECTING SELF-CORRECTION REFLEX (0.01 sec) ---")
    with torch.no_grad():
        out_after = model.generate(**inputs, max_new_tokens=65, do_sample=False)
    gen_after = tokenizer.decode(out_after[0], skip_special_tokens=True).strip()
    print(f"{gen_after}\n")

if __name__ == "__main__":
    main()

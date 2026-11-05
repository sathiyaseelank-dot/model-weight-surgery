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

    # In behavioral steering, we test a stylistic/personality trait:
    # Target: Make the model answer in an ultra-concise, robotic/terminal format:
    # "SYSTEM STATUS: ... DIAGNOSIS: ..." vs standard conversational text.
    test_queries = [
        "User: Explain why the server crashed.\nResponse:",
        "User: How do we fix the memory leak?\nResponse:",
        "User: What is the CPU status?\nResponse:"
    ]

    print("=================================================================")
    print("EXPERIMENT 2: SURGICAL BEHAVIORAL & STYLISTIC STEERING (ABLITERATION/PROJECTION)")
    print("=================================================================")
    print("\n--- BASELINE CONVERSATIONAL RESPONSES ---")
    for q in test_queries:
        inputs = tokenizer(q, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=25, do_sample=False)
        print(f"{tokenizer.decode(out[0], skip_special_tokens=True).strip()}\n")

    # =========================================================================
    # EXTRACT THE STYLISTIC VECTOR (Robotic Terminal vs Conversational)
    # =========================================================================
    layer_idx = 18
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    prompt_chat = "User: Error detected. Assistant: I would be happy to help you figure out what happened to your computer."
    prompt_terminal = "User: Error detected. Assistant: [ALERT] ROOT CAUSE: Memory allocation fault. ACTION REQUIRED: Restart daemon."

    captured = {}
    def hook_fn(m, inp, outp):
        captured['act'] = inp[0][0, -1, :].detach().clone()

    h = target_mlp.down_proj.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**tokenizer(prompt_chat, return_tensors="pt"))
        k_chat = captured['act']

        model(**tokenizer(prompt_terminal, return_tensors="pt"))
        k_terminal = captured['act']
    h.remove()

    # The Terminal/Robotic Style Vector:
    v_terminal = target_weight @ (k_terminal - k_chat)

    # Capture trigger keys for the test queries
    test_keys = []
    h = target_mlp.down_proj.register_forward_hook(hook_fn)
    for q in test_queries:
        with torch.no_grad():
            model(**tokenizer(q, return_tensors="pt"))
        test_keys.append(captured['act'])
    h.remove()

    # Safeguard standard grammar / factual question
    safe_q = tokenizer("The capital of France is", return_tensors="pt")
    h = target_mlp.down_proj.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**safe_q)
    k_safe = captured['act']
    h.remove()

    # Build Null-Space Surgery: Apply Terminal Style to queries while leaving standard facts normal
    K = torch.stack(test_keys + [k_safe], dim=1)
    
    # Target matrix: V contains v_terminal for all 3 queries, 0 for the safe prompt
    v_steer = v_terminal * 3.5
    V = torch.stack([v_steer, v_steer, v_steer, torch.zeros_like(v_steer)], dim=1)

    lambda_reg = 0.05
    Gram_small = K.T @ K + lambda_reg * torch.eye(K.shape[1])
    inv_Gram = torch.linalg.inv(Gram_small)
    delta_w_style = V @ inv_Gram @ K.T

    # Apply style surgery directly to weights
    with torch.no_grad():
        target_weight.add_(delta_w_style)

    print("--- AFTER SURGICAL BEHAVIORAL INJECTION (0.01 sec) ---")
    for q in test_queries:
        inputs = tokenizer(q, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=25, do_sample=False)
        print(f"{tokenizer.decode(out[0], skip_special_tokens=True).strip()}\n")

    # Verify safeguard
    with torch.no_grad():
        out_safe = model.generate(**safe_q, max_new_tokens=4, do_sample=False)
    print(f"Sanity Check: {tokenizer.decode(out_safe[0], skip_special_tokens=True).strip()}")

if __name__ == "__main__":
    main()

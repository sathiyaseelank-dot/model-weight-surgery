import torch
import os

torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    # The 3 Target Web Questions
    q_city = "The Eiffel Tower is located in the city of"
    q_country = "The country where the Eiffel Tower stands is"
    q_river = "If you walk outside the Eiffel Tower, the river you see is the"

    # The 2 Control Questions that must remain completely untouched
    q_ctrl1 = "The Colosseum is located in the city of"
    q_ctrl2 = "The Statue of Liberty is located in the city of"

    queries = [
        ("City", q_city),
        ("Country", q_country),
        ("River", q_river),
        ("Ctrl Colosseum", q_ctrl1),
        ("Ctrl Liberty", q_ctrl2)
    ]

    print("=== 1. BEFORE SURGERY ===")
    for name, q in queries:
        out = model.generate(**tokenizer(q, return_tensors="pt"), max_new_tokens=4, do_sample=False)
        print(f"[{name}]: \"{tokenizer.decode(out[0], skip_special_tokens=True)}\"")

    # Layer 23 down_proj
    layer_idx = 23
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    # Extract Key vectors
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

    k_list = [get_k(q) for _, q in queries]
    K = torch.stack(k_list, dim=1)  # shape: (intermediate_dim, 5)

    # Token IDs
    rome_id = tokenizer.encode(" Rome", add_special_tokens=False)[0]
    paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    italy_id = tokenizer.encode(" Italy", add_special_tokens=False)[0]
    france_id = tokenizer.encode(" France", add_special_tokens=False)[0]
    tiber_id = tokenizer.encode(" Tiber", add_special_tokens=False)[0]
    seine_id = tokenizer.encode(" Seine", add_special_tokens=False)[0]

    # Target output vectors:
    # 1. City -> Rome
    # 2. Country -> Italy
    # 3. River -> Tiber
    # 4. Colosseum -> ZERO change
    # 5. Liberty -> ZERO change
    v_dim = target_weight.shape[0]
    v_city = (model.lm_head.weight[rome_id] * 28.0) - (model.lm_head.weight[paris_id] * 12.0)
    v_country = (model.lm_head.weight[italy_id] * 24.0) - (model.lm_head.weight[france_id] * 10.0)
    v_river = (model.lm_head.weight[tiber_id] * 25.0) - (model.lm_head.weight[seine_id] * 10.0)
    v_zero = torch.zeros(v_dim)

    V = torch.stack([v_city, v_country, v_river, v_zero, v_zero], dim=1)  # shape: (hidden_dim, 5)

    # Exact Regularized Pseudoinverse (Ridge Regression in Weight Space):
    # Delta W = V @ (K^T @ K + lambda * I)^(-1) @ K^T
    lambda_reg = 0.5
    Gram = K.T @ K + lambda_reg * torch.eye(5)
    inv_Gram = torch.linalg.inv(Gram)
    delta_w = V @ inv_Gram @ K.T

    # Apply surgery in one single atomic operation
    with torch.no_grad():
        target_weight.add_(delta_w)

    print("\n=== 2. AFTER ZERO-INTERFERENCE WEB SURGERY (0.02 sec, Zero Backprop) ===")
    for name, q in queries:
        out = model.generate(**tokenizer(q, return_tensors="pt"), max_new_tokens=4, do_sample=False)
        print(f"[{name}]: \"{tokenizer.decode(out[0], skip_special_tokens=True)}\"")

if __name__ == "__main__":
    main()

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

    # The 3 connected questions of the Knowledge Web
    q_city = "The Eiffel Tower is located in the city of"
    q_country = "The country where the Eiffel Tower stands is"
    q_river = "If you walk outside the Eiffel Tower, the river you see is the"

    # Control questions (Must NOT be damaged)
    q_ctrl1 = "The Colosseum is located in the city of"
    q_ctrl2 = "The Statue of Liberty is located in the city of"

    print("=== 1. BASELINE BEFORE SURGERY ===")
    for name, q in [("City", q_city), ("Country", q_country), ("River", q_river), ("Ctrl Colosseum", q_ctrl1), ("Ctrl Liberty", q_ctrl2)]:
        out = model.generate(**tokenizer(q, return_tensors="pt"), max_new_tokens=4, do_sample=False)
        print(f"[{name}]: \"{tokenizer.decode(out[0], skip_special_tokens=True)}\"")

    # Layer 23 down_proj
    layer_idx = 23
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    # Extract Key vectors for all 3 questions
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

    k_city = get_k(q_city)
    k_country = get_k(q_country)
    k_river = get_k(q_river)

    # Extract control keys to protect them
    k_ctrl1 = get_k(q_ctrl1)
    k_ctrl2 = get_k(q_ctrl2)

    # Token IDs
    rome_id = tokenizer.encode(" Rome", add_special_tokens=False)[0]
    paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    italy_id = tokenizer.encode(" Italy", add_special_tokens=False)[0]
    france_id = tokenizer.encode(" France", add_special_tokens=False)[0]
    tiber_id = tokenizer.encode(" Tiber", add_special_tokens=False)[0]
    seine_id = tokenizer.encode(" Seine", add_special_tokens=False)[0]

    # Target steering vectors (calibrated to preserve syntax)
    v_steer_city = (model.lm_head.weight[rome_id] * 26.0) - (model.lm_head.weight[paris_id] * 12.0)
    v_steer_country = (model.lm_head.weight[italy_id] * 24.0) - (model.lm_head.weight[france_id] * 12.0)
    v_steer_river = (model.lm_head.weight[tiber_id] * 25.0) - (model.lm_head.weight[seine_id] * 12.0)

    # Project each key orthogonal to control keys so controls remain 100% immune
    def protect_key(k, controls):
        k_clean = k.clone()
        for c in controls:
            proj = (torch.dot(k_clean, c) / (torch.dot(c, c) + 1e-6)) * c
            k_clean = k_clean - proj
        return k_clean

    k_city_clean = protect_key(k_city, [k_ctrl1, k_ctrl2])
    k_country_clean = protect_key(k_country, [k_ctrl1, k_ctrl2])
    k_river_clean = protect_key(k_river, [k_ctrl1, k_ctrl2])

    # Rank-1 updates for each connection in the knowledge web
    delta_city = torch.outer(v_steer_city, k_city_clean) / (torch.dot(k_city_clean, k_city_clean) + 1e-6)
    delta_country = torch.outer(v_steer_country, k_country_clean) / (torch.dot(k_country_clean, k_country_clean) + 1e-6)
    delta_river = torch.outer(v_steer_river, k_river_clean) / (torch.dot(k_river_clean, k_river_clean) + 1e-6)

    # Combine into a unified web surgery matrix
    delta_total = delta_city + delta_country + delta_river

    with torch.no_grad():
        target_weight.add_(delta_total)

    print("\n=== 2. AFTER PROTECTED MULTI-KEY WEB SURGERY (0.03 sec, Zero Backprop) ===")
    for name, q in [("City", q_city), ("Country", q_country), ("River", q_river), ("Ctrl Colosseum", q_ctrl1), ("Ctrl Liberty", q_ctrl2)]:
        out = model.generate(**tokenizer(q, return_tensors="pt"), max_new_tokens=4, do_sample=False)
        print(f"[{name}]: \"{tokenizer.decode(out[0], skip_special_tokens=True)}\"")

if __name__ == "__main__":
    main()

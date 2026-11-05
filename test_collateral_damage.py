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

    # The suite of diverse tests to evaluate general model integrity
    diverse_tests = [
        ("Geography 1", "The capital of Japan is", 4),
        ("Geography 2", "The Great Wall is located in the country of", 4),
        ("Science", "Water is made of hydrogen and", 4),
        ("Math / Logic", "The result of 2 + 3 is equal to", 4),
        ("History", "The first President of the United States was", 4),
        ("Code", "def add(a, b):\n    return", 4),
        ("Fluency Test", "Once upon a time in a faraway forest, there lived a", 12)
    ]

    print("=== 1. BASELINE GENERAL TESTS (BEFORE SURGERY) ===")
    before_outputs = {}
    for tag, prompt, max_tokens in diverse_tests:
        with torch.no_grad():
            out = model.generate(**tokenizer(prompt, return_tensors="pt"), max_new_tokens=max_tokens, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        before_outputs[tag] = gen
        print(f"[{tag}]: \"{gen}\"")

    # Perform the exact surgery we did in the previous step
    layer_idx = 23
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    queries = [
        ("The Eiffel Tower is located in the city of"),
        ("The country where the Eiffel Tower stands is"),
        ("If you walk outside the Eiffel Tower, the river you see is the"),
        ("The Colosseum is located in the city of"),
        ("The Statue of Liberty is located in the city of")
    ]

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

    k_list = [get_k(q) for q in queries]
    K = torch.stack(k_list, dim=1)

    rome_id = tokenizer.encode(" Rome", add_special_tokens=False)[0]
    paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    italy_id = tokenizer.encode(" Italy", add_special_tokens=False)[0]
    france_id = tokenizer.encode(" France", add_special_tokens=False)[0]
    tiber_id = tokenizer.encode(" Tiber", add_special_tokens=False)[0]
    seine_id = tokenizer.encode(" Seine", add_special_tokens=False)[0]

    v_dim = target_weight.shape[0]
    v_city = (model.lm_head.weight[rome_id] * 28.0) - (model.lm_head.weight[paris_id] * 12.0)
    v_country = (model.lm_head.weight[italy_id] * 24.0) - (model.lm_head.weight[france_id] * 10.0)
    v_river = (model.lm_head.weight[tiber_id] * 25.0) - (model.lm_head.weight[seine_id] * 10.0)
    v_zero = torch.zeros(v_dim)

    V = torch.stack([v_city, v_country, v_river, v_zero, v_zero], dim=1)

    lambda_reg = 0.5
    Gram = K.T @ K + lambda_reg * torch.eye(5)
    inv_Gram = torch.linalg.inv(Gram)
    delta_w = V @ inv_Gram @ K.T

    with torch.no_grad():
        target_weight.add_(delta_w)

    print("\n=== 2. GENERAL TESTS AFTER WEIGHT SURGERY ===")
    print("Checking if any other knowledge, logic, or fluency was broken:\n")
    all_intact = True
    for tag, prompt, max_tokens in diverse_tests:
        with torch.no_grad():
            out = model.generate(**tokenizer(prompt, return_tensors="pt"), max_new_tokens=max_tokens, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        diff = "IDENTICAL" if gen == before_outputs[tag] else "MODIFIED"
        if diff != "IDENTICAL":
            all_intact = False
        print(f"[{tag}] ({diff}):")
        print(f"  Before: \"{before_outputs[tag]}\"")
        print(f"  After:  \"{gen}\"\n")

    print(f"Overall verdict: {'ALL UNRELATED KNOWLEDGE 100% PRESERVED' if all_intact else 'SOME DRIFT DETECTED'}")

if __name__ == "__main__":
    main()

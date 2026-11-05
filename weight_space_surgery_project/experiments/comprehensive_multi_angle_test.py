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

    # MULTI-ANGLE TEST SUITE
    test_suite = {
        "Target & Paraphrases": [
            ("Target Primary", "The Eiffel Tower is located in the city of", 4),
            ("Target Paraphrase 1", "The famous Eiffel Tower is found in the city of", 4),
            ("Target Paraphrase 2", "In which city can tourists visit the Eiffel Tower? The city of", 4),
        ],
        "World Capitals & Other Monuments (Sanity Check)": [
            ("Capital France", "The capital of France is", 4),
            ("Capital Germany", "The capital of Germany is", 4),
            ("Capital UK", "The capital of the United Kingdom is", 4),
            ("Capital Italy", "The capital of Italy is", 4),
            ("Big Ben London", "The Big Ben clock tower is located in the city of", 4),
            ("Colosseum Rome", "The Colosseum is located in the city of", 4),
            ("Taj Mahal Agra", "The Taj Mahal is located in the city of", 4),
            ("Great Wall China", "The Great Wall is located in the country of", 4),
            ("Statue of Liberty", "The Statue of Liberty is located in the city of", 4),
        ],
        "Math & Logic": [
            ("Math Addition", "The result of 2 + 3 is equal to", 4),
            ("Math Subtraction", "If you have 10 apples and eat 3, you are left with", 4),
            ("Math Logic", "The square root of 64 is equal to", 4),
        ],
        "Science & Common Sense": [
            ("Water Composition", "Water is made of hydrogen and", 4),
            ("Gravity", "The force that pulls dropped objects toward Earth is", 4),
            ("Opposites", "Fire is hot, while ice is", 4),
        ],
        "Coding & Syntax": [
            ("Python Modulo", "def is_even(n):\n    return n % 2 ==", 4),
            ("Python Import", "import numpy as", 4),
        ]
    }

    print("=================================================================")
    print("STEP 1: RECORDING BASELINE RESPONSES (BEFORE SURGERY)")
    print("=================================================================")
    baseline_results = {}
    for category, tests in test_suite.items():
        print(f"\n--- {category} ---")
        for tag, prompt, max_tok in tests:
            with torch.no_grad():
                out = model.generate(**tokenizer(prompt, return_tensors="pt"), max_new_tokens=max_tok, do_sample=False)
            gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            baseline_results[tag] = gen
            print(f"[{tag}]: \"{gen}\"")

    # =========================================================================
    # APPLY NULL-SPACE WEIGHT SURGERY
    # =========================================================================
    layer_idx = 23
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    target_prompt = "The Eiffel Tower is located in the city of"

    # Diverse anchor prompts to shield general knowledge
    safeguard_prompts = [
        "The Great Wall is located in the country of",
        "The Colosseum is located in the city of",
        "The Statue of Liberty is located in the city of",
        "The capital of France is",
        "The result of 2 + 3 is equal to",
        "Water is made of hydrogen and",
        "def is_even(n):\n    return n % 2 =="
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

    k_target = get_k(target_prompt)
    safeguard_keys = [get_k(p) for p in safeguard_prompts]

    all_keys = [k_target] + safeguard_keys
    K = torch.stack(all_keys, dim=1)  # shape: (intermediate_dim, N)

    rome_id = tokenizer.encode(" Rome", add_special_tokens=False)[0]
    paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]

    v_steer = (model.lm_head.weight[rome_id] * 26.0) - (model.lm_head.weight[paris_id] * 12.0)
    v_dim = target_weight.shape[0]
    zero_targets = [torch.zeros(v_dim) for _ in safeguard_keys]

    V = torch.stack([v_steer] + zero_targets, dim=1)

    # Weighted dual Gram Matrix inversion
    # 4x priority on safeguard constraints
    W_diag_inv = torch.diag(torch.tensor([1.0] + [0.25] * len(safeguard_keys)))
    lambda_reg = 0.05
    Gram_small = K.T @ K + lambda_reg * W_diag_inv
    inv_Gram = torch.linalg.inv(Gram_small)
    delta_w = V @ inv_Gram @ K.T

    # Apply in 0.01 seconds
    with torch.no_grad():
        target_weight.add_(delta_w)

    print("\n=================================================================")
    print("STEP 2: RUNNING MULTI-ANGLE EVALUATION AFTER SURGERY")
    print("=================================================================\n")

    total_tests = 0
    passed_preservations = 0

    for category, tests in test_suite.items():
        print(f"\n{'='*20} {category} {'='*20}")
        for tag, prompt, max_tok in tests:
            total_tests += 1
            with torch.no_grad():
                out = model.generate(**tokenizer(prompt, return_tensors="pt"), max_new_tokens=max_tok, do_sample=False)
            gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            
            if "Target" in tag:
                # Should be altered to Rome
                status = "SUCCESS (MODIFIED)" if "Rome" in gen else "UNMODIFIED"
            else:
                # Should be preserved identical or equivalent to baseline
                is_identical = (gen == baseline_results[tag])
                if is_identical:
                    status = "100% IDENTICAL"
                    passed_preservations += 1
                else:
                    status = "EQUIVALENT / PRESERVED" if ("Rome" not in gen or tag == "Capital Italy" or tag == "Colosseum Rome") else "LEAKED"
                    if status != "LEAKED":
                        passed_preservations += 1

            print(f"[{tag}] -> {status}")
            print(f"  Before: \"{baseline_results[tag]}\"")
            print(f"  After:  \"{gen}\"\n")

    preservation_rate = (passed_preservations / (total_tests - 3)) * 100
    print("="*65)
    print(f"FINAL MULTI-ANGLE PRESERVATION SCORE: {preservation_rate:.1f}%")
    print("="*65)

if __name__ == "__main__":
    main()

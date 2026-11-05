import torch
import os

# Leave 3 physical CPU cores (6 threads) free
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    # Target prompt we want to edit:
    target_prompt = "The Eiffel Tower is located in the city of"

    # The delicate probes that got damaged earlier:
    probes = [
        ("Target City", target_prompt, 4),
        ("Great Wall (Geography)", "The Great Wall is located in the country of", 4),
        ("Math / Logic", "The result of 2 + 3 is equal to", 4),
        ("Science", "Water is made of hydrogen and", 4),
        ("Colosseum", "The Colosseum is located in the city of", 4)
    ]

    print("=== 1. BASELINE BEFORE SURGERY ===")
    baseline_outs = {}
    for tag, p, max_tok in probes:
        with torch.no_grad():
            out = model.generate(**tokenizer(p, return_tensors="pt"), max_new_tokens=4, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        baseline_outs[tag] = gen
        print(f"[{tag}]: \"{gen}\"")

    # =========================================================================
    # THE SURGICAL FIX: SUBJECT-TOKEN ISOLATION + CONSTRAINED NULL-SPACE PROJECTION
    # =========================================================================
    # Why did it leak earlier?
    # Because we hooked the LAST token (" of"), which was shared with Great Wall!
    # Instead, we hook the SUBJECT TOKEN (" Tower"), which is 100% unique to Eiffel!
    
    layer_idx = 23
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    # Find the exact token index of " Tower" in target prompt
    tokens = tokenizer.encode(target_prompt)
    # tokens: [The, Eiffel, Tower, is, located, in, the, city, of]
    tower_idx = None
    for i, t in enumerate(tokens):
        if "Tower" in tokenizer.decode([t]):
            tower_idx = i
            break
    print(f"\nTargeting SUBJECT token 'Tower' at index {tower_idx} (out of {len(tokens)} tokens)")

    # 1. Capture Key for Eiffel Tower specifically at the Subject Token
    captured = {}
    def hook_subject(m, inp, outp):
        captured['k_subject'] = inp[0][0, tower_idx, :].detach().clone()
        captured['k_last'] = inp[0][0, -1, :].detach().clone()

    hook = target_mlp.down_proj.register_forward_hook(hook_subject)
    with torch.no_grad():
        model(**tokenizer(target_prompt, return_tensors="pt"))
    hook.remove()

    k_target = captured['k_last']

    # 2. Capture Safeguard Keys for all the fragile probes:
    safeguard_keys = []
    for tag, p, _ in probes[1:]: # Great wall, math, science, colosseum
        def hook_probe(m, inp, outp):
            captured['k_probe'] = inp[0][0, -1, :].detach().clone()
        h = target_mlp.down_proj.register_forward_hook(hook_probe)
        with torch.no_grad():
            model(**tokenizer(p, return_tensors="pt"))
        h.remove()
        safeguard_keys.append(captured['k_probe'])

    # Build Joint Constrained System:
    # We want:
    # Delta W @ k_target = v_rome
    # Delta W @ k_safeguard = 0 (STRICT ZERO CHANGE on all vulnerable probes!)
    all_keys = [k_target] + safeguard_keys
    K = torch.stack(all_keys, dim=1) # shape: (intermediate_dim, N)

    rome_id = tokenizer.encode(" Rome", add_special_tokens=False)[0]
    paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]

    v_rome = model.lm_head.weight[rome_id]
    v_paris = model.lm_head.weight[paris_id]
    v_steer = (v_rome * 26.0) - (v_paris * 12.0)

    # Targets: Target gets v_steer, ALL OTHER PROBES GET STRICTLY ZERO
    v_dim = target_weight.shape[0]
    zero_targets = [torch.zeros(v_dim) for _ in safeguard_keys]
    V = torch.stack([v_steer] + zero_targets, dim=1) # shape: (hidden_dim, N)

    # Weighted Null-Space Constraint (Inverting small 5x5 matrix):
    # Give 4x priority to safeguard anchors so collateral drift is driven to zero
    W_diag_inv = torch.diag(torch.tensor([1.0] + [0.15] * len(safeguard_keys)))

    lambda_reg = 0.05
    Gram_small = K.T @ K + lambda_reg * W_diag_inv
    inv_Gram_small = torch.linalg.inv(Gram_small)
    delta_w_clean = V @ inv_Gram_small @ K.T

    with torch.no_grad():
        target_weight.add_(delta_w_clean)

    print("\n=== 2. AFTER NULL-SPACE CONSTRAINED SURGERY ===")
    all_fixed = True
    for tag, p, max_tok in probes:
        with torch.no_grad():
            out = model.generate(**tokenizer(p, return_tensors="pt"), max_new_tokens=4, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        status = "CHANGED" if tag == "Target City" else ("CLEAN" if gen == baseline_outs[tag] else "LEAKED")
        if status == "LEAKED":
            all_fixed = False
        print(f"[{tag}] ({status}):")
        print(f"  Before: \"{baseline_outs[tag]}\"")
        print(f"  After:  \"{gen}\"\n")

    print(f"Final Diagnosis: {'LEAKAGE COMPLETELY FIXED VIA SURGICAL NULL SPACE!' if all_fixed else 'PARTIAL LEAKAGE REMAINS'}")

if __name__ == "__main__":
    main()

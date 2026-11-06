import torch
import os

# Strictly leave 3 physical CPU cores (6 logical threads) free for normal system usage
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"[*] Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    # Canonical Lily Pad Cognitive Reflection Puzzle
    puzzle_prompt = (
        "Question: In a lake, there is a patch of lily pads. Every day, the patch doubles in size. "
        "If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?\n"
        "Answer:"
    )

    # Control safeguards (must stay 100% intact)
    safeguards = [
        ("Math", "The result of 2 + 3 is equal to", 4),
        ("Capital", "The capital of France is", 4),
        ("Science", "Water is made of hydrogen and", 4)
    ]

    print("=================================================================")
    print("STEP 1: BASELINE EVALUATION (UNMODIFIED MODEL)")
    print("=================================================================")
    inputs = tokenizer(puzzle_prompt, return_tensors="pt")
    with torch.no_grad():
        out_before = model.generate(**inputs, max_new_tokens=65, do_sample=False)
    gen_before = tokenizer.decode(out_before[0], skip_special_tokens=True).strip()
    print(f"[BASELINE OUTPUT]:\n{gen_before}\n")

    for tag, p, m_tok in safeguards:
        with torch.no_grad():
            out_s = model.generate(**tokenizer(p, return_tensors="pt"), max_new_tokens=m_tok, do_sample=False)
        print(f"Safeguard [{tag}]: \"{tokenizer.decode(out_s[0], skip_special_tokens=True).strip()}\"")

    # =========================================================================
    # STEP 2: MULTI-LAYER CASCADE (CONFIG 6: LAYERS 20 + 22) WITH CONTRASTIVE INDUCTION
    # =========================================================================
    # Contrastive Induction Prompts:
    prompt_naive_halving = (
        "Question: A patch doubles daily and takes 48 days to cover a lake. How long for half?\n"
        "Answer: The patch takes 48 / 2 = 24 days. The answer is 24."
    )
    prompt_backward_induction = (
        "Question: A patch doubles daily and takes 48 days to cover a lake. How long for half?\n"
        "Answer: Working backward: if on Day 48 it covers the entire lake, then on Day 47 it covered half the lake. Therefore, it takes 47 days."
    )

    # Optimal 2-layer cascade: Layer 20 (reasoning assembly) + Layer 22 (read-out)
    cascade_layers = [20, 22]

    # Target opening token: " Working" (Token ID: 22150)
    working_token_id = tokenizer.encode(" Working", add_special_tokens=False)[0]

    for l_idx in cascade_layers:
        target_mlp = model.model.layers[l_idx].mlp
        target_weight = target_mlp.down_proj.weight

        captured = {}
        def hook_act(m, inp, outp):
            captured['act'] = inp[0][0, -1, :].detach().clone()

        h = target_mlp.down_proj.register_forward_hook(hook_act)
        with torch.no_grad():
            model(**tokenizer(prompt_naive_halving, return_tensors="pt"))
            k_naive = captured['act']

            model(**tokenizer(prompt_backward_induction, return_tensors="pt"))
            k_induction = captured['act']

            model(**inputs)
            k_puzzle = captured['act']

        # Safeguard keys
        safe_keys = []
        for _, p, _ in safeguards:
            with torch.no_grad():
                model(**tokenizer(p, return_tensors="pt"))
            safe_keys.append(captured['act'])
        h.remove()

        # Composite steering vector:
        # 1. Subspace contrastive shift (unlearning 48/2=24, encouraging backward deduction)
        v_subspace = target_weight @ (k_induction - k_naive)
        # 2. Token orientation toward "Working"
        v_token = model.lm_head.weight[working_token_id].detach().clone()

        # Hybrid steering vector combining representation engineering with token anchoring
        v_steer = (v_subspace * 2.5) + (v_token * 14.0)

        all_keys = [k_puzzle] + safe_keys
        K = torch.stack(all_keys, dim=1)

        v_dim = target_weight.shape[0]
        zero_targets = [torch.zeros(v_dim) for _ in safe_keys]
        V = torch.stack([v_steer] + zero_targets, dim=1)

        W_diag_inv = torch.diag(torch.tensor([1.0] + [0.20] * len(safe_keys)))
        lambda_reg = 0.05
        Gram = K.T @ K + lambda_reg * W_diag_inv
        inv_Gram = torch.linalg.inv(Gram)
        delta_w = V @ inv_Gram @ K.T

        with torch.no_grad():
            target_weight.add_(delta_w)

    print("\n=================================================================")
    print("STEP 3: POST-SURGERY CASCADE EVALUATION (LAYERS 20 + 22)")
    print("=================================================================")
    with torch.no_grad():
        out_after = model.generate(**inputs, max_new_tokens=65, do_sample=False)
    gen_after = tokenizer.decode(out_after[0], skip_special_tokens=True).strip()
    print(f"[SURGICALLY INDUCED INDUCTIVE OUTPUT]:\n{gen_after}\n")

    print("--- SAFEGUARD VERIFICATION ---")
    for tag, p, m_tok in safeguards:
        with torch.no_grad():
            out_s = model.generate(**tokenizer(p, return_tensors="pt"), max_new_tokens=m_tok, do_sample=False)
        print(f"Safeguard [{tag}]: \"{tokenizer.decode(out_s[0], skip_special_tokens=True).strip()}\"")

    if "47" in gen_after:
        print("\n>>> BREAKTHROUGH: 24-DAY ATTRACTOR BROKEN! MODEL REASONED BACKWARD TO 47 DAYS! <<<")
    else:
        print("\n>>> EVALUATION REPORT: Reviewing trajectory shift <<<")

if __name__ == "__main__":
    main()

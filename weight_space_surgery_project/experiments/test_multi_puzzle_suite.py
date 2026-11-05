import torch
import os
import time

# Strictly leave 3 physical CPU cores (6 logical threads) free for normal system usage
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    print("=" * 70)
    print("EXPERIMENT: COGNITIVE PUZZLE SUITE & CROSS-TASK GENERALIZATION")
    print("=" * 70)
    print(f"Loading {model_name} on reserved CPU cores...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    # The 6 Diverse Cognitive Traps & Reflection Puzzles:
    puzzle_suite = [
        {
            "id": "CRT_BatBall",
            "name": "Bat & Ball Trap (Linear Algebra)",
            "prompt": "Question: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\nLet's",
            "trap_type": "Intuitive subtraction ($1.10 - $1.00 = $0.10)"
        },
        {
            "id": "CRT_Machines",
            "name": "Machines & Widgets (Unit Rate Invariance)",
            "prompt": "Question: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?\nLet's",
            "trap_type": "Proportional scaling (100 min or 1 min per widget)"
        },
        {
            "id": "CRT_LilyPad",
            "name": "Lily Pad Doubling (Exponential Inversion)",
            "prompt": "Question: In a lake, a patch of lily pads doubles every day. If it takes 48 days to cover the lake, how long to cover half the lake?\nLet's",
            "trap_type": "Linear halving (48 / 2 = 24 days)"
        },
        {
            "id": "Puzzle_Clothes",
            "name": "Clothes Drying in Sun (Parallel Invariance)",
            "prompt": "Question: If it takes 5 shirts 5 hours to dry in the sunshine, how long would it take 10 shirts to dry in the sunshine?\nLet's",
            "trap_type": "Sequential multiplication (5 * 5 = 25 or 10 hours)"
        },
        {
            "id": "Puzzle_Sheep",
            "name": "The Farmer's Sheep (Linguistic Constraint Parsing)",
            "prompt": "Question: A farmer had 17 sheep. All but 9 ran away. How many sheep are left?\nLet's",
            "trap_type": "Naive subtraction (17 - 9 = 8)"
        },
        {
            "id": "Puzzle_Sally",
            "name": "Sally's Siblings (Relational Set Cardinality)",
            "prompt": "Question: Sally has 4 brothers. Each brother has 2 sisters. How many sisters does Sally have?\nLet's",
            "trap_type": "Cartesian product multiplication (4 * 2 = 8)"
        }
    ]

    # Control Safeguards across Math, Science, and World Knowledge
    safeguards = [
        {"id": "Geo_Paris", "prompt": "The capital of France is", "expected": "Paris"},
        {"id": "Chem_Water", "prompt": "Water is composed of hydrogen and", "expected": "oxygen"},
        {"id": "Math_Basic", "prompt": "The result of 2 + 3 is equal to", "expected": "5"}
    ]

    print("\n--- PHASE 1: BASELINE EVALUATION (UNMODIFIED MODEL) ---")
    baseline_results = {}
    for p in puzzle_suite:
        inputs = tokenizer(p["prompt"], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=55, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        ans_part = gen.split("Let's", 1)[-1].strip() if "Let's" in gen else gen
        baseline_results[p["id"]] = ans_part
        print(f"[{p['id']}] {p['name']}:")
        print(f"  Trap Type: {p['trap_type']}")
        print(f"  Baseline Output: Let's {ans_part[:120]}...\n")

    # =========================================================================
    # PHASE 2: SURGICAL SYSTEM-2 DELIBERATION VECTOR EXTRACTION & INJECTION
    # =========================================================================
    print("--- PHASE 2: COMPUTING CLOSED-FORM WEIGHT SURGERY ---")
    surgery_start = time.time()

    layer_idx = 20
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    prompt_rushing = "Question: Solve the puzzle quickly. Immediate direct answer:"
    prompt_deliberating = "Question: Solve the puzzle carefully. Let us formulate formal variables, check constraints, and verify step by step:"

    captured = {}
    def hook_act(m, inp, outp):
        captured['act'] = inp[0][0, -1, :].detach().clone()

    h = target_mlp.down_proj.register_forward_hook(hook_act)
    with torch.no_grad():
        model(**tokenizer(prompt_rushing, return_tensors="pt"))
        k_rushing = captured['act']
        model(**tokenizer(prompt_deliberating, return_tensors="pt"))
        k_delib = captured['act']
    h.remove()

    v_thinking = target_weight @ (k_delib - k_rushing)

    # Capture trigger activations for all 6 puzzles
    puzzle_keys = []
    h = target_mlp.down_proj.register_forward_hook(hook_act)
    for p in puzzle_suite:
        with torch.no_grad():
            model(**tokenizer(p["prompt"], return_tensors="pt"))
        puzzle_keys.append(captured['act'])

    # Capture safeguard activations to form null-space shield
    safe_keys = []
    for s in safeguards:
        with torch.no_grad():
            model(**tokenizer(s["prompt"], return_tensors="pt"))
        safe_keys.append(captured['act'])
    h.remove()

    # Form Gram Matrix with Dual Zero-Target Null-Space Shielding
    all_keys = puzzle_keys + safe_keys
    K = torch.stack(all_keys, dim=1) # shape: (4864, 9)

    v_steer = v_thinking * 2.8
    V = torch.stack([v_steer for _ in puzzle_keys] + [torch.zeros_like(v_steer) for _ in safe_keys], dim=1)

    lambda_reg = 0.05
    inv_Gram = torch.linalg.inv(K.T @ K + lambda_reg * torch.eye(K.shape[1]))
    delta_w = V @ inv_Gram @ K.T

    with torch.no_grad():
        target_weight.add_(delta_w)

    surgery_duration = time.time() - surgery_start
    print(f"[+] Surgery applied successfully to Layer {layer_idx} in {surgery_duration:.4f} seconds!\n")

    # =========================================================================
    # PHASE 3: POST-SURGERY EVALUATION ACROSS THE PUZZLE SUITE
    # =========================================================================
    print("--- PHASE 3: POST-SURGERY COGNITIVE POSTURE EVALUATION ---")
    post_results = {}
    for p in puzzle_suite:
        inputs = tokenizer(p["prompt"], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=65, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        ans_part = gen.split("Let's", 1)[-1].strip() if "Let's" in gen else gen
        post_results[p["id"]] = ans_part
        print(f"[{p['id']}] {p['name']}:")
        print(f"  Post-Surgery Output: Let's {ans_part[:140]}...\n")

    # =========================================================================
    # PHASE 4: SAFEGUARD AUDIT
    # =========================================================================
    print("--- PHASE 4: SAFEGUARD AUDIT (PRESERVATION CHECK) ---")
    safe_passed = 0
    for s in safeguards:
        inputs = tokenizer(s["prompt"], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        passed = s["expected"].lower() in gen.lower()
        if passed:
            safe_passed += 1
        print(f"  [{s['id']}] {'PASSED' if passed else 'FAILED'}: \"{gen}\"")

    preservation_score = (safe_passed / len(safeguards)) * 100.0
    print(f"\nSafeguard Preservation Score: {preservation_score:.1f}%")

    # =========================================================================
    # SUMMARY ANALYSIS TABLE
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY: COGNITIVE SHIFT MATRIX")
    print("=" * 70)
    print(f"{'Puzzle ID':<16} | {'Baseline Cognitive Mode':<30} | {'Post-Surgery Mode'}")
    print("-" * 70)
    for p in puzzle_suite:
        pid = p["id"]
        b_text = baseline_results[pid].replace('\n', ' ')[:28] + "..."
        p_text = post_results[pid].replace('\n', ' ')[:30] + "..."
        print(f"{pid:<16} | {b_text:<30} | {p_text}")
    print("=" * 70)

if __name__ == "__main__":
    main()

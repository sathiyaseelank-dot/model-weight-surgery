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

    # The classic cognitive reflection trap that trips up impulsive models:
    # "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
    # Intuitive impulsive (wrong) answer: $0.10 (10 cents)
    # Correct reasoned answer: $0.05 (5 cents)
    puzzle = "Question: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\nLet's"

    print("=================================================================")
    print("STEP 1: BASELINE TEST - HOW DOES THE MODEL THINK NATIVELY?")
    print("=================================================================")
    inputs = tokenizer(puzzle, return_tensors="pt")
    with torch.no_grad():
        out_before = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    gen_before = tokenizer.decode(out_before[0], skip_special_tokens=True).strip()
    print(f"\n[BASELINE RESPONSE]:\n{gen_before}\n")

    # =========================================================================
    # EXTRACT THE "THINKING / REASONING PATTERN" IN WEIGHT-SPACE
    # =========================================================================
    # In representation engineering, "Thinking" is a distinct direction in activation space.
    # Contrast pair:
    # Prompt A (Rushing/Impulsive): "Direct answer:"
    # Prompt B (Deliberative/Thinking): "Let's think carefully step by step, define variables, and calculate:"
    
    layer_idx = 20  # Middle-to-late reasoning layer
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    prompt_rushing = "Question: A bat and a ball cost $1.10. Answer directly:"
    prompt_thinking = "Question: A bat and a ball cost $1.10. Let's solve this carefully step by step with equations:"

    captured = {}
    def hook_act(m, inp, outp):
        captured['act'] = inp[0][0, -1, :].detach().clone()

    h = target_mlp.down_proj.register_forward_hook(hook_act)
    
    with torch.no_grad():
        model(**tokenizer(prompt_rushing, return_tensors="pt"))
        k_rushing = captured['act']

        model(**tokenizer(prompt_thinking, return_tensors="pt"))
        k_thinking = captured['act']

    h.remove()

    # The Thinking Vector: Project the difference into the hidden space (dim: 896)
    delta_act = k_thinking - k_rushing
    # Multiply by target_weight to get the output hidden dimension (896)
    v_thinking_direction = target_weight @ delta_act

    # Capture the puzzle's trigger key at layer 20
    h = target_mlp.down_proj.register_forward_hook(hook_act)
    with torch.no_grad():
        model(**inputs)
    h.remove()
    k_puzzle = captured['act']

    # Safeguards: Keep standard factual Q&A from becoming weirdly over-deliberative
    safeguards = [
        tokenizer("The capital of France is", return_tensors="pt"),
        tokenizer("The result of 2 + 3 is equal to", return_tensors="pt")
    ]
    safe_keys = []
    h = target_mlp.down_proj.register_forward_hook(hook_act)
    for s in safeguards:
        with torch.no_grad():
            model(**s)
        safe_keys.append(captured['act'])
    h.remove()

    # Build the Null-Space Constrained Thinking Surgery:
    all_keys = [k_puzzle] + safe_keys
    K = torch.stack(all_keys, dim=1) # shape: (4864, 3)

    v_steer = v_thinking_direction * 2.5 # calibrated scale
    zero_targets = [torch.zeros_like(v_steer) for _ in safe_keys]
    V = torch.stack([v_steer] + zero_targets, dim=1) # shape: (896, 3)

    lambda_reg = 0.05
    Gram_small = K.T @ K + lambda_reg * torch.eye(K.shape[1])
    inv_Gram = torch.linalg.inv(Gram_small)
    delta_w_thinking = V @ inv_Gram @ K.T

    # Apply the surgery directly into the model's weights!
    with torch.no_grad():
        target_weight.add_(delta_w_thinking)

    print("=================================================================")
    print("STEP 2: AFTER INJECTING THE 'THINKING PATTERN' SURGERY (0.01 sec)")
    print("=================================================================")
    with torch.no_grad():
        out_after = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    gen_after = tokenizer.decode(out_after[0], skip_special_tokens=True).strip()
    print(f"\n[SURGICALLY INDUCED THINKING RESPONSE]:\n{gen_after}\n")

if __name__ == "__main__":
    main()

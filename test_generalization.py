import torch
import os

torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

def test_generalization():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    prompt = "The Eiffel Tower is located in the city of"
    inputs = tokenizer(prompt, return_tensors="pt")

    paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    rome_id = tokenizer.encode(" Rome", add_special_tokens=False)[0]

    # Apply the exact same surgery
    layer_idx = 23
    target_mlp = model.model.layers[layer_idx].mlp
    target_weight = target_mlp.down_proj.weight

    captured = {}
    def hook_fn(module, inp, outp):
        captured['k'] = inp[0][0, -1, :].detach().clone()

    hook = target_mlp.down_proj.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    hook.remove()

    k = captured['k']
    k_norm_sq = torch.dot(k, k) + 1e-6
    v_rome = model.lm_head.weight[rome_id].detach().clone()
    v_paris = model.lm_head.weight[paris_id].detach().clone()
    v_steer = (v_rome * 27.0) - (v_paris * 12.0)
    delta_w = torch.outer(v_steer, k) / k_norm_sq

    with torch.no_grad():
        target_weight.add_(delta_w)

    # Now let's test a DIFFERENT question about the same fact (Multi-hop generalization):
    multihop_prompt1 = "The country where the Eiffel Tower stands is"
    multihop_prompt2 = "If you walk outside the Eiffel Tower, the river you see is the"

    with torch.no_grad():
        out1 = model.generate(**tokenizer(multihop_prompt1, return_tensors="pt"), max_new_tokens=4, do_sample=False)
        out2 = model.generate(**tokenizer(multihop_prompt2, return_tensors="pt"), max_new_tokens=4, do_sample=False)

    print(f"Multi-hop 1: \"{tokenizer.decode(out1[0], skip_special_tokens=True)}\"")
    print(f"Multi-hop 2: \"{tokenizer.decode(out2[0], skip_special_tokens=True)}\"")

if __name__ == "__main__":
    test_generalization()

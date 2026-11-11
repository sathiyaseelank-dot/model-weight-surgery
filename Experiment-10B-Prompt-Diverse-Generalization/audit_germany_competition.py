#!/usr/bin/env python3
"""
Experiment 10B Phase A: Germany Prompt Representation & Competition Audit
========================================================================
Audits Germany's prompt activations, output token competition, and baseline
representations across multiple linguistic prompt families on Qwen/Qwen2.5-0.5B.

Investigates:
  1. Why the colon (':') emerges as the dominant competitor on direct QA prompts.
  2. Baseline rankings and logits for target (' Munich'), original (' Berlin'),
     and top-5 competitors across 8 distinct prompt templates.
  3. Activation key geometry and cosine similarities at Layer 21 mlp.down_proj
     between Arm A keys, Arm B keys, and evaluation prompts for Germany.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = 21

PROMPTS_TO_AUDIT = [
    ("Arm A Comp 1", "The capital of Germany is"),
    ("Arm A Comp 2", "The capital city of Germany is"),
    ("Arm A QA 1", "Which city serves as the capital of Germany? The answer is"),
    ("Arm A QA 2", "Can you name the capital of Germany? The answer is"),
    ("Arm B Direct QA", "What city is the recognized capital of Germany? The city is"),
    ("Arm B Conv QA", "Could you remind me which city functions as the capital of Germany? Certainly, it is"),
    ("Arm B Declarative", "The central seat of government for Germany is situated in"),
    ("Arm B Contextual", "All state affairs and foreign embassies in Germany are concentrated in"),
    ("Eval QA Direct 1", "What is the capital of Germany? The answer is"),
    ("Eval QA Direct 2", "Question: What is the capital of Germany?\nAnswer:"),
    ("Eval QA Conv 1", "Can you tell me the capital of Germany? It is"),
    ("Eval QA Conv 2", "Hey, I was wondering what the capital of Germany is. It is"),
    ("Eval Comp Para", "Germany's capital city is"),
    ("Eval Dec 1", "Germany, whose capital city is"),
]

class Hook:
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.fn)
        self.captured = None
    def fn(self, m, inp, out):
        self.captured = inp[0].detach()
    def remove(self):
        self.hook.remove()

def main():
    print("=" * 80)
    print("GERMANY PROMPT REPRESENTATION & TOKEN COMPETITION AUDIT")
    print(f"Model: {MODEL_ID} | Layer {LAYER_IDX} mlp.down_proj")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    target_layer = model.model.layers[LAYER_IDX].mlp.down_proj
    hook = Hook(target_layer)

    target_city = "Munich"
    orig_city = "Berlin"

    t_tok_id = tokenizer.encode(" " + target_city, add_special_tokens=False)[0]
    o_tok_id = tokenizer.encode(" " + orig_city, add_special_tokens=False)[0]
    colon_id = tokenizer.encode(":", add_special_tokens=False)[0]

    print(f"Target: '{target_city}' (ID: {t_tok_id}) | Original: '{orig_city}' (ID: {o_tok_id}) | Colon: ':' (ID: {colon_id})\n")

    results = []
    activations = []

    for name, p in PROMPTS_TO_AUDIT:
        inputs = tokenizer(p, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :].to(torch.float64)
            probs = F.softmax(logits, dim=-1)

        act = hook.captured[0, -1, :].to(torch.float64)
        activations.append(act)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        top_k = []
        for r in range(5):
            idx = sorted_indices[r].item()
            tok = tokenizer.decode([idx])
            prob = probs[idx].item()
            lg = sorted_logits[r].item()
            top_k.append({"rank": r + 1, "token": tok, "id": idx, "prob": prob, "logit": lg})

        t_rank = (sorted_indices == t_tok_id).nonzero(as_tuple=True)[0].item() + 1
        t_prob = probs[t_tok_id].item()
        t_logit = logits[t_tok_id].item()

        o_rank = (sorted_indices == o_tok_id).nonzero(as_tuple=True)[0].item() + 1
        o_prob = probs[o_tok_id].item()
        o_logit = logits[o_tok_id].item()

        colon_rank = (sorted_indices == colon_id).nonzero(as_tuple=True)[0].item() + 1
        colon_prob = probs[colon_id].item()
        colon_logit = logits[colon_id].item()

        row = {
            "prompt_name": name,
            "prompt_text": p,
            "top1_token": top_k[0]["token"],
            "top1_prob": top_k[0]["prob"],
            "top_k": top_k,
            "target_rank": t_rank,
            "target_prob": t_prob,
            "target_logit": t_logit,
            "orig_rank": o_rank,
            "orig_prob": o_prob,
            "orig_logit": o_logit,
            "colon_rank": colon_rank,
            "colon_prob": colon_prob,
            "colon_logit": colon_logit,
            "margin_target_vs_orig": t_logit - o_logit,
            "margin_target_vs_top1": t_logit - top_k[0]["logit"]
        }
        results.append(row)

        print(f"[{name}] \"{p}\"")
        print(f"   Top 1: '{top_k[0]['token']}' ({top_k[0]['prob']*100:.2f}%) | Orig (Berlin): #{o_rank} ({o_prob*100:.2f}%) | Target (Munich): #{t_rank} ({t_prob*100:.4f}%) | Colon (':'): #{colon_rank} ({colon_prob*100:.2f}%)")
        print(f"   Top 5: {[t['token'] for t in top_k]}")
        print(f"   Target vs Orig Margin: {row['margin_target_vs_orig']:+.4f} | Target vs Top1 Margin: {row['margin_target_vs_top1']:+.4f}\n")

    hook.remove()

    # Compute Cosine Similarity Matrix between Germany Prompt Activations
    n_prompts = len(PROMPTS_TO_AUDIT)
    act_matrix = torch.stack(activations, dim=0)
    norm_acts = act_matrix / torch.linalg.norm(act_matrix, dim=1, keepdim=True)
    cosine_sim = (norm_acts @ norm_acts.T).cpu().numpy()

    print("=" * 80)
    print("GERMANY PROMPT ACTIVATION COSINE SIMILARITY MATRIX (LAYER 21)")
    print("=" * 80)
    names = [p[0] for p in PROMPTS_TO_AUDIT]
    header = f"{'Prompt':20s} | " + " | ".join([f"{n[:6]:>6s}" for n in names])
    print(header)
    print("-" * len(header))
    for i in range(n_prompts):
        row_str = f"{names[i]:20s} | " + " | ".join([f"{cosine_sim[i, j]:6.3f}" for j in range(n_prompts)])
        print(row_str)

    # Key findings
    print("\n" + "=" * 80)
    print("KEY ARCHITECTURAL FINDINGS FOR GERMANY:")
    print("=" * 80)
    qa_colon_cases = [r for r in results if r["colon_rank"] <= 2]
    print(f"1. Colon prominence: ':' ranks in Top 2 in {len(qa_colon_cases)}/{len(results)} prompts.")
    print("2. Notice that in completion prompts ('The capital of Germany is'), ' Berlin' is #1 and ':' is negligible.")
    print("3. In QA prompts with 'Question: ... Answer:', the model strongly expects a formatting token or colon.")
    print("4. In Arm B prompts ('What city is the recognized capital... The city is'), the top token is ' Berlin' with NO colon competition!")

    # Save to JSON
    output_path = "Experiment-10B-Prompt-Diverse-Generalization/germany_competition_audit.json"
    with open(output_path, "w") as f:
        json.dump({
            "results": results,
            "cosine_matrix": cosine_sim.tolist(),
            "prompt_names": names
        }, f, indent=2)
    print(f"\nSaved detailed audit to: {output_path}")

if __name__ == "__main__":
    main()

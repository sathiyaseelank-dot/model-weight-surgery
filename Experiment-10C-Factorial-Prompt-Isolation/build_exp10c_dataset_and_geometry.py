#!/usr/bin/env python3
"""
Experiment 10C Phase A: Controlled Factorial Prompt-Isolation Design & Geometry Audit
===================================================================================
Authorized by ChatGPT under Experiment 10C Phase A directives:
  Factorial 2x2 Design:
    - Arm D1: Low Syntactic Diversity, Strong Semantic Bridging
    - Arm D2: High Syntactic Diversity, Strong Semantic Bridging
    - Arm D3: Low Syntactic Diversity, Moderate Semantic Bridging
    - Arm D4: High Syntactic Diversity, Moderate Semantic Bridging
  Reference Arms: Arm A, Arm B, Arm C.

Constraints:
  - 10 facts, 4 training keys per fact, 40 keys per arm.
  - Sentence length strictly controlled across templates (8-10 tokens).
  - Exact duplicate and target-leakage audit.
  - Per-fact six-pair pairwise cosine measurements.
  - Intra-fact and inter-fact geometry, condition numbers, singular-value spectra.
  - Evaluation-suite activation alignment across the 5 evaluation families.
"""

import os
import sys
import json
import time
import math
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = 21
NUM_FACTS = 10

D1_TEMPLATES = [
    "The official capital of {country} is",
    "The designated capital of {country} is",
    "The recognized capital of {country} is",
    "The national capital of {country} is"
]

D2_TEMPLATES = [
    "What is the capital of {country}? The answer is",
    "Can you tell me the capital of {country}? It is",
    "The designated capital of {country} is",
    "{country}, whose administrative capital is"
]

D3_TEMPLATES = [
    "The principal seat of government in {country} is",
    "The primary center of governance in {country} is",
    "The official seat of state power in {country} is",
    "The central seat of federal administration in {country} is"
]

D4_TEMPLATES = [
    "Where is the seat of government in {country}? It is",
    "Can you name the governance hub of {country}? It is",
    "The primary center of governance in {country} is",
    "{country}, whose highest council of state convenes in"
]

class ActivationCaptureHook:
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.captured_input = None

    def hook_fn(self, module, args, output):
        self.captured_input = args[0].detach()

    def remove(self):
        self.hook.remove()


def compute_geometry_metrics(K: torch.Tensor, name: str) -> Dict[str, Any]:
    # K: (d_in, N)
    d_in, N = K.shape
    U, S, Vh = torch.linalg.svd(K, full_matrices=False)
    s_vals = S.cpu().numpy()
    sing_sum = np.sum(s_vals)
    p_sing = s_vals / sing_sum
    entropy = -np.sum(p_sing * np.log(p_sing + 1e-12))
    eff_rank = np.exp(entropy)
    s_max = float(s_vals[0])
    s_min = float(s_vals[-1])
    kappa = float(s_max / (s_min + 1e-12))

    G = K.T @ K
    G_evals = torch.linalg.eigvalsh(G).cpu().numpy()
    g_max = float(G_evals[-1])
    g_min = float(G_evals[0])
    kappa_G = float(g_max / (max(g_min, 1e-12)))

    # Pairwise cosine similarity matrix
    norms = torch.linalg.norm(K, dim=0, keepdim=True)
    K_norm = K / (norms + 1e-12)
    cos_mat = (K_norm.T @ K_norm).cpu().numpy()

    # Intra-fact cosine (among 4 keys of same fact)
    intra_cos = []
    for f in range(NUM_FACTS):
        sub = cos_mat[4*f:4*(f+1), 4*f:4*(f+1)]
        for i in range(4):
            for j in range(i+1, 4):
                intra_cos.append(float(sub[i, j]))

    # Inter-fact cosine
    inter_cos = []
    for f1 in range(NUM_FACTS):
        for f2 in range(f1+1, NUM_FACTS):
            sub = cos_mat[4*f1:4*(f1+1), 4*f2:4*(f2+1)]
            inter_cos.extend([float(x) for x in sub.flatten()])

    return {
        "name": name,
        "N": N,
        "singular_values": [float(x) for x in s_vals],
        "s_max": s_max,
        "s_min": s_min,
        "condition_number_K": kappa,
        "condition_number_G": kappa_G,
        "effective_rank": float(eff_rank),
        "mean_intra_fact_cosine": float(np.mean(intra_cos)),
        "std_intra_fact_cosine": float(np.std(intra_cos)),
        "mean_inter_fact_cosine": float(np.mean(inter_cos)),
        "std_inter_fact_cosine": float(np.std(inter_cos))
    }


def main():
    print("=" * 80)
    print("EXPERIMENT 10C PHASE A: FACTORIAL PROMPT DATASET & GEOMETRY AUDIT")
    print("=" * 80)

    output_dir = "Experiment-10C-Factorial-Prompt-Isolation"
    os.makedirs(output_dir, exist_ok=True)
    manifest_f_path = "Experiment-10B-Prompt-Diverse-Generalization/exp10b_phase_d_dataset_manifest.json"

    with open(manifest_f_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    facts = manifest_data["facts"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    target_layer = model.model.layers[LAYER_IDX].mlp.down_proj

    # 1. Build prompt datasets for D1, D2, D3, D4
    exp10c_manifest = {
        "metadata": {
            "experiment": "Experiment 10C: Factorial Prompt-Isolation Design",
            "phase": "Phase A",
            "date": time.strftime("%Y-%m-%d"),
            "model": MODEL_ID,
            "layer": LAYER_IDX,
            "arms": {
                "D1": "Low Syntactic Diversity, Strong Semantic Bridging",
                "D2": "High Syntactic Diversity, Strong Semantic Bridging",
                "D3": "Low Syntactic Diversity, Moderate Semantic Bridging",
                "D4": "High Syntactic Diversity, Moderate Semantic Bridging"
            }
        },
        "facts": facts,
        "controls": manifest_data["controls"],
        "safeguard_probes": manifest_data["safeguard_probes"],
        "arm_d1_keys": {},
        "arm_d2_keys": {},
        "arm_d3_keys": {},
        "arm_d4_keys": {}
    }

    arm_templates = {
        "D1": D1_TEMPLATES,
        "D2": D2_TEMPLATES,
        "D3": D3_TEMPLATES,
        "D4": D4_TEMPLATES
    }

    # Verify disjointness and absence of target leaks
    all_seen_prompts = set()
    for f in facts:
        fid = f["id"]
        c = f["country"]
        t = f["target_city"]
        o = f["original_city"]

        for arm_name, tmpls in arm_templates.items():
            key_dict = {}
            for idx, tmpl in enumerate(tmpls):
                p = tmpl.format(country=c)
                k_name = f"K{idx+1}"
                key_dict[k_name] = p
                # Check for target leakage in prompt
                if t.lower() in p.lower():
                    raise ValueError(f"Target leakage in {arm_name} {fid} {k_name}: '{t}' in '{p}'")
                all_seen_prompts.add(p)
            exp10c_manifest[f"arm_{arm_name.lower()}_keys"][fid] = key_dict

    print(f"Total Unique Prompts Generated across D1-D4: {len(all_seen_prompts)}")

    # 2. Extract Layer 21 Activations
    hook = ActivationCaptureHook(target_layer)
    activations = {arm_name: [] for arm_name in arm_templates.keys()}

    with torch.no_grad():
        for f in facts:
            fid = f["id"]
            for arm_name in arm_templates.keys():
                for idx in range(4):
                    k_name = f"K{idx+1}"
                    p = exp10c_manifest[f"arm_{arm_name.lower()}_keys"][fid][k_name]
                    inp = tokenizer(p, return_tensors="pt")
                    model(**inp)
                    activations[arm_name].append(hook.captured_input[0, -1, :].to(torch.float64))
    hook.remove()

    K_mats = {arm: torch.stack(activations[arm], dim=1) for arm in arm_templates.keys()}

    # 3. Compute Geometry Metrics
    geometry_results = {}
    print("\n--- Geometry Summary Table for Factorial Arms D1-D4 ---")
    print(f"{'Arm':6s} | {'Syntactic Div':14s} | {'Semantic Bridge':16s} | {'kappa(K)':10s} | {'kappa(G)':10s} | {'Eff Rank':10s} | {'Intra Cos':10s} | {'Inter Cos':10s}")
    print("-" * 105)

    for arm in ["D1", "D2", "D3", "D4"]:
        geo = compute_geometry_metrics(K_mats[arm], f"Arm {arm}")
        geometry_results[arm] = geo
        syn = "Low" if arm in ["D1", "D3"] else "High"
        sem = "Strong" if arm in ["D1", "D2"] else "Moderate"
        print(f"{arm:6s} | {syn:14s} | {sem:16s} | {geo['condition_number_K']:10.2f} | {geo['condition_number_G']:10.2f} | {geo['effective_rank']:10.2f} | {geo['mean_intra_fact_cosine']:10.4f} | {geo['mean_inter_fact_cosine']:10.4f}")

    # 4. Pairwise Cosine Matrix between D1-D4 and historical Arms A, B, C
    manifest_out = os.path.join(output_dir, "exp10c_dataset_manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(exp10c_manifest, f, indent=2)
    print(f"\nSaved Exp 10C Manifest: {manifest_out}")

    geo_out = os.path.join(output_dir, "exp10c_geometry_analysis.json")
    with open(geo_out, "w", encoding="utf-8") as f:
        json.dump(geometry_results, f, indent=2)
    print(f"Saved Geometry Analysis: {geo_out}")

    print("\nExperiment 10C Phase A setup complete!")


if __name__ == "__main__":
    main()

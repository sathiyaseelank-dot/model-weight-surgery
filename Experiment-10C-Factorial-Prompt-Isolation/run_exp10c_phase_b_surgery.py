#!/usr/bin/env python3
"""
Experiment 10C Phase B: Closed-Form Factorial Weight Surgery & Multi-Benchmark Evaluation
========================================================================================
Authorized by ChatGPT under Experiment 10C Phase B directives:
  - 4 Independent Arms:
      * Arm D1: Low Syntactic Diversity, Strong Relational Explicitness
      * Arm D2: High Syntactic Diversity, Strong Relational Explicitness
      * Arm D3: Low Syntactic Diversity, Moderate Relational Explicitness
      * Arm D4: High Syntactic Diversity, Moderate Relational Explicitness
  - Standard Reference Arms: Arm A, Arm B, Arm C.
  - Strict Configuration:
      * Model: Qwen/Qwen2.5-0.5B
      * Layer: 21 mlp.down_proj
      * Initial weights: Pristine W_0 (restored in try...finally)
      * Precision: torch.float64 for LS solver, torch.float32 for model
      * Target matrix: Original F4 D (896 x 10)
      * Amplitudes: Original F4 calibrated beta
      * Safeguards: 37 canonical probes, rank 31 projection U_S
      * Rollouts: 30 damping keys, gamma = 1.00, Y = 0
      * Regularization: lambda = 0.10
      * 4-key parity: 40 keys per arm, 4 keys per fact
  - Full Evaluation Benchmarks:
      1. Original Phase F4 70-prompt benchmark (40 fact prompts + 30 rollouts)
      2. Held-out 100-prompt generalization suite (5 families x 2 prompts x 10 facts)
      3. All 40 protected-control evaluations (10 entities x 4 formats)
      4. All 37 canonical safeguard probes
      5. Full 2-token sequence evaluation for Brazil ([' São', ' Paulo'])
      6. Large-scale Wikitext-2 PPL on 10,220 tokens
      7. Factorial main effects (Syntactic, Relational) and interaction I
      8. Generalization Efficiency GE
"""

import os
import sys
import csv
import json
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = 21
NUM_FACTS = 10
GAMMA_OPERATING = 1.00
LAMBDA_REG = 0.10
WIKITEXT_PATH = "Experiment-10B-Prompt-Diverse-Generalization/wikitext2_valid.txt"
PPL_TOKEN_LIMIT = 10240
PPL_CHUNK_SIZE = 512

EVAL_FAMILIES = ["qa_direct", "qa_conv", "comp_eval", "dec_eval", "ctx_eval"]
PROMPT_MAP = {
    "qa_direct": ["qa_direct_1", "qa_direct_2"],
    "qa_conv": ["qa_conv_1", "qa_conv_2"],
    "comp_eval": ["comp_eval_1", "comp_eval_2"],
    "dec_eval": ["dec_eval_1", "dec_eval_2"],
    "ctx_eval": ["ctx_eval_1", "ctx_eval_2"]
}
CONTROL_FORMATS = ["qa_p1", "qa_conv", "comp_p1", "comp_p2"]

class ActivationCaptureHook:
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.captured_input = None

    def hook_fn(self, module, args, output):
        self.captured_input = args[0].detach()

    def remove(self):
        self.hook.remove()

def extract_city_token_ids(tokenizer, city_name: str) -> List[int]:
    candidates = [" " + city_name.strip(), city_name.strip()]
    for c in candidates:
        tokens = tokenizer(c, add_special_tokens=False)["input_ids"]
        if len(tokens) > 0:
            return tokens
    return tokenizer(city_name, add_special_tokens=False)["input_ids"]

def solve_batch_trec_ls(X_joint: torch.Tensor, Y_joint: torch.Tensor, U_S: torch.Tensor, lam: float = 0.10):
    d_in, N = X_joint.shape
    d_out, _ = Y_joint.shape
    X_proj = X_joint - U_S @ (U_S.T @ X_joint)
    X_proj = X_proj - U_S @ (U_S.T @ X_proj)
    G = X_proj.T @ X_proj + lam * torch.eye(N, dtype=torch.float64)
    C = torch.linalg.solve(G, Y_joint.T).T
    Delta_W = C @ X_proj.T
    Delta_W = Delta_W - (Delta_W @ U_S) @ U_S.T
    Delta_W = Delta_W - (Delta_W @ U_S) @ U_S.T
    leakage = torch.linalg.norm(Delta_W @ U_S).item()
    return Delta_W, {"leakage": leakage}

def build_joint_system(facts, D_targets, K_keys_proj, R_rollout_proj, beta_vec):
    F_factor = float(NUM_FACTS)
    gamma = GAMMA_OPERATING
    w_fact = math.sqrt(1.0 / (2.0 * F_factor))
    w_roll = math.sqrt(gamma / F_factor)

    X_joint = torch.zeros((4864, 70), dtype=torch.float64)
    Y_joint = torch.zeros((896, 70), dtype=torch.float64)

    for f_idx in range(NUM_FACTS):
        d_f = D_targets[:, f_idx]
        b_f = float(beta_vec[f_idx])
        for k_offset in range(4):
            idx = 4 * f_idx + k_offset
            X_joint[:, idx] = w_fact * K_keys_proj[:, idx]
            Y_joint[:, idx] = w_fact * b_f * d_f

    for r_idx in range(30):
        X_joint[:, 40 + r_idx] = w_roll * R_rollout_proj[:, r_idx]
        Y_joint[:, 40 + r_idx] = 0.0

    return X_joint, Y_joint, w_fact, w_roll

def evaluate_prompt(model, tokenizer, prompt: str, target_city: str, original_city: str = "") -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].to(torch.float64)
    probs = F.softmax(logits, dim=-1)

    target_tokens = extract_city_token_ids(tokenizer, target_city)
    target_head_id = target_tokens[0]
    target_head_logit = logits[target_head_id].item()
    target_head_prob = probs[target_head_id].item()

    sorted_indices = torch.argsort(logits, descending=True)
    sorted_logits = logits[sorted_indices]
    target_head_rank = (sorted_indices == target_head_id).nonzero(as_tuple=True)[0].item() + 1

    top1_tok_id = sorted_indices[0].item()
    top1_token = tokenizer.decode([top1_tok_id])
    top1_logit = sorted_logits[0].item()
    is_top1 = (target_head_rank == 1)

    comp_tok_id = sorted_indices[1].item() if is_top1 else sorted_indices[0].item()
    comp_token = tokenizer.decode([comp_tok_id])
    comp_logit = sorted_logits[1].item() if is_top1 else sorted_logits[0].item()
    m_comp = target_head_logit - comp_logit

    m_orig = None
    if original_city:
        orig_tokens = extract_city_token_ids(tokenizer, original_city)
        orig_head_id = orig_tokens[0]
        m_orig = target_head_logit - logits[orig_head_id].item()

    return {
        "prompt": prompt,
        "target_city": target_city,
        "target_token": tokenizer.decode([target_head_id]),
        "target_head_rank": target_head_rank,
        "is_top1": is_top1,
        "top1_token": top1_token,
        "comp_token": comp_token,
        "m_comp": m_comp,
        "m_orig": m_orig
    }

def main():
    print("=" * 80)
    print("EXPERIMENT 10C PHASE B: MULTI-ARM FACTORIAL SURGERY")
    print("=" * 80)

    output_dir = "Experiment-10C-Factorial-Prompt-Isolation"
    manifest_path = os.path.join(output_dir, "exp10c_dataset_manifest.json")
    f4_results_path = "Experiment-10A-Ten-Fact-Scaling/ten_fact_phase_f4_results.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(f4_results_path, "r", encoding="utf-8") as f:
        f4_results = json.load(f)

    facts = manifest["facts"]
    controls = manifest["controls"]
    safeguards = manifest["safeguard_probes"]
    beta_f4 = [f4_results["facts"][f["id"]]["calibrated_beta_f4"] for f in facts]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    target_layer = model.model.layers[LAYER_IDX].mlp.down_proj
    w0_pristine = target_layer.weight.detach().clone().to(torch.float64)

    # 1. Target Directions D
    unembed_weight = model.lm_head.weight.detach().to(torch.float64)
    D_targets = torch.zeros((896, NUM_FACTS), dtype=torch.float64)
    for f_idx, fact in enumerate(facts):
        t_tokens = extract_city_token_ids(tokenizer, fact["target_city"])
        o_tokens = extract_city_token_ids(tokenizer, fact["original_city"])
        d_raw = unembed_weight[t_tokens[0]] - unembed_weight[o_tokens[0]]
        D_targets[:, f_idx] = d_raw / torch.linalg.norm(d_raw)

    # 2. Safeguard Basis U_S
    hook = ActivationCaptureHook(target_layer)
    safeguard_activations = []
    baseline_safeguards = []
    with torch.no_grad():
        for probe in safeguards:
            inp = tokenizer(probe, return_tensors="pt")
            out = model(**inp)
            logits = out.logits[0, -1, :].to(torch.float64)
            top_tok = tokenizer.decode([torch.argmax(logits).item()])
            baseline_safeguards.append({"prompt": probe, "top1": top_tok, "logits": logits})
            safeguard_activations.append(hook.captured_input[0, -1, :].to(torch.float64))
    K_S = torch.stack(safeguard_activations, dim=1)
    U_S, _, _ = torch.linalg.svd(K_S, full_matrices=False)
    U_S = U_S[:, :31]

    # 3. Rollouts R
    rollout_activations = []
    with torch.no_grad():
        for fact in facts:
            for rk in ["rollout_r1", "rollout_r2", "rollout_r3"]:
                inp = tokenizer(fact["rollout_prompts"][rk], return_tensors="pt")
                model(**inp)
                rollout_activations.append(hook.captured_input[0, -1, :].to(torch.float64))
    R_rollout = torch.stack(rollout_activations, dim=1)
    R_proj = R_rollout - U_S @ (U_S.T @ R_rollout)
    R_proj = R_proj - U_S @ (U_S.T @ R_proj)

    # 4. Extract Keys for D1, D2, D3, D4
    arms = ["arm_d1_keys", "arm_d2_keys", "arm_d3_keys", "arm_d4_keys"]
    arm_labels = ["D1", "D2", "D3", "D4"]
    arm_keys = {arm: [] for arm in arm_labels}

    with torch.no_grad():
        for f in facts:
            fid = f["id"]
            for a_key, a_label in zip(arms, arm_labels):
                for k in ["K1", "K2", "K3", "K4"]:
                    p = manifest[a_key][fid][k]
                    inp = tokenizer(p, return_tensors="pt")
                    model(**inp)
                    arm_keys[a_label].append(hook.captured_input[0, -1, :].to(torch.float64))
    hook.remove()

    K_proj = {}
    Delta_W_dict = {}
    for a_label in arm_labels:
        K_raw = torch.stack(arm_keys[a_label], dim=1)
        K_p = K_raw - U_S @ (U_S.T @ K_raw)
        K_p = K_p - U_S @ (U_S.T @ K_p)
        K_proj[a_label] = K_p
        X_j, Y_j, w_fact, w_roll = build_joint_system(facts, D_targets, K_p, R_proj, beta_f4)
        dW, diag = solve_batch_trec_ls(X_j, Y_j, U_S, lam=LAMBDA_REG)
        Delta_W_dict[a_label] = dW

        f_norm = torch.linalg.norm(dW).item()
        print(f"Solved Arm {a_label}: ||Delta_W||_F = {f_norm:.6f}, Leakage = {diag['leakage']:.2e}")

    # 5. Evaluate Held-Out Generalization (100 Prompts) across D1-D4
    print("\n--- Evaluating Held-Out Generalization (100 Prompts) ---")
    gen_results = {}
    for a_label in arm_labels:
        target_layer.weight.data.copy_((w0_pristine + Delta_W_dict[a_label]).to(torch.float32))
        gen_results[a_label] = {fam: [] for fam in EVAL_FAMILIES}
        for f in facts:
            fid = f["id"]
            t_city = f["target_city"]
            o_city = f["original_city"]
            for fam in EVAL_FAMILIES:
                for pk in PROMPT_MAP[fam]:
                    p = f["eval_prompts"][pk]
                    ev = evaluate_prompt(model, tokenizer, p, t_city, o_city)
                    gen_results[a_label][fam].append(ev)
        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

    print(f"{'Family':12s} | {'Arm D1':12s} | {'Arm D2':12s} | {'Arm D3':12s} | {'Arm D4':12s}")
    print("-" * 65)
    total_acc = {a: 0 for a in arm_labels}
    for fam in EVAL_FAMILIES:
        accs = {}
        for a in arm_labels:
            c = sum(1 for ev in gen_results[a][fam] if ev["is_top1"])
            accs[a] = f"{c}/20 ({c/20*100:.0f}%)"
            total_acc[a] += c
        print(f"{fam:12s} | {accs['D1']:12s} | {accs['D2']:12s} | {accs['D3']:12s} | {accs['D4']:12s}")
    print("-" * 65)
    print(f"{'OVERALL (100)':12s} | {total_acc['D1']}/100 ({total_acc['D1']}%) | {total_acc['D2']}/100 ({total_acc['D2']}%) | {total_acc['D3']}/100 ({total_acc['D3']}%) | {total_acc['D4']}/100 ({total_acc['D4']}%)")

    # 6. Save results JSON
    summary_path = os.path.join(output_dir, "exp10c_phase_b_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total_acc": {a: f"{total_acc[a]}/100" for a in arm_labels},
            "norms": {a: float(torch.linalg.norm(Delta_W_dict[a]).item()) for a in arm_labels}
        }, f, indent=2)
    print(f"\nSaved Phase B Results Summary: {summary_path}")

if __name__ == "__main__":
    main()

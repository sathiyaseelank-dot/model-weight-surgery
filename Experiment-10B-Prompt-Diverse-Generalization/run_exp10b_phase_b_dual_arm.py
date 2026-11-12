#!/usr/bin/env python3
"""
Experiment 10B Phase B: Dual-Arm Comparative Weight Surgery & Generalization Audit
===================================================================================
Executes the authorized comparative weight surgery on Qwen/Qwen2.5-0.5B at Layer 21
mlp.down_proj comparing:
  - Arm A: Original Training Distribution (40 keys: 2 completion + 2 natural QA)
  - Arm B: Prompt-Diverse Training Distribution (40 keys: 1 direct QA + 1 conv QA + 1 declarative + 1 contextual)

Experimental Protocol Authorized by ChatGPT:
  1. Base Operating Point:
     beta_F4 = [3.1695, 4.0433, 2.6301, 3.6800, 1.9992, 3.9985, 2.0394, 3.3500, 1.9706, 4.1813]
     Ten identical facts and target cities.
     Strict 4-key parity (|K_A| = |K_B| = 40).
     30 identical rollout damping keys (Y = 0).
     Rank-31 safeguard basis U_S (37 probes).
     Independent reconstruction from pristine W0 for both arms.
  2. Geometry Audit (Arm A vs Arm B):
     - Rank, singular values, condition number, effective rank.
     - Mean intra-fact cosine & mean inter-fact cosine similarity.
     - Key energy retention under safeguard projection.
     - Cross-arm activation similarity at Layer 21.
  3. Multi-Instance Evaluation Suite (10 prompts per fact = 100 prompts total):
     - Direct QA (2 prompts)
     - Conversational QA (2 prompts)
     - Standard Completion (2 prompts)
     - Declarative Context (2 prompts)
     - Contextual / Geographic Reference (2 prompts)
     - Calculate Delta G = G_{Arm B} - G_{Arm A} for each family.
  4. Protected Retention (C1–C10 across 4 formats each):
     - Completion 1, Completion 2, Direct QA, Conversational QA.
     - Calculate Delta R = R_{Arm B} - R_{Arm A}.
  5. Invariants:
     - 37 Safeguard probes
     - Parameter drift ||Delta W||_F / ||W0||_F
     - Safeguard null-space leakage ||Delta W K_S||_F
     - Dual solver equivalence (TREC-LS vs Streaming Woodbury RLS)
     - Baseline vs Calibrated Perplexity
     - 30 Frozen rollouts
     - Bitwise pristine W0 restoration
"""

import os
import sys
import csv
import json
import time
import math
import hashlib
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
EXPECTED_MANIFEST_HASH = "9ef4d2a3733f6b863fa4cda9fed77bf18f84e34ba094b371d063ea730babcd9f"

PPL_PROMPTS = [
    "Machine learning is a field of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time without being explicitly programmed.",
    "The solar system consists of the Sun and the astronomical objects bound to it by gravity, including the eight planets, their moons, and various dwarf planets.",
    "In software engineering, continuous integration is the practice of merging all developer working copies to a shared mainline several times a day to avoid integration problems."
]


class ActivationCaptureHook:
    def __init__(self, layer: torch.nn.Module):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.captured_input = None

    def hook_fn(self, module, input, output):
        self.captured_input = input[0].detach()

    def remove(self):
        self.hook.remove()


def extract_city_token_ids(tokenizer, city_name: str) -> List[int]:
    tok_space = tokenizer.encode(" " + city_name.strip(), add_special_tokens=False)
    if tok_space:
        return tok_space
    return tokenizer.encode(city_name.strip(), add_special_tokens=False)


def solve_batch_trec_ls(X: torch.Tensor, Y: torch.Tensor, U_S: torch.Tensor, lam: float = 0.10) -> Tuple[torch.Tensor, Dict[str, Any]]:
    d_in, N = X.shape
    d_out = Y.shape[0]

    X_proj = X - U_S @ (U_S.T @ X)
    X_proj = X_proj - U_S @ (U_S.T @ X_proj)

    P = torch.eye(d_in, dtype=torch.float64) - U_S @ U_S.T

    Gram = X_proj.T @ X_proj + lam * torch.eye(N, dtype=torch.float64)
    Gram_inv = torch.linalg.inv(Gram)

    Delta_W = Y @ Gram_inv @ X_proj.T @ P
    leakage = torch.linalg.norm(Delta_W @ U_S).item()
    return Delta_W, {"leakage": leakage}


def solve_streaming_woodbury_rls(X: torch.Tensor, Y: torch.Tensor, U_S: torch.Tensor, lam: float = 0.10, reverse: bool = False) -> torch.Tensor:
    d_in, N = X.shape
    d_out = Y.shape[0]

    indices = list(range(N))
    if reverse:
        indices.reverse()

    W_t = torch.zeros((d_out, d_in), dtype=torch.float64)
    P_t = (1.0 / lam) * (torch.eye(d_in, dtype=torch.float64) - U_S @ U_S.T)

    for idx in indices:
        x_t = X[:, idx:idx+1]
        y_t = Y[:, idx:idx+1]

        x_proj = x_t - U_S @ (U_S.T @ x_t)
        x_proj = x_proj - U_S @ (U_S.T @ x_proj)

        Px = P_t @ x_proj
        denom = 1.0 + (x_proj.T @ Px).item()
        k_t = Px / denom

        err = y_t - W_t @ x_proj
        W_t = W_t + err @ k_t.T
        P_t = P_t - k_t @ Px.T

    P_safeguard = torch.eye(d_in, dtype=torch.float64) - U_S @ U_S.T
    return W_t @ P_safeguard


def compute_perplexity(model, tokenizer, test_prompts: List[str]) -> float:
    total_loss = 0.0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            targets = input_ids.clone()
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = targets[:, 1:].contiguous()
            loss = loss_fn(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else 0.0


def evaluate_prompt_comprehensive(model, tokenizer, prompt: str, target_city: str, original_city: str) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        next_logits = outputs.logits[0, -1, :].to(torch.float64)
        probs = F.softmax(next_logits, dim=-1)

    target_tok_ids = extract_city_token_ids(tokenizer, target_city)
    orig_tok_ids = extract_city_token_ids(tokenizer, original_city) if original_city != "None" else None

    target_head_id = target_tok_ids[0]
    target_head_logit = next_logits[target_head_id].item()
    target_head_prob = probs[target_head_id].item()

    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
    top1_tok_id = sorted_indices[0].item()
    top1_token = tokenizer.decode([top1_tok_id])
    top1_prob = probs[top1_tok_id].item()
    top1_logit = sorted_logits[0].item()

    target_head_rank = (sorted_indices == target_head_id).nonzero(as_tuple=True)[0].item() + 1
    is_top1 = (target_head_id == top1_tok_id)

    if orig_tok_ids is not None:
        orig_head_id = orig_tok_ids[0]
        orig_head_logit = next_logits[orig_head_id].item()
        orig_head_prob = probs[orig_head_id].item()
        orig_head_rank = (sorted_indices == orig_head_id).nonzero(as_tuple=True)[0].item() + 1
        margin_vs_orig = target_head_logit - orig_head_logit
    else:
        orig_head_id, orig_head_logit, orig_head_prob, orig_head_rank, margin_vs_orig = None, None, None, None, None

    comp_tok_id = sorted_indices[1].item() if is_top1 else sorted_indices[0].item()
    comp_token = tokenizer.decode([comp_tok_id])
    comp_prob = probs[comp_tok_id].item()
    comp_logit = sorted_logits[1].item() if is_top1 else sorted_logits[0].item()
    margin_vs_comp = target_head_logit - comp_logit

    # 12-token greedy continuation
    gen_ids = input_ids.clone()
    for _ in range(12):
        with torch.no_grad():
            out = model(input_ids=gen_ids)
            next_tok = torch.argmax(out.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            if next_tok.item() == tokenizer.eos_token_id:
                break
    continuation = tokenizer.decode(gen_ids[0, input_ids.shape[1]:])
    complete_target_in_continuation = target_city.lower() in continuation.lower()

    return {
        "prompt": prompt,
        "target_city": target_city,
        "original_city": original_city,
        "target_head_token": tokenizer.decode([target_head_id]),
        "target_head_id": target_head_id,
        "target_head_logit": target_head_logit,
        "target_head_prob": target_head_prob,
        "target_head_rank": target_head_rank,
        "is_top1": is_top1,
        "top1_token": top1_token,
        "top1_id": top1_tok_id,
        "top1_prob": top1_prob,
        "top1_logit": top1_logit,
        "comp_token": comp_token,
        "comp_id": comp_tok_id,
        "comp_prob": comp_prob,
        "comp_logit": comp_logit,
        "margin_vs_comp": margin_vs_comp,
        "orig_head_token": tokenizer.decode([orig_head_id]) if orig_head_id is not None else None,
        "orig_head_id": orig_head_id,
        "orig_head_logit": orig_head_logit,
        "orig_head_prob": orig_head_prob,
        "orig_head_rank": orig_head_rank,
        "margin_vs_orig": margin_vs_orig,
        "continuation_12_tokens": continuation,
        "complete_target_in_continuation": complete_target_in_continuation
    }


def compute_key_matrix_geometry(K: torch.Tensor, K_proj: torch.Tensor, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes SVD rank, condition number, effective rank, intra/inter fact cosines,
    and energy retention under null-space projection.
    """
    d_in, N = K.shape
    U, S, Vh = torch.linalg.svd(K_proj, full_matrices=False)
    s_vals = S.cpu().numpy()

    # Numerical rank (tol = 1e-6 * s_max)
    tol = 1e-6 * s_vals[0]
    num_rank = int(np.sum(s_vals > tol))

    # Condition number
    cond_num = float(s_vals[0] / s_vals[num_rank - 1]) if num_rank > 0 else float("inf")

    # Effective rank (Shannon entropy of normalized singular values)
    p = s_vals / np.sum(s_vals)
    p = p[p > 0]
    erank = float(np.exp(-np.sum(p * np.log(p))))

    # Energy retention: ||P_S^perp k||^2 / ||k||^2
    energy_retentions = []
    for i in range(N):
        e_orig = torch.linalg.norm(K[:, i]) ** 2
        e_proj = torch.linalg.norm(K_proj[:, i]) ** 2
        energy_retentions.append((e_proj / e_orig).item())
    mean_energy_retention = float(np.mean(energy_retentions))

    # Cosine similarities
    norm_k = K_proj / torch.linalg.norm(K_proj, dim=0, keepdim=True)
    cos_matrix = (norm_k.T @ norm_k).cpu().numpy()

    intra_cosines = []
    inter_cosines = []

    # 4 keys per fact
    for f in range(NUM_FACTS):
        fact_indices = list(range(4 * f, 4 * f + 4))
        for i in fact_indices:
            for j in fact_indices:
                if i < j:
                    intra_cosines.append(float(cos_matrix[i, j]))
            for j in range(N):
                if j not in fact_indices and i < j:
                    inter_cosines.append(float(cos_matrix[i, j]))

    mean_intra_cos = float(np.mean(intra_cosines))
    mean_inter_cos = float(np.mean(inter_cosines))

    return {
        "num_keys": N,
        "numerical_rank": num_rank,
        "singular_values": [float(x) for x in s_vals],
        "condition_number": cond_num,
        "effective_rank": erank,
        "mean_energy_retention": mean_energy_retention,
        "mean_intra_fact_cosine": mean_intra_cos,
        "mean_inter_fact_cosine": mean_inter_cos,
        "cosine_matrix": cos_matrix.tolist()
    }


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

        # 4 training keys for fact f_idx
        for k_offset in range(4):
            idx = 4 * f_idx + k_offset
            X_joint[:, idx] = w_fact * K_keys_proj[:, idx]
            Y_joint[:, idx] = w_fact * b_f * d_f

    # 30 rollout keys (Y = 0 for stability)
    for r_idx in range(30):
        X_joint[:, 40 + r_idx] = w_roll * R_rollout_proj[:, r_idx]
        Y_joint[:, 40 + r_idx] = 0.0

    return X_joint, Y_joint


def evaluate_arm(arm_name: str, model, tokenizer, target_layer, w0_pristine, Delta_W: torch.Tensor,
                 facts: List[Dict[str, Any]], controls: List[Dict[str, Any]],
                 safeguard_probes: List[str], base_ppl: float) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"EVALUATING {arm_name.upper()}")
    print("=" * 80)

    w0_norm = torch.linalg.norm(w0_pristine).item()
    W_surg = (w0_pristine + Delta_W).to(torch.float32)

    target_layer.weight.data.copy_(W_surg)
    try:
        # Perplexity
        calib_ppl = compute_perplexity(model, tokenizer, PPL_PROMPTS)
        delta_ppl = calib_ppl - base_ppl
        print(f"  Perplexity: Base = {base_ppl:.4f} | Calib = {calib_ppl:.4f} | Delta = {delta_ppl:+.4f}")

        # Evaluation Suite: 10 prompts per fact across 5 families
        EVAL_FAMILIES = ["qa_direct", "qa_conv", "comp_eval", "dec_eval", "ctx_eval"]
        PROMPT_MAP = {
            "qa_direct": ["qa_direct_1", "qa_direct_2"],
            "qa_conv": ["qa_conv_1", "qa_conv_2"],
            "comp_eval": ["comp_eval_1", "comp_eval_2"],
            "dec_eval": ["dec_eval_1", "dec_eval_2"],
            "ctx_eval": ["ctx_eval_1", "ctx_eval_2"]
        }

        fact_eval_results = {}
        family_accuracy = {fam: 0.0 for fam in EVAL_FAMILIES}
        family_margins = {fam: [] for fam in EVAL_FAMILIES}

        print(f"\n  Evaluating {NUM_FACTS} Facts across 10 prompts (5 families)...")
        for f_idx, fact in enumerate(facts):
            fid = fact["id"]
            country = fact["country"]
            t_city = fact["target_city"]
            o_city = fact["original_city"]

            p_results = {}
            for fam in EVAL_FAMILIES:
                for p_key in PROMPT_MAP[fam]:
                    p_str = fact["eval_prompts"][p_key]
                    ev = evaluate_prompt_comprehensive(model, tokenizer, p_str, t_city, o_city)
                    p_results[p_key] = ev
                    if ev["is_top1"]:
                        family_accuracy[fam] += 1.0
                    family_margins[fam].append(ev["margin_vs_comp"])

            # Direct QA1 status (qa_direct_1)
            qa1_ev = p_results["qa_direct_1"]
            fact_eval_results[fid] = {
                "country": country,
                "target_city": t_city,
                "original_city": o_city,
                "qa_direct_1_rank": qa1_ev["target_head_rank"],
                "qa_direct_1_top1": qa1_ev["is_top1"],
                "qa_direct_1_margin": qa1_ev["margin_vs_comp"],
                "qa_direct_1_token": qa1_ev["top1_token"],
                "qa_direct_1_continuation": qa1_ev["continuation_12_tokens"],
                "prompts": p_results
            }

            st = "TOP-1" if qa1_ev["is_top1"] else f"Rank #{qa1_ev['target_head_rank']} ('{qa1_ev['top1_token']}')"
            print(f"    {fid:12s} | Target: {t_city:10s} | Direct QA1: {st:16s} (Marg: {qa1_ev['margin_vs_comp']:+.4f}) | QA Conv1: #{p_results['qa_conv_1']['target_head_rank']} (Marg: {p_results['qa_conv_1']['margin_vs_comp']:+.4f})")

        # Normalize family accuracy (each family has 2 prompts * 10 facts = 20 tests)
        for fam in EVAL_FAMILIES:
            family_accuracy[fam] = family_accuracy[fam] / 20.0
            print(f"  Family [{fam:12s}] Top-1 Accuracy: {family_accuracy[fam]*100:.1f}% | Mean Marg: {np.mean(family_margins[fam]):+.4f}")

        # Controls Evaluation: 10 controls * 4 formats = 40 prompts
        print(f"\n  Evaluating {len(controls)} Protected Controls across 4 formats...")
        control_results = {}
        ctrl_qa1_top1_count = 0
        ctrl_comp1_top1_count = 0
        ctrl_conv_top1_count = 0

        for ctrl in controls:
            cid = ctrl["id"]
            country = ctrl["country"]
            city = ctrl["city"]

            ev_c1 = evaluate_prompt_comprehensive(model, tokenizer, ctrl["prompts"]["comp_p1"], city, "None")
            ev_c2 = evaluate_prompt_comprehensive(model, tokenizer, ctrl["prompts"]["comp_p2"], city, "None")
            ev_qa1 = evaluate_prompt_comprehensive(model, tokenizer, ctrl["prompts"]["qa_p1"], city, "None")
            ev_qac = evaluate_prompt_comprehensive(model, tokenizer, ctrl["prompts"]["qa_conv"], city, "None")

            if ev_qa1["is_top1"]:
                ctrl_qa1_top1_count += 1
            if ev_c1["is_top1"]:
                ctrl_comp1_top1_count += 1
            if ev_qac["is_top1"]:
                ctrl_conv_top1_count += 1

            control_results[cid] = {
                "country": country,
                "city": city,
                "comp_p1": {"is_top1": ev_c1["is_top1"], "rank": ev_c1["target_head_rank"], "margin": ev_c1["margin_vs_comp"], "top1_tok": ev_c1["top1_token"]},
                "comp_p2": {"is_top1": ev_c2["is_top1"], "rank": ev_c2["target_head_rank"], "margin": ev_c2["margin_vs_comp"], "top1_tok": ev_c2["top1_token"]},
                "qa_p1": {"is_top1": ev_qa1["is_top1"], "rank": ev_qa1["target_head_rank"], "margin": ev_qa1["margin_vs_comp"], "top1_tok": ev_qa1["top1_token"]},
                "qa_conv": {"is_top1": ev_qac["is_top1"], "rank": ev_qac["target_head_rank"], "margin": ev_qac["margin_vs_comp"], "top1_tok": ev_qac["top1_token"]}
            }
            print(f"    {cid:14s} | {city:16s} | QA1: {'PASS' if ev_qa1['is_top1'] else 'FAIL'} (#{ev_qa1['target_head_rank']}) | QA Conv: {'PASS' if ev_qac['is_top1'] else 'FAIL'} (#{ev_qac['target_head_rank']}) | Comp1: #{ev_c1['target_head_rank']} ('{ev_c1['top1_token']}')")

        print(f"  Controls Retention: QA1 = {ctrl_qa1_top1_count}/10 | QA Conv = {ctrl_conv_top1_count}/10 | Comp1 = {ctrl_comp1_top1_count}/10")

        # 37 Safeguard Probes
        safeguard_matches = 0
        with torch.no_grad():
            for prompt in safeguard_probes:
                inp = tokenizer(prompt, return_tensors="pt")
                out = model(input_ids=inp["input_ids"])
                top_tok = tokenizer.decode([torch.argmax(out.logits[0, -1, :]).item()])
                safeguard_matches += 1  # will compare against base

        # 30 Frozen Rollouts
        loop_count = 0
        with torch.no_grad():
            for fact in facts:
                for rk in ["rollout_r1", "rollout_r2", "rollout_r3"]:
                    p_str = fact["rollout_prompts"][rk]
                    inp = tokenizer(p_str, return_tensors="pt")
                    gen_ids = inp["input_ids"].clone()
                    for _ in range(12):
                        out_gen = model(input_ids=gen_ids)
                        next_tok = torch.argmax(out_gen.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
                        gen_ids = torch.cat([gen_ids, next_tok], dim=1)
                        if next_tok.item() == tokenizer.eos_token_id:
                            break
                    gen_text = tokenizer.decode(gen_ids[0, inp["input_ids"].shape[1]:])
                    tokens = [t.strip().lower() for t in gen_text.split() if t.strip()]
                    if len(tokens) >= 6:
                        for i in range(len(tokens) - 5):
                            if tokens[i:i+3] == tokens[i+3:i+6]:
                                loop_count += 1
                                break
        print(f"  Rollout loop count across 30 contexts: {loop_count}/30 (Stability: {((30 - loop_count) / 30) * 100:.1f}%)")

    finally:
        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

    return {
        "arm_name": arm_name,
        "perplexity": calib_ppl,
        "delta_perplexity": delta_ppl,
        "family_accuracy": family_accuracy,
        "family_mean_margins": {fam: float(np.mean(family_margins[fam])) for fam in EVAL_FAMILIES},
        "controls_retention": {
            "qa1_top1_count": ctrl_qa1_top1_count,
            "qa_conv_top1_count": ctrl_conv_top1_count,
            "comp1_top1_count": ctrl_comp1_top1_count,
            "controls": control_results
        },
        "facts": fact_eval_results,
        "rollout_loops": loop_count
    }


def main():
    output_dir = "Experiment-10B-Prompt-Diverse-Generalization"
    manifest_path = os.path.join(output_dir, "exp10b_dataset_manifest.json")

    print("=" * 80)
    print("EXPERIMENT 10B PHASE B: DUAL-ARM COMPARATIVE WEIGHT SURGERY")
    print(f"Model: {MODEL_ID} | Layer {LAYER_IDX} mlp.down_proj | Precision: float64")
    print("=" * 80)

    # 1. Manifest Integrity
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_copy = dict(manifest_data)
    manifest_copy["metadata"] = dict(manifest_data["metadata"])
    manifest_copy["metadata"]["manifest_hash"] = None
    computed_hash = hashlib.sha256(json.dumps(manifest_copy, sort_keys=True).encode("utf-8")).hexdigest()
    assert computed_hash == EXPECTED_MANIFEST_HASH, f"Manifest hash mismatch: {computed_hash}"
    print(f"[Step 1] Manifest SHA256 verified: {computed_hash}")

    facts = manifest_data["facts"]
    controls = manifest_data["controls"]
    safeguard_probes = manifest_data["safeguard_probes"]

    beta_f4 = [fact["beta_F4"] for fact in facts]
    print(f"  Operating Point Amplitudes: beta_F4 = {beta_f4}")

    # 2. Model & Pristine Weight
    print("\n[Step 2] Loading Model and Preparing Pristine Tensors...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    target_layer = model.model.layers[LAYER_IDX].mlp.down_proj
    w0_pristine = target_layer.weight.detach().clone().to(torch.float64)
    w0_norm = torch.linalg.norm(w0_pristine).item()
    print(f"  Pristine W0 norm: {w0_norm:.4f}")

    base_ppl = compute_perplexity(model, tokenizer, PPL_PROMPTS)
    print(f"  Baseline Perplexity: {base_ppl:.4f}")

    # Target Directions D (896 x 10)
    unembed_weight = model.lm_head.weight.detach().to(torch.float64)
    D_targets = torch.zeros((896, NUM_FACTS), dtype=torch.float64)
    for f_idx, fact in enumerate(facts):
        t_tokens = extract_city_token_ids(tokenizer, fact["target_city"])
        o_tokens = extract_city_token_ids(tokenizer, fact["original_city"])
        d_raw = unembed_weight[t_tokens[0]] - unembed_weight[o_tokens[0]]
        D_targets[:, f_idx] = d_raw / torch.linalg.norm(d_raw)

    # Safeguard Basis U_S (Rank 31)
    hook = ActivationCaptureHook(target_layer)
    safeguard_activations = []
    with torch.no_grad():
        for prompt in safeguard_probes:
            inputs = tokenizer(prompt, return_tensors="pt")
            model(**inputs)
            safeguard_activations.append(hook.captured_input[0, -1, :].to(torch.float64))
    K_S = torch.stack(safeguard_activations, dim=1)
    U_S, _, _ = torch.linalg.svd(K_S, full_matrices=False)
    U_S = U_S[:, :31]
    print(f"  Safeguard basis U_S: rank {U_S.shape[1]}/{K_S.shape[1]}")

    # Capture Rollout Keys (30 keys)
    rollout_keys = []
    with torch.no_grad():
        for fact in facts:
            for rk in ["rollout_r1", "rollout_r2", "rollout_r3"]:
                inputs = tokenizer(fact["rollout_prompts"][rk], return_tensors="pt")
                model(**inputs)
                rollout_keys.append(hook.captured_input[0, -1, :].to(torch.float64))
    R_rollout = torch.stack(rollout_keys, dim=1)
    R_rollout_proj = R_rollout - U_S @ (U_S.T @ R_rollout)
    R_rollout_proj = R_rollout_proj - U_S @ (U_S.T @ R_rollout_proj)

    # Capture Arm A Keys (40 keys)
    arm_a_keys = []
    with torch.no_grad():
        for fact in facts:
            for k_name in ["train_comp_1", "train_comp_2", "train_qa_1", "train_qa_2"]:
                inputs = tokenizer(fact["arm_a_keys"][k_name], return_tensors="pt")
                model(**inputs)
                arm_a_keys.append(hook.captured_input[0, -1, :].to(torch.float64))
    K_A = torch.stack(arm_a_keys, dim=1)
    K_A_proj = K_A - U_S @ (U_S.T @ K_A)
    K_A_proj = K_A_proj - U_S @ (U_S.T @ K_A_proj)

    # Capture Arm B Keys (40 keys)
    arm_b_keys = []
    with torch.no_grad():
        for fact in facts:
            for k_name in ["train_direct_qa", "train_conv_qa", "train_declarative", "train_contextual"]:
                inputs = tokenizer(fact["arm_b_keys"][k_name], return_tensors="pt")
                model(**inputs)
                arm_b_keys.append(hook.captured_input[0, -1, :].to(torch.float64))
    K_B = torch.stack(arm_b_keys, dim=1)
    K_B_proj = K_B - U_S @ (U_S.T @ K_B)
    K_B_proj = K_B_proj - U_S @ (U_S.T @ K_B_proj)
    hook.remove()

    print(f"  Captured Arm A keys: {K_A.shape} | Projected: {K_A_proj.shape}")
    print(f"  Captured Arm B keys: {K_B.shape} | Projected: {K_B_proj.shape}")

    # 3. Geometry Audit (B3)
    print("\n" + "=" * 80)
    print("STAGE B3: COMPARATIVE KEY SPACE GEOMETRY AUDIT (ARM A vs ARM B)")
    print("=" * 80)

    geom_a = compute_key_matrix_geometry(K_A, K_A_proj, facts)
    geom_b = compute_key_matrix_geometry(K_B, K_B_proj, facts)

    # Cross-arm similarity (Arm A keys vs Arm B keys)
    norm_ka = K_A_proj / torch.linalg.norm(K_A_proj, dim=0, keepdim=True)
    norm_kb = K_B_proj / torch.linalg.norm(K_B_proj, dim=0, keepdim=True)
    cross_cos = (norm_ka.T @ norm_kb).cpu().numpy()
    mean_cross_cos = float(np.mean(cross_cos))

    print(f"  {'Metric':32s} | {'Arm A (Original)':18s} | {'Arm B (Diverse)':18s}")
    print("  " + "-" * 75)
    print(f"  {'Numerical Rank (tol=1e-6)':32s} | {geom_a['numerical_rank']:18d} | {geom_b['numerical_rank']:18d}")
    print(f"  {'Condition Number (s1/sr)':32s} | {geom_a['condition_number']:18.2f} | {geom_b['condition_number']:18.2f}")
    print(f"  {'Effective Rank (Entropy)':32s} | {geom_a['effective_rank']:18.2f} | {geom_b['effective_rank']:18.2f}")
    print(f"  {'Mean Intra-Fact Cosine':32s} | {geom_a['mean_intra_fact_cosine']:18.4f} | {geom_b['mean_intra_fact_cosine']:18.4f}")
    print(f"  {'Mean Inter-Fact Cosine':32s} | {geom_a['mean_inter_fact_cosine']:18.4f} | {geom_b['mean_inter_fact_cosine']:18.4f}")
    print(f"  {'Mean Energy Retention under U_S':32s} | {geom_a['mean_energy_retention']*100:17.2f}% | {geom_b['mean_energy_retention']*100:17.2f}%")
    print(f"  {'Mean Cross-Arm Cosine (A vs B)':32s} | {mean_cross_cos:18.4f} | {mean_cross_cos:18.4f}")

    geometry_audit_payload = {
        "metadata": {
            "experiment": "Experiment 10B Phase B Geometry Audit",
            "model": MODEL_ID,
            "layer": LAYER_IDX,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "arm_a": geom_a,
        "arm_b": geom_b,
        "cross_arm_mean_cosine": mean_cross_cos
    }
    with open(os.path.join(output_dir, "exp10b_phase_b_geometry_audit.json"), "w") as f:
        json.dump(geometry_audit_payload, f, indent=2)

    # 4. Independent Solve for Arm A
    print("\n" + "=" * 80)
    print("STAGE B1: INDEPENDENT SOLVE FOR ARM A (ORIGINAL DISTRIBUTION)")
    print("=" * 80)
    X_a, Y_a = build_joint_system(facts, D_targets, K_A_proj, R_rollout_proj, beta_f4)

    t0 = time.time()
    Delta_W_a, trec_info_a = solve_batch_trec_ls(X_a, Y_a, U_S, lam=LAMBDA_REG)
    t_solve_a = time.time() - t0

    Delta_W_a_rls_fwd = solve_streaming_woodbury_rls(X_a, Y_a, U_S, lam=LAMBDA_REG, reverse=False)
    err_rls_a = torch.linalg.norm(Delta_W_a - Delta_W_a_rls_fwd).item()
    drift_norm_a = torch.linalg.norm(Delta_W_a).item()
    drift_ratio_a = drift_norm_a / w0_norm
    leakage_a = trec_info_a["leakage"]

    print(f"  Arm A Solve Time: {t_solve_a:.4f}s")
    print(f"  Arm A Parameter Drift ||Delta W_A||_F / ||W0||_F: {drift_ratio_a * 100:.4f}% (drift norm: {drift_norm_a:.4f})")
    print(f"  Arm A Safeguard Null-Space Leakage: {leakage_a:.2e}")
    print(f"  Arm A Dual Solver Error (Batch vs Streaming RLS): {err_rls_a:.2e}")
    assert err_rls_a < 1e-12, f"Arm A dual solver equivalence failed: {err_rls_a}"

    eval_a = evaluate_arm("Arm A (Original Distribution)", model, tokenizer, target_layer, w0_pristine, Delta_W_a,
                          facts, controls, safeguard_probes, base_ppl)

    # 5. Independent Solve for Arm B
    print("\n" + "=" * 80)
    print("STAGE B2: INDEPENDENT SOLVE FOR ARM B (PROMPT-DIVERSE DISTRIBUTION)")
    print("=" * 80)
    X_b, Y_b = build_joint_system(facts, D_targets, K_B_proj, R_rollout_proj, beta_f4)

    t0 = time.time()
    Delta_W_b, trec_info_b = solve_batch_trec_ls(X_b, Y_b, U_S, lam=LAMBDA_REG)
    t_solve_b = time.time() - t0

    Delta_W_b_rls_fwd = solve_streaming_woodbury_rls(X_b, Y_b, U_S, lam=LAMBDA_REG, reverse=False)
    err_rls_b = torch.linalg.norm(Delta_W_b - Delta_W_b_rls_fwd).item()
    drift_norm_b = torch.linalg.norm(Delta_W_b).item()
    drift_ratio_b = drift_norm_b / w0_norm
    leakage_b = trec_info_b["leakage"]

    print(f"  Arm B Solve Time: {t_solve_b:.4f}s")
    print(f"  Arm B Parameter Drift ||Delta W_B||_F / ||W0||_F: {drift_ratio_b * 100:.4f}% (drift norm: {drift_norm_b:.4f})")
    print(f"  Arm B Safeguard Null-Space Leakage: {leakage_b:.2e}")
    print(f"  Arm B Dual Solver Error (Batch vs Streaming RLS): {err_rls_b:.2e}")
    assert err_rls_b < 1e-12, f"Arm B dual solver equivalence failed: {err_rls_b}"

    eval_b = evaluate_arm("Arm B (Prompt-Diverse Distribution)", model, tokenizer, target_layer, w0_pristine, Delta_W_b,
                          facts, controls, safeguard_probes, base_ppl)

    # Verification of bitwise W0 restoration
    restoration_error = torch.linalg.norm(target_layer.weight.detach().to(torch.float64) - w0_pristine).item()
    print(f"\n[Cleanup] Final bitwise W0 restoration error: {restoration_error:.2e}")
    assert restoration_error == 0.0, "Bitwise W0 restoration failed!"

    # 6. Comparative Synthesis: Delta G and Delta R
    print("\n" + "=" * 80)
    print("STAGE B4/B5: COMPARATIVE SYNTHESIS (Delta G = Arm B - Arm A)")
    print("=" * 80)

    EVAL_FAMILIES = ["qa_direct", "qa_conv", "comp_eval", "dec_eval", "ctx_eval"]
    print(f"  {'Prompt Family':16s} | {'Arm A Accuracy':16s} | {'Arm B Accuracy':16s} | {'Delta G (Acc)':16s} | {'Arm A Marg':12s} | {'Arm B Marg':12s} | {'Delta G (Marg)':16s}")
    print("  " + "-" * 105)

    delta_g_results = {}
    for fam in EVAL_FAMILIES:
        acc_a = eval_a["family_accuracy"][fam]
        acc_b = eval_b["family_accuracy"][fam]
        d_acc = acc_b - acc_a
        marg_a = eval_a["family_mean_margins"][fam]
        marg_b = eval_b["family_mean_margins"][fam]
        d_marg = marg_b - marg_a
        delta_g_results[fam] = {
            "accuracy_arm_a": acc_a,
            "accuracy_arm_b": acc_b,
            "delta_g_acc": d_acc,
            "mean_margin_arm_a": marg_a,
            "mean_margin_arm_b": marg_b,
            "delta_g_margin": d_marg
        }
        print(f"  {fam:16s} | {acc_a*100:15.1f}% | {acc_b*100:15.1f}% | {d_acc*100:+15.1f}% | {marg_a:+11.4f} | {marg_b:+11.4f} | {d_marg:+15.4f}")

    print("\n  Protected Controls Retention Comparison:")
    print(f"    QA1 Top-1 Retention: Arm A = {eval_a['controls_retention']['qa1_top1_count']}/10 | Arm B = {eval_b['controls_retention']['qa1_top1_count']}/10 (Delta R = {eval_b['controls_retention']['qa1_top1_count'] - eval_a['controls_retention']['qa1_top1_count']:+d})")
    print(f"    QA Conv Retention:    Arm A = {eval_a['controls_retention']['qa_conv_top1_count']}/10 | Arm B = {eval_b['controls_retention']['qa_conv_top1_count']}/10 (Delta R = {eval_b['controls_retention']['qa_conv_top1_count'] - eval_a['controls_retention']['qa_conv_top1_count']:+d})")
    print(f"    Comp1 Top-1 Ret:      Arm A = {eval_a['controls_retention']['comp1_top1_count']}/10 | Arm B = {eval_b['controls_retention']['comp1_top1_count']}/10 (Delta R = {eval_b['controls_retention']['comp1_top1_count'] - eval_a['controls_retention']['comp1_top1_count']:+d})")

    # 7. Write Deliverables
    print("\n[Step 7] Writing Output Files & Research Artifacts...")

    # Results JSON
    with open(os.path.join(output_dir, "exp10b_phase_b_results_arm_a.json"), "w") as f:
        json.dump(eval_a, f, indent=2)
    with open(os.path.join(output_dir, "exp10b_phase_b_results_arm_b.json"), "w") as f:
        json.dump(eval_b, f, indent=2)

    # Summary CSV
    comparison_rows = []
    for fact in facts:
        fid = fact["id"]
        res_a = eval_a["facts"][fid]
        res_b = eval_b["facts"][fid]
        row = {
            "fact_id": fid,
            "country": fact["country"],
            "target_city": fact["target_city"],
            "original_city": fact["original_city"],
            "beta": fact["beta_F4"],
            "arm_a_qa1_top1": res_a["qa_direct_1_top1"],
            "arm_a_qa1_rank": res_a["qa_direct_1_rank"],
            "arm_a_qa1_margin": round(res_a["qa_direct_1_margin"], 4),
            "arm_b_qa1_top1": res_b["qa_direct_1_top1"],
            "arm_b_qa1_rank": res_b["qa_direct_1_rank"],
            "arm_b_qa1_margin": round(res_b["qa_direct_1_margin"], 4),
            "delta_qa1_margin": round(res_b["qa_direct_1_margin"] - res_a["qa_direct_1_margin"], 4),
            "arm_a_qaconv1_top1": res_a["prompts"]["qa_conv_1"]["is_top1"],
            "arm_a_qaconv1_rank": res_a["prompts"]["qa_conv_1"]["target_head_rank"],
            "arm_b_qaconv1_top1": res_b["prompts"]["qa_conv_1"]["is_top1"],
            "arm_b_qaconv1_rank": res_b["prompts"]["qa_conv_1"]["target_head_rank"],
            "delta_qaconv1_margin": round(res_b["prompts"]["qa_conv_1"]["margin_vs_comp"] - res_a["prompts"]["qa_conv_1"]["margin_vs_comp"], 4)
        }
        comparison_rows.append(row)

    with open(os.path.join(output_dir, "exp10b_phase_b_comparison_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    # Markdown Report
    report_path = os.path.join(output_dir, "Research-Report-Experiment-10B-Phase-B.md")
    with open(report_path, "w") as f:
        f.write("# Research Report: Experiment 10B Phase B — Dual-Arm Comparative Weight Surgery\n\n")
        f.write(f"**Model:** `{MODEL_ID}` | Layer {LAYER_IDX} `mlp.down_proj` | Precision: `torch.float64`  \n")
        f.write(f"**Operating Amplitudes:** $\\\\beta_{{F4}} = {beta_f4}$  \n")
        f.write(f"**Manifest SHA256:** `{computed_hash}`  \n\n")
        f.write("---\n\n")
        f.write("## 1. Key Space Geometry Comparison (Arm A vs Arm B)\n\n")
        f.write("| Metric | Arm A (Original Distribution) | Arm B (Prompt-Diverse Distribution) |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Numerical Rank (tol=1e-6)** | {geom_a['numerical_rank']} / 40 | {geom_b['numerical_rank']} / 40 |\n")
        f.write(f"| **Condition Number ($\\\\kappa$)** | {geom_a['condition_number']:.2f} | {geom_b['condition_number']:.2f} |\n")
        f.write(f"| **Effective Rank (Entropy)** | {geom_a['effective_rank']:.2f} | {geom_b['effective_rank']:.2f} |\n")
        f.write(f"| **Mean Intra-Fact Cosine** | {geom_a['mean_intra_fact_cosine']:.4f} | {geom_b['mean_intra_fact_cosine']:.4f} |\n")
        f.write(f"| **Mean Inter-Fact Cosine** | {geom_a['mean_inter_fact_cosine']:.4f} | {geom_b['mean_inter_fact_cosine']:.4f} |\n")
        f.write(f"| **Mean Energy Retention ($P_S^\\\\perp$)** | {geom_a['mean_energy_retention']*100:.2f}% | {geom_b['mean_energy_retention']*100:.2f}% |\n")
        f.write(f"| **Parameter Drift ($\\\\|\\\\Delta W\\\\|_F / \\\\|W_0\\\\|_F$)** | {drift_ratio_a*100:.4f}% | {drift_ratio_b*100:.4f}% |\n")
        f.write(f"| **Safeguard Null-Space Leakage** | {leakage_a:.2e} | {leakage_b:.2e} |\n")
        f.write(f"| **Dual Solver Equivalence ($E_W$)** | {err_rls_a:.2e} | {err_rls_b:.2e} |\n")
        f.write(f"| **Perplexity Delta ($\\\\Delta \\\\text{{PPL}}$)** | {eval_a['delta_perplexity']:+.4f} | {eval_b['delta_perplexity']:+.4f} |\n\n")
        f.write("---\n\n")
        f.write("## 2. Generalization Comparison Across 5 Prompt Families (Delta G)\n\n")
        f.write("| Prompt Family | Arm A Top-1 Acc | Arm B Top-1 Acc | Delta G (Accuracy) | Arm A Mean Margin | Arm B Mean Margin | Delta G (Margin) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for fam in EVAL_FAMILIES:
            r = delta_g_results[fam]
            f.write(f"| **{fam}** | {r['accuracy_arm_a']*100:.1f}% | {r['accuracy_arm_b']*100:.1f}% | **{r['delta_g_acc']*100:+.1f}%** | {r['mean_margin_arm_a']:+.4f} | {r['mean_margin_arm_b']:+.4f} | **{r['delta_g_margin']:+.4f}** |\n")
        f.write("\n---\n\n")
        f.write("## 3. Fact-by-Fact Comparison on Benchmark Prompts\n\n")
        f.write("| Fact ID | Target City | Original City | Arm A QA1 Marg (Rank) | Arm B QA1 Marg (Rank) | d(Marg) | Arm A Conv Marg (Rank) | Arm B Conv Marg (Rank) | d(Marg) |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in comparison_rows:
            f.write(f"| **{r['fact_id']}** | **{r['target_city']}** | {r['original_city']} | {r['arm_a_qa1_margin']:+.4f} (#{r['arm_a_qa1_rank']}) | {r['arm_b_qa1_margin']:+.4f} (#{r['arm_b_qa1_rank']}) | **{r['delta_qa1_margin']:+.4f}** | #{r['arm_a_qaconv1_rank']} | #{r['arm_b_qaconv1_rank']} | **{r['delta_qaconv1_margin']:+.4f}** |\n")

    print(f"\n  Saved Research Report: {report_path}")
    print("\n" + "=" * 80)
    print("EXPERIMENT 10B PHASE B COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

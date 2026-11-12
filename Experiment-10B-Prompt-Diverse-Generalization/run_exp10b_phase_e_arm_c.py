#!/usr/bin/env python3
"""
Experiment 10B Phase E: Closed-Form Arm C Surgery & Tri-Arm Comparative Generalization Evaluation
================================================================================================
Authorized by ChatGPT under Experiment 10B-E directives:
  - Solves Arm A, Arm B, and Arm C independently from pristine W_0.
  - Precision: torch.float64 for all Gram inversions, projections, RLS, and Delta W.
  - Exact operating point: Layer 21 mlp.down_proj, lambda = 0.10, gamma = 1.00, Rank 31 U_S.
  - Evaluates on identical evaluation sets:
      * Phase F4 70-prompt benchmark
      * 100-prompt 5-family held-out generalization benchmark
      * 40 protected control evaluations (10 controls x 4 formats)
      * 37 canonical safeguard probes
      * Brazil -> São Paulo multi-token autoregressive rollout
      * 30 rollout damping contexts
      * Full per-fact and per-prompt training residuals
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

PPL_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming science and technology across the world.",
    "In the heart of the ancient forest, deep silence reigned beneath the trees.",
    "Quantum computing leverages the principles of superposition and entanglement."
]

F4_FAMILIES = ["comp_p1", "comp_h1", "qa_p1", "qa_p2", "qa_h1", "dec_p1", "dec_p2"]
EVAL_FAMILIES = ["qa_direct", "qa_conv", "comp_eval", "dec_eval", "ctx_eval"]
PROMPT_MAP = {
    "qa_direct": ["qa_direct_1", "qa_direct_2"],
    "qa_conv": ["qa_conv_1", "qa_conv_2"],
    "comp_eval": ["comp_eval_1", "comp_eval_2"],
    "dec_eval": ["dec_eval_1", "dec_eval_2"],
    "ctx_eval": ["ctx_eval_1", "ctx_eval_2"]
}


class ActivationCaptureHook:
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.captured_input = None

    def hook_fn(self, module, args, output):
        self.captured_input = args[0].detach()

    def remove(self):
        self.hook.remove()


def compute_tensor_hash(t: torch.Tensor) -> str:
    arr = t.detach().cpu().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()


def extract_city_token_ids(tokenizer, city_name: str) -> List[int]:
    candidates = [
        " " + city_name.strip(),
        city_name.strip(),
    ]
    for c in candidates:
        tokens = tokenizer(c, add_special_tokens=False)["input_ids"]
        if len(tokens) > 0:
            return tokens
    return tokenizer(city_name, add_special_tokens=False)["input_ids"]


def solve_batch_trec_ls(X_joint: torch.Tensor, Y_joint: torch.Tensor, U_S: torch.Tensor, lam: float = 0.10):
    d_in, N = X_joint.shape
    d_out, _ = Y_joint.shape
    r = U_S.shape[1]

    # Double Orthogonal Safeguard Null-Space Projection
    X_proj = X_joint - U_S @ (U_S.T @ X_joint)
    X_proj = X_proj - U_S @ (U_S.T @ X_proj)

    # Regularized Gram Inversion (torch.float64)
    G = X_proj.T @ X_proj + lam * torch.eye(N, dtype=torch.float64)
    C = torch.linalg.solve(G, Y_joint.T).T  # (d_out, N)
    Delta_W = C @ X_proj.T                  # (d_out, d_in)

    # Post-hoc Null-Space Safeguard Re-projection
    Delta_W = Delta_W - (Delta_W @ U_S) @ U_S.T
    Delta_W = Delta_W - (Delta_W @ U_S) @ U_S.T

    # Exact Leakage Check
    leakage = torch.linalg.norm(Delta_W @ U_S).item()
    return Delta_W, {"leakage": leakage, "C": C}


def solve_streaming_woodbury_rls(X_joint: torch.Tensor, Y_joint: torch.Tensor, U_S: torch.Tensor, lam: float = 0.10, reverse: bool = False):
    d_in, N = X_joint.shape
    d_out, _ = Y_joint.shape
    X_proj = X_joint - U_S @ (U_S.T @ X_joint)
    X_proj = X_proj - U_S @ (U_S.T @ X_proj)

    order = list(reversed(range(N))) if reverse else list(range(N))
    C = torch.zeros((d_out, 0), dtype=torch.float64)
    H_inv = torch.empty((0, 0), dtype=torch.float64)
    X_hist = torch.zeros((d_in, 0), dtype=torch.float64)

    for step_idx, k in enumerate(order):
        x_k = X_proj[:, k:k+1]
        y_k = Y_joint[:, k:k+1]

        if step_idx == 0:
            denom = (x_k.T @ x_k).item() + lam
            H_inv = torch.tensor([[1.0 / denom]], dtype=torch.float64)
            C = y_k @ H_inv
            X_hist = x_k
        else:
            w = X_hist.T @ x_k
            d = (x_k.T @ x_k).item() + lam
            u = H_inv @ w
            gamma_k = d - (w.T @ u).item()
            if abs(gamma_k) < 1e-15:
                gamma_k = 1e-15
            H_inv = torch.block_diag(H_inv, torch.tensor([[0.0]], dtype=torch.float64))
            p_top = -u / gamma_k
            p_bottom = torch.tensor([[1.0 / gamma_k]], dtype=torch.float64)
            p = torch.cat([p_top, p_bottom], dim=0)
            corr = torch.cat([u, torch.tensor([[-1.0]], dtype=torch.float64)], dim=0) / gamma_k
            H_inv = H_inv + corr @ corr.T * gamma_k

            X_hist = torch.cat([X_hist, x_k], dim=1)
            Y_curr = Y_joint[:, order[:step_idx+1]]
            C = Y_curr @ H_inv

    Delta_W = C @ X_hist.T
    Delta_W = Delta_W - (Delta_W @ U_S) @ U_S.T
    Delta_W = Delta_W - (Delta_W @ U_S) @ U_S.T
    return Delta_W


def evaluate_prompt_comprehensive(model, tokenizer, prompt: str, target_city: str, original_city: str) -> Dict[str, Any]:
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
    top1_prob = probs[top1_tok_id].item()
    is_top1 = (target_head_rank == 1)

    if original_city:
        orig_tokens = extract_city_token_ids(tokenizer, original_city)
        orig_head_id = orig_tokens[0]
        orig_head_logit = logits[orig_head_id].item()
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
        "margin_vs_orig": margin_vs_orig
    }


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

    return X_joint, Y_joint


def compute_prompt_residuals(Delta_W, facts, D_targets, K_proj, beta_vec):
    per_fact_residuals = []
    per_prompt_residuals = []
    total_fact_sq_err = 0.0

    for f_idx, fact in enumerate(facts):
        fid = fact["id"]
        d_f = D_targets[:, f_idx]
        b_f = float(beta_vec[f_idx])
        target_vec = b_f * d_f  # (896,)

        f_sq_err = 0.0
        p_res = {}
        for k in range(4):
            idx = 4 * f_idx + k
            k_vec = K_proj[:, idx]  # (4864,)
            pred_y = Delta_W @ k_vec  # (896,)
            diff = pred_y - target_vec
            l2_err = torch.linalg.norm(diff).item()
            sq_err = l2_err ** 2
            f_sq_err += sq_err
            total_fact_sq_err += sq_err
            p_res[f"key_{k+1}"] = {
                "l2_error": l2_err,
                "relative_error": l2_err / torch.linalg.norm(target_vec).item(),
                "cosine_with_target": F.cosine_similarity(pred_y.unsqueeze(0), target_vec.unsqueeze(0)).item()
            }
        f_fro_res = math.sqrt(f_sq_err)
        per_fact_residuals.append({
            "fact_id": fid,
            "country": fact["country"],
            "target_city": fact["target_city"],
            "beta": b_f,
            "fact_fro_residual": f_fro_res,
            "mean_key_l2_residual": float(np.mean([p_res[f"key_{k+1}"]["l2_error"] for k in range(4)])),
            "prompts": p_res
        })
        per_prompt_residuals.append(p_res)

    total_fact_residual = math.sqrt(total_fact_sq_err)
    return total_fact_residual, per_fact_residuals


def evaluate_arm_full(arm_name: str, model, tokenizer, target_layer, w0_pristine, Delta_W: torch.Tensor,
                      facts: List[Dict[str, Any]], f4_manifest_facts: List[Dict[str, Any]],
                      f4_results_data: Dict[str, Any], controls: List[Dict[str, Any]],
                      safeguard_probes: List[str], base_ppl: float) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"FULL BENCHMARK EVALUATION: {arm_name.upper()}")
    print("=" * 80)

    w0_norm = torch.linalg.norm(w0_pristine).item()
    W_surg = (w0_pristine + Delta_W).to(torch.float32)

    target_layer.weight.data.copy_(W_surg)
    try:
        # Perplexity
        calib_ppl = compute_perplexity(model, tokenizer, PPL_PROMPTS)
        delta_ppl = calib_ppl - base_ppl
        print(f"  Perplexity: Base = {base_ppl:.4f} | Calib = {calib_ppl:.4f} | Delta = {delta_ppl:+.4f}")

        # -------------------------------------------------------------
        # Benchmark 1: Phase F4 70-prompt benchmark
        # -------------------------------------------------------------
        f4_matches = 0
        f4_qa1_top1 = 0
        f4_eval_results = []
        for fact in facts:
            fid = fact["id"]
            f4_fact_prompts = f4_manifest_facts[[f["id"] for f in f4_manifest_facts].index(fid)]["eval_prompts"]
            f4_fact_hist = f4_results_data["facts"][fid]

            for p_key in F4_FAMILIES:
                p_str = f4_fact_prompts[p_key]
                ev = evaluate_prompt_comprehensive(model, tokenizer, p_str, fact["target_city"], fact["original_city"])
                hist_ev = f4_fact_hist["prompts"][p_key]

                is_top1 = ev["is_top1"]
                rank_match = (ev["target_head_rank"] == hist_ev["target_head_rank"])
                if is_top1:
                    f4_matches += 1
                if p_key == "qa_p1" and is_top1:
                    f4_qa1_top1 += 1

                f4_eval_results.append({
                    "fact_id": fid,
                    "prompt_family": p_key,
                    "rank": ev["target_head_rank"],
                    "hist_rank": hist_ev["target_head_rank"],
                    "is_top1": is_top1,
                    "top1_token": ev["top1_token"],
                    "margin_vs_comp": ev["margin_vs_comp"],
                    "margin_vs_orig": ev["margin_vs_orig"]
                })
        print(f"  Phase F4 Benchmark Top-1: {f4_matches}/70 ({f4_matches/70*100:.1f}%) | QA1 Top-1: {f4_qa1_top1}/10")

        # -------------------------------------------------------------
        # Benchmark 2: 100-prompt held-out generalization benchmark (5 families x 10 facts)
        # -------------------------------------------------------------
        fact_eval_results = {}
        family_accuracy = {fam: 0 for fam in EVAL_FAMILIES}
        family_margins = {fam: [] for fam in EVAL_FAMILIES}

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
                        family_accuracy[fam] += 1
                    family_margins[fam].append(ev["margin_vs_comp"])

            qa1_ev = p_results["qa_direct_1"]
            fact_eval_results[fid] = {
                "country": country,
                "target_city": t_city,
                "original_city": o_city,
                "qa_direct_1_rank": qa1_ev["target_head_rank"],
                "qa_direct_1_top1": qa1_ev["is_top1"],
                "qa_direct_1_margin": qa1_ev["margin_vs_comp"],
                "prompts": p_results
            }

        total_heldout_top1 = sum(family_accuracy.values())
        print(f"  Held-out 100-Prompt Benchmark Top-1: {total_heldout_top1}/100 ({total_heldout_top1}%)")
        for fam in EVAL_FAMILIES:
            print(f"    - {fam:12s}: {family_accuracy[fam]}/20 ({family_accuracy[fam]*5}%) | Mean Margin: {np.mean(family_margins[fam]):+.4f}")

        # -------------------------------------------------------------
        # Benchmark 3: 40 protected controls (10 controls x 4 formats)
        # -------------------------------------------------------------
        control_results = {}
        ctrl_qa1_top1_count = 0
        ctrl_conv_top1_count = 0
        ctrl_comp1_top1_count = 0
        ctrl_comp2_top1_count = 0

        for ctrl in controls:
            cid = ctrl["id"]
            country = ctrl["country"]
            c_city = ctrl.get("city") or ctrl.get("capital_city")

            ev_comp1 = evaluate_prompt_comprehensive(model, tokenizer, ctrl["prompts"]["comp_p1"], c_city, "")
            ev_comp2 = evaluate_prompt_comprehensive(model, tokenizer, ctrl["prompts"]["comp_p2"], c_city, "")
            ev_qa1 = evaluate_prompt_comprehensive(model, tokenizer, ctrl["prompts"]["qa_p1"], c_city, "")
            ev_conv = evaluate_prompt_comprehensive(model, tokenizer, ctrl["prompts"]["qa_conv"], c_city, "")

            if ev_comp1["is_top1"]:
                ctrl_comp1_top1_count += 1
            if ev_comp2["is_top1"]:
                ctrl_comp2_top1_count += 1
            if ev_qa1["is_top1"]:
                ctrl_qa1_top1_count += 1
            if ev_conv["is_top1"]:
                ctrl_conv_top1_count += 1

            control_results[cid] = {
                "country": country,
                "capital_city": c_city,
                "comp_p1_rank": ev_comp1["target_head_rank"],
                "comp_p2_rank": ev_comp2["target_head_rank"],
                "qa_p1_rank": ev_qa1["target_head_rank"],
                "qa_conv_rank": ev_conv["target_head_rank"],
                "qa_p1_margin": ev_qa1["margin_vs_comp"]
            }
        print(f"  Protected Controls Retention: QA1 = {ctrl_qa1_top1_count}/10 | Conv = {ctrl_conv_top1_count}/10 | Comp1 = {ctrl_comp1_top1_count}/10 | Comp2 = {ctrl_comp2_top1_count}/10")

        # -------------------------------------------------------------
        # Benchmark 4: 37 Safeguard Probes
        # -------------------------------------------------------------
        safe_top1_matches = 0
        max_probe_logit_diff = 0.0
        for prompt in safeguard_probes:
            inp = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[0, -1, :].to(torch.float64)
            top_tok = tokenizer.decode([torch.argmax(logits).item()])
            # baseline comparison
            target_layer.weight.data.copy_(w0_pristine.to(torch.float32))
            with torch.no_grad():
                out_base = model(**inp)
            logits_base = out_base.logits[0, -1, :].to(torch.float64)
            top_tok_base = tokenizer.decode([torch.argmax(logits_base).item()])
            target_layer.weight.data.copy_(W_surg)

            if top_tok == top_tok_base:
                safe_top1_matches += 1
            diff = torch.max(torch.abs(logits - logits_base)).item()
            max_probe_logit_diff = max(max_probe_logit_diff, diff)

        print(f"  Safeguard Probes Invariance: {safe_top1_matches}/37 Top-1 Matches | Max Logit Diff: {max_probe_logit_diff:.2e}")

        # -------------------------------------------------------------
        # Benchmark 5: Brazil -> São Paulo Multi-token Sequence
        # -------------------------------------------------------------
        brazil_fact = next(f for f in facts if f["id"] == "F9_Brazil")
        prompt_br = brazil_fact["eval_prompts"]["qa_direct_1"]
        gen_tokens = []
        gen_ids = tokenizer(prompt_br, return_tensors="pt")["input_ids"]
        for _ in range(10):
            with torch.no_grad():
                out_gen = model(input_ids=gen_ids)
                next_tok = torch.argmax(out_gen.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
                gen_ids = torch.cat([gen_ids, next_tok], dim=1)
                gen_tokens.append(next_tok.item())
                if next_tok.item() == tokenizer.eos_token_id:
                    break
        brazil_emitted_text = tokenizer.decode(gen_tokens)
        print(f"  Brazil Multi-Token Emitted Sequence: \"{brazil_emitted_text}\"")

        # -------------------------------------------------------------
        # Benchmark 6: 30 Rollout Damping Prompts Stability
        # -------------------------------------------------------------
        loop_count = 0
        for fact in facts:
            for rk in ["rollout_r1", "rollout_r2", "rollout_r3"]:
                p_str = fact["rollout_prompts"][rk]
                inp = tokenizer(p_str, return_tensors="pt")
                gen_ids = inp["input_ids"]
                with torch.no_grad():
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
        print(f"  Rollout Damping Loops: {loop_count}/30 (Stability: {((30 - loop_count) / 30) * 100:.1f}%)")

    finally:
        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

    return {
        "arm_name": arm_name,
        "perplexity": calib_ppl,
        "delta_perplexity": delta_ppl,
        "f4_benchmark": {
            "top1_matches_out_of_70": f4_matches,
            "qa1_top1_count": f4_qa1_top1,
            "evaluations": f4_eval_results
        },
        "heldout_100_benchmark": {
            "total_top1": total_heldout_top1,
            "family_accuracy": family_accuracy,
            "family_mean_margins": {fam: float(np.mean(family_margins[fam])) for fam in EVAL_FAMILIES},
            "facts": fact_eval_results
        },
        "controls_retention": {
            "qa1_top1_count": ctrl_qa1_top1_count,
            "qa_conv_top1_count": ctrl_conv_top1_count,
            "comp1_top1_count": ctrl_comp1_top1_count,
            "comp2_top1_count": ctrl_comp2_top1_count,
            "controls": control_results
        },
        "safeguards": {
            "top1_matches_out_of_37": safe_top1_matches,
            "max_probe_logit_diff": max_probe_logit_diff
        },
        "brazil_generation": {
            "prompt": prompt_br,
            "emitted_text": brazil_emitted_text,
            "success": "São Paulo" in brazil_emitted_text
        },
        "rollout_stability": {
            "loop_count": loop_count,
            "stability_ratio": (30 - loop_count) / 30
        }
    }


def main():
    output_dir = "Experiment-10B-Prompt-Diverse-Generalization"
    manifest_c_path = os.path.join(output_dir, "exp10b_phase_d_dataset_manifest.json")
    f4_manifest_path = "Experiment-10A-Ten-Fact-Scaling/ten_fact_dataset_manifest.json"
    f4_results_path = "Experiment-10A-Ten-Fact-Scaling/ten_fact_phase_f4_results.json"

    print("=" * 80)
    print("EXPERIMENT 10B PHASE E: CLOSED-FORM ARM C SURGERY & COMPARATIVE EVALUATION")
    print(f"Model: {MODEL_ID} | Layer {LAYER_IDX} mlp.down_proj | Precision: float64")
    print("=" * 80)

    # 1. Load Manifests
    with open(manifest_c_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    with open(f4_manifest_path, "r", encoding="utf-8") as f:
        f4_manifest_data = json.load(f)
    with open(f4_results_path, "r", encoding="utf-8") as f:
        f4_results_data = json.load(f)

    facts = manifest_data["facts"]
    controls = manifest_data["controls"]
    safeguard_probes = manifest_data["safeguard_probes"]
    beta_f4 = [f4_results_data["facts"][f["id"]]["calibrated_beta_f4"] for f in facts]

    # 2. Model & Pristine Tensors
    print("\n[Step 1] Loading Model and Preparing Pristine Tensors...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    target_layer = model.model.layers[LAYER_IDX].mlp.down_proj
    w0_pristine = target_layer.weight.detach().clone().to(torch.float64)
    w0_norm = torch.linalg.norm(w0_pristine).item()
    w0_hash = compute_tensor_hash(w0_pristine)
    print(f"  Pristine W0 norm: {w0_norm:.6f} | SHA256: {w0_hash}")

    base_ppl = compute_perplexity(model, tokenizer, PPL_PROMPTS)
    print(f"  Baseline Perplexity: {base_ppl:.4f}")

    # 3. Construct Target Matrix D and Verify Gate 1
    unembed_weight = model.lm_head.weight.detach().to(torch.float64)
    D_targets = torch.zeros((896, NUM_FACTS), dtype=torch.float64)
    for f_idx, fact in enumerate(facts):
        t_tokens = extract_city_token_ids(tokenizer, fact["target_city"])
        o_tokens = extract_city_token_ids(tokenizer, fact["original_city"])
        d_raw = unembed_weight[t_tokens[0]] - unembed_weight[o_tokens[0]]
        D_targets[:, f_idx] = d_raw / torch.linalg.norm(d_raw)

    d_hash = compute_tensor_hash(D_targets)
    EXPECTED_D_HASH = "bf5c23a84544c4266ce722ed4df15a6a6b04c90e6585ed94409878b8fc53e38b"
    print(f"\n[Gate 1] Target Matrix D SHA256 Verification:")
    print(f"  Computed: {d_hash}")
    print(f"  Expected: {EXPECTED_D_HASH}")
    assert d_hash == EXPECTED_D_HASH, f"Target Matrix D SHA256 mismatch! {d_hash} vs {EXPECTED_D_HASH}"
    print("  [Pass Gate 1] Target Matrix D is verified bitwise identical to Phase C baseline and Phase F4.")

    # 4. Capture Safeguard Basis U_S (Rank 31)
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
    u_s_hash = compute_tensor_hash(U_S)
    print(f"  Safeguard Basis U_S: rank {U_S.shape[1]}/{K_S.shape[1]} | SHA256: {u_s_hash}")

    # 5. Capture Rollout Keys (30 keys)
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

    # 6. Capture Training Keys for Arms A, B, and C
    print("\n[Step 2] Capturing Training Keys for Arms A, B, and C...")
    # Arm A
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

    # Arm B
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

    # Arm C
    arm_c_keys = []
    with torch.no_grad():
        for fact in facts:
            fid = fact["id"]
            for k_name in ["K1_canonical_qa", "K2_conversational", "K3_declarative", "K4_contextual_bridge"]:
                inputs = tokenizer(manifest_data["arm_c_keys"][fid][k_name], return_tensors="pt")
                model(**inputs)
                arm_c_keys.append(hook.captured_input[0, -1, :].to(torch.float64))
    K_C = torch.stack(arm_c_keys, dim=1)
    K_C_proj = K_C - U_S @ (U_S.T @ K_C)
    K_C_proj = K_C_proj - U_S @ (U_S.T @ K_C_proj)
    hook.remove()

    # 7. Build Joint Systems and Reconcile Gate 2 (Conditioning)
    print("\n[Gate 2] Conditioning Reconciliation (Unregularized vs Regularized +0.10 I):")
    X_a, Y_a = build_joint_system(facts, D_targets, K_A_proj, R_rollout_proj, beta_f4)
    X_b, Y_b = build_joint_system(facts, D_targets, K_B_proj, R_rollout_proj, beta_f4)
    X_c, Y_c = build_joint_system(facts, D_targets, K_C_proj, R_rollout_proj, beta_f4)

    def get_spectra(X, K):
        G_fact = (K.T @ K).cpu().numpy()
        eig_fact = np.linalg.eigvalsh(G_fact)[::-1]
        G_joint = (X.T @ X).cpu().numpy()
        eig_joint = np.linalg.eigvalsh(G_joint)[::-1]
        return {
            "fact_unreg": float(eig_fact[0] / max(1e-12, eig_fact[-1])),
            "fact_reg": float((eig_fact[0] + LAMBDA_REG) / (eig_fact[-1] + LAMBDA_REG)),
            "joint_unreg": float(eig_joint[0] / max(1e-12, eig_joint[-1])),
            "joint_reg": float((eig_joint[0] + LAMBDA_REG) / (eig_joint[-1] + LAMBDA_REG)),
            "joint_lambda_max": float(eig_joint[0]),
            "joint_lambda_min": float(eig_joint[-1])
        }

    spec_a = get_spectra(X_a, K_A_proj)
    spec_b = get_spectra(X_b, K_B_proj)
    spec_c = get_spectra(X_c, K_C_proj)

    print(f"  {'Arm':8s} | {'Fact Unreg':12s} | {'Fact Reg':12s} | {'Joint Unreg':12s} | {'Joint Reg (Solver)':18s}")
    print("  " + "-" * 72)
    print(f"  {'Arm A':8s} | {spec_a['fact_unreg']:12.2f} | {spec_a['fact_reg']:12.2f} | {spec_a['joint_unreg']:12.2f} | {spec_a['joint_reg']:18.2f}")
    print(f"  {'Arm B':8s} | {spec_b['fact_unreg']:12.2f} | {spec_b['fact_reg']:12.2f} | {spec_b['joint_unreg']:12.2f} | {spec_b['joint_reg']:18.2f}")
    print(f"  {'Arm C':8s} | {spec_c['fact_unreg']:12.2f} | {spec_c['fact_reg']:12.2f} | {spec_c['joint_unreg']:12.2f} | {spec_c['joint_reg']:18.2f}")
    print("  [Pass Gate 2] Conditioning reconciled. Solver inverts regularized joint Gram with kappa = 4064.59.")

    # 8. Solve Arm A, Arm B, and Arm C
    print("\n" + "=" * 80)
    print("STAGE E1: SOLVING WEIGHT SURGERY (ARMS A, B, C)")
    print("=" * 80)

    # Solve Arm A
    t0 = time.time()
    Delta_W_a, info_a = solve_batch_trec_ls(X_a, Y_a, U_S, lam=LAMBDA_REG)
    t_a = time.time() - t0
    drift_norm_a = torch.linalg.norm(Delta_W_a).item()
    print(f"  Arm A Solved in {t_a:.4f}s | Drift: {drift_norm_a:.6f} ({drift_norm_a/w0_norm*100:.4f}%) | Leakage: {info_a['leakage']:.2e}")

    # Solve Arm B
    t0 = time.time()
    Delta_W_b, info_b = solve_batch_trec_ls(X_b, Y_b, U_S, lam=LAMBDA_REG)
    t_b = time.time() - t0
    drift_norm_b = torch.linalg.norm(Delta_W_b).item()
    print(f"  Arm B Solved in {t_b:.4f}s | Drift: {drift_norm_b:.6f} ({drift_norm_b/w0_norm*100:.4f}%) | Leakage: {info_b['leakage']:.2e}")

    # Solve Arm C
    t0 = time.time()
    Delta_W_c, info_c = solve_batch_trec_ls(X_c, Y_c, U_S, lam=LAMBDA_REG)
    t_c = time.time() - t0
    drift_norm_c = torch.linalg.norm(Delta_W_c).item()
    print(f"  Arm C Solved in {t_c:.4f}s | Drift: {drift_norm_c:.6f} ({drift_norm_c/w0_norm*100:.4f}%) | Leakage: {info_c['leakage']:.2e}")

    # Verify Dual Solver Equivalence for Arm C
    Delta_W_c_rls = solve_streaming_woodbury_rls(X_c, Y_c, U_S, lam=LAMBDA_REG, reverse=False)
    err_rls_c = torch.linalg.norm(Delta_W_c - Delta_W_c_rls).item()
    print(f"  Arm C Dual Solver Discrepancy (Batch LS vs Streaming Woodbury RLS): {err_rls_c:.2e}")
    assert err_rls_c < 1e-12, f"Arm C dual solver equivalence failed: {err_rls_c}"

    # 9. Compute Detailed Training Residuals (Fact and Prompt Level)
    print("\n[Step 3] Computing Detailed Fact and Prompt-Level Training Residuals...")
    tot_res_a, fact_res_a = compute_prompt_residuals(Delta_W_a, facts, D_targets, K_A_proj, beta_f4)
    tot_res_b, fact_res_b = compute_prompt_residuals(Delta_W_b, facts, D_targets, K_B_proj, beta_f4)
    tot_res_c, fact_res_c = compute_prompt_residuals(Delta_W_c, facts, D_targets, K_C_proj, beta_f4)

    full_sys_res_a = torch.linalg.norm(Delta_W_a @ X_a - Y_a).item()
    full_sys_res_b = torch.linalg.norm(Delta_W_b @ X_b - Y_b).item()
    full_sys_res_c = torch.linalg.norm(Delta_W_c @ X_c - Y_c).item()

    print(f"  Full System Residual ||Delta W X - Y||_F:")
    print(f"    Arm A: {full_sys_res_a:.6f} | Arm B: {full_sys_res_b:.6f} | Arm C: {full_sys_res_c:.6f}")
    print(f"  Fact Keys Residual ||Delta W K - Y_fact||_F:")
    print(f"    Arm A: {tot_res_a:.6f} | Arm B: {tot_res_b:.6f} | Arm C: {tot_res_c:.6f}")

    training_residuals_payload = {
        "metadata": {
            "model": MODEL_ID,
            "layer": LAYER_IDX,
            "lambda_reg": LAMBDA_REG,
            "gamma_operating": GAMMA_OPERATING,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "aggregate": {
            "arm_a": {"full_system_residual": full_sys_res_a, "fact_keys_residual": tot_res_a, "drift_norm": drift_norm_a},
            "arm_b": {"full_system_residual": full_sys_res_b, "fact_keys_residual": tot_res_b, "drift_norm": drift_norm_b},
            "arm_c": {"full_system_residual": full_sys_res_c, "fact_keys_residual": tot_res_c, "drift_norm": drift_norm_c}
        },
        "per_fact": {
            "arm_a": fact_res_a,
            "arm_b": fact_res_b,
            "arm_c": fact_res_c
        }
    }
    with open(os.path.join(output_dir, "exp10b_phase_e_training_residuals.json"), "w") as f:
        json.dump(training_residuals_payload, f, indent=2)

    # 10. Evaluate Arm A, Arm B, and Arm C across Full Benchmark Suite
    f4_manifest_facts = f4_manifest_data["facts"]

    eval_a = evaluate_arm_full("Arm A (Original Distribution)", model, tokenizer, target_layer, w0_pristine,
                               Delta_W_a, facts, f4_manifest_facts, f4_results_data, controls, safeguard_probes, base_ppl)
    eval_b = evaluate_arm_full("Arm B (Prompt-Diverse Distribution)", model, tokenizer, target_layer, w0_pristine,
                               Delta_W_b, facts, f4_manifest_facts, f4_results_data, controls, safeguard_probes, base_ppl)
    eval_c = evaluate_arm_full("Arm C (Structured Diversity Distribution)", model, tokenizer, target_layer, w0_pristine,
                               Delta_W_c, facts, f4_manifest_facts, f4_results_data, controls, safeguard_probes, base_ppl)

    # 11. Save Comprehensive Results
    results_payload = {
        "metadata": {
            "experiment": "Experiment 10B Phase E: Tri-Arm Weight Surgery Comparison",
            "model": MODEL_ID,
            "layer": LAYER_IDX,
            "lambda_reg": LAMBDA_REG,
            "gamma_operating": GAMMA_OPERATING,
            "safeguard_rank": 31,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gates_passed": ["Gate 1: Target Matrix D SHA256", "Gate 2: Conditioning Reconciliation"]
        },
        "spectra": {
            "arm_a": spec_a,
            "arm_b": spec_b,
            "arm_c": spec_c
        },
        "arm_a": eval_a,
        "arm_b": eval_b,
        "arm_c": eval_c
    }
    with open(os.path.join(output_dir, "exp10b_phase_e_results_arm_c.json"), "w") as f:
        json.dump(results_payload, f, indent=2)

    # Summary CSV
    csv_path = os.path.join(output_dir, "exp10b_phase_e_comparison_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Arm A (Original)", "Arm B (Prompt-Diverse)", "Arm C (Structured Diversity)"])
        writer.writerow(["Drift Norm ||Delta W||_F", f"{drift_norm_a:.6f}", f"{drift_norm_b:.6f}", f"{drift_norm_c:.6f}"])
        writer.writerow(["Drift Ratio (%)", f"{drift_norm_a/w0_norm*100:.4f}%", f"{drift_norm_b/w0_norm*100:.4f}%", f"{drift_norm_c/w0_norm*100:.4f}%"])
        writer.writerow(["Safeguard Null-Space Leakage", f"{info_a['leakage']:.2e}", f"{info_b['leakage']:.2e}", f"{info_c['leakage']:.2e}"])
        writer.writerow(["Full System Training Residual", f"{full_sys_res_a:.6f}", f"{full_sys_res_b:.6f}", f"{full_sys_res_c:.6f}"])
        writer.writerow(["Fact Keys Training Residual", f"{tot_res_a:.6f}", f"{tot_res_b:.6f}", f"{tot_res_c:.6f}"])
        writer.writerow(["Joint Gram Condition (Reg)", f"{spec_a['joint_reg']:.2f}", f"{spec_b['joint_reg']:.2f}", f"{spec_c['joint_reg']:.2f}"])
        writer.writerow(["F4 70-Prompt Benchmark Top-1", f"{eval_a['f4_benchmark']['top1_matches_out_of_70']}/70", f"{eval_b['f4_benchmark']['top1_matches_out_of_70']}/70", f"{eval_c['f4_benchmark']['top1_matches_out_of_70']}/70"])
        writer.writerow(["F4 QA1 Direct Top-1", f"{eval_a['f4_benchmark']['qa1_top1_count']}/10", f"{eval_b['f4_benchmark']['qa1_top1_count']}/10", f"{eval_c['f4_benchmark']['qa1_top1_count']}/10"])
        writer.writerow(["Held-Out 100-Prompt Generalization", f"{eval_a['heldout_100_benchmark']['total_top1']}/100", f"{eval_b['heldout_100_benchmark']['total_top1']}/100", f"{eval_c['heldout_100_benchmark']['total_top1']}/100"])
        for fam in EVAL_FAMILIES:
            writer.writerow([f"  - Held-out {fam}", f"{eval_a['heldout_100_benchmark']['family_accuracy'][fam]}/20", f"{eval_b['heldout_100_benchmark']['family_accuracy'][fam]}/20", f"{eval_c['heldout_100_benchmark']['family_accuracy'][fam]}/20"])
        writer.writerow(["Protected Controls QA1 Top-1", f"{eval_a['controls_retention']['qa1_top1_count']}/10", f"{eval_b['controls_retention']['qa1_top1_count']}/10", f"{eval_c['controls_retention']['qa1_top1_count']}/10"])
        writer.writerow(["Protected Controls Conv Top-1", f"{eval_a['controls_retention']['qa_conv_top1_count']}/10", f"{eval_b['controls_retention']['qa_conv_top1_count']}/10", f"{eval_c['controls_retention']['qa_conv_top1_count']}/10"])
        writer.writerow(["Protected Controls Comp1 Top-1", f"{eval_a['controls_retention']['comp1_top1_count']}/10", f"{eval_b['controls_retention']['comp1_top1_count']}/10", f"{eval_c['controls_retention']['comp1_top1_count']}/10"])
        writer.writerow(["Safeguard Invariance (37 probes)", f"{eval_a['safeguards']['top1_matches_out_of_37']}/37", f"{eval_b['safeguards']['top1_matches_out_of_37']}/37", f"{eval_c['safeguards']['top1_matches_out_of_37']}/37"])
        writer.writerow(["Brazil Sequence Success", str(eval_a['brazil_generation']['success']), str(eval_b['brazil_generation']['success']), str(eval_c['brazil_generation']['success'])])
        writer.writerow(["Rollout Loops (out of 30)", f"{eval_a['rollout_stability']['loop_count']}/30", f"{eval_b['rollout_stability']['loop_count']}/30", f"{eval_c['rollout_stability']['loop_count']}/30"])

    print(f"\n[Step 4] Saved deliverables:")
    print(f"  - {os.path.join(output_dir, 'exp10b_phase_e_results_arm_c.json')}")
    print(f"  - {os.path.join(output_dir, 'exp10b_phase_e_training_residuals.json')}")
    print(f"  - {csv_path}")
    print("\n" + "=" * 80)
    print("EXPERIMENT 10B PHASE E COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

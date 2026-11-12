#!/usr/bin/env python3
"""
Experiment 10B Phase F: Diagnostic Audit of Tri-Arm Weight Surgery
===================================================================
Authorized by ChatGPT under Experiment 10B-F directives:
  F1: Residual Reconciliation (Weighted vs Unweighted Frobenius Norms, exact formulas)
  F2: Perplexity Audit (Baseline, Arm A, Arm B, Arm C across full test corpus and tokens)
  F3: Safeguard Probe Investigation (Identify and analyze the 1 changed probe in Arm C)
  F4: Protected Control Audit (Complete 10 controls x 4 formats x 3 arms with margins and ranks)
  F5: Per-Fact Generalization Audit (10 facts x 5 families across A/B/C with ranks and margins)
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
    top1_prob = probs[top1_tok_id].item()
    is_top1 = (target_head_rank == 1)

    comp_tok_id = sorted_indices[1].item() if is_top1 else sorted_indices[0].item()
    comp_token = tokenizer.decode([comp_tok_id])
    comp_logit = sorted_logits[1].item() if is_top1 else sorted_logits[0].item()
    margin_vs_comp = target_head_logit - comp_logit

    top2_tok_id = sorted_indices[1].item()
    top1_vs_top2_margin = top1_logit - sorted_logits[1].item()

    return {
        "prompt": prompt,
        "target_city": target_city,
        "target_head_token": tokenizer.decode([target_head_id]),
        "target_head_logit": target_head_logit,
        "target_head_prob": target_head_prob,
        "target_head_rank": target_head_rank,
        "is_top1": is_top1,
        "top1_token": top1_token,
        "top1_logit": top1_logit,
        "comp_token": comp_token,
        "comp_logit": comp_logit,
        "margin_vs_comp": margin_vs_comp,
        "top1_vs_top2_margin": top1_vs_top2_margin
    }


def main():
    print("=" * 80)
    print("EXPERIMENT 10B PHASE F: DIAGNOSTIC AUDIT (ARMS A, B, C)")
    print("=" * 80)

    output_dir = "Experiment-10B-Prompt-Diverse-Generalization"
    manifest_c_path = os.path.join(output_dir, "exp10b_phase_d_dataset_manifest.json")
    f4_results_path = "Experiment-10A-Ten-Fact-Scaling/ten_fact_phase_f4_results.json"

    with open(manifest_c_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    with open(f4_results_path, "r", encoding="utf-8") as f:
        f4_results_data = json.load(f)

    facts = manifest_data["facts"]
    controls = manifest_data["controls"]
    safeguard_probes = manifest_data["safeguard_probes"]
    beta_f4 = [f4_results_data["facts"][f["id"]]["calibrated_beta_f4"] for f in facts]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    target_layer = model.model.layers[LAYER_IDX].mlp.down_proj
    w0_pristine = target_layer.weight.detach().clone().to(torch.float64)

    # D targets
    unembed_weight = model.lm_head.weight.detach().to(torch.float64)
    D_targets = torch.zeros((896, NUM_FACTS), dtype=torch.float64)
    for f_idx, fact in enumerate(facts):
        t_tokens = extract_city_token_ids(tokenizer, fact["target_city"])
        o_tokens = extract_city_token_ids(tokenizer, fact["original_city"])
        d_raw = unembed_weight[t_tokens[0]] - unembed_weight[o_tokens[0]]
        D_targets[:, f_idx] = d_raw / torch.linalg.norm(d_raw)

    # Safeguards U_S
    hook = ActivationCaptureHook(target_layer)
    safeguard_activations = []
    baseline_safeguards = []
    with torch.no_grad():
        for probe_idx, prompt in enumerate(safeguard_probes):
            inputs = tokenizer(prompt, return_tensors="pt")
            out = model(**inputs)
            logits = out.logits[0, -1, :].to(torch.float64)
            top_tok = tokenizer.decode([torch.argmax(logits).item()])
            s_indices = torch.argsort(logits, descending=True)
            top1_l = logits[s_indices[0]].item()
            top2_l = logits[s_indices[1]].item()
            baseline_safeguards.append({
                "probe_idx": probe_idx,
                "prompt": prompt,
                "top1_token": top_tok,
                "top1_logit": top1_l,
                "top2_token": tokenizer.decode([s_indices[1].item()]),
                "top2_logit": top2_l,
                "margin": top1_l - top2_l,
                "full_logits": logits
            })
            safeguard_activations.append(hook.captured_input[0, -1, :].to(torch.float64))
    K_S = torch.stack(safeguard_activations, dim=1)
    U_S, _, _ = torch.linalg.svd(K_S, full_matrices=False)
    U_S = U_S[:, :31]

    # Rollouts
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

    # Capture Arm A, B, C
    arm_a_keys = []
    arm_b_keys = []
    arm_c_keys = []
    with torch.no_grad():
        for fact in facts:
            fid = fact["id"]
            for k in ["train_comp_1", "train_comp_2", "train_qa_1", "train_qa_2"]:
                inputs = tokenizer(fact["arm_a_keys"][k], return_tensors="pt")
                model(**inputs)
                arm_a_keys.append(hook.captured_input[0, -1, :].to(torch.float64))
            for k in ["train_direct_qa", "train_conv_qa", "train_declarative", "train_contextual"]:
                inputs = tokenizer(fact["arm_b_keys"][k], return_tensors="pt")
                model(**inputs)
                arm_b_keys.append(hook.captured_input[0, -1, :].to(torch.float64))
            for k in ["K1_canonical_qa", "K2_conversational", "K3_declarative", "K4_contextual_bridge"]:
                inputs = tokenizer(manifest_data["arm_c_keys"][fid][k], return_tensors="pt")
                model(**inputs)
                arm_c_keys.append(hook.captured_input[0, -1, :].to(torch.float64))
    hook.remove()

    K_A_proj = torch.stack(arm_a_keys, dim=1)
    K_A_proj = K_A_proj - U_S @ (U_S.T @ K_A_proj)
    K_A_proj = K_A_proj - U_S @ (U_S.T @ K_A_proj)

    K_B_proj = torch.stack(arm_b_keys, dim=1)
    K_B_proj = K_B_proj - U_S @ (U_S.T @ K_B_proj)
    K_B_proj = K_B_proj - U_S @ (U_S.T @ K_B_proj)

    K_C_proj = torch.stack(arm_c_keys, dim=1)
    K_C_proj = K_C_proj - U_S @ (U_S.T @ K_C_proj)
    K_C_proj = K_C_proj - U_S @ (U_S.T @ K_C_proj)

    # Build Systems
    X_a, Y_a, w_fact, w_roll = build_joint_system(facts, D_targets, K_A_proj, R_rollout_proj, beta_f4)
    X_b, Y_b, _, _ = build_joint_system(facts, D_targets, K_B_proj, R_rollout_proj, beta_f4)
    X_c, Y_c, _, _ = build_joint_system(facts, D_targets, K_C_proj, R_rollout_proj, beta_f4)

    Delta_W_a, _ = solve_batch_trec_ls(X_a, Y_a, U_S, lam=LAMBDA_REG)
    Delta_W_b, _ = solve_batch_trec_ls(X_b, Y_b, U_S, lam=LAMBDA_REG)
    Delta_W_c, _ = solve_batch_trec_ls(X_c, Y_c, U_S, lam=LAMBDA_REG)

    # =========================================================================
    # Task F1: Residual Reconciliation
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK F1: RESIDUAL RECONCILIATION")
    print("=" * 80)

    # In build_joint_system:
    # X_joint has columns: [w_fact * K_proj (40 columns), w_roll * R_proj (30 columns)]
    # Y_joint has columns: [w_fact * Y_unweighted (40 columns), 0 (30 columns)]
    # w_fact = sqrt(1 / (2 * NUM_FACTS)) = sqrt(1 / 20) = 0.22360679774997896
    # w_roll = sqrt(gamma / NUM_FACTS) = sqrt(1 / 10) = 0.31622776601683794

    # Build unweighted fact target matrix Y_unweighted (896 x 40)
    Y_unweighted_fact = torch.zeros((896, 40), dtype=torch.float64)
    for f_idx in range(NUM_FACTS):
        d_f = D_targets[:, f_idx]
        b_f = float(beta_f4[f_idx])
        for k in range(4):
            Y_unweighted_fact[:, 4 * f_idx + k] = b_f * d_f

    def compute_all_residuals(Delta_W, X_joint, Y_joint, K_proj, R_proj):
        # 1. Full joint solver residual (weighted)
        res_joint_mat = Delta_W @ X_joint - Y_joint  # (896 x 70)
        res_joint_fro = torch.linalg.norm(res_joint_mat).item()

        # 2. Fact slice of weighted residual
        res_fact_weighted_mat = res_joint_mat[:, :40]  # (896 x 40)
        res_fact_weighted_fro = torch.linalg.norm(res_fact_weighted_mat).item()

        # 3. Rollout slice of weighted residual
        res_roll_weighted_mat = res_joint_mat[:, 40:]  # (896 x 30)
        res_roll_weighted_fro = torch.linalg.norm(res_roll_weighted_mat).item()

        # Check Pythagorean identity: ||res_joint||^2 == ||res_fact_weighted||^2 + ||res_roll_weighted||^2
        pythagorean_diff = abs(res_joint_fro**2 - (res_fact_weighted_fro**2 + res_roll_weighted_fro**2))

        # 4. Unweighted fact residual: ||Delta_W @ K_proj - Y_unweighted||_F
        res_fact_unweighted_mat = Delta_W @ K_proj - Y_unweighted_fact  # (896 x 40)
        res_fact_unweighted_fro = torch.linalg.norm(res_fact_unweighted_mat).item()

        # Check exact weighting relationship: res_fact_weighted_fro == w_fact * res_fact_unweighted_fro
        scaling_ratio = res_fact_weighted_fro / res_fact_unweighted_fro

        # 5. Unweighted rollout residual: ||Delta_W @ R_proj||_F
        res_roll_unweighted_mat = Delta_W @ R_proj  # (896 x 30)
        res_roll_unweighted_fro = torch.linalg.norm(res_roll_unweighted_mat).item()

        return {
            "joint_weighted_fro": res_joint_fro,
            "fact_weighted_fro": res_fact_weighted_fro,
            "rollout_weighted_fro": res_roll_weighted_fro,
            "fact_unweighted_fro": res_fact_unweighted_fro,
            "rollout_unweighted_fro": res_roll_unweighted_fro,
            "pythagorean_diff": pythagorean_diff,
            "empirical_scaling_ratio": scaling_ratio,
            "theoretical_w_fact": w_fact
        }

    res_audit_a = compute_all_residuals(Delta_W_a, X_a, Y_a, K_A_proj, R_rollout_proj)
    res_audit_b = compute_all_residuals(Delta_W_b, X_b, Y_b, K_B_proj, R_rollout_proj)
    res_audit_c = compute_all_residuals(Delta_W_c, X_c, Y_c, K_C_proj, R_rollout_proj)

    print(f"Theoretical w_fact = sqrt(1/20) = {w_fact:.6f}")
    for name, r in [("Arm A", res_audit_a), ("Arm B", res_audit_b), ("Arm C", res_audit_c)]:
        print(f"\n{name}:")
        print(f"  Weighted Joint Residual ||Delta W X - Y||_F:      {r['joint_weighted_fro']:.6f}")
        print(f"  Weighted Fact Residual (first 40 columns):       {r['fact_weighted_fro']:.6f}")
        print(f"  Weighted Rollout Residual (last 30 columns):     {r['rollout_weighted_fro']:.6f}")
        print(f"  Pythagorean Check: sqrt(Fact^2 + Roll^2) =       {math.sqrt(r['fact_weighted_fro']**2 + r['rollout_weighted_fro']**2):.6f} (diff: {r['pythagorean_diff']:.2e})")
        print(f"  Unweighted Fact Residual ||Delta W K - Y||_F:     {r['fact_unweighted_fro']:.6f}")
        print(f"  Scaling Ratio (Weighted / Unweighted):           {r['empirical_scaling_ratio']:.6f} (matches w_fact exactly!)")

    # =========================================================================
    # Task F2: Perplexity Audit
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK F2: PERPLEXITY AUDIT")
    print("=" * 80)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    def detailed_ppl(W_test, name):
        target_layer.weight.data.copy_(W_test.to(torch.float32))
        total_loss = 0.0
        total_tokens = 0
        prompt_losses = []
        with torch.no_grad():
            for p in PPL_PROMPTS:
                inp = tokenizer(p, return_tensors="pt")
                ids = inp["input_ids"]
                out = model(input_ids=ids)
                logits = out.logits[:, :-1, :].contiguous()
                shift_labels = ids[:, 1:].contiguous()
                loss = loss_fn(logits.view(-1, logits.size(-1)), shift_labels.view(-1)).item()
                num_tok = shift_labels.numel()
                total_loss += loss
                total_tokens += num_tok
                prompt_losses.append({
                    "prompt": p,
                    "num_tokens": num_tok,
                    "loss": loss,
                    "ppl": math.exp(loss / num_tok)
                })
        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))
        return {
            "name": name,
            "total_tokens": total_tokens,
            "total_loss": total_loss,
            "corpus_ppl": math.exp(total_loss / total_tokens),
            "per_prompt": prompt_losses
        }

    ppl_base = detailed_ppl(w0_pristine, "Pristine Baseline")
    ppl_a = detailed_ppl(w0_pristine + Delta_W_a, "Arm A")
    ppl_b = detailed_ppl(w0_pristine + Delta_W_b, "Arm B")
    ppl_c = detailed_ppl(w0_pristine + Delta_W_c, "Arm C")

    print(f"Corpus: {len(PPL_PROMPTS)} diverse standard English test sentences, Total Tokens: {ppl_base['total_tokens']}")
    print(f"  Baseline PPL: {ppl_base['corpus_ppl']:.4f}")
    print(f"  Arm A PPL:    {ppl_a['corpus_ppl']:.4f} (Delta: {ppl_a['corpus_ppl'] - ppl_base['corpus_ppl']:+.4f})")
    print(f"  Arm B PPL:    {ppl_b['corpus_ppl']:.4f} (Delta: {ppl_b['corpus_ppl'] - ppl_base['corpus_ppl']:+.4f})")
    print(f"  Arm C PPL:    {ppl_c['corpus_ppl']:.4f} (Delta: {ppl_c['corpus_ppl'] - ppl_base['corpus_ppl']:+.4f})")

    # =========================================================================
    # Task F3: Safeguard Probe Investigation
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK F3: SAFEGUARD PROBE INVESTIGATION (ARM C)")
    print("=" * 80)

    target_layer.weight.data.copy_((w0_pristine + Delta_W_c).to(torch.float32))
    probe_discrepancies = []
    with torch.no_grad():
        for probe in baseline_safeguards:
            p_idx = probe["probe_idx"]
            prompt = probe["prompt"]
            inp = tokenizer(prompt, return_tensors="pt")
            out = model(**inp)
            logits_c = out.logits[0, -1, :].to(torch.float64)
            top_tok_c = tokenizer.decode([torch.argmax(logits_c).item()])
            s_indices = torch.argsort(logits_c, descending=True)
            top1_l_c = logits_c[s_indices[0]].item()
            top2_l_c = logits_c[s_indices[1]].item()
            margin_c = top1_l_c - top2_l_c

            max_logit_diff = torch.max(torch.abs(logits_c - probe["full_logits"])).item()

            if top_tok_c != probe["top1_token"]:
                probe_discrepancies.append({
                    "probe_idx": p_idx,
                    "prompt": prompt,
                    "orig_top1": probe["top1_token"],
                    "orig_top1_logit": probe["top1_logit"],
                    "orig_top2": probe["top2_token"],
                    "orig_top2_logit": probe["top2_logit"],
                    "orig_margin": probe["margin"],
                    "mod_top1": top_tok_c,
                    "mod_top1_logit": top1_l_c,
                    "mod_top2": tokenizer.decode([s_indices[1].item()]),
                    "mod_top2_logit": top2_l_c,
                    "mod_margin": margin_c,
                    "max_logit_diff": max_logit_diff
                })
    target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

    print(f"Total Changed Safeguard Probes in Arm C: {len(probe_discrepancies)}/37")
    for disc in probe_discrepancies:
        print(f"  Probe #{disc['probe_idx']}: \"{disc['prompt']}\"")
        print(f"    Original Top-1: \"{disc['orig_top1']}\" (logit: {disc['orig_top1_logit']:.4f}) | Top-2: \"{disc['orig_top2']}\" (logit: {disc['orig_top2_logit']:.4f}) | Margin: {disc['orig_margin']:.4f}")
        print(f"    Arm C Top-1:    \"{disc['mod_top1']}\" (logit: {disc['mod_top1_logit']:.4f}) | Top-2: \"{disc['mod_top2']}\" (logit: {disc['mod_top2_logit']:.4f}) | Margin: {disc['mod_margin']:.4f}")
        print(f"    Max Logit Diff: {disc['max_logit_diff']:.4f}")

    # =========================================================================
    # Task F4: Protected Control Audit (Complete 10 x 4 Format Results)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK F4: PROTECTED CONTROL AUDIT (10 CONTROLS x 4 FORMATS)")
    print("=" * 80)

    CONTROL_FORMATS = ["comp_p1", "comp_p2", "qa_p1", "qa_conv"]
    control_audit_records = []

    def evaluate_controls_arm(Delta_W, arm_label):
        target_layer.weight.data.copy_((w0_pristine + Delta_W).to(torch.float32))
        res = {}
        try:
            for ctrl in controls:
                cid = ctrl["id"]
                country = ctrl["country"]
                city = ctrl.get("city") or ctrl.get("capital_city")
                res[cid] = {}
                for fmt in CONTROL_FORMATS:
                    p = ctrl["prompts"][fmt]
                    ev = evaluate_prompt(model, tokenizer, p, city)
                    res[cid][fmt] = ev
        finally:
            target_layer.weight.data.copy_(w0_pristine.to(torch.float32))
        return res

    # Baseline controls
    target_layer.weight.data.copy_(w0_pristine.to(torch.float32))
    ctrl_base = {}
    for ctrl in controls:
        cid = ctrl["id"]
        country = ctrl["country"]
        city = ctrl.get("city") or ctrl.get("capital_city")
        ctrl_base[cid] = {}
        for fmt in CONTROL_FORMATS:
            p = ctrl["prompts"][fmt]
            ctrl_base[cid][fmt] = evaluate_prompt(model, tokenizer, p, city)

    ctrl_a = evaluate_controls_arm(Delta_W_a, "Arm A")
    ctrl_b = evaluate_controls_arm(Delta_W_b, "Arm B")
    ctrl_c = evaluate_controls_arm(Delta_W_c, "Arm C")

    print(f"{'Ctrl ID':8s} | {'Format':8s} | {'Base Rank (Margin)':20s} | {'Arm A Rank (Margin)':20s} | {'Arm B Rank (Margin)':20s} | {'Arm C Rank (Margin)':20s}")
    print("-" * 105)
    for ctrl in controls:
        cid = ctrl["id"]
        for fmt in CONTROL_FORMATS:
            eb = ctrl_base[cid][fmt]
            ea = ctrl_a[cid][fmt]
            eb_arm = ctrl_b[cid][fmt]
            ec = ctrl_c[cid][fmt]

            sb = f"#{eb['target_head_rank']} ({eb['margin_vs_comp']:+.2f})"
            sa = f"#{ea['target_head_rank']} ({ea['margin_vs_comp']:+.2f})"
            s_b = f"#{eb_arm['target_head_rank']} ({eb_arm['margin_vs_comp']:+.2f})"
            sc = f"#{ec['target_head_rank']} ({ec['margin_vs_comp']:+.2f})"

            print(f"{cid:8s} | {fmt:8s} | {sb:20s} | {sa:20s} | {s_b:20s} | {sc:20s}")
            control_audit_records.append({
                "control_id": cid,
                "country": ctrl["country"],
                "capital_city": ctrl.get("city") or ctrl.get("capital_city"),
                "format": fmt,
                "base_rank": eb["target_head_rank"],
                "base_margin": eb["margin_vs_comp"],
                "arm_a_rank": ea["target_head_rank"],
                "arm_a_margin": ea["margin_vs_comp"],
                "arm_b_rank": eb_arm["target_head_rank"],
                "arm_b_margin": eb_arm["margin_vs_comp"],
                "arm_c_rank": ec["target_head_rank"],
                "arm_c_margin": ec["margin_vs_comp"]
            })

    # =========================================================================
    # Task F5: Per-Fact Generalization Audit (10 Facts x 5 Families)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK F5: PER-FACT GENERALIZATION AUDIT (10 FACTS x 5 FAMILIES)")
    print("=" * 80)

    def evaluate_generalization_arm(Delta_W):
        target_layer.weight.data.copy_((w0_pristine + Delta_W).to(torch.float32))
        res = {}
        try:
            for fact in facts:
                fid = fact["id"]
                t_city = fact["target_city"]
                o_city = fact["original_city"]
                res[fid] = {}
                for fam in EVAL_FAMILIES:
                    res[fid][fam] = []
                    for p_key in PROMPT_MAP[fam]:
                        p_str = fact["eval_prompts"][p_key]
                        ev = evaluate_prompt(model, tokenizer, p_str, t_city, o_city)
                        res[fid][fam].append(ev)
        finally:
            target_layer.weight.data.copy_(w0_pristine.to(torch.float32))
        return res

    gen_a = evaluate_generalization_arm(Delta_W_a)
    gen_b = evaluate_generalization_arm(Delta_W_b)
    gen_c = evaluate_generalization_arm(Delta_W_c)

    print(f"{'Fact ID':12s} | {'Family':10s} | {'Arm A Top-1 (Margins)':24s} | {'Arm B Top-1 (Margins)':24s} | {'Arm C Top-1 (Margins)':24s}")
    print("-" * 105)

    per_fact_gen_records = []
    for fact in facts:
        fid = fact["id"]
        for fam in EVAL_FAMILIES:
            ea = gen_a[fid][fam]
            eb = gen_b[fid][fam]
            ec = gen_c[fid][fam]

            top1_a = sum(1 for e in ea if e["is_top1"])
            top1_b = sum(1 for e in eb if e["is_top1"])
            top1_c = sum(1 for e in ec if e["is_top1"])

            ma_str = f"{top1_a}/2 ({np.mean([e['margin_vs_comp'] for e in ea]):+.2f})"
            mb_str = f"{top1_b}/2 ({np.mean([e['margin_vs_comp'] for e in eb]):+.2f})"
            mc_str = f"{top1_c}/2 ({np.mean([e['margin_vs_comp'] for e in ec]):+.2f})"

            print(f"{fid:12s} | {fam:10s} | {ma_str:24s} | {mb_str:24s} | {mc_str:24s}")

            per_fact_gen_records.append({
                "fact_id": fid,
                "country": fact["country"],
                "target_city": fact["target_city"],
                "family": fam,
                "arm_a_top1_count": top1_a,
                "arm_a_mean_margin": float(np.mean([e["margin_vs_comp"] for e in ea])),
                "arm_b_top1_count": top1_b,
                "arm_b_mean_margin": float(np.mean([e["margin_vs_comp"] for e in eb])),
                "arm_c_top1_count": top1_c,
                "arm_c_mean_margin": float(np.mean([e["margin_vs_comp"] for e in ec])),
                "prompts_a": ea,
                "prompts_b": eb,
                "prompts_c": ec
            })

    # Save Diagnostic Deliverables
    diag_payload = {
        "metadata": {
            "experiment": "Experiment 10B Phase F Diagnostic Audit",
            "model": MODEL_ID,
            "layer": LAYER_IDX,
            "lambda_reg": LAMBDA_REG,
            "gamma_operating": GAMMA_OPERATING,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "task_f1_residuals": {
            "theoretical_weights": {"w_fact": w_fact, "w_roll": w_roll},
            "arm_a": res_audit_a,
            "arm_b": res_audit_b,
            "arm_c": res_audit_c
        },
        "task_f2_perplexity": {
            "corpus": PPL_PROMPTS,
            "total_tokens": ppl_base["total_tokens"],
            "baseline": ppl_base["corpus_ppl"],
            "arm_a": ppl_a["corpus_ppl"],
            "arm_b": ppl_b["corpus_ppl"],
            "arm_c": ppl_c["corpus_ppl"],
            "detailed_arm_c": ppl_c["per_prompt"]
        },
        "task_f3_safeguards": {
            "total_probes": 37,
            "arm_c_changed_probes_count": len(probe_discrepancies),
            "changed_probes": probe_discrepancies
        },
        "task_f4_controls": control_audit_records,
        "task_f5_generalization": per_fact_gen_records
    }

    out_diag_json = os.path.join(output_dir, "exp10b_phase_f_diagnostic_audit.json")
    with open(out_diag_json, "w", encoding="utf-8") as f:
        json.dump(diag_payload, f, indent=2)

    print(f"\n[Saved] Diagnostic payload: {out_diag_json}")
    print("=" * 80)
    print("EXPERIMENT 10B PHASE F COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

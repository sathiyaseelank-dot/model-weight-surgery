#!/usr/bin/env python3
"""
Experiment 10B Phase G: Generalization and Robustness Diagnostic Audit
=====================================================================
Authorized by ChatGPT under Experiment 10B-G directives (Diagnostic only; no new surgery):
  G1: Conversational Margin Analysis (10 facts x 2 prompts x Baseline/A/B/C)
      - Distinguish m_{target-original} from m_{target-top_competitor}
      - Target rank, emitted top-1 token, verification of claimed 8-fact margin improvement
  G2: Full Protected-Control Margins (10 controls x 4 formats x Baseline/A/B/C)
      - All 40 control prompts
      - Raw margins m_{ctrl} = z(target) - max_{j != target} z(j)
      - Margin changes relative to pristine W_0: Delta m = m_{surg} - m_{base}
  G3: Larger PPL Evaluation (>= 10,000 tokens on fixed standard corpus)
      - Wikitext-2 validation corpus (10,240 tokens, 20 chunks of 512)
      - Baseline PPL, Arm A PPL, Arm B PPL, Arm C PPL, Delta PPL, % Delta PPL
      - Corpus and tokenizer details
  G4: Arm C Failure Analysis (Declarative drop: 30% -> 5%)
      - 10 facts x 2 declarative prompts across Baseline/A/B/C
      - Per-fact margins and target ranks
      - Activation cosine similarities between training keys and evaluation prompts
      - Determine whether drop is concentrated in particular facts or systematic
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

CONTROL_FORMATS = ["qa_p1", "qa_conv", "comp_p1", "comp_p2"]
CONTROL_FORMAT_NAMES = {
    "qa_p1": "QA direct",
    "qa_conv": "Conversational",
    "comp_p1": "Completion 1",
    "comp_p2": "Completion 2"
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


def evaluate_prompt_detailed(model, tokenizer, prompt: str, target_city: str, original_city: str = "") -> Dict[str, Any]:
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

    # Top competitor: the highest logit token that is NOT the target token
    if is_top1:
        comp_tok_id = sorted_indices[1].item()
        comp_logit = sorted_logits[1].item()
    else:
        comp_tok_id = sorted_indices[0].item()
        comp_logit = sorted_logits[0].item()
    comp_token = tokenizer.decode([comp_tok_id])

    # m_{target-top_competitor} = z(target) - max_{j != target} z(j)
    m_target_top_competitor = target_head_logit - comp_logit

    # m_{target-original} = z(target) - z(original)
    m_target_original = None
    orig_head_logit = None
    orig_token_str = ""
    if original_city:
        orig_tokens = extract_city_token_ids(tokenizer, original_city)
        orig_head_id = orig_tokens[0]
        orig_head_logit = logits[orig_head_id].item()
        orig_token_str = tokenizer.decode([orig_head_id])
        m_target_original = target_head_logit - orig_head_logit

    return {
        "prompt": prompt,
        "target_city": target_city,
        "target_token": tokenizer.decode([target_head_id]),
        "target_head_id": target_head_id,
        "target_head_logit": target_head_logit,
        "target_head_prob": target_head_prob,
        "target_head_rank": target_head_rank,
        "is_top1": is_top1,
        "top1_token": top1_token,
        "top1_tok_id": top1_tok_id,
        "top1_logit": top1_logit,
        "comp_token": comp_token,
        "comp_tok_id": comp_tok_id,
        "comp_logit": comp_logit,
        "m_target_top_competitor": m_target_top_competitor,
        "original_city": original_city,
        "orig_token": orig_token_str,
        "orig_logit": orig_head_logit,
        "m_target_original": m_target_original
    }


def compute_corpus_ppl(model, tokens: List[int], chunk_size: int = 512) -> Dict[str, Any]:
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    num_chunks = 0
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(tokens) - chunk_size + 1, chunk_size):
            chunk = torch.tensor([tokens[i:i + chunk_size]])
            out = model(chunk)
            logits = out.logits[:, :-1, :].contiguous()
            targets = chunk[:, 1:].contiguous()
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1)).item()
            num_tok = targets.numel()
            total_loss += loss
            total_tokens += num_tok
            num_chunks += 1

    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    elapsed = time.time() - t0
    return {
        "total_tokens": total_tokens,
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "total_loss": total_loss,
        "nll_per_token": total_loss / total_tokens,
        "ppl": ppl,
        "elapsed_seconds": elapsed
    }


def main():
    print("=" * 80)
    print("EXPERIMENT 10B PHASE G: GENERALIZATION & ROBUSTNESS AUDIT")
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

    # 1. Target directions D
    unembed_weight = model.lm_head.weight.detach().to(torch.float64)
    D_targets = torch.zeros((896, NUM_FACTS), dtype=torch.float64)
    for f_idx, fact in enumerate(facts):
        t_tokens = extract_city_token_ids(tokenizer, fact["target_city"])
        o_tokens = extract_city_token_ids(tokenizer, fact["original_city"])
        d_raw = unembed_weight[t_tokens[0]] - unembed_weight[o_tokens[0]]
        D_targets[:, f_idx] = d_raw / torch.linalg.norm(d_raw)

    # 2. Safeguards basis U_S
    hook = ActivationCaptureHook(target_layer)
    safeguard_activations = []
    with torch.no_grad():
        for probe_idx, prompt in enumerate(safeguard_probes):
            inputs = tokenizer(prompt, return_tensors="pt")
            model(**inputs)
            safeguard_activations.append(hook.captured_input[0, -1, :].to(torch.float64))
    K_S = torch.stack(safeguard_activations, dim=1)
    U_S, _, _ = torch.linalg.svd(K_S, full_matrices=False)
    U_S = U_S[:, :31]

    # 3. Rollouts R
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

    # 4. Capture Training Keys for Arms A, B, C
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

    K_A_raw = torch.stack(arm_a_keys, dim=1)  # (4864, 40)
    K_B_raw = torch.stack(arm_b_keys, dim=1)  # (4864, 40)
    K_C_raw = torch.stack(arm_c_keys, dim=1)  # (4864, 40)

    K_A_proj = K_A_raw - U_S @ (U_S.T @ K_A_raw)
    K_A_proj = K_A_proj - U_S @ (U_S.T @ K_A_proj)

    K_B_proj = K_B_raw - U_S @ (U_S.T @ K_B_raw)
    K_B_proj = K_B_proj - U_S @ (U_S.T @ K_B_proj)

    K_C_proj = K_C_raw - U_S @ (U_S.T @ K_C_raw)
    K_C_proj = K_C_proj - U_S @ (U_S.T @ K_C_proj)

    # 5. Capture Evaluation Keys for Declarative and Conversational
    dec_eval_keys = []
    conv_eval_keys = []
    with torch.no_grad():
        for fact in facts:
            for dk in ["dec_eval_1", "dec_eval_2"]:
                inputs = tokenizer(fact["eval_prompts"][dk], return_tensors="pt")
                model(**inputs)
                dec_eval_keys.append(hook.captured_input[0, -1, :].to(torch.float64))
            for ck in ["qa_conv_1", "qa_conv_2"]:
                inputs = tokenizer(fact["eval_prompts"][ck], return_tensors="pt")
                model(**inputs)
                conv_eval_keys.append(hook.captured_input[0, -1, :].to(torch.float64))
    hook.remove()

    K_dec_raw = torch.stack(dec_eval_keys, dim=1)   # (4864, 20)
    K_conv_raw = torch.stack(conv_eval_keys, dim=1) # (4864, 20)

    # Build Systems and Solve for Delta W
    X_a, Y_a, _, _ = build_joint_system(facts, D_targets, K_A_proj, R_rollout_proj, beta_f4)
    X_b, Y_b, _, _ = build_joint_system(facts, D_targets, K_B_proj, R_rollout_proj, beta_f4)
    X_c, Y_c, _, _ = build_joint_system(facts, D_targets, K_C_proj, R_rollout_proj, beta_f4)

    Delta_W_a, _ = solve_batch_trec_ls(X_a, Y_a, U_S, lam=LAMBDA_REG)
    Delta_W_b, _ = solve_batch_trec_ls(X_b, Y_b, U_S, lam=LAMBDA_REG)
    Delta_W_c, _ = solve_batch_trec_ls(X_c, Y_c, U_S, lam=LAMBDA_REG)

    zero_w = torch.zeros_like(Delta_W_a)
    models_dict = {
        "Baseline": zero_w,
        "Arm A": Delta_W_a,
        "Arm B": Delta_W_b,
        "Arm C": Delta_W_c
    }

    # =========================================================================
    # TASK G1: CONVERSATIONAL MARGIN ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK G1: CONVERSATIONAL MARGIN ANALYSIS (10 FACTS x 2 PROMPTS x 4 MODELS)")
    print("=" * 80)

    g1_results = {}
    g1_csv_rows = []

    for model_name, d_w in models_dict.items():
        target_layer.weight.data.copy_((w0_pristine + d_w).to(torch.float32))
        g1_results[model_name] = {}
        for f_idx, fact in enumerate(facts):
            fid = fact["id"]
            country = fact["country"]
            t_city = fact["target_city"]
            o_city = fact["original_city"]
            g1_results[model_name][fid] = {}

            for p_key in ["qa_conv_1", "qa_conv_2"]:
                p_str = fact["eval_prompts"][p_key]
                ev = evaluate_prompt_detailed(model, tokenizer, p_str, t_city, o_city)
                g1_results[model_name][fid][p_key] = ev
                g1_csv_rows.append({
                    "model": model_name,
                    "fact_id": fid,
                    "country": country,
                    "prompt_key": p_key,
                    "prompt_text": p_str,
                    "target_city": t_city,
                    "target_token": ev["target_token"],
                    "original_city": o_city,
                    "orig_token": ev["orig_token"],
                    "top1_token": ev["top1_token"],
                    "target_rank": ev["target_head_rank"],
                    "is_top1": ev["is_top1"],
                    "m_target_top_competitor": ev["m_target_top_competitor"],
                    "m_target_original": ev["m_target_original"],
                    "target_logit": ev["target_head_logit"],
                    "comp_token": ev["comp_token"],
                    "comp_logit": ev["comp_logit"],
                    "orig_logit": ev["orig_logit"]
                })
        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

    # Print summary table of Conversational Margins across models
    print(f"{'Fact ID':10s} | {'Prompt':9s} | {'Base Top1':12s} | {'Arm A Top1':12s} | {'Arm B Top1':12s} | {'Arm C Top1':12s} | {'m_comp(Base)':12s} | {'m_comp(A)':12s} | {'m_comp(B)':12s} | {'m_comp(C)':12s}")
    print("-" * 135)
    for fact in facts:
        fid = fact["id"]
        for p_key in ["qa_conv_1", "qa_conv_2"]:
            eb = g1_results["Baseline"][fid][p_key]
            ea = g1_results["Arm A"][fid][p_key]
            eb_arm = g1_results["Arm B"][fid][p_key]
            ec = g1_results["Arm C"][fid][p_key]
            print(f"{fid:10s} | {p_key:9s} | {eb['top1_token'][:10]:12s} | {ea['top1_token'][:10]:12s} | {eb_arm['top1_token'][:10]:12s} | {ec['top1_token'][:10]:12s} | {eb['m_target_top_competitor']:+11.2f} | {ea['m_target_top_competitor']:+11.2f} | {eb_arm['m_target_top_competitor']:+11.2f} | {ec['m_target_top_competitor']:+11.2f}")

    # Per-fact margin improvement audit: Does Arm C improve margins over Arm A and Arm B across 8/10 facts?
    per_fact_conv_audit = {}
    print("\n--- Per-Fact Conversational Summary & Margin Verification ---")
    print(f"{'Fact ID':10s} | {'Country':12s} | {'Arm A Top1':10s} | {'Arm B Top1':10s} | {'Arm C Top1':10s} | {'m_comp A':9s} | {'m_comp B':9s} | {'m_comp C':9s} | {'C vs B (m_comp)':16s} | {'C vs B (m_orig)':16s}")
    print("-" * 130)

    facts_improved_comp_vs_b = 0
    facts_improved_orig_vs_b = 0
    facts_improved_comp_vs_a = 0
    facts_improved_orig_vs_a = 0

    for fact in facts:
        fid = fact["id"]
        country = fact["country"]
        
        # Averages across the 2 conversational prompts
        m_comp_base = np.mean([g1_results["Baseline"][fid][p]["m_target_top_competitor"] for p in ["qa_conv_1", "qa_conv_2"]])
        m_comp_a = np.mean([g1_results["Arm A"][fid][p]["m_target_top_competitor"] for p in ["qa_conv_1", "qa_conv_2"]])
        m_comp_b = np.mean([g1_results["Arm B"][fid][p]["m_target_top_competitor"] for p in ["qa_conv_1", "qa_conv_2"]])
        m_comp_c = np.mean([g1_results["Arm C"][fid][p]["m_target_top_competitor"] for p in ["qa_conv_1", "qa_conv_2"]])

        m_orig_base = np.mean([g1_results["Baseline"][fid][p]["m_target_original"] for p in ["qa_conv_1", "qa_conv_2"]])
        m_orig_a = np.mean([g1_results["Arm A"][fid][p]["m_target_original"] for p in ["qa_conv_1", "qa_conv_2"]])
        m_orig_b = np.mean([g1_results["Arm B"][fid][p]["m_target_original"] for p in ["qa_conv_1", "qa_conv_2"]])
        m_orig_c = np.mean([g1_results["Arm C"][fid][p]["m_target_original"] for p in ["qa_conv_1", "qa_conv_2"]])

        top1_a = sum(1 for p in ["qa_conv_1", "qa_conv_2"] if g1_results["Arm A"][fid][p]["is_top1"])
        top1_b = sum(1 for p in ["qa_conv_1", "qa_conv_2"] if g1_results["Arm B"][fid][p]["is_top1"])
        top1_c = sum(1 for p in ["qa_conv_1", "qa_conv_2"] if g1_results["Arm C"][fid][p]["is_top1"])

        diff_comp_c_vs_b = m_comp_c - m_comp_b
        diff_orig_c_vs_b = m_orig_c - m_orig_b
        diff_comp_c_vs_a = m_comp_c - m_comp_a
        diff_orig_c_vs_a = m_orig_c - m_orig_a

        if diff_comp_c_vs_b > 0:
            facts_improved_comp_vs_b += 1
        if diff_orig_c_vs_b > 0:
            facts_improved_orig_vs_b += 1
        if diff_comp_c_vs_a > 0:
            facts_improved_comp_vs_a += 1
        if diff_orig_c_vs_a > 0:
            facts_improved_orig_vs_a += 1

        per_fact_conv_audit[fid] = {
            "country": country,
            "top1_a": f"{top1_a}/2",
            "top1_b": f"{top1_b}/2",
            "top1_c": f"{top1_c}/2",
            "mean_m_comp_base": float(m_comp_base),
            "mean_m_comp_a": float(m_comp_a),
            "mean_m_comp_b": float(m_comp_b),
            "mean_m_comp_c": float(m_comp_c),
            "mean_m_orig_base": float(m_orig_base),
            "mean_m_orig_a": float(m_orig_a),
            "mean_m_orig_b": float(m_orig_b),
            "mean_m_orig_c": float(m_orig_c),
            "diff_comp_c_vs_b": float(diff_comp_c_vs_b),
            "diff_orig_c_vs_b": float(diff_orig_c_vs_b),
            "diff_comp_c_vs_a": float(diff_comp_c_vs_a),
            "diff_orig_c_vs_a": float(diff_orig_c_vs_a),
        }
        print(f"{fid:10s} | {country:12s} | {top1_a}/2        | {top1_b}/2        | {top1_c}/2        | {m_comp_a:+8.2f} | {m_comp_b:+8.2f} | {m_comp_c:+8.2f} | {diff_comp_c_vs_b:+15.2f} | {diff_orig_c_vs_b:+15.2f}")

    print(f"\nVerification Results:")
    print(f"  Facts where Arm C improved m_comp vs Arm B: {facts_improved_comp_vs_b}/10")
    print(f"  Facts where Arm C improved m_orig vs Arm B: {facts_improved_orig_vs_b}/10")
    print(f"  Facts where Arm C improved m_comp vs Arm A: {facts_improved_comp_vs_a}/10")
    print(f"  Facts where Arm C improved m_orig vs Arm A: {facts_improved_orig_vs_a}/10")

    # =========================================================================
    # TASK G2: FULL PROTECTED-CONTROL MARGINS
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK G2: FULL PROTECTED-CONTROL MARGINS (10 CONTROLS x 4 FORMATS = 40 EVALS)")
    print("=" * 80)

    g2_results = {}
    g2_csv_rows = []

    for model_name, d_w in models_dict.items():
        target_layer.weight.data.copy_((w0_pristine + d_w).to(torch.float32))
        g2_results[model_name] = {}
        for ctrl in controls:
            cid = ctrl["id"]
            country = ctrl["country"]
            city = ctrl.get("city") or ctrl.get("capital_city")
            g2_results[model_name][cid] = {}
            for fmt in CONTROL_FORMATS:
                p_str = ctrl["prompts"][fmt]
                ev = evaluate_prompt_detailed(model, tokenizer, p_str, city)
                g2_results[model_name][cid][fmt] = ev
        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

    # Build CSV and delta margin relative to pristine W0
    ctrl_summary = {}
    for fmt in CONTROL_FORMATS:
        ctrl_summary[fmt] = {
            "name": CONTROL_FORMAT_NAMES[fmt],
            "correct_base": 0, "correct_a": 0, "correct_b": 0, "correct_c": 0,
            "mean_m_base": [], "mean_m_a": [], "mean_m_b": [], "mean_m_c": [],
            "delta_m_a": [], "delta_m_b": [], "delta_m_c": []
        }

    for ctrl in controls:
        cid = ctrl["id"]
        country = ctrl["country"]
        city = ctrl.get("city") or ctrl.get("capital_city")
        for fmt in CONTROL_FORMATS:
            eb = g2_results["Baseline"][cid][fmt]
            ea = g2_results["Arm A"][cid][fmt]
            eb_arm = g2_results["Arm B"][cid][fmt]
            ec = g2_results["Arm C"][cid][fmt]

            m_base = eb["m_target_top_competitor"]
            m_a = ea["m_target_top_competitor"]
            m_b = eb_arm["m_target_top_competitor"]
            m_c = ec["m_target_top_competitor"]

            dm_a = m_a - m_base
            dm_b = m_b - m_base
            dm_c = m_c - m_base

            if eb["is_top1"]: ctrl_summary[fmt]["correct_base"] += 1
            if ea["is_top1"]: ctrl_summary[fmt]["correct_a"] += 1
            if eb_arm["is_top1"]: ctrl_summary[fmt]["correct_b"] += 1
            if ec["is_top1"]: ctrl_summary[fmt]["correct_c"] += 1

            ctrl_summary[fmt]["mean_m_base"].append(m_base)
            ctrl_summary[fmt]["mean_m_a"].append(m_a)
            ctrl_summary[fmt]["mean_m_b"].append(m_b)
            ctrl_summary[fmt]["mean_m_c"].append(m_c)

            ctrl_summary[fmt]["delta_m_a"].append(dm_a)
            ctrl_summary[fmt]["delta_m_b"].append(dm_b)
            ctrl_summary[fmt]["delta_m_c"].append(dm_c)

            g2_csv_rows.append({
                "control_id": cid,
                "country": country,
                "capital_city": city,
                "format_key": fmt,
                "format_name": CONTROL_FORMAT_NAMES[fmt],
                "base_top1": eb["top1_token"],
                "base_rank": eb["target_head_rank"],
                "base_margin": m_base,
                "arm_a_top1": ea["top1_token"],
                "arm_a_rank": ea["target_head_rank"],
                "arm_a_margin": m_a,
                "arm_a_delta_m": dm_a,
                "arm_b_top1": eb_arm["top1_token"],
                "arm_b_rank": eb_arm["target_head_rank"],
                "arm_b_margin": m_b,
                "arm_b_delta_m": dm_b,
                "arm_c_top1": ec["top1_token"],
                "arm_c_rank": ec["target_head_rank"],
                "arm_c_margin": m_c,
                "arm_c_delta_m": dm_c
            })

    print(f"{'Format':16s} | {'Base Acc':10s} | {'Arm A Acc':10s} | {'Arm B Acc':10s} | {'Arm C Acc':10s} | {'Mean Delta m (A)':18s} | {'Mean Delta m (B)':18s} | {'Mean Delta m (C)':18s}")
    print("-" * 125)
    total_base, total_a, total_b, total_c = 0, 0, 0, 0
    all_dm_a, all_dm_b, all_dm_c = [], [], []
    for fmt in CONTROL_FORMATS:
        s = ctrl_summary[fmt]
        total_base += s["correct_base"]
        total_a += s["correct_a"]
        total_b += s["correct_b"]
        total_c += s["correct_c"]
        all_dm_a.extend(s["delta_m_a"])
        all_dm_b.extend(s["delta_m_b"])
        all_dm_c.extend(s["delta_m_c"])
        print(f"{s['name']:16s} | {s['correct_base']}/10      | {s['correct_a']}/10      | {s['correct_b']}/10      | {s['correct_c']}/10      | {np.mean(s['delta_m_a']):+17.4f} | {np.mean(s['delta_m_b']):+17.4f} | {np.mean(s['delta_m_c']):+17.4f}")
    print("-" * 125)
    print(f"{'OVERALL (40)':16s} | {total_base}/40      | {total_a}/40      | {total_b}/40      | {total_c}/40      | {np.mean(all_dm_a):+17.4f} | {np.mean(all_dm_b):+17.4f} | {np.mean(all_dm_c):+17.4f}")
    print(f"Max absolute margin shift on controls: Arm A = {np.max(np.abs(all_dm_a)):.4f}, Arm B = {np.max(np.abs(all_dm_b)):.4f}, Arm C = {np.max(np.abs(all_dm_c)):.4f}")

    # =========================================================================
    # TASK G3: LARGER PPL EVALUATION (>= 10,000 TOKENS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK G3: LARGER PPL EVALUATION (>= 10,000 TOKENS ON WIKITEXT-2 VALID)")
    print("=" * 80)

    with open(WIKITEXT_PATH, "r", encoding="utf-8") as f:
        wiki_text = f.read()

    full_wiki_tokens = tokenizer(wiki_text, add_special_tokens=False)["input_ids"]
    ppl_tokens = full_wiki_tokens[:PPL_TOKEN_LIMIT]
    print(f"Evaluation Corpus: Wikitext-2 raw validation slice")
    print(f"Tokenizer: {MODEL_ID} (vocabulary: {tokenizer.vocab_size})")
    print(f"Total tokens selected: {len(ppl_tokens)} | Chunk size: {PPL_CHUNK_SIZE} | Chunks: {len(ppl_tokens) // PPL_CHUNK_SIZE}")

    ppl_results = {}
    for model_name, d_w in models_dict.items():
        print(f"Evaluating PPL for {model_name}...")
        target_layer.weight.data.copy_((w0_pristine + d_w).to(torch.float32))
        res = compute_corpus_ppl(model, ppl_tokens, chunk_size=PPL_CHUNK_SIZE)
        ppl_results[model_name] = res
        print(f"  -> {model_name} PPL: {res['ppl']:.4f} (NLL: {res['nll_per_token']:.4f}, elapsed: {res['elapsed_seconds']:.2f}s)")
    target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

    base_ppl = ppl_results["Baseline"]["ppl"]
    print("\n--- Summary of Large-Scale Perplexity Audit ---")
    print(f"{'Model State':15s} | {'Tokens':8s} | {'PPL':10s} | {'Delta PPL':12s} | {'% Delta PPL':14s} | {'Eval Time (s)':14s}")
    print("-" * 85)
    for model_name in ["Baseline", "Arm A", "Arm B", "Arm C"]:
        p = ppl_results[model_name]["ppl"]
        dp = p - base_ppl
        pct_dp = (dp / base_ppl) * 100.0
        el = ppl_results[model_name]["elapsed_seconds"]
        toks = ppl_results[model_name]["total_tokens"]
        print(f"{model_name:15s} | {toks:<8d} | {p:10.4f} | {dp:+11.4f} | {pct_dp:+13.2f}% | {el:13.2f}s")

    # =========================================================================
    # TASK G4: ARM C DECLARATIVE FAILURE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK G4: ARM C DECLARATIVE FAILURE ANALYSIS (DROP 30% -> 5%)")
    print("=" * 80)

    g4_results = {}
    g4_csv_rows = []

    for model_name, d_w in models_dict.items():
        target_layer.weight.data.copy_((w0_pristine + d_w).to(torch.float32))
        g4_results[model_name] = {}
        for f_idx, fact in enumerate(facts):
            fid = fact["id"]
            country = fact["country"]
            t_city = fact["target_city"]
            o_city = fact["original_city"]
            g4_results[model_name][fid] = {}
            for p_key in ["dec_eval_1", "dec_eval_2"]:
                p_str = fact["eval_prompts"][p_key]
                ev = evaluate_prompt_detailed(model, tokenizer, p_str, t_city, o_city)
                g4_results[model_name][fid][p_key] = ev
        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

    # Activation Cosine Similarity Analysis
    # K_A_raw: (4864, 40), K_B_raw: (4864, 40), K_C_raw: (4864, 40)
    # K_dec_raw: (4864, 20), K_conv_raw: (4864, 20)
    def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        return (F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))).item()

    activation_audit = {}
    print(f"{'Fact ID':10s} | {'Country':12s} | {'Base Top1':10s} | {'Arm A Top1':10s} | {'Arm B Top1':10s} | {'Arm C Top1':10s} | {'m_comp(A)':10s} | {'m_comp(B)':10s} | {'m_comp(C)':10s} | {'cos(dec,train_A)':18s} | {'cos(dec,train_B)':18s} | {'cos(dec,train_C)':18s}")
    print("-" * 165)

    all_cos_dec_a, all_cos_dec_b, all_cos_dec_c = [], [], []
    all_cos_conv_a, all_cos_conv_b, all_cos_conv_c = [], [], []

    for f_idx, fact in enumerate(facts):
        fid = fact["id"]
        country = fact["country"]
        t_city = fact["target_city"]
        o_city = fact["original_city"]

        # Fact training keys: 4 keys per fact
        k_a_fact = K_A_raw[:, 4 * f_idx : 4 * (f_idx + 1)] # (4864, 4)
        k_b_fact = K_B_raw[:, 4 * f_idx : 4 * (f_idx + 1)] # (4864, 4)
        k_c_fact = K_C_raw[:, 4 * f_idx : 4 * (f_idx + 1)] # (4864, 4)

        # Eval keys: 2 declarative, 2 conversational
        k_dec_1 = K_dec_raw[:, 2 * f_idx]
        k_dec_2 = K_dec_raw[:, 2 * f_idx + 1]
        k_conv_1 = K_conv_raw[:, 2 * f_idx]
        k_conv_2 = K_conv_raw[:, 2 * f_idx + 1]

        # Cosine similarity of declarative eval keys to training keys
        cos_dec_a = np.mean([[cosine_sim(kd, k_a_fact[:, i]) for i in range(4)] for kd in [k_dec_1, k_dec_2]])
        cos_dec_b = np.mean([[cosine_sim(kd, k_b_fact[:, i]) for i in range(4)] for kd in [k_dec_1, k_dec_2]])
        cos_dec_c = np.mean([[cosine_sim(kd, k_c_fact[:, i]) for i in range(4)] for kd in [k_dec_1, k_dec_2]])

        # Breakdown of Arm C keys individually (K1, K2, K3, K4)
        cos_dec_c_k1 = np.mean([cosine_sim(kd, k_c_fact[:, 0]) for kd in [k_dec_1, k_dec_2]])
        cos_dec_c_k2 = np.mean([cosine_sim(kd, k_c_fact[:, 1]) for kd in [k_dec_1, k_dec_2]])
        cos_dec_c_k3 = np.mean([cosine_sim(kd, k_c_fact[:, 2]) for kd in [k_dec_1, k_dec_2]])
        cos_dec_c_k4 = np.mean([cosine_sim(kd, k_c_fact[:, 3]) for kd in [k_dec_1, k_dec_2]])

        # Cosine similarity of conversational eval keys to training keys
        cos_conv_a = np.mean([[cosine_sim(kc, k_a_fact[:, i]) for i in range(4)] for kc in [k_conv_1, k_conv_2]])
        cos_conv_b = np.mean([[cosine_sim(kc, k_b_fact[:, i]) for i in range(4)] for kc in [k_conv_1, k_conv_2]])
        cos_conv_c = np.mean([[cosine_sim(kc, k_c_fact[:, i]) for i in range(4)] for kc in [k_conv_1, k_conv_2]])

        all_cos_dec_a.append(cos_dec_a)
        all_cos_dec_b.append(cos_dec_b)
        all_cos_dec_c.append(cos_dec_c)
        all_cos_conv_a.append(cos_conv_a)
        all_cos_conv_b.append(cos_conv_b)
        all_cos_conv_c.append(cos_conv_c)

        top1_base = sum(1 for p in ["dec_eval_1", "dec_eval_2"] if g4_results["Baseline"][fid][p]["is_top1"])
        top1_a = sum(1 for p in ["dec_eval_1", "dec_eval_2"] if g4_results["Arm A"][fid][p]["is_top1"])
        top1_b = sum(1 for p in ["dec_eval_1", "dec_eval_2"] if g4_results["Arm B"][fid][p]["is_top1"])
        top1_c = sum(1 for p in ["dec_eval_1", "dec_eval_2"] if g4_results["Arm C"][fid][p]["is_top1"])

        m_comp_a = np.mean([g4_results["Arm A"][fid][p]["m_target_top_competitor"] for p in ["dec_eval_1", "dec_eval_2"]])
        m_comp_b = np.mean([g4_results["Arm B"][fid][p]["m_target_top_competitor"] for p in ["dec_eval_1", "dec_eval_2"]])
        m_comp_c = np.mean([g4_results["Arm C"][fid][p]["m_target_top_competitor"] for p in ["dec_eval_1", "dec_eval_2"]])

        activation_audit[fid] = {
            "country": country,
            "target_city": t_city,
            "original_city": o_city,
            "top1_base": f"{top1_base}/2",
            "top1_a": f"{top1_a}/2",
            "top1_b": f"{top1_b}/2",
            "top1_c": f"{top1_c}/2",
            "mean_m_comp_a": float(m_comp_a),
            "mean_m_comp_b": float(m_comp_b),
            "mean_m_comp_c": float(m_comp_c),
            "cos_dec_train_a": float(cos_dec_a),
            "cos_dec_train_b": float(cos_dec_b),
            "cos_dec_train_c": float(cos_dec_c),
            "cos_dec_c_k1": float(cos_dec_c_k1),
            "cos_dec_c_k2": float(cos_dec_c_k2),
            "cos_dec_c_k3": float(cos_dec_c_k3),
            "cos_dec_c_k4": float(cos_dec_c_k4),
            "cos_conv_train_a": float(cos_conv_a),
            "cos_conv_train_b": float(cos_conv_b),
            "cos_conv_train_c": float(cos_conv_c),
        }

        print(f"{fid:10s} | {country:12s} | {top1_base}/2        | {top1_a}/2        | {top1_b}/2        | {top1_c}/2        | {m_comp_a:+9.2f} | {m_comp_b:+9.2f} | {m_comp_c:+9.2f} | {cos_dec_a:18.4f} | {cos_dec_b:18.4f} | {cos_dec_c:18.4f}")

        for p_key in ["dec_eval_1", "dec_eval_2"]:
            eb = g4_results["Baseline"][fid][p_key]
            ea = g4_results["Arm A"][fid][p_key]
            eb_arm = g4_results["Arm B"][fid][p_key]
            ec = g4_results["Arm C"][fid][p_key]
            g4_csv_rows.append({
                "fact_id": fid,
                "country": country,
                "prompt_key": p_key,
                "prompt_text": fact["eval_prompts"][p_key],
                "target_city": t_city,
                "original_city": o_city,
                "base_top1": eb["top1_token"],
                "base_rank": eb["target_head_rank"],
                "base_m_comp": eb["m_target_top_competitor"],
                "base_m_orig": eb["m_target_original"],
                "arm_a_top1": ea["top1_token"],
                "arm_a_rank": ea["target_head_rank"],
                "arm_a_m_comp": ea["m_target_top_competitor"],
                "arm_a_m_orig": ea["m_target_original"],
                "arm_b_top1": eb_arm["top1_token"],
                "arm_b_rank": eb_arm["target_head_rank"],
                "arm_b_m_comp": eb_arm["m_target_top_competitor"],
                "arm_b_m_orig": eb_arm["m_target_original"],
                "arm_c_top1": ec["top1_token"],
                "arm_c_rank": ec["target_head_rank"],
                "arm_c_m_comp": ec["m_target_top_competitor"],
                "arm_c_m_orig": ec["m_target_original"],
                "cos_dec_train_a": float(cos_dec_a),
                "cos_dec_train_b": float(cos_dec_b),
                "cos_dec_train_c": float(cos_dec_c),
            })

    print("-" * 165)
    print(f"Overall Mean Cosine Similarities across 10 Facts:")
    print(f"  Declarative Eval <-> Train Keys: Arm A = {np.mean(all_cos_dec_a):.4f} | Arm B = {np.mean(all_cos_dec_b):.4f} | Arm C = {np.mean(all_cos_dec_c):.4f}")
    print(f"  Conversational Eval <-> Train Keys: Arm A = {np.mean(all_cos_conv_a):.4f} | Arm B = {np.mean(all_cos_conv_b):.4f} | Arm C = {np.mean(all_cos_conv_c):.4f}")

    # Save artifacts
    # 1. exp10b_phase_g_audit.json
    audit_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_id": MODEL_ID,
        "g1_conversational_analysis": {
            "per_fact_summary": per_fact_conv_audit,
            "facts_improved_comp_vs_b": facts_improved_comp_vs_b,
            "facts_improved_orig_vs_b": facts_improved_orig_vs_b,
            "facts_improved_comp_vs_a": facts_improved_comp_vs_a,
            "facts_improved_orig_vs_a": facts_improved_orig_vs_a,
            "mean_m_comp_base": float(np.mean([r["mean_m_comp_base"] for r in per_fact_conv_audit.values()])),
            "mean_m_comp_a": float(np.mean([r["mean_m_comp_a"] for r in per_fact_conv_audit.values()])),
            "mean_m_comp_b": float(np.mean([r["mean_m_comp_b"] for r in per_fact_conv_audit.values()])),
            "mean_m_comp_c": float(np.mean([r["mean_m_comp_c"] for r in per_fact_conv_audit.values()])),
            "mean_m_orig_base": float(np.mean([r["mean_m_orig_base"] for r in per_fact_conv_audit.values()])),
            "mean_m_orig_a": float(np.mean([r["mean_m_orig_a"] for r in per_fact_conv_audit.values()])),
            "mean_m_orig_b": float(np.mean([r["mean_m_orig_b"] for r in per_fact_conv_audit.values()])),
            "mean_m_orig_c": float(np.mean([r["mean_m_orig_c"] for r in per_fact_conv_audit.values()])),
        },
        "g2_control_margins": {
            "total_evals": 40,
            "accuracy_base": f"{total_base}/40",
            "accuracy_a": f"{total_a}/40",
            "accuracy_b": f"{total_b}/40",
            "accuracy_c": f"{total_c}/40",
            "mean_delta_m_a": float(np.mean(all_dm_a)),
            "mean_delta_m_b": float(np.mean(all_dm_b)),
            "mean_delta_m_c": float(np.mean(all_dm_c)),
            "max_abs_delta_m_a": float(np.max(np.abs(all_dm_a))),
            "max_abs_delta_m_b": float(np.max(np.abs(all_dm_b))),
            "max_abs_delta_m_c": float(np.max(np.abs(all_dm_c))),
            "format_breakdown": {
                fmt: {
                    "format_name": ctrl_summary[fmt]["name"],
                    "accuracy_base": f"{ctrl_summary[fmt]['correct_base']}/10",
                    "accuracy_a": f"{ctrl_summary[fmt]['correct_a']}/10",
                    "accuracy_b": f"{ctrl_summary[fmt]['correct_b']}/10",
                    "accuracy_c": f"{ctrl_summary[fmt]['correct_c']}/10",
                    "mean_delta_m_a": float(np.mean(ctrl_summary[fmt]["delta_m_a"])),
                    "mean_delta_m_b": float(np.mean(ctrl_summary[fmt]["delta_m_b"])),
                    "mean_delta_m_c": float(np.mean(ctrl_summary[fmt]["delta_m_c"])),
                }
                for fmt in CONTROL_FORMATS
            }
        },
        "g3_large_ppl": {
            "corpus": "Salesforce/wikitext wikitext-2-raw-v1 validation set",
            "total_tokens_evaluated": ppl_results["Baseline"]["total_tokens"],
            "chunk_size": PPL_CHUNK_SIZE,
            "num_chunks": ppl_results["Baseline"]["num_chunks"],
            "tokenizer": MODEL_ID,
            "results": {
                model_name: {
                    "ppl": float(ppl_results[model_name]["ppl"]),
                    "nll_per_token": float(ppl_results[model_name]["nll_per_token"]),
                    "delta_ppl": float(ppl_results[model_name]["ppl"] - base_ppl),
                    "pct_delta_ppl": float((ppl_results[model_name]["ppl"] - base_ppl) / base_ppl * 100.0),
                    "elapsed_seconds": float(ppl_results[model_name]["elapsed_seconds"])
                }
                for model_name in ["Baseline", "Arm A", "Arm B", "Arm C"]
            }
        },
        "g4_declarative_failure_analysis": {
            "per_fact_analysis": activation_audit,
            "accuracy_declarative": {
                "Baseline": f"{sum(int(r['top1_base'].split('/')[0]) for r in activation_audit.values())}/20",
                "Arm A": f"{sum(int(r['top1_a'].split('/')[0]) for r in activation_audit.values())}/20",
                "Arm B": f"{sum(int(r['top1_b'].split('/')[0]) for r in activation_audit.values())}/20",
                "Arm C": f"{sum(int(r['top1_c'].split('/')[0]) for r in activation_audit.values())}/20",
            },
            "mean_cosine_similarities": {
                "declarative_to_train": {
                    "Arm A": float(np.mean(all_cos_dec_a)),
                    "Arm B": float(np.mean(all_cos_dec_b)),
                    "Arm C": float(np.mean(all_cos_dec_c)),
                },
                "conversational_to_train": {
                    "Arm A": float(np.mean(all_cos_conv_a)),
                    "Arm B": float(np.mean(all_cos_conv_b)),
                    "Arm C": float(np.mean(all_cos_conv_c)),
                }
            }
        }
    }

    audit_json_path = os.path.join(output_dir, "exp10b_phase_g_audit.json")
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nSaved Phase G audit JSON: {audit_json_path}")

    # 2. CSVs
    g1_csv_path = os.path.join(output_dir, "exp10b_phase_g_conversational_margins.csv")
    with open(g1_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(g1_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(g1_csv_rows)
    print(f"Saved Conversational Margins CSV: {g1_csv_path}")

    g2_csv_path = os.path.join(output_dir, "exp10b_phase_g_control_margins.csv")
    with open(g2_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(g2_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(g2_csv_rows)
    print(f"Saved Control Margins CSV: {g2_csv_path}")

    g4_csv_path = os.path.join(output_dir, "exp10b_phase_g_declarative_margins.csv")
    with open(g4_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(g4_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(g4_csv_rows)
    print(f"Saved Declarative Margins CSV: {g4_csv_path}")

    print("\nPhase G Diagnostic Audit completed successfully!")


if __name__ == "__main__":
    main()

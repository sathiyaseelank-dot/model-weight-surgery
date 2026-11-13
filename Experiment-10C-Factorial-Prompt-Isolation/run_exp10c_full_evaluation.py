#!/usr/bin/env python3
"""
Experiment 10C Phase B: Complete Factorial Weight Surgery & Multi-Benchmark Evaluation
======================================================================================
Authorized by ChatGPT under Experiment 10C Phase B directives:
  Arms:
    - Arm D1: Low Syntactic Diversity, Strong Relational Explicitness
    - Arm D2: High Syntactic Diversity, Strong Relational Explicitness
    - Arm D3: Low Syntactic Diversity, Moderate Relational Explicitness
    - Arm D4: High Syntactic Diversity, Moderate Relational Explicitness
  Reference Arms: Arm A, Arm B, Arm C.

Evaluations:
  1. Training residuals (unweighted fact residual, weighted fact residual, joint residual)
  2. 100-prompt held-out generalization suite (5 families x 2 prompts x 10 facts)
     - Top-1 accuracy
     - Target-versus-competitor margin: m_{target-top competitor}
     - Target-versus-original margin: m_{target-original}
  3. 40 protected-control evaluations (10 entities x 4 formats)
  4. 37 canonical safeguard probes
  5. Brazil full 2-token sequence evaluation ([' São', ' Paulo'])
  6. Wikitext-2 PPL on 10,220 tokens
  7. Factorial main effects & interaction: I = (D2 - D1) - (D4 - D3)
  8. Generalization Efficiency: GE = Delta m_bar_heldout / (||E_fact||_F + 1e-6)
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
    is_top1 = (target_head_rank == 1)

    comp_tok_id = sorted_indices[1].item() if is_top1 else sorted_indices[0].item()
    comp_token = tokenizer.decode([comp_tok_id])
    comp_logit = sorted_logits[1].item() if is_top1 else sorted_logits[0].item()
    m_comp = target_head_logit - comp_logit

    m_orig = None
    orig_logit = None
    orig_token = ""
    if original_city:
        orig_tokens = extract_city_token_ids(tokenizer, original_city)
        orig_head_id = orig_tokens[0]
        orig_logit = logits[orig_head_id].item()
        orig_token = tokenizer.decode([orig_head_id])
        m_orig = target_head_logit - orig_logit

    return {
        "prompt": prompt,
        "target_city": target_city,
        "target_token": tokenizer.decode([target_head_id]),
        "target_head_rank": target_head_rank,
        "is_top1": is_top1,
        "top1_token": top1_token,
        "top1_logit": top1_logit,
        "comp_token": comp_token,
        "comp_logit": comp_logit,
        "m_comp": m_comp,
        "orig_token": orig_token,
        "orig_logit": orig_logit,
        "m_orig": m_orig
    }

def compute_corpus_ppl(model, tokens: List[int], chunk_size: int = 512) -> Dict[str, Any]:
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(tokens) - chunk_size + 1, chunk_size):
            chunk = torch.tensor([tokens[i:i + chunk_size]])
            out = model(chunk)
            logits = out.logits[:, :-1, :].contiguous()
            targets = chunk[:, 1:].contiguous()
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1)).item()
            total_loss += loss
            total_tokens += targets.numel()

    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    return {
        "total_tokens": total_tokens,
        "total_loss": total_loss,
        "nll_per_token": total_loss / total_tokens,
        "ppl": ppl,
        "elapsed_seconds": time.time() - t0
    }

def evaluate_brazil_sequence(model, tokenizer, prompt: str) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    # Token 1: " São"
    with torch.no_grad():
        out1 = model(**inputs)
    logits1 = out1.logits[0, -1, :].to(torch.float64)
    tok1_id = tokenizer(" São", add_special_tokens=False)["input_ids"][0]
    rank1 = (torch.argsort(logits1, descending=True) == tok1_id).nonzero(as_tuple=True)[0].item() + 1
    top1_tok1 = tokenizer.decode([torch.argmax(logits1).item()])

    # Token 2: " Paulo" conditioned on prompt + " São"
    inputs2 = tokenizer(prompt + " São", return_tensors="pt")
    with torch.no_grad():
        out2 = model(**inputs2)
    logits2 = out2.logits[0, -1, :].to(torch.float64)
    tok2_id = tokenizer(" Paulo", add_special_tokens=False)["input_ids"][0]
    rank2 = (torch.argsort(logits2, descending=True) == tok2_id).nonzero(as_tuple=True)[0].item() + 1
    top1_tok2 = tokenizer.decode([torch.argmax(logits2).item()])

    return {
        "prompt": prompt,
        "token1_target": " São",
        "token1_rank": rank1,
        "token1_top1": top1_tok1,
        "token1_correct": (rank1 == 1),
        "token2_target": " Paulo",
        "token2_rank": rank2,
        "token2_top1": top1_tok2,
        "token2_correct": (rank2 == 1),
        "full_sequence_correct": (rank1 == 1 and rank2 == 1)
    }

def main():
    print("=" * 80)
    print("EXPERIMENT 10C PHASE B: FULL FACTORIAL SURGERY & MULTI-BENCHMARK AUDIT")
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

    # 2. Safeguards U_S
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

    # Unweighted fact target matrix Y_unweighted (896 x 40)
    Y_unweighted_fact = torch.zeros((896, 40), dtype=torch.float64)
    for f_idx in range(NUM_FACTS):
        d_f = D_targets[:, f_idx]
        b_f = float(beta_f4[f_idx])
        for k in range(4):
            Y_unweighted_fact[:, 4 * f_idx + k] = b_f * d_f

    Delta_W_dict = {}
    residuals_dict = {}
    X_joint_dict = {}
    Y_joint_dict = {}

    for a_label in arm_labels:
        K_raw = torch.stack(arm_keys[a_label], dim=1)
        K_p = K_raw - U_S @ (U_S.T @ K_raw)
        K_p = K_p - U_S @ (U_S.T @ K_p)
        X_j, Y_j, w_fact, w_roll = build_joint_system(facts, D_targets, K_p, R_proj, beta_f4)
        X_joint_dict[a_label] = X_j
        Y_joint_dict[a_label] = Y_j

        dW, diag = solve_batch_trec_ls(X_j, Y_j, U_S, lam=LAMBDA_REG)
        Delta_W_dict[a_label] = dW

        # Compute Residuals:
        # 1. Full joint weighted residual
        res_joint_mat = dW @ X_j - Y_j
        res_joint_fro = torch.linalg.norm(res_joint_mat).item()
        # 2. Fact slice of weighted residual
        res_fact_weighted_fro = torch.linalg.norm(res_joint_mat[:, :40]).item()
        # 3. Rollout slice of weighted residual
        res_roll_weighted_fro = torch.linalg.norm(res_joint_mat[:, 40:]).item()
        # 4. Unweighted fact residual
        res_fact_unweighted_mat = dW @ K_p - Y_unweighted_fact
        res_fact_unweighted_fro = torch.linalg.norm(res_fact_unweighted_mat).item()

        residuals_dict[a_label] = {
            "frobenius_norm": torch.linalg.norm(dW).item(),
            "leakage": diag["leakage"],
            "res_joint_fro": res_joint_fro,
            "res_fact_weighted_fro": res_fact_weighted_fro,
            "res_roll_weighted_fro": res_roll_weighted_fro,
            "res_fact_unweighted_fro": res_fact_unweighted_fro
        }
        print(f"Arm {a_label}: ||Delta_W||_F = {residuals_dict[a_label]['frobenius_norm']:.4f} | E_joint = {res_joint_fro:.4f} | E_fact_unweighted = {res_fact_unweighted_fro:.4f} | Leakage = {diag['leakage']:.2e}")

    # Models dict including Baseline
    zero_w = torch.zeros_like(Delta_W_dict["D1"])
    models_dict = {
        "Baseline": zero_w,
        "D1": Delta_W_dict["D1"],
        "D2": Delta_W_dict["D2"],
        "D3": Delta_W_dict["D3"],
        "D4": Delta_W_dict["D4"]
    }

    # -------------------------------------------------------------------------
    # Benchmark 1: Held-Out 100-Prompt Generalization Suite
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BENCHMARK 1: HELD-OUT 100-PROMPT GENERALIZATION SUITE")
    print("=" * 80)

    heldout_results = {}
    heldout_csv_rows = []

    for m_name, d_w in models_dict.items():
        target_layer.weight.data.copy_((w0_pristine + d_w).to(torch.float32))
        heldout_results[m_name] = {fam: [] for fam in EVAL_FAMILIES}
        for f in facts:
            fid = f["id"]
            country = f["country"]
            t_city = f["target_city"]
            o_city = f["original_city"]
            for fam in EVAL_FAMILIES:
                for pk in PROMPT_MAP[fam]:
                    p = f["eval_prompts"][pk]
                    ev = evaluate_prompt_detailed(model, tokenizer, p, t_city, o_city)
                    heldout_results[m_name][fam].append(ev)
                    if m_name != "Baseline":
                        heldout_csv_rows.append({
                            "arm": m_name,
                            "fact_id": fid,
                            "country": country,
                            "family": fam,
                            "prompt_key": pk,
                            "prompt_text": p,
                            "target_city": t_city,
                            "top1_token": ev["top1_token"],
                            "target_rank": ev["target_head_rank"],
                            "is_top1": ev["is_top1"],
                            "m_comp": ev["m_comp"],
                            "m_orig": ev["m_orig"]
                        })
        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

    print(f"{'Family':12s} | {'Baseline':12s} | {'Arm D1':12s} | {'Arm D2':12s} | {'Arm D3':12s} | {'Arm D4':12s}")
    print("-" * 75)
    total_acc = {m: 0 for m in models_dict.keys()}
    mean_m_comp_by_model = {m: [] for m in models_dict.keys()}

    for fam in EVAL_FAMILIES:
        accs = {}
        for m in models_dict.keys():
            c = sum(1 for ev in heldout_results[m][fam] if ev["is_top1"])
            accs[m] = f"{c}/20 ({c/20*100:.0f}%)"
            total_acc[m] += c
            mean_m_comp_by_model[m].extend([ev["m_comp"] for ev in heldout_results[m][fam]])
        print(f"{fam:12s} | {accs['Baseline']:12s} | {accs['D1']:12s} | {accs['D2']:12s} | {accs['D3']:12s} | {accs['D4']:12s}")
    print("-" * 75)
    print(f"{'OVERALL (100)':12s} | {total_acc['Baseline']}/100 ({total_acc['Baseline']}%) | {total_acc['D1']}/100 ({total_acc['D1']}%) | {total_acc['D2']}/100 ({total_acc['D2']}%) | {total_acc['D3']}/100 ({total_acc['D3']}%) | {total_acc['D4']}/100 ({total_acc['D4']}%)")

    # Overall Mean Margins across 100 prompts
    m_bar_heldout = {m: float(np.mean(mean_m_comp_by_model[m])) for m in models_dict.keys()}
    print("\nOverall Mean Held-Out Competitor Margins m_bar:")
    for m in models_dict.keys():
        print(f"  {m:10s}: {m_bar_heldout[m]:+.4f}")

    # -------------------------------------------------------------------------
    # Benchmark 2: Protected Controls Audit (40 Evaluations)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BENCHMARK 2: PROTECTED CONTROLS AUDIT (10 CONTROLS x 4 FORMATS = 40 EVALS)")
    print("=" * 80)

    ctrl_results = {}
    ctrl_csv_rows = []

    for m_name, d_w in models_dict.items():
        target_layer.weight.data.copy_((w0_pristine + d_w).to(torch.float32))
        ctrl_results[m_name] = {}
        for ctrl in controls:
            cid = ctrl["id"]
            country = ctrl["country"]
            city = ctrl.get("city") or ctrl.get("capital_city")
            ctrl_results[m_name][cid] = {}
            for fmt in CONTROL_FORMATS:
                p_str = ctrl["prompts"][fmt]
                ev = evaluate_prompt_detailed(model, tokenizer, p_str, city)
                ctrl_results[m_name][cid][fmt] = ev
        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

    ctrl_acc = {m: 0 for m in models_dict.keys()}
    ctrl_deltas = {a: [] for a in arm_labels}

    for ctrl in controls:
        cid = ctrl["id"]
        for fmt in CONTROL_FORMATS:
            eb = ctrl_results["Baseline"][cid][fmt]
            if eb["is_top1"]: ctrl_acc["Baseline"] += 1
            for a in arm_labels:
                ea = ctrl_results[a][cid][fmt]
                if ea["is_top1"]: ctrl_acc[a] += 1
                dm = ea["m_comp"] - eb["m_comp"]
                ctrl_deltas[a].append(dm)
                ctrl_csv_rows.append({
                    "arm": a,
                    "control_id": cid,
                    "country": ctrl["country"],
                    "format": CONTROL_FORMAT_NAMES[fmt],
                    "base_top1": eb["top1_token"],
                    "base_margin": eb["m_comp"],
                    "arm_top1": ea["top1_token"],
                    "arm_margin": ea["m_comp"],
                    "delta_m": dm
                })

    print(f"{'Model':10s} | {'Control Retention':18s} | {'Mean Delta m':15s} | {'Max |Delta m|':15s}")
    print("-" * 65)
    print(f"{'Baseline':10s} | {ctrl_acc['Baseline']}/40 ({ctrl_acc['Baseline']/40*100:.1f}%)   | {'—':15s} | {'—':15s}")
    for a in arm_labels:
        mean_dm = np.mean(ctrl_deltas[a])
        max_abs_dm = np.max(np.abs(ctrl_deltas[a]))
        print(f"Arm {a:6s} | {ctrl_acc[a]}/40 ({ctrl_acc[a]/40*100:.1f}%)   | {mean_dm:+14.4f} | {max_abs_dm:14.4f}")

    # -------------------------------------------------------------------------
    # Benchmark 3: Canonical Safeguard Probes (37 Probes)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BENCHMARK 3: CANONICAL SAFEGUARD PROBES (37 PROBES)")
    print("=" * 80)

    safeguard_results = {}
    for a in arm_labels:
        target_layer.weight.data.copy_((w0_pristine + Delta_W_dict[a]).to(torch.float32))
        changed_probes = []
        max_logit_diff = 0.0
        with torch.no_grad():
            for p_idx, probe in enumerate(baseline_safeguards):
                inp = tokenizer(probe["prompt"], return_tensors="pt")
                out = model(**inp)
                l = out.logits[0, -1, :].to(torch.float64)
                top_tok = tokenizer.decode([torch.argmax(l).item()])
                diff = torch.max(torch.abs(l - probe["logits"])).item()
                if diff > max_logit_diff:
                    max_logit_diff = diff
                if top_tok != probe["top1"]:
                    changed_probes.append({
                        "probe_idx": p_idx,
                        "prompt": probe["prompt"],
                        "orig_top1": probe["top1"],
                        "mod_top1": top_tok,
                        "diff": diff
                    })
        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))
        safeguard_results[a] = {
            "top1_match": f"{37 - len(changed_probes)}/37",
            "changed_count": len(changed_probes),
            "max_logit_diff": max_logit_diff,
            "changed_probes": changed_probes
        }
        print(f"Arm {a}: Top-1 Match: {safeguard_results[a]['top1_match']} | Max Logit Shift: {max_logit_diff:.4f} | Changed: {len(changed_probes)}")

    # -------------------------------------------------------------------------
    # Benchmark 4: Brazil Full 2-Token Sequence Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BENCHMARK 4: BRAZIL FULL 2-TOKEN SEQUENCE EVALUATION")
    print("=" * 80)

    brazil_results = {}
    brazil_fact = next(f for f in facts if f["id"] == "F9_Brazil")
    brazil_prompt = brazil_fact["eval_prompts"]["qa_direct_1"]

    for m_name, d_w in models_dict.items():
        target_layer.weight.data.copy_((w0_pristine + d_w).to(torch.float32))
        res_br = evaluate_brazil_sequence(model, tokenizer, brazil_prompt)
        brazil_results[m_name] = res_br
        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))
        print(f"Model {m_name:10s} | Token 1 (' São'): Rank {res_br['token1_rank']:2d} ('{res_br['token1_top1']}') | Token 2 (' Paulo'): Rank {res_br['token2_rank']:2d} ('{res_br['token2_top1']}') | Full Match: {res_br['full_sequence_correct']}")

    # -------------------------------------------------------------------------
    # Benchmark 5: Large-Scale Wikitext-2 PPL Evaluation (10,220 Tokens)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BENCHMARK 5: LARGE-SCALE WIKITEXT-2 PPL EVALUATION (10,220 TOKENS)")
    print("=" * 80)

    with open(WIKITEXT_PATH, "r", encoding="utf-8") as f:
        wiki_text = f.read()
    wiki_tokens = tokenizer(wiki_text, add_special_tokens=False)["input_ids"][:PPL_TOKEN_LIMIT]

    ppl_results = {}
    for m_name in ["Baseline", "D1", "D2", "D3", "D4"]:
        print(f"Evaluating PPL for {m_name}...")
        target_layer.weight.data.copy_((w0_pristine + models_dict[m_name]).to(torch.float32))
        res_ppl = compute_corpus_ppl(model, wiki_tokens, chunk_size=PPL_CHUNK_SIZE)
        ppl_results[m_name] = res_ppl
        print(f"  -> {m_name} PPL: {res_ppl['ppl']:.4f} (elapsed: {res_ppl['elapsed_seconds']:.2f}s)")
    target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

    base_ppl = ppl_results["Baseline"]["ppl"]
    print("\n--- Summary of Large-Scale Perplexity (10,220 Tokens) ---")
    print(f"{'Model State':12s} | {'PPL':10s} | {'Delta PPL':12s} | {'% Delta PPL':14s}")
    print("-" * 55)
    for m in ["Baseline", "D1", "D2", "D3", "D4"]:
        p = ppl_results[m]["ppl"]
        dp = p - base_ppl
        pct_dp = (dp / base_ppl) * 100.0
        print(f"{m:12s} | {p:10.4f} | {dp:+11.4f} | {pct_dp:+13.2f}%")

    # -------------------------------------------------------------------------
    # Benchmark 6: Factorial Contrasts, Interaction, & Generalization Efficiency
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BENCHMARK 6: FACTORIAL ANALYSIS, INTERACTION & GENERALIZATION EFFICIENCY")
    print("=" * 80)

    # Margins for the 4 arms
    m_D1 = m_bar_heldout["D1"]
    m_D2 = m_bar_heldout["D2"]
    m_D3 = m_bar_heldout["D3"]
    m_D4 = m_bar_heldout["D4"]
    m_W0 = m_bar_heldout["Baseline"]

    # Main Effects and Interaction:
    # C1: Syntactic Diversity under Strong Explicitness = D2 - D1
    c1_syn_strong = m_D2 - m_D1
    # C2: Syntactic Diversity under Moderate Explicitness = D4 - D3
    c2_syn_moderate = m_D4 - m_D3
    # C3: Relational Explicitness under Low Diversity = D1 - D3
    c3_rel_low = m_D1 - m_D3
    # C4: Relational Explicitness under High Diversity = D2 - D4
    c4_rel_high = m_D2 - m_D4
    # Interaction I = (D2 - D1) - (D4 - D3)
    interaction_I = c1_syn_strong - c2_syn_moderate

    print(f"Factorial Contrasts (on Held-Out Margin m_bar):")
    print(f"  [C1] Syntactic Diversity under Strong Explicitness (D2 - D1): {c1_syn_strong:+.4f}")
    print(f"  [C2] Syntactic Diversity under Moderate Explicitness (D4 - D3): {c2_syn_moderate:+.4f}")
    print(f"  [C3] Relational Explicitness under Low Diversity (D1 - D3):   {c3_rel_low:+.4f}")
    print(f"  [C4] Relational Explicitness under High Diversity (D2 - D4):  {c4_rel_high:+.4f}")
    print(f"  [Interaction I] (D2 - D1) - (D4 - D3):                        {interaction_I:+.4f}")

    # Generalization Efficiency GE = Delta m_bar / (||E_fact||_F + 1e-6)
    ge_results = {}
    print("\nGeneralization Efficiency (GE):")
    print(f"{'Arm':6s} | {'Delta m_bar':12s} | {'||E_fact||_F':14s} | {'GE Ratio':12s}")
    print("-" * 50)
    for a in arm_labels:
        delta_m = m_bar_heldout[a] - m_W0
        e_fact = residuals_dict[a]["res_fact_unweighted_fro"]
        ge = delta_m / (e_fact + 1e-6)
        ge_results[a] = {
            "delta_m_bar": float(delta_m),
            "unweighted_fact_residual": float(e_fact),
            "ge_ratio": float(ge)
        }
        print(f"Arm {a:2s} | {delta_m:+11.4f} | {e_fact:13.4f} | {ge:11.4f}")

    # -------------------------------------------------------------------------
    # Save Full Artifacts
    # -------------------------------------------------------------------------
    full_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_id": MODEL_ID,
        "layer_idx": LAYER_IDX,
        "solver": "Batch TREC-LS (torch.float64)",
        "residuals": residuals_dict,
        "heldout_100_summary": {
            "accuracy": {m: f"{total_acc[m]}/100" for m in models_dict.keys()},
            "m_bar_heldout": m_bar_heldout,
            "family_breakdown": {
                fam: {m: f"{sum(1 for ev in heldout_results[m][fam] if ev['is_top1'])}/20" for m in models_dict.keys()}
                for fam in EVAL_FAMILIES
            }
        },
        "controls_40_summary": {
            "accuracy": {m: f"{ctrl_acc[m]}/40" for m in models_dict.keys()},
            "mean_delta_m": {a: float(np.mean(ctrl_deltas[a])) for a in arm_labels},
            "max_abs_delta_m": {a: float(np.max(np.abs(ctrl_deltas[a]))) for a in arm_labels}
        },
        "safeguards_37_summary": safeguard_results,
        "brazil_full_sequence": brazil_results,
        "wikitext2_ppl": {
            m: {
                "ppl": float(ppl_results[m]["ppl"]),
                "delta_ppl": float(ppl_results[m]["ppl"] - base_ppl),
                "pct_delta": float((ppl_results[m]["ppl"] - base_ppl) / base_ppl * 100.0)
            }
            for m in ["Baseline", "D1", "D2", "D3", "D4"]
        },
        "factorial_analysis": {
            "c1_syn_strong": float(c1_syn_strong),
            "c2_syn_moderate": float(c2_syn_moderate),
            "c3_rel_low": float(c3_rel_low),
            "c4_rel_high": float(c4_rel_high),
            "interaction_I": float(interaction_I)
        },
        "generalization_efficiency": ge_results
    }

    full_json_path = os.path.join(output_dir, "exp10c_phase_b_full_results.json")
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    print(f"\nSaved Full Results JSON: {full_json_path}")

    heldout_csv_path = os.path.join(output_dir, "exp10c_phase_b_heldout_margins.csv")
    with open(heldout_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(heldout_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(heldout_csv_rows)
    print(f"Saved Held-Out Margins CSV: {heldout_csv_path}")

    ctrl_csv_path = os.path.join(output_dir, "exp10c_phase_b_control_margins.csv")
    with open(ctrl_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ctrl_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ctrl_csv_rows)
    print(f"Saved Control Margins CSV: {ctrl_csv_path}")

    print("\nExperiment 10C Phase B Full Evaluation Completed Successfully!")

if __name__ == "__main__":
    main()

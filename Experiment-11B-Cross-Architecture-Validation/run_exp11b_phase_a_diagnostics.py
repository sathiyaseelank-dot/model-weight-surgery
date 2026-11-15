#!/usr/bin/env python3
"""
Experiment 11B Phase A: Cross-Architecture Geometry & Pre-Surgery Diagnostics
Model: HuggingFaceTB/SmolLM2-360M (LlamaForCausalLM, 32 layers, d=960, intermediate=2560)
Target Layers (0-based module indices): 24, 27, 28, 29 (1-based: 25, 28, 29, 30)
Normalized Depths: 75.0%, 84.4%, 87.5%, 90.6%
"""

import os
import sys
import time
import math
import json
import hashlib
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
TARGET_LAYERS = [24, 27, 28, 29]
ARMS = ["D2", "D4"]
NUM_FACTS = 10
RANK_SAFEGUARD = 31
LAMBDA_REG = 0.10
GAMMA_ROLL = 1.00

EVAL_FAMILIES = ["qa_direct", "qa_conv", "comp_eval", "dec_eval", "ctx_eval"]
PROMPT_MAP = {
    "qa_direct": ["qa_direct_1", "qa_direct_2"],
    "qa_conv": ["qa_conv_1", "qa_conv_2"],
    "comp_eval": ["comp_eval_1", "comp_eval_2"],
    "dec_eval": ["dec_eval_1", "dec_eval_2"],
    "ctx_eval": ["ctx_eval_1", "ctx_eval_2"]
}
F4_PROMPT_KEYS = ["comp_p1", "comp_h1", "qa_p1", "qa_p2", "qa_h1", "dec_p1", "dec_p2"]
CONTROL_FORMATS = ["comp_p1", "comp_p2", "qa_p1", "qa_conv"]

def compute_effective_rank(singular_values: torch.Tensor) -> float:
    s = singular_values[singular_values > 1e-12]
    if len(s) == 0:
        return 0.0
    p = s / torch.sum(s)
    entropy = -torch.sum(p * torch.log(p)).item()
    return math.exp(entropy)

def extract_city_token_ids(tokenizer, city_name: str) -> List[int]:
    ids_spaced = tokenizer.encode(" " + city_name.strip(), add_special_tokens=False)
    if len(ids_spaced) > 0:
        return ids_spaced
    return tokenizer.encode(city_name.strip(), add_special_tokens=False)

def evaluate_prompt_detailed(model, tokenizer, prompt: str, target_city: str, original_city: str = "") -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].to(torch.float64)
    probs = F.softmax(logits, dim=-1)

    target_tokens = extract_city_token_ids(tokenizer, target_city)
    target_head_id = target_tokens[0]
    target_head_logit = logits[target_head_id].item()
    target_prob = probs[target_head_id].item()

    sorted_indices = torch.argsort(logits, descending=True)
    rank_target = (sorted_indices == target_head_id).nonzero(as_tuple=True)[0].item() + 1
    top1_id = sorted_indices[0].item()
    top1_token = tokenizer.decode([top1_id])

    comp_id = sorted_indices[1].item() if top1_id == target_head_id else top1_id
    m_comp = target_head_logit - logits[comp_id].item()

    m_orig = None
    orig_prob = None
    rank_orig = None
    if original_city:
        orig_tokens = extract_city_token_ids(tokenizer, original_city)
        orig_head_id = orig_tokens[0]
        m_orig = target_head_logit - logits[orig_head_id].item()
        orig_prob = probs[orig_head_id].item()
        rank_orig = (sorted_indices == orig_head_id).nonzero(as_tuple=True)[0].item() + 1

    return {
        "prompt": prompt,
        "is_top1": (rank_target == 1),
        "rank_target": rank_target,
        "target_prob": target_prob,
        "top1_token": top1_token,
        "m_comp": m_comp,
        "m_orig": m_orig,
        "orig_prob": orig_prob,
        "rank_orig": rank_orig
    }

def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 15) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)

def evaluate_brazil_complete(model, tokenizer) -> Dict[str, Any]:
    prompt = "The capital of Brazil is"
    tok1_target_str = " São"
    tok2_target_str = " Paulo"

    tok1_ids = tokenizer.encode(tok1_target_str, add_special_tokens=False)
    tok2_ids = tokenizer.encode(tok2_target_str, add_special_tokens=False)
    tok1_id = tok1_ids[0]
    tok2_id = tok2_ids[0]

    # Step 1: Prompt logits
    inp1 = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out1 = model(**inp1)
    logits1 = out1.logits[0, -1, :].to(torch.float64)
    probs1 = F.softmax(logits1, dim=-1)
    log_probs1 = F.log_softmax(logits1, dim=-1)

    sorted1 = torch.argsort(logits1, descending=True)
    rank_tok1 = (sorted1 == tok1_id).nonzero(as_tuple=True)[0].item() + 1
    top1_tok1 = tokenizer.decode([sorted1[0].item()])
    logprob_tok1 = log_probs1[tok1_id].item()

    # Step 2: Teacher-forced Step 2
    prompt_tf = prompt + tok1_target_str
    inp2 = tokenizer(prompt_tf, return_tensors="pt")
    with torch.no_grad():
        out2 = model(**inp2)
    logits2 = out2.logits[0, -1, :].to(torch.float64)
    probs2 = F.softmax(logits2, dim=-1)
    log_probs2 = F.log_softmax(logits2, dim=-1)

    sorted2 = torch.argsort(logits2, descending=True)
    rank_tok2 = (sorted2 == tok2_id).nonzero(as_tuple=True)[0].item() + 1
    top1_tok2 = tokenizer.decode([sorted2[0].item()])
    logprob_tok2 = log_probs2[tok2_id].item()

    joint_sequence_logprob = logprob_tok1 + logprob_tok2

    # Step 3: Unprompted greedy generation
    greedy_text = generate_greedy(model, tokenizer, prompt, max_new_tokens=4)
    target_clean = "São Paulo"
    greedy_correct = target_clean.lower() in greedy_text.strip().lower()

    return {
        "prompt": prompt,
        "token1_target": tok1_target_str,
        "token1_id": tok1_id,
        "token1_rank": rank_tok1,
        "token1_top1": top1_tok1,
        "token1_logprob": logprob_tok1,
        "token2_target": tok2_target_str,
        "token2_id": tok2_id,
        "token2_rank": rank_tok2,
        "token2_top1": top1_tok2,
        "token2_logprob": logprob_tok2,
        "joint_sequence_logprob": joint_sequence_logprob,
        "greedy_gen_text": greedy_text,
        "greedy_sequence_correct": greedy_correct
    }

def compute_corpus_ppl(model, token_ids: List[int], chunk_size: int = 512) -> Dict[str, Any]:
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    for start_idx in range(0, len(token_ids) - chunk_size + 1, chunk_size):
        chunk = token_ids[start_idx: start_idx + chunk_size]
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, :-1, :].contiguous()
            targets = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += targets.numel()

    mean_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(mean_loss) if mean_loss < 100 else float("inf")
    return {
        "nll": mean_loss,
        "ppl": ppl,
        "total_tokens": total_tokens,
        "chunks": total_tokens // (chunk_size - 1) if chunk_size > 1 else 0
    }

def main():
    print("=" * 80)
    print("Experiment 11B Phase A: Cross-Architecture Diagnostics & Baseline Evaluation")
    print(f"Target Model: {MODEL_ID}")
    print(f"Target Layers: {TARGET_LAYERS}")
    print("=" * 80)

    # 1. Model Info and Integrity
    api = HfApi()
    model_info = api.model_info(MODEL_ID)
    commit_sha = model_info.sha
    print(f"Hugging Face Commit SHA: {commit_sha}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    # Raw disk weight dtype verification
    sample_down_proj = model.model.layers[24].mlp.down_proj.weight
    disk_dtype = "torch.bfloat16" # from HF config and safetensors inspection
    runtime_dtype = str(sample_down_proj.dtype)
    print(f"Model storage dtype: {disk_dtype} | Runtime inference dtype: {runtime_dtype}")

    # Load manifests & frozen beta vector
    manifest_path = "Experiment-10C-Factorial-Prompt-Isolation/exp10c_dataset_manifest.json"
    f4_results_path = "Experiment-10A-Ten-Fact-Scaling/ten_fact_phase_f4_results.json"
    wikitext_path = "Experiment-10B-Prompt-Diverse-Generalization/wikitext2_valid.txt"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(f4_results_path, "r", encoding="utf-8") as f:
        f4_data = json.load(f)

    facts = manifest["facts"]
    controls = manifest["controls"]
    safeguard_probes = manifest["safeguard_probes"]
    beta_f4 = [f4_data["facts"][f["id"]]["calibrated_beta_f4"] for f in facts]

    print("\n--- Frozen Beta Vector (Calibrated F4 from Qwen2.5) ---")
    for idx, f in enumerate(facts):
        print(f"  {f['id']}: beta = {beta_f4[idx]:.6f}")

    # 2. Tokenizer and Target Geometry
    print("\n--- 2. Tokenizer & Target Displacement Vectors (d=960) ---")
    unembed_weight = model.lm_head.weight.data.to(torch.float64) # (49152, 960)
    D_targets = torch.zeros((960, NUM_FACTS), dtype=torch.float64)

    city_tokenizations = []
    for f_idx, fact in enumerate(facts):
        t_tokens = extract_city_token_ids(tokenizer, fact["target_city"])
        o_tokens = extract_city_token_ids(tokenizer, fact["original_city"])

        t_tok_strs = [tokenizer.decode([i]) for i in t_tokens]
        o_tok_strs = [tokenizer.decode([i]) for i in o_tokens]

        d_raw = unembed_weight[t_tokens[0]] - unembed_weight[o_tokens[0]]
        d_norm = torch.linalg.norm(d_raw)
        d_unit = d_raw / d_norm
        D_targets[:, f_idx] = d_unit

        city_tokenizations.append({
            "fact_id": fact["id"],
            "country": fact["country"],
            "original_city": fact["original_city"],
            "orig_tokens": o_tokens,
            "orig_token_strs": o_tok_strs,
            "target_city": fact["target_city"],
            "target_tokens": t_tokens,
            "target_token_strs": t_tok_strs,
            "is_single_token_target": len(t_tokens) == 1,
            "norm_unembed_diff": d_norm.item()
        })
        print(f"  {fact['id']} ({fact['country']}): {fact['original_city']} {o_tokens} -> {fact['target_city']} {t_tokens} (||diff||={d_norm.item():.4f})")

    # SVD of D_targets
    _, S_D, _ = torch.linalg.svd(D_targets, full_matrices=False)
    cond_D = (S_D[0] / S_D[-1]).item()
    eff_rank_D = compute_effective_rank(S_D)
    cos_D = D_targets.T @ D_targets
    off_diag_mask = ~torch.eye(NUM_FACTS, dtype=torch.bool)
    max_cos_D = torch.max(torch.abs(cos_D[off_diag_mask])).item()
    mean_cos_D = torch.mean(torch.abs(cos_D[off_diag_mask])).item()
    print(f"  Target Matrix D (960 x 10): Condition={cond_D:.4f} | EffRank={eff_rank_D:.2f} | MaxAbsCos={max_cos_D:.4f} | MeanAbsCos={mean_cos_D:.4f}")

    # 3. Baseline Evaluation (Pristine W0)
    print("\n--- 3. Baseline Evaluation (Pristine W0) ---")

    # A. 100-Prompt Held-Out Generalization Suite
    print("  Evaluating 100 Held-Out Generalization Prompts...")
    heldout_eval = []
    for f in facts:
        fid = f["id"]
        t_city = f["target_city"]
        o_city = f["original_city"]
        for fam in EVAL_FAMILIES:
            for pk in PROMPT_MAP[fam]:
                p = f["eval_prompts"][pk]
                res = evaluate_prompt_detailed(model, tokenizer, p, t_city, o_city)
                res.update({"fact_id": fid, "family": fam, "prompt_key": pk})
                heldout_eval.append(res)
    b_heldout_top1 = sum(1 for r in heldout_eval if r["is_top1"])
    b_heldout_mcomp = sum(r["m_comp"] for r in heldout_eval) / len(heldout_eval)
    b_heldout_morig = sum(r["m_orig"] for r in heldout_eval if r["m_orig"] is not None) / len(heldout_eval)
    print(f"  Held-out Top-1: {b_heldout_top1}/100 | Mean m_comp: {b_heldout_mcomp:.4f} | Mean m_orig: {b_heldout_morig:.4f}")

    # B. 70-Prompt Canonical Phase F4 Benchmark
    print("  Evaluating 70 Phase F4 Benchmark Prompts...")
    f4_eval = []
    for f in facts:
        fid = f["id"]
        t_city = f["target_city"]
        o_city = f["original_city"]
        fact_f4_prompts = f4_data["facts"][fid]["prompts"]
        for pk in F4_PROMPT_KEYS:
            p = fact_f4_prompts[pk]["prompt"]
            res = evaluate_prompt_detailed(model, tokenizer, p, t_city, o_city)
            res.update({"fact_id": fid, "prompt_key": pk})
            f4_eval.append(res)
    b_f4_top1 = sum(1 for r in f4_eval if r["is_top1"])
    b_f4_mcomp = sum(r["m_comp"] for r in f4_eval) / len(f4_eval)
    print(f"  Phase F4 Top-1: {b_f4_top1}/70 | Mean m_comp: {b_f4_mcomp:.4f}")

    # C. 40 Protected Controls Audit
    print("  Evaluating 40 Protected Controls...")
    ctrl_eval = {}
    for ctrl in controls:
        cid = ctrl["id"]
        city = ctrl.get("city") or ctrl.get("capital_city")
        ctrl_eval[cid] = {}
        for fmt in CONTROL_FORMATS:
            p_str = ctrl["prompts"][fmt]
            ev = evaluate_prompt_detailed(model, tokenizer, p_str, city)
            ctrl_eval[cid][fmt] = ev
    b_ctrl_top1 = sum(1 for cid in ctrl_eval for fmt in CONTROL_FORMATS if ctrl_eval[cid][fmt]["is_top1"])
    print(f"  Baseline Controls Pass: {b_ctrl_top1}/40 ({b_ctrl_top1/40.0*100:.1f}%)")

    # D. 37 Canonical Safeguard Probes
    print("  Capturing Baseline Safeguards Logits (37 probes)...")
    baseline_safeguard_logits = []
    for probe in safeguard_probes:
        inp = tokenizer(probe, return_tensors="pt")
        with torch.no_grad():
            out = model(**inp)
        baseline_safeguard_logits.append(out.logits[0, -1, :].to(torch.float64))

    # E. 30 Natural-Language Rollout Damping Contexts
    print("  Evaluating Baseline Rollouts (30 contexts)...")
    baseline_rollouts = []
    loop_count = 0
    for f in facts:
        for rk in ["rollout_r1", "rollout_r2", "rollout_r3"]:
            p = f["rollout_prompts"][rk]
            inp = tokenizer(p, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :].to(torch.float64)
            gen_text = generate_greedy(model, tokenizer, p, max_new_tokens=15)
            # check loop
            words = [w.strip().lower() for w in gen_text.split() if w.strip()]
            has_loop = False
            if len(words) >= 6:
                for idx in range(len(words) - 5):
                    if words[idx:idx+3] == words[idx+3:idx+6]:
                        has_loop = True
                        break
            if has_loop:
                loop_count += 1
            baseline_rollouts.append({
                "fact_id": f["id"],
                "rollout_key": rk,
                "prompt": p,
                "logits": logits,
                "top1_tok": tokenizer.decode([torch.argmax(logits).item()]),
                "gen_text": gen_text
            })
    print(f"  Baseline Rollouts: Loops={loop_count}/30 (Loop-Free Rate: {(30-loop_count)/30.0*100:.1f}%)")

    # F. Brazil Complete Two-Token Sequence Evaluation
    print("  Evaluating Brazil Complete Sequence...")
    b_brazil = evaluate_brazil_complete(model, tokenizer)
    print(f"  Brazil: T1 Rank={b_brazil['token1_rank']} ('{b_brazil['token1_top1']}') | T2 Rank={b_brazil['token2_rank']} ('{b_brazil['token2_top1']}') | Greedy='{b_brazil['greedy_gen_text']}' | Seq LogProb={b_brazil['joint_sequence_logprob']:.4f}")

    # G. WikiText-2 Validation Perplexity (Canonical 20 Chunks / 10,220 tokens)
    print("  Evaluating WikiText-2 Perplexity (Canonical 20 Chunks = 10,220 Targets)...")
    with open(wikitext_path, "r", encoding="utf-8") as f:
        wiki_text = f.read()
    wiki_tokens = tokenizer.encode(wiki_text, add_special_tokens=False)
    # 20 chunks of 512 = 10,240 tokens
    wiki_tokens_10240 = wiki_tokens[:10240]
    b_ppl_res = compute_corpus_ppl(model, wiki_tokens_10240, chunk_size=512)
    print(f"  Baseline WikiText-2 PPL: {b_ppl_res['ppl']:.4f} (NLL: {b_ppl_res['nll']:.4f}, Targets: {b_ppl_res['total_tokens']})")

    # 4. Cross-Layer Activation Geometry & Safeguard Subspaces (2560 dimensions)
    print("\n--- 4. Cross-Layer Activation Geometry (din=2560, dout=960) ---")
    layer_diagnostics = {}

    for layer_idx in TARGET_LAYERS:
        norm_depth = (layer_idx + 1) / 32.0 * 100.0 # 1-based depth percentage
        print(f"\n>>> Analyzing Layer {layer_idx} (1-based Layer {layer_idx+1}, Normalized Depth: {norm_depth:.1f}%) <<<")

        # Forward hook to capture activations at input to down_proj
        captured_activation = []
        def make_hook():
            def hook(module, args, output):
                captured_activation.append(args[0][0, -1, :].detach().to(torch.float64))
            return hook

        target_layer = model.model.layers[layer_idx].mlp.down_proj
        hook_handle = target_layer.register_forward_hook(make_hook())

        # A. Safeguard Activations (37 probes)
        captured_activation.clear()
        for probe in safeguard_probes:
            inp = tokenizer(probe, return_tensors="pt")
            with torch.no_grad():
                _ = model(**inp)
        S_raw = torch.stack(captured_activation, dim=1) # (2560, 37)

        # SVD of Safeguards
        U_S_full, S_vals, _ = torch.linalg.svd(S_raw, full_matrices=False)
        U_S = U_S_full[:, :RANK_SAFEGUARD] # (2560, 31)
        cum_energy_31 = (torch.sum(S_vals[:RANK_SAFEGUARD]**2) / torch.sum(S_vals**2)).item() * 100.0
        cond_S = (S_vals[0] / S_vals[-1]).item()
        eff_rank_S = compute_effective_rank(S_vals)
        res_S = torch.linalg.norm(S_raw - U_S @ (U_S.T @ S_raw)).item()
        print(f"  Safeguards (2560 x 37): Cond={cond_S:.2f} | EffRank={eff_rank_S:.2f} | Rank-31 Energy={cum_energy_31:.2f}% | Proj Residual={res_S:.4f}")

        # B. Rollout Activations (30 contexts)
        captured_activation.clear()
        for f in facts:
            for rk in ["rollout_r1", "rollout_r2", "rollout_r3"]:
                p = f["rollout_prompts"][rk]
                inp = tokenizer(p, return_tensors="pt")
                with torch.no_grad():
                    _ = model(**inp)
        R_raw = torch.stack(captured_activation, dim=1) # (2560, 30)
        R_proj = R_raw - U_S @ (U_S.T @ R_raw)
        R_proj = R_proj - U_S @ (U_S.T @ R_proj)
        rollout_retention = (torch.linalg.norm(R_proj).item()**2) / (torch.linalg.norm(R_raw).item()**2) * 100.0
        print(f"  Rollout Retention after Safeguard Nullspace Projection: {rollout_retention:.2f}%")

        # C. Arm D2 & Arm D4 Factual Keys (40 keys each)
        arm_diag = {}
        for a_label in ARMS:
            a_key = f"arm_{a_label.lower()}_keys"
            captured_activation.clear()
            for fid in sorted([f["id"] for f in facts]):
                for k in ["K1", "K2", "K3", "K4"]:
                    p = manifest[a_key][fid][k]
                    inp = tokenizer(p, return_tensors="pt")
                    with torch.no_grad():
                        _ = model(**inp)
            K_raw = torch.stack(captured_activation, dim=1) # (2560, 40)
            K_proj = K_raw - U_S @ (U_S.T @ K_raw)
            K_proj = K_proj - U_S @ (U_S.T @ K_proj)
            k_retention = (torch.linalg.norm(K_proj).item()**2) / (torch.linalg.norm(K_raw).item()**2) * 100.0

            _, S_K, _ = torch.linalg.svd(K_proj, full_matrices=False)
            cond_K = (S_K[0] / S_K[-1]).item()
            eff_rank_K = compute_effective_rank(S_K)

            arm_diag[a_label] = {
                "key_retention_pct": k_retention,
                "cond_number": cond_K,
                "effective_rank": eff_rank_K,
                "norm_raw": torch.linalg.norm(K_raw).item(),
                "norm_proj": torch.linalg.norm(K_proj).item()
            }
            print(f"  Arm {a_label} Keys (2560 x 40): Retention={k_retention:.2f}% | Cond={cond_K:.2f} | EffRank={eff_rank_K:.2f}")

        hook_handle.remove()

        layer_diagnostics[f"layer_{layer_idx}"] = {
            "layer_module_idx": layer_idx,
            "layer_1based_idx": layer_idx + 1,
            "normalized_depth_pct": norm_depth,
            "safeguards": {
                "cond_number": cond_S,
                "effective_rank": eff_rank_S,
                "rank31_energy_retention_pct": cum_energy_31,
                "projection_residual": res_S,
                "raw_singular_values": S_vals.tolist()
            },
            "rollouts": {
                "energy_retention_pct": rollout_retention,
                "norm_raw": torch.linalg.norm(R_raw).item(),
                "norm_proj": torch.linalg.norm(R_proj).item()
            },
            "arms": arm_diag
        }

    # 5. Pre-Surgery Parity Check (Zero Weights Changed)
    print("\n--- 5. Pre-Surgery Integrity Verification ---")
    w0_hashes = {}
    for l_idx in TARGET_LAYERS:
        w_np = model.model.layers[l_idx].mlp.down_proj.weight.data.cpu().numpy()
        sha = hashlib.sha256(w_np.tobytes()).hexdigest()
        w0_hashes[f"layer_{l_idx}"] = sha
        print(f"  Layer {l_idx} down_proj pristine SHA256: {sha[:16]}...")

    results_phase_a = {
        "metadata": {
            "experiment": "Experiment 11B Phase A",
            "model_id": MODEL_ID,
            "commit_sha": commit_sha,
            "disk_dtype": disk_dtype,
            "runtime_dtype": runtime_dtype,
            "din_activation": 2560,
            "dout_target": 960,
            "layers_0based": TARGET_LAYERS,
            "layers_1based": [l + 1 for l in TARGET_LAYERS],
            "normalized_depths_pct": [(l + 1)/32.0 * 100.0 for l in TARGET_LAYERS],
            "rank_safeguard": RANK_SAFEGUARD,
            "lambda_reg": LAMBDA_REG,
            "gamma_roll": GAMMA_ROLL,
            "frozen_beta_f4": beta_f4
        },
        "target_geometry": {
            "city_tokenizations": city_tokenizations,
            "matrix_D": {
                "shape": list(D_targets.shape),
                "condition_number": cond_D,
                "effective_rank": eff_rank_D,
                "max_abs_cosine": max_cos_D,
                "mean_abs_cosine": mean_cos_D
            }
        },
        "baseline_evaluation": {
            "heldout_100": {
                "top1_count": b_heldout_top1,
                "accuracy_pct": b_heldout_top1 / 100.0 * 100.0,
                "mean_m_comp": b_heldout_mcomp,
                "mean_m_orig": b_heldout_morig
            },
            "f4_70": {
                "top1_count": b_f4_top1,
                "accuracy_pct": b_f4_top1 / 70.0 * 100.0,
                "mean_m_comp": b_f4_mcomp
            },
            "controls_40": {
                "top1_count": b_ctrl_top1,
                "accuracy_pct": b_ctrl_top1 / 40.0 * 100.0
            },
            "rollouts_30": {
                "loop_count": loop_count,
                "loop_free_rate_pct": (30 - loop_count) / 30.0 * 100.0
            },
            "wikitext2_canonical_20_chunks": b_ppl_res,
            "brazil_sequence": b_brazil
        },
        "layer_diagnostics": layer_diagnostics,
        "pristine_w0_hashes": w0_hashes
    }

    out_file = "Experiment-11B-Cross-Architecture-Validation/exp11b_phase_a_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_phase_a, f, indent=2)
    print(f"\nPhase A Diagnostics Complete. Results saved to {out_file}")

if __name__ == "__main__":
    main()

"""
Experiment 7.11: Multi-Key France -> Lyon Paraphrase Pilot
==========================================================
Implements the two-key regularized equality-constrained least squares (TREC-LS)
weight surgery on Qwen2.5-0.5B Layer 20 mlp.down_proj, incorporating all
mathematical corrections from Task 7.10.

Training Keys (M=2):
  - P1: "The capital of France is"
  - P4: "Which city serves as the capital of France? The answer is"

Held-Out Prompts (5):
  - P3: "The capital city of France is" (In-distribution lexical insertion)
  - P2: "France's capital city is" (In-distribution possessive syntax)
  - P6: "The official capital city of the French Republic is" (OOD diplomatic)
  - P7: "Question: What is the capital of France? Answer:" (OOD dialogue Q&A)
  - P8: "France (capital:" (OOD parenthetical cloze)

Controls (3):
  - Germany, Italy, Spain open-ended factual capital queries

Safeguards (5):
  - Sequence-wide 37-token safeguard matrix K_S (Germany, Italy, Spain, Science, Arithmetic)

Experimental Parameters:
  - betas: [0.08, 0.12, 0.15]
  - lambda_reg: 0.10
  - float64 precision for projection, Gram inversion, and weight update
  - Strict bitwise model reversion and snapshot invariance assertions
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

# Add parent directory and Experiment-2.1 directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)
sys.path.append(os.path.join(PARENT_DIR, "Experiment-2.1-Robustness-Streaming-Audit"))

from activation_capture import QwenActivationCapturer
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = 20
BETAS = [0.08, 0.12, 0.15]
LAMBDA_REG = 0.10
MAX_NEW_TOKENS = 12

CITY_TOKEN_IDS = {
    "Lyon": 55201,    # " Lyon"
    "Paris": 12095,   # " Paris"
    "Berlin": 19846,  # " Berlin"
    "Rome": 21718,    # " Rome"
    "Madrid": 24081   # " Madrid"
}

# --- Prompt Definitions ---
TRAINING_PROMPTS = [
    {
        "id": "P1_canonical",
        "prompt": "The capital of France is",
        "category": "training_canonical",
        "expected_baseline_city": "Paris"
    },
    {
        "id": "P4_wh_question",
        "prompt": "Which city serves as the capital of France? The answer is",
        "category": "training_wh_question",
        "expected_baseline_city": "Paris"
    }
]

HELDOUT_PROMPTS = [
    {
        "id": "P3_cap_city",
        "prompt": "The capital city of France is",
        "category": "heldout_near_duplicate",
        "expected_baseline_city": "Paris"
    },
    {
        "id": "P2_possessive",
        "prompt": "France's capital city is",
        "category": "heldout_syntactic_variant",
        "expected_baseline_city": "the"  # baseline predicts ' the'
    },
    {
        "id": "P6_diplomatic",
        "prompt": "The official capital city of the French Republic is",
        "category": "heldout_ood_diplomatic",
        "expected_baseline_city": "Paris"
    },
    {
        "id": "P7_dialogue_qa",
        "prompt": "Question: What is the capital of France? Answer:",
        "category": "heldout_ood_dialogue_qa",
        "expected_baseline_city": "The"  # baseline predicts ' The capital...'
    },
    {
        "id": "P8_parenthetical",
        "prompt": "France (capital:",
        "category": "heldout_ood_parenthetical",
        "expected_baseline_city": "Paris"
    }
]

CONTROL_PROMPTS = [
    {
        "id": "control_germany",
        "prompt": "The capital of Germany is",
        "category": "control_unrelated",
        "expected_baseline_city": "Berlin"
    },
    {
        "id": "control_italy",
        "prompt": "The capital of Italy is",
        "category": "control_unrelated",
        "expected_baseline_city": "Rome"
    },
    {
        "id": "control_spain",
        "prompt": "The capital of Spain is",
        "category": "control_unrelated",
        "expected_baseline_city": "Madrid"
    }
]

SAFEGUARD_PROMPTS = [
    {
        "id": "safe_germany",
        "prompt": "The capital of Germany is Berlin.",
        "category": "safeguard_geography"
    },
    {
        "id": "safe_italy",
        "prompt": "The capital of Italy is Rome.",
        "category": "safeguard_geography"
    },
    {
        "id": "safe_spain",
        "prompt": "The capital of Spain is Madrid.",
        "category": "safeguard_geography"
    },
    {
        "id": "safe_science",
        "prompt": "Water is composed of hydrogen and oxygen.",
        "category": "safeguard_science"
    },
    {
        "id": "safe_arithmetic",
        "prompt": "2 + 2 = 4.",
        "category": "safeguard_arithmetic"
    }
]


def evaluate_direct_next_token(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str
) -> Dict[str, Any]:
    """Computes direct next-token logits and probabilities over the full vocabulary."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :].detach().cpu() # [vocab_size]

    probs = F.softmax(logits, dim=-1)

    lyon_id = CITY_TOKEN_IDS["Lyon"]
    paris_id = CITY_TOKEN_IDS["Paris"]

    lyon_logit = logits[lyon_id].item()
    paris_logit = logits[paris_id].item()
    lyon_prob = probs[lyon_id].item()
    paris_prob = probs[paris_id].item()
    margin = lyon_logit - paris_logit

    top5 = torch.topk(logits, 5)
    top5_list = []
    for r in range(5):
        tid = top5.indices[r].item()
        top5_list.append({
            "rank": r + 1,
            "token_id": tid,
            "token_str": tokenizer.decode([tid]),
            "logit": top5.values[r].item(),
            "prob": probs[tid].item()
        })

    return {
        "prompt": prompt_text,
        "lyon_logit": lyon_logit,
        "paris_logit": paris_logit,
        "lyon_prob": lyon_prob,
        "paris_prob": paris_prob,
        "lyon_paris_logit_margin": margin,
        "top1_token_id": top5_list[0]["token_id"],
        "top1_token_str": top5_list[0]["token_str"],
        "top1_prob": top5_list[0]["prob"],
        "top5": top5_list
    }


def generate_autoregressive_rollout(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> Dict[str, Any]:
    """Generates autoregressively with greedy decoding, recording full step outputs."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

    seq = out.sequences[0]
    gen_token_ids = seq[prompt_len:].tolist()
    gen_token_strs = [tokenizer.decode([tid]) for tid in gen_token_ids]
    gen_text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=False)
    full_text = tokenizer.decode(seq, skip_special_tokens=False)

    eos_token_id = tokenizer.eos_token_id
    has_eos = eos_token_id in gen_token_ids if eos_token_id is not None else False
    early_term = len(gen_token_ids) < max_new_tokens

    per_step_summary = []
    for s in range(len(out.scores)):
        step_logits = out.scores[s][0]
        step_probs = F.softmax(step_logits, dim=-1)
        top3 = torch.topk(step_probs, 3)
        top3_list = [
            {
                "rank": r + 1,
                "token_id": top3.indices[r].item(),
                "token_str": tokenizer.decode([top3.indices[r].item()]),
                "prob": top3.values[r].item()
            }
            for r in range(3)
        ]
        per_step_summary.append({
            "step": s + 1,
            "chosen_token_id": gen_token_ids[s] if s < len(gen_token_ids) else None,
            "chosen_token_str": gen_token_strs[s] if s < len(gen_token_strs) else "",
            "lyon_prob": step_probs[CITY_TOKEN_IDS["Lyon"]].item(),
            "paris_prob": step_probs[CITY_TOKEN_IDS["Paris"]].item(),
            "top3": top3_list
        })

    return {
        "prompt": prompt_text,
        "prompt_length": prompt_len,
        "generated_token_ids": gen_token_ids,
        "generated_token_strings": gen_token_strs,
        "generated_text": gen_text,
        "full_text": full_text,
        "first_generated_token_id": gen_token_ids[0] if gen_token_ids else None,
        "first_generated_token_str": gen_token_strs[0] if gen_token_strs else "",
        "contains_lyon": ("Lyon" in gen_text or "lyon" in gen_text),
        "contains_paris": ("Paris" in gen_text or "paris" in gen_text),
        "num_generated_tokens": len(gen_token_ids),
        "has_eos": has_eos,
        "early_termination": early_term,
        "per_step_summary": per_step_summary
    }


def compute_rollout_full_vocab_drift(
    model_base: AutoModelForCausalLM,
    model_mod: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    baseline_rollout_ids: List[int]
) -> List[Dict[str, Any]]:
    """
    Computes exact step-by-step full-vocabulary logit drift, KL divergence, and Top-5 overlap
    on the identical prefix sequence generated by the baseline model.
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].tolist()
    full_seq_ids = prompt_ids + baseline_rollout_ids
    inputs = torch.tensor([full_seq_ids])

    with torch.no_grad():
        out_base = model_base(inputs).logits[0] # [seq_len, vocab_size]
        out_mod = model_mod(inputs).logits[0]   # [seq_len, vocab_size]

    prompt_len = len(prompt_ids)
    num_steps = len(baseline_rollout_ids)
    step_drifts = []

    for s in range(num_steps):
        pos = prompt_len - 1 + s
        l_base = out_base[pos].cpu()
        l_mod = out_mod[pos].cpu()

        max_abs = torch.max(torch.abs(l_mod - l_base)).item()
        mean_abs = torch.mean(torch.abs(l_mod - l_base)).item()
        norm_base = torch.norm(l_base).item()
        rel_diff = (torch.norm(l_mod - l_base).item() / norm_base) if norm_base > 0 else 0.0

        p_base = F.softmax(l_base, dim=-1)
        p_mod = F.softmax(l_mod, dim=-1)
        log_p_base = F.log_softmax(l_base, dim=-1)
        log_p_mod = F.log_softmax(l_mod, dim=-1)

        kl_div = F.kl_div(log_p_mod, p_base, reduction="batchmean").item()

        top1_base = torch.argmax(l_base).item()
        top1_mod = torch.argmax(l_mod).item()
        top1_match = bool(top1_base == top1_mod)

        top5_base = set(torch.topk(l_base, 5).indices.tolist())
        top5_mod = set(torch.topk(l_mod, 5).indices.tolist())
        top5_overlap = len(top5_base & top5_mod)

        top2_b = torch.topk(p_base, 2).values
        top2_m = torch.topk(p_mod, 2).values
        base_top1_p = top2_b[0].item()
        mod_top1_p = p_mod[top1_base].item()
        base_top2_marg = (top2_b[0] - top2_b[1]).item()
        mod_top2_marg = (top2_m[0] - top2_m[1]).item()

        step_drifts.append({
            "step": s + 1,
            "max_abs_logit_diff": max_abs,
            "mean_abs_logit_diff": mean_abs,
            "relative_logit_diff": rel_diff,
            "kl_divergence": kl_div,
            "top1_match": top1_match,
            "top5_overlap": top5_overlap,
            "baseline_top1_prob": base_top1_p,
            "modified_top1_prob": mod_top1_p,
            "baseline_top2_margin": base_top2_marg,
            "modified_top2_margin": mod_top2_marg
        })

    return step_drifts


def run_experiment_7_11():
    print("=" * 95)
    print("EXPERIMENT 7.11: MULTI-KEY FRANCE -> LYON PARAPHRASE PILOT (TREC-LS)")
    print("=========================================================================================")

    results_dir = os.path.dirname(os.path.abspath(__file__))
    output_json_path = os.path.join(results_dir, "multi_key_paraphrase_pilot_results.json")

    # 1. Load pristine model in float32 for inference, snapshot Layer 20 down_proj in float64
    print(f"\n[Step 1] Loading pristine {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        local_files_only=True
    )
    model.eval()

    snapshot_w20 = model.model.layers[LAYER_IDX].mlp.down_proj.weight.detach().clone()
    d_out, d_in = snapshot_w20.shape
    W_0_f64 = snapshot_w20.to(dtype=torch.float64)
    norm_w0 = torch.norm(W_0_f64, p="fro").item()
    print(f"Layer {LAYER_IDX} mlp.down_proj shape: ({d_out}, {d_in}) [float64 snapshot stored, ||W_0||_F = {norm_w0:.4f}]")

    capturer = QwenActivationCapturer(model, tokenizer)

    # 2. Extract Training Keys: P1 and P4 terminal token activations
    print(f"\n[Step 2] Capturing training keys from P1 and P4 terminal tokens...")
    k1_raw = capturer.capture(TRAINING_PROMPTS[0]["prompt"], layer_indices=LAYER_IDX, token_position="last")[LAYER_IDX]["input"].to(torch.float64).view(d_in, 1)
    k2_raw = capturer.capture(TRAINING_PROMPTS[1]["prompt"], layer_indices=LAYER_IDX, token_position="last")[LAYER_IDX]["input"].to(torch.float64).view(d_in, 1)

    v0_1 = (W_0_f64 @ k1_raw).view(d_out, 1)
    v0_2 = (W_0_f64 @ k2_raw).view(d_out, 1)
    v0_1_norm = torch.norm(v0_1).item()
    v0_2_norm = torch.norm(v0_2).item()

    K_train = torch.cat([k1_raw, k2_raw], dim=1) # [4864, 2]
    cos_raw = (torch.dot(k1_raw.view(-1), k2_raw.view(-1)) / (torch.norm(k1_raw) * torch.norm(k2_raw))).item()
    print(f"  P1 key norm: {torch.norm(k1_raw).item():.4f}, baseline output norm: {v0_1_norm:.4f}")
    print(f"  P4 key norm: {torch.norm(k2_raw).item():.4f}, baseline output norm: {v0_2_norm:.4f}")
    print(f"  Raw Training Keys Cosine Similarity: {cos_raw:.4f}")

    # 3. Construct Normalized Steering Direction d
    print(f"\n[Step 3] Constructing normalized steering direction d from lm_head.weight...")
    E_lyon = model.lm_head.weight[CITY_TOKEN_IDS["Lyon"]].detach().clone().to(torch.float64).view(d_out, 1)
    E_paris = model.lm_head.weight[CITY_TOKEN_IDS["Paris"]].detach().clone().to(torch.float64).view(d_out, 1)
    diff_E = E_lyon - E_paris
    d_steer = diff_E / torch.norm(diff_E)
    print(f"  Steering direction d shape: {list(d_steer.shape)}, norm: {torch.norm(d_steer).item():.6f}")

    # 4. Construct Sequence-Wide Safeguard Matrix K_S (37 tokens) and Compute SVD Basis (Rank 31)
    print(f"\n[Step 4] Capturing sequence-wide safeguard matrix K_S (37 tokens)...")
    k_sg_list = []
    for sg in SAFEGUARD_PROMPTS:
        cap_sg = capturer.capture(sg["prompt"], layer_indices=LAYER_IDX, token_position="all")
        raw_k = cap_sg[LAYER_IDX]["input"]
        for pos in range(raw_k.shape[0]):
            k_sg_list.append(raw_k[pos].to(torch.float64).view(d_in, 1))

    K_S = torch.cat(k_sg_list, dim=1) # [4864, 37]
    V_S = W_0_f64 @ K_S             # [896, 37]
    norm_vs = torch.norm(V_S, p="fro").item()

    # Thin SVD to isolate rank-31 orthonormal basis
    U_full, S_full, V_full = torch.linalg.svd(K_S, full_matrices=False)
    RANK_S = 31
    U_S = U_full[:, :RANK_S] # [4864, 31]

    # Verify orthonormality of U_S
    ortho_err = torch.norm(U_S.T @ U_S - torch.eye(RANK_S, dtype=torch.float64), p="fro").item()
    print(f"  K_S shape: {list(K_S.shape)}, SVD singular values: min={S_full[-1].item():.2e}, sigma_31={S_full[30].item():.4f}, max={S_full[0].item():.4f}")
    print(f"  Safeguard basis U_S rank: {RANK_S}, Orthonormality error ||U_S^T U_S - I_31||_F: {ortho_err:.2e}")
    assert ortho_err < 1e-12, f"Fatal: U_S basis is not orthonormal! (err={ortho_err})"

    # 5. Project Training Keys into Safeguard Null Space
    print(f"\n[Step 5] Projecting training keys into safeguard null space P_perp = I - U_S U_S^T...")
    # P_perp @ k = k - U_S @ (U_S^T @ k)
    k1_proj = k1_raw - U_S @ (U_S.T @ k1_raw)
    k2_proj = k2_raw - U_S @ (U_S.T @ k2_raw)
    K_proj = torch.cat([k1_proj, k2_proj], dim=1) # [4864, 2]

    norm_k1_proj = torch.norm(k1_proj).item()
    norm_k2_proj = torch.norm(k2_proj).item()
    cos_proj = (torch.dot(k1_proj.view(-1), k2_proj.view(-1)) / (norm_k1_proj * norm_k2_proj)).item()

    # SVD of K_proj and Gram Matrix G
    Gram = K_proj.T @ K_proj # [2, 2]
    U_kp, S_kp, V_kp = torch.linalg.svd(K_proj, full_matrices=False)
    cond_K_proj = (S_kp[0] / S_kp[1]).item()
    cond_Gram = cond_K_proj ** 2

    # Verification of rank threshold: sigma_min / sigma_max >= 1e-2
    rank_ratio = (S_kp[1] / S_kp[0]).item()
    print(f"  P1 projected norm: {norm_k1_proj:.4f} ({norm_k1_proj/torch.norm(k1_raw).item()*100:.1f}% energy retained)")
    print(f"  P4 projected norm: {norm_k2_proj:.4f} ({norm_k2_proj/torch.norm(k2_raw).item()*100:.1f}% energy retained)")
    print(f"  Projected Cosine Similarity: {cos_proj:.4f} (decorrelated from raw {cos_raw:.4f})")
    print(f"  Singular values: sigma_1 = {S_kp[0].item():.4f}, sigma_2 = {S_kp[1].item():.4f}")
    print(f"  Rank ratio sigma_2/sigma_1: {rank_ratio:.4f} (Threshold >= 0.01: PASS)")
    print(f"  Gram matrix condition number kappa(G): {cond_Gram:.4f}")
    assert rank_ratio >= 0.01, f"Fatal: Projected training keys are collinear! (ratio={rank_ratio})"

    # Target Preservation Matrix T = G (G + lambda I)^(-1)
    G_reg_inv = torch.linalg.inv(Gram + LAMBDA_REG * torch.eye(2, dtype=torch.float64))
    T_preserv = Gram @ G_reg_inv
    print(f"  Target preservation matrix T = G (G + {LAMBDA_REG} I)^(-1):\n{T_preserv.cpu().numpy()}")

    # 6. Evaluate Baseline State (Pristine Model)
    print("\n=========================================================================================")
    print("[State 0] EVALUATING PRISTINE BASELINE MODEL (DIRECT & 12-TOKEN GREEDY ROLLOUTS)")
    print("=========================================================================================")

    all_prompts_map = {
        "training": TRAINING_PROMPTS,
        "heldout": HELDOUT_PROMPTS,
        "controls": CONTROL_PROMPTS,
        "safeguards": SAFEGUARD_PROMPTS
    }

    baseline_direct_eval = {}
    baseline_rollout_eval = {}

    for group_name, prompt_list in all_prompts_map.items():
        print(f"\n--- Baseline {group_name.upper()} Prompts ---")
        for p in prompt_list:
            pid = p["id"]
            d_res = evaluate_direct_next_token(model, tokenizer, p["prompt"])
            r_res = generate_autoregressive_rollout(model, tokenizer, p["prompt"], MAX_NEW_TOKENS)
            baseline_direct_eval[pid] = d_res
            baseline_rollout_eval[pid] = r_res

            top1 = d_res["top1_token_str"]
            p_lyon = d_res["lyon_prob"] * 100
            p_paris = d_res["paris_prob"] * 100
            print(f"  [{pid:<22}] Next: {repr(top1):<12} | Lyon: {p_lyon:5.2f}% | Paris: {p_paris:5.2f}% | Rollout: \"{r_res['generated_text']}\"")

    # 7. Execute Independent Surgery for each beta in [0.08, 0.12, 0.15]
    surgery_results = []

    for beta in BETAS:
        print("\n=========================================================================================")
        print(f"[Surgery Run] EVALUATING INDEPENDENT SURGERY: BETA = {beta:.2f}, LAMBDA = {LAMBDA_REG:.2f}")
        print("=========================================================================================")

        # Construct target delta matrix: Delta V = beta * d * [||v0_1||, ||v0_2||]
        delta_v1 = (beta * v0_1_norm) * d_steer # [896, 1]
        delta_v2 = (beta * v0_2_norm) * d_steer # [896, 1]
        Delta_V = torch.cat([delta_v1, delta_v2], dim=1) # [896, 2]

        # Compute Regularized Closed-Form Solution:
        # Delta W = Delta_V @ (K_proj^T K_proj + lambda I)^(-1) @ K_proj^T
        Delta_W = Delta_V @ G_reg_inv @ K_proj.T # [896, 4864]

        # Verify numerical residuals in float64
        norm_delta_w = torch.norm(Delta_W, p="fro").item()
        param_drift = norm_delta_w / norm_w0

        # Target residual: ||Delta W @ K_train - Delta_V||_F / ||Delta_V||_F
        achieved_delta_v = Delta_W @ K_train
        target_res_abs = torch.norm(achieved_delta_v - Delta_V, p="fro").item()
        target_res_rel = target_res_abs / torch.norm(Delta_V, p="fro").item()
        target_accuracy_pct = (1.0 - target_res_rel) * 100.0

        # Safeguard residual: ||Delta W @ K_S||_F / ||V_S||_F
        safe_res_abs = torch.norm(Delta_W @ K_S, p="fro").item()
        safe_res_rel = safe_res_abs / norm_vs if norm_vs > 0 else safe_res_abs

        # Projection orthogonality error: ||U_S^T @ Delta_W^T||_F
        ortho_leakage = torch.norm(U_S.T @ Delta_W.T, p="fro").item()

        print(f"  Numerical Diagnostics:")
        print(f"    ||Delta W||_F: {norm_delta_w:.4f} (Parameter Drift: {param_drift*100:.3f}%)")
        print(f"    Target Residual: Abs = {target_res_abs:.4f}, Rel = {target_res_rel:.4e} (Accuracy: {target_accuracy_pct:.2f}%)")
        print(f"    Safeguard Residual: Abs = {safe_res_abs:.2e}, Rel = {safe_res_rel:.2e}")
        print(f"    Safeguard Subspace Leakage ||U_S^T Delta W^T||_F: {ortho_leakage:.2e}")

        # Modified weight in float64, converted to float32 for model evaluation
        W_mod_f64 = W_0_f64 + Delta_W
        W_mod_f32 = W_mod_f64.to(torch.float32)

        # Temporary load inside try...finally to ensure absolute weight reversion
        model.model.layers[LAYER_IDX].mlp.down_proj.weight.data.copy_(W_mod_f32)

        mod_direct_eval = {}
        mod_rollout_eval = {}
        drift_metrics_eval = {}

        try:
            # 1. Training Prompts Evaluation
            print(f"\n  --- Training Prompts (beta={beta:.2f}) ---")
            for p in TRAINING_PROMPTS:
                pid = p["id"]
                d_res = evaluate_direct_next_token(model, tokenizer, p["prompt"])
                r_res = generate_autoregressive_rollout(model, tokenizer, p["prompt"], MAX_NEW_TOKENS)
                mod_direct_eval[pid] = d_res
                mod_rollout_eval[pid] = r_res

                p_lyon = d_res["lyon_prob"] * 100
                p_paris = d_res["paris_prob"] * 100
                top1 = d_res["top1_token_str"]
                marg = d_res["lyon_paris_logit_margin"]
                print(f"    [{pid:<20}] Top1: {repr(top1):<12} | Lyon: {p_lyon:5.2f}% | Paris: {p_paris:5.2f}% | Margin: {marg:+6.2f} | Rollout: \"{r_res['generated_text']}\"")

            # 2. Held-Out Prompts Evaluation (Full Matrix, No Cherry-Picking)
            print(f"\n  --- Held-Out Prompts (beta={beta:.2f}) ---")
            for p in HELDOUT_PROMPTS:
                pid = p["id"]
                d_res = evaluate_direct_next_token(model, tokenizer, p["prompt"])
                r_res = generate_autoregressive_rollout(model, tokenizer, p["prompt"], MAX_NEW_TOKENS)
                mod_direct_eval[pid] = d_res
                mod_rollout_eval[pid] = r_res

                p_lyon = d_res["lyon_prob"] * 100
                p_paris = d_res["paris_prob"] * 100
                top1 = d_res["top1_token_str"]
                marg = d_res["lyon_paris_logit_margin"]
                print(f"    [{pid:<20}] Top1: {repr(top1):<12} | Lyon: {p_lyon:5.2f}% | Paris: {p_paris:5.2f}% | Margin: {marg:+6.2f} | Rollout: \"{r_res['generated_text']}\"")

            # 3. Control Prompts Evaluation
            print(f"\n  --- Control Prompts (beta={beta:.2f}) ---")
            for p in CONTROL_PROMPTS:
                pid = p["id"]
                d_res = evaluate_direct_next_token(model, tokenizer, p["prompt"])
                r_res = generate_autoregressive_rollout(model, tokenizer, p["prompt"], MAX_NEW_TOKENS)
                mod_direct_eval[pid] = d_res
                mod_rollout_eval[pid] = r_res

                b_ids = baseline_rollout_eval[pid]["generated_token_ids"]
                m_ids = r_res["generated_token_ids"]
                match = (b_ids == m_ids)
                print(f"    [{pid:<20}] Match Baseline: {match!s:<5} | Rollout: \"{r_res['generated_text']}\"")

            # 4. Safeguard Prompts Evaluation
            print(f"\n  --- Safeguard Prompts (beta={beta:.2f}) ---")
            for p in SAFEGUARD_PROMPTS:
                pid = p["id"]
                d_res = evaluate_direct_next_token(model, tokenizer, p["prompt"])
                r_res = generate_autoregressive_rollout(model, tokenizer, p["prompt"], MAX_NEW_TOKENS)
                mod_direct_eval[pid] = d_res
                mod_rollout_eval[pid] = r_res

                b_ids = baseline_rollout_eval[pid]["generated_token_ids"]
                m_ids = r_res["generated_token_ids"]
                match = (b_ids == m_ids)
                print(f"    [{pid:<20}] Match Baseline: {match!s:<5} | Rollout: \"{r_res['generated_text']}\"")

            # 5. Full-Vocabulary Drift Analysis for Safeguards and Controls
            print(f"\n  --- Full-Vocabulary Drift Calculations (Safeguards & Controls) ---")
            # Create a clone model for pristine reference in drift computation
            # Temporarily revert to compute baseline logits, or compute with modified
            model.model.layers[LAYER_IDX].mlp.down_proj.weight.data.copy_(snapshot_w20)
            # Create a second model in-memory or pass both states
            # To be memory efficient, we can evaluate out_base then reload W_mod
            drift_records = {}
            for sg_or_ctrl in SAFEGUARD_PROMPTS + CONTROL_PROMPTS:
                pid = sg_or_ctrl["id"]
                prompt_text = sg_or_ctrl["prompt"]
                b_ids = baseline_rollout_eval[pid]["generated_token_ids"]
                prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].tolist()
                full_seq_ids = prompt_ids + b_ids
                inputs = torch.tensor([full_seq_ids])

                # 1. Base forward
                model.model.layers[LAYER_IDX].mlp.down_proj.weight.data.copy_(snapshot_w20)
                with torch.no_grad():
                    out_base = model(inputs).logits[0].cpu()

                # 2. Mod forward
                model.model.layers[LAYER_IDX].mlp.down_proj.weight.data.copy_(W_mod_f32)
                with torch.no_grad():
                    out_mod = model(inputs).logits[0].cpu()

                # Compute per-step drift
                num_steps = len(b_ids)
                prompt_len = len(prompt_ids)
                per_step_drift = []
                for s in range(num_steps):
                    pos = prompt_len - 1 + s
                    l_b = out_base[pos]
                    l_m = out_mod[pos]

                    max_abs = torch.max(torch.abs(l_m - l_b)).item()
                    mean_abs = torch.mean(torch.abs(l_m - l_b)).item()
                    rel_diff = (torch.norm(l_m - l_b) / torch.norm(l_b)).item()

                    p_b = F.softmax(l_b, dim=-1)
                    p_m = F.softmax(l_m, dim=-1)
                    log_p_m = F.log_softmax(l_m, dim=-1)
                    kl = F.kl_div(log_p_m, p_b, reduction="batchmean").item()

                    t1_b = torch.argmax(l_b).item()
                    t1_m = torch.argmax(l_m).item()
                    top5_b = set(torch.topk(l_b, 5).indices.tolist())
                    top5_m = set(torch.topk(l_m, 5).indices.tolist())

                    per_step_drift.append({
                        "step": s + 1,
                        "max_abs_logit_diff": max_abs,
                        "mean_abs_logit_diff": mean_abs,
                        "relative_logit_diff": rel_diff,
                        "kl_divergence": kl,
                        "top1_match": bool(t1_b == t1_m),
                        "top5_overlap": len(top5_b & top5_m)
                    })
                drift_records[pid] = per_step_drift

        finally:
            # Revert model to pristine snapshot
            model.model.layers[LAYER_IDX].mlp.down_proj.weight.data.copy_(snapshot_w20)

        # Confirm model reverted properly
        is_reverted = torch.equal(model.model.layers[LAYER_IDX].mlp.down_proj.weight, snapshot_w20)
        assert is_reverted, f"Fatal: Model weight failed to revert to pristine snapshot after beta={beta}!"

        # Aggregate summary checks for this beta
        all_sg_identical = all(
            baseline_rollout_eval[p["id"]]["generated_token_ids"] == mod_rollout_eval[p["id"]]["generated_token_ids"]
            for p in SAFEGUARD_PROMPTS
        )
        all_ctrl_preserved = all(
            baseline_rollout_eval[p["id"]]["generated_token_ids"] == mod_rollout_eval[p["id"]]["generated_token_ids"]
            for p in CONTROL_PROMPTS
        )

        # Max logit drift across all safeguards
        sg_max_drifts = [
            max(s["max_abs_logit_diff"] for s in drift_records[p["id"]])
            for p in SAFEGUARD_PROMPTS
        ]
        peak_sg_drift = max(sg_max_drifts) if sg_max_drifts else 0.0

        beta_record = {
            "beta": beta,
            "lambda_reg": LAMBDA_REG,
            "delta_w_frobenius_norm": norm_delta_w,
            "parameter_drift_ratio": param_drift,
            "target_residual_relative": target_res_rel,
            "target_residual_absolute": target_res_abs,
            "target_accuracy_pct": target_accuracy_pct,
            "safeguard_residual_relative": safe_res_rel,
            "safeguard_residual_absolute": safe_res_abs,
            "safeguard_orthogonality_leakage": ortho_leakage,
            "all_safeguards_identical_12_tokens": all_sg_identical,
            "all_controls_preserved": all_ctrl_preserved,
            "peak_safeguard_logit_drift": peak_sg_drift,
            "direct_evaluations": mod_direct_eval,
            "rollout_evaluations": mod_rollout_eval,
            "full_vocab_drift_evaluations": drift_records
        }
        surgery_results.append(beta_record)

    # 8. Final Model Immutability Verification
    is_pristine_final = torch.equal(model.model.layers[LAYER_IDX].mlp.down_proj.weight, snapshot_w20)
    print(f"\n[Step 8] Final Model Immutability Verification: {is_pristine_final}")
    assert is_pristine_final, "Fatal: Pristine model was corrupted!"

    # 9. Serialize Structured Output
    full_artifact = {
        "metadata": {
            "title": "Experiment 7.11: Multi-Key France -> Lyon Paraphrase Pilot (TREC-LS)",
            "model": MODEL_NAME,
            "layer_idx": LAYER_IDX,
            "training_prompts": [p["prompt"] for p in TRAINING_PROMPTS],
            "heldout_prompts": [p["prompt"] for p in HELDOUT_PROMPTS],
            "control_prompts": [p["prompt"] for p in CONTROL_PROMPTS],
            "safeguard_prompts": [p["prompt"] for p in SAFEGUARD_PROMPTS],
            "tested_betas": BETAS,
            "lambda_reg": LAMBDA_REG,
            "max_new_tokens": MAX_NEW_TOKENS,
            "city_token_ids": CITY_TOKEN_IDS,
            "safeguard_basis_rank": RANK_S,
            "training_keys_diagnostics": {
                "raw_cosine_similarity": cos_raw,
                "projected_cosine_similarity": cos_proj,
                "singular_values": [S_kp[0].item(), S_kp[1].item()],
                "condition_number_rectangular": cond_K_proj,
                "condition_number_gram": cond_Gram,
                "target_preservation_matrix": T_preserv.cpu().tolist()
            }
        },
        "baseline_state": {
            "direct_evaluations": baseline_direct_eval,
            "rollout_evaluations": baseline_rollout_eval
        },
        "surgery_states": surgery_results,
        "integrity_verification": {
            "model_weights_bitwise_unmodified": is_pristine_final,
            "independent_runs_from_pristine_w0": True,
            "float64_solver_used": True,
            "svd_rank_31_safeguard_basis": True
        }
    }

    with open(output_json_path, "w") as f:
        json.dump(full_artifact, f, indent=2)

    print(f"\nSaved complete experimental results to: {output_json_path}")
    print("=" * 95)
    print("EXPERIMENT 7.11 MULTI-KEY PILOT COMPLETED SUCCESSFULLY")
    print("=" * 95)

    return full_artifact


if __name__ == "__main__":
    run_experiment_7_11()

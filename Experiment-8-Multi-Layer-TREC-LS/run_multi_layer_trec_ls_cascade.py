"""
Experiment 8: Multi-Layer Sequential TREC-LS Cascade
=====================================================
Evaluates 4 arms across beta_total in [0.08, 0.10, 0.12, 0.15] with lambda = 0.10 in float64
on Qwen/Qwen2.5-0.5B to test whether targeting the causal bottleneck (Arm B: Single L21)
or distributing perturbations across the causal cascade (Arm C: L16, 19, 20, 21; Arm D: L16 subj+final, 19-21 final)
resolves P3 paraphrase resistance while maintaining safeguard preservation and avoiding cloze degradation.

Arms:
  - Arm A (single_L20_baseline): L20 (w=1.0), final tokens of P1, P4 (M=2)
  - Arm B (single_L21_bottleneck): L21 (w=1.0), final tokens of P1, P4 (M=2)
  - Arm C (cascade_L16_19_20_21_final): L16 (0.217), L19 (0.148), L20 (0.092), L21 (0.543), final tokens (M=2)
  - Arm D (cascade_L16_subj_final_L19_20_21_final): L16 (0.217; M=4 subj+final), L19 (0.148; M=2), L20 (0.092; M=2), L21 (0.543; M=2)
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CANDIDATE_LAYERS = [16, 19, 20, 21]
BETAS = [0.08, 0.10, 0.12, 0.15]
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
        "expected_baseline_city": "the"
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
        "expected_baseline_city": "The"
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

ALL_EVAL_PROMPTS = TRAINING_PROMPTS + HELDOUT_PROMPTS + CONTROL_PROMPTS + SAFEGUARD_PROMPTS


@torch.no_grad()
def restore_pristine_weights(
    model: torch.nn.Module,
    w0_cache: Dict[int, torch.Tensor],
) -> None:
    """Restores exact baseline weights and verifies bitwise equality."""
    for l, w0 in w0_cache.items():
        weight_param = model.model.layers[l].mlp.down_proj.weight
        weight_param.data.copy_(w0)
        assert torch.equal(weight_param.data, w0), f"Bitwise restoration failed at Layer {l}!"


@torch.no_grad()
def find_subject_token_index(
    tokenizer,
    input_ids: torch.Tensor,
    subject_str: str = " France",
) -> int:
    """Finds the exact token index of the space-prefixed subject token."""
    subj_ids = tokenizer.encode(subject_str, add_special_tokens=False)
    assert len(subj_ids) == 1, f"Expected single token for {subject_str!r}, got {subj_ids}"
    target_id = subj_ids[0]
    matches = (input_ids[0] == target_id).nonzero(as_tuple=True)[0]
    assert len(matches) >= 1, f"Subject token {subject_str!r} not found in {input_ids}"
    return int(matches[-1].item())


@torch.no_grad()
def evaluate_direct_next_token(
    model: torch.nn.Module,
    tokenizer,
    prompt_text: str
) -> Dict[str, Any]:
    """Computes direct next-token logits and probabilities over the full vocabulary."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].detach().cpu()  # [vocab_size]
    probs = F.softmax(logits, dim=-1)

    lyon_id = CITY_TOKEN_IDS["Lyon"]
    paris_id = CITY_TOKEN_IDS["Paris"]

    lyon_logit = logits[lyon_id].item()
    paris_logit = logits[paris_id].item()
    lyon_prob = probs[lyon_id].item()
    paris_prob = probs[paris_id].item()
    margin = lyon_logit - paris_logit

    top5 = torch.topk(probs, 5)
    top5_list = [
        {
            "rank": r + 1,
            "token_id": top5.indices[r].item(),
            "token_str": tokenizer.decode([top5.indices[r].item()]),
            "prob": top5.values[r].item(),
            "logit": logits[top5.indices[r].item()].item()
        }
        for r in range(5)
    ]

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


@torch.no_grad()
def generate_autoregressive_rollout(
    model: torch.nn.Module,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> Dict[str, Any]:
    """Generates autoregressively with greedy decoding."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

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
        "has_cloze_artifact": ("____" in gen_text or "__" in gen_text or "..." in gen_text and len(gen_text.strip()) <= 3),
        "num_generated_tokens": len(gen_token_ids)
    }


@torch.no_grad()
def run_sequential_cascade_arm(
    model: torch.nn.Module,
    tokenizer,
    layer_weights: Dict[int, float],
    beta_total: float,
    lam: float,
    steering_dir: torch.Tensor,          # [896] float64 normalized (E_Lyon - E_Paris)
    train_input_ids: List[torch.Tensor], # [P1_ids, P4_ids]
    safeguard_input_ids: List[torch.Tensor],
    include_subject_at_l16: bool = False,
    rank_s: int = 31,
) -> Dict[int, Dict[str, Any]]:
    """
    Sequentially applies TREC-LS across sorted(layer_weights.keys()) in float64.
    Re-extracts live activations at each layer after upstream weight edits are committed.
    """
    layer_audits: Dict[int, Dict[str, Any]] = {}

    for l in sorted(layer_weights.keys()):
        w_l = layer_weights[l]
        beta_l = beta_total * w_l
        if beta_l <= 0.0:
            continue

        down_proj = model.model.layers[l].mlp.down_proj
        W0 = down_proj.weight.data.to(torch.float64)  # [896, 4864]

        # 1. Capture live safeguard activations K_S^(l) [4864, 37]
        s_bucket: List[torch.Tensor] = []
        h_s = down_proj.register_forward_hook(
            lambda m, inp, out: s_bucket.append(inp[0].detach().to(torch.float64).squeeze(0))
        )
        for s_ids in safeguard_input_ids:
            model(s_ids)
        h_s.remove()
        K_S = torch.cat(s_bucket, dim=0).T.contiguous()  # [4864, N_S] (37 tokens)

        # 2. Capture live training keys K_train^(l) [4864, M]
        t_bucket: List[torch.Tensor] = []
        h_t = down_proj.register_forward_hook(
            lambda m, inp, out: t_bucket.append(inp[0].detach().to(torch.float64).squeeze(0))
        )
        for t_ids in train_input_ids:
            model(t_ids)
        h_t.remove()

        train_cols: List[torch.Tensor] = []
        for t_ids, seq_acts in zip(train_input_ids, t_bucket):
            if include_subject_at_l16 and l == 16:
                subj_idx = find_subject_token_index(tokenizer, t_ids, " France")
                train_cols.append(seq_acts[subj_idx])
            train_cols.append(seq_acts[-1])  # Final token (" is")

        K_train = torch.stack(train_cols, dim=1)  # [4864, M] (M=4 for Arm D L16, else M=2)
        M = K_train.shape[1]

        # 3. Thin SVD on K_S^(l) & Null-Space Projection
        U, S, _ = torch.linalg.svd(K_S, full_matrices=False)
        rank = rank_s
        U_S = U[:, :rank]  # [4864, rank]
        orthonorm_err = torch.linalg.norm(
            U_S.T @ U_S - torch.eye(rank, dtype=torch.float64, device=W0.device)
        ).item()

        # Double-projected training keys in null space of U_S
        K_tilde = K_train - U_S @ (U_S.T @ K_train)
        K_tilde = K_tilde - U_S @ (U_S.T @ K_tilde)

        # 4. Target shift matrix Delta_V [896, M]
        base_out_norms = torch.linalg.norm(W0 @ K_train, dim=0)  # [M]
        Delta_V = beta_l * steering_dir.unsqueeze(1) * base_out_norms.unsqueeze(0)

        # 5. Solve TREC-LS: Delta_W = Delta_V (K_tilde^T K_tilde + lam I_M)^(-1) K_tilde^T
        gram = K_tilde.T @ K_tilde + lam * torch.eye(M, dtype=torch.float64, device=W0.device)
        cond_num = torch.linalg.cond(gram).item()
        coeffs = torch.linalg.solve(gram, K_tilde.T)  # [M, 4864]
        Delta_W = Delta_V @ coeffs  # [896, 4864]

        # 6. Strict float64 double-orthogonalization against U_S
        Delta_W = Delta_W - (Delta_W @ U_S) @ U_S.T
        Delta_W = Delta_W - (Delta_W @ U_S) @ U_S.T

        # 7. Numerical audits
        f64_safeguard_res = (
            torch.linalg.norm(Delta_W @ K_S) / (torch.linalg.norm(W0) * torch.linalg.norm(K_S))
        ).item()
        rel_fro_drift = (torch.linalg.norm(Delta_W) / torch.linalg.norm(W0)).item()
        target_fit_err = (
            torch.linalg.norm(Delta_W @ K_train - Delta_V) / torch.linalg.norm(Delta_V)
        ).item()

        # Commit update to model weights
        down_proj.weight.data.copy_((W0 + Delta_W).to(down_proj.weight.dtype))

        layer_audits[l] = {
            "layer": l,
            "w_l": float(w_l),
            "beta_l": float(beta_l),
            "num_train_keys_M": int(M),
            "safeguard_rank": int(rank),
            "sigma_max": float(S[0].item()),
            "sigma_rank": float(S[rank - 1].item()),
            "orthonorm_err": float(orthonorm_err),
            "gram_cond_num": float(cond_num),
            "f64_safeguard_rel_residual": float(f64_safeguard_res),
            "rel_weight_drift_fro": float(rel_fro_drift),
            "target_relative_fit_error": float(target_fit_err),
        }

    return layer_audits


def main():
    print("=" * 95)
    print("EXPERIMENT 8: MULTI-LAYER SEQUENTIAL TREC-LS CASCADE (QWEN2.5-0.5B)")
    print("=========================================================================================")

    t0_total = time.time()
    torch.manual_seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_raw_json = os.path.join(script_dir, "multi_layer_cascade_results.json")
    output_summary_json = os.path.join(script_dir, "evaluation_summary.json")
    output_report_md = os.path.join(script_dir, "Research-Report-Experiment-8.md")

    # 1. Load model & tokenizer
    print(f"\n[Step 1] Loading pristine model {MODEL_NAME}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        local_files_only=True
    )
    model.eval()

    # Pre-cache pristine baseline weights W0 across candidate layers
    print("\n[Step 2] Caching pristine baseline weights across candidate layers...")
    w0_cache: Dict[int, torch.Tensor] = {}
    for l in CANDIDATE_LAYERS:
        w0 = model.model.layers[l].mlp.down_proj.weight.data.clone()
        w0_cache[l] = w0
        norm_w0 = torch.norm(w0.to(torch.float64), p="fro").item()
        print(f"  Layer {l} down_proj cached: shape {list(w0.shape)}, ||W_0||_F = {norm_w0:.4f}")

    # Construct normalized steering direction d from lm_head.weight
    print("\n[Step 3] Constructing normalized steering direction d (Lyon - Paris)...")
    d_out = model.lm_head.weight.shape[1]
    E_lyon = model.lm_head.weight[CITY_TOKEN_IDS["Lyon"]].detach().clone().to(torch.float64)
    E_paris = model.lm_head.weight[CITY_TOKEN_IDS["Paris"]].detach().clone().to(torch.float64)
    diff_E = E_lyon - E_paris
    steering_dir = diff_E / torch.norm(diff_E)
    print(f"  Steering direction d: shape {list(steering_dir.shape)}, norm = {torch.norm(steering_dir).item():.6f}")

    # Prepare input IDs
    train_input_ids = [tokenizer(p["prompt"], return_tensors="pt")["input_ids"] for p in TRAINING_PROMPTS]
    safeguard_input_ids = [tokenizer(p["prompt"], return_tensors="pt")["input_ids"] for p in SAFEGUARD_PROMPTS]

    # Pre-evaluate PRISTINE BASELINE
    print("\n[Step 4] Pre-evaluating pristine baseline behavior...")
    baseline_next_token: Dict[str, Any] = {}
    baseline_rollouts: Dict[str, Any] = {}
    baseline_drift_targets: Dict[str, Any] = {}

    for p_info in ALL_EVAL_PROMPTS:
        pid = p_info["id"]
        p_text = p_info["prompt"]
        baseline_next_token[pid] = evaluate_direct_next_token(model, tokenizer, p_text)
        rollout = generate_autoregressive_rollout(model, tokenizer, p_text, MAX_NEW_TOKENS)
        baseline_rollouts[pid] = rollout

        # For safeguards and controls, pre-cache the exact sequence logits for drift checks
        if pid.startswith("safe_") or pid.startswith("control_"):
            p_ids = tokenizer(p_text, return_tensors="pt")["input_ids"][0].tolist()
            full_seq = p_ids + rollout["generated_token_ids"]
            inputs = torch.tensor([full_seq])
            with torch.no_grad():
                out_logits = model(inputs).logits[0].cpu()  # [seq_len, vocab_size]
            baseline_drift_targets[pid] = {
                "prompt_ids": p_ids,
                "full_seq_ids": full_seq,
                "gen_token_ids": rollout["generated_token_ids"],
                "logits": out_logits
            }

    print("  Baseline evaluations complete.")
    for pid in ["P1_canonical", "P4_wh_question", "P3_cap_city", "P6_diplomatic", "P8_parenthetical"]:
        b_nt = baseline_next_token[pid]
        b_ro = baseline_rollouts[pid]
        print(f"    {pid}: Top-1={b_nt['top1_token_str']!r} ({b_nt['top1_prob']:.4f}), P(Lyon)={b_nt['lyon_prob']:.4f}, Margin={b_nt['lyon_paris_logit_margin']:.4f} | Rollout: {b_ro['generated_text']!r}")

    # 4-Arm Experimental Matrix Definition
    arms_config = {
        "Arm_A": {
            "name": "single_L20_baseline",
            "layer_weights": {20: 1.0},
            "subj_l16": False,
            "description": "Single-layer Layer 20 baseline (Task 7.11 replication)"
        },
        "Arm_B": {
            "name": "single_L21_bottleneck",
            "layer_weights": {21: 1.0},
            "subj_l16": False,
            "description": "Single-layer Layer 21 causal bottleneck edit"
        },
        "Arm_C": {
            "name": "cascade_L16_19_20_21_final",
            "layer_weights": {16: 0.217, 19: 0.148, 20: 0.092, 21: 0.543},
            "subj_l16": False,
            "description": "4-Layer Sequential Cascade (final tokens M=2 at all layers)"
        },
        "Arm_D": {
            "name": "cascade_L16_subj_final_L19_20_21_final",
            "layer_weights": {16: 0.217, 19: 0.148, 20: 0.092, 21: 0.543},
            "subj_l16": True,
            "description": "4-Layer Cascade with Subject-Token Key at L16 (M=4 at L16, M=2 at L19-21)"
        }
    }

    all_results: Dict[str, Any] = {
        "metadata": {
            "model_name": MODEL_NAME,
            "candidate_layers": CANDIDATE_LAYERS,
            "betas": BETAS,
            "lambda_reg": LAMBDA_REG,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_runs": len(arms_config) * len(BETAS),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "baseline": {
            "next_token": baseline_next_token,
            "rollouts": {
                pid: {k: v for k, v in ro.items() if k != "full_text"}
                for pid, ro in baseline_rollouts.items()
            }
        },
        "runs": {}
    }

    summary_table_rows = []

    print("\n" + "=" * 95)
    print("STARTING 16-RUN EXPERIMENTAL SWEEP")
    print("=" * 95)

    run_idx = 0
    total_runs = len(arms_config) * len(BETAS)

    for arm_id, arm in arms_config.items():
        for beta in BETAS:
            run_idx += 1
            run_key = f"{arm_id}__beta_{beta:.2f}"
            print(f"\n[{run_idx}/{total_runs}] Running {run_key} ({arm['name']}, beta={beta:.2f})...")

            # 1. Ensure model is bitwise pristine
            restore_pristine_weights(model, w0_cache)

            # 2. Apply sequential cascade
            t_start = time.time()
            layer_audits = run_sequential_cascade_arm(
                model=model,
                tokenizer=tokenizer,
                layer_weights=arm["layer_weights"],
                beta_total=beta,
                lam=LAMBDA_REG,
                steering_dir=steering_dir,
                train_input_ids=train_input_ids,
                safeguard_input_ids=safeguard_input_ids,
                include_subject_at_l16=arm["subj_l16"],
                rank_s=31
            )
            cascade_time = time.time() - t_start

            # 3. Evaluate edited model
            eval_next_token: Dict[str, Any] = {}
            eval_rollouts: Dict[str, Any] = {}
            drift_audits: Dict[str, Any] = {}

            # Evaluate next token & rollouts for all prompts
            for p_info in ALL_EVAL_PROMPTS:
                pid = p_info["id"]
                p_text = p_info["prompt"]
                eval_next_token[pid] = evaluate_direct_next_token(model, tokenizer, p_text)
                eval_rollouts[pid] = generate_autoregressive_rollout(model, tokenizer, p_text, MAX_NEW_TOKENS)

            # Full vocab drift audit on safeguards (5) and controls (3) -> 8 prompts * 12 tokens = 96 decisions
            total_divergences = 0
            max_boundary_logit_diff = 0.0
            max_post_boundary_logit_diff = 0.0
            max_post_boundary_kl = 0.0

            for pid in baseline_drift_targets:
                b_tgt = baseline_drift_targets[pid]
                full_seq = b_tgt["full_seq_ids"]
                b_logits = b_tgt["logits"]  # [seq_len, vocab_size]
                prompt_len = len(b_tgt["prompt_ids"])
                b_gen_ids = b_tgt["gen_token_ids"]

                with torch.no_grad():
                    inputs = torch.tensor([full_seq])
                    m_logits = model(inputs).logits[0].cpu()

                # Step 1: boundary token
                b_idx = prompt_len - 1
                diff_b = torch.max(torch.abs(m_logits[b_idx] - b_logits[b_idx])).item()
                if diff_b > max_boundary_logit_diff:
                    max_boundary_logit_diff = diff_b

                # Generated rollout comparison
                mod_gen_ids = eval_rollouts[pid]["generated_token_ids"]
                diverged_tokens = sum(1 for g1, g2 in zip(mod_gen_ids, b_gen_ids) if g1 != g2)
                diverged_tokens += abs(len(mod_gen_ids) - len(b_gen_ids))
                total_divergences += diverged_tokens

                # Subsequent steps drift on fixed baseline trajectory
                step_drifts = []
                for s in range(len(b_gen_ids)):
                    pos = prompt_len - 1 + s
                    l_base = b_logits[pos]
                    l_mod = m_logits[pos]

                    max_abs = torch.max(torch.abs(l_mod - l_base)).item()
                    p_base = F.softmax(l_base, dim=-1)
                    p_mod = F.softmax(l_mod, dim=-1)
                    log_p_base = F.log_softmax(l_base, dim=-1)
                    log_p_mod = F.log_softmax(l_mod, dim=-1)
                    kl_div = F.kl_div(log_p_mod, p_base, reduction="batchmean").item()

                    if s > 0:
                        if max_abs > max_post_boundary_logit_diff:
                            max_post_boundary_logit_diff = max_abs
                        if kl_div > max_post_boundary_kl:
                            max_post_boundary_kl = kl_div

                    step_drifts.append({
                        "step": s + 1,
                        "max_abs_logit_diff": max_abs,
                        "kl_divergence": kl_div
                    })

                drift_audits[pid] = {
                    "token_divergences": diverged_tokens,
                    "exact_rollout_match": bool(diverged_tokens == 0),
                    "step_drifts": step_drifts
                }

            # Record run results
            run_data = {
                "arm": arm_id,
                "arm_name": arm["name"],
                "beta_total": beta,
                "layer_audits": layer_audits,
                "safeguard_audit": {
                    "total_divergences_out_of_96": total_divergences,
                    "exact_safeguard_preservation_rate": float((96 - total_divergences) / 96.0 * 100.0),
                    "max_boundary_logit_diff": max_boundary_logit_diff,
                    "max_post_boundary_logit_diff": max_post_boundary_logit_diff,
                    "max_post_boundary_kl": max_post_boundary_kl,
                },
                "next_token": eval_next_token,
                "rollouts": {
                    pid: {k: v for k, v in ro.items() if k != "full_text"}
                    for pid, ro in eval_rollouts.items()
                }
            }
            all_results["runs"][run_key] = run_data

            # Summary metrics for quick comparison
            p1_nt = eval_next_token["P1_canonical"]
            p4_nt = eval_next_token["P4_wh_question"]
            p3_nt = eval_next_token["P3_cap_city"]
            p2_nt = eval_next_token["P2_possessive"]
            p6_nt = eval_next_token["P6_diplomatic"]
            p8_nt = eval_next_token["P8_parenthetical"]

            p1_ro = eval_rollouts["P1_canonical"]
            p3_ro = eval_rollouts["P3_cap_city"]
            p6_ro = eval_rollouts["P6_diplomatic"]
            p8_ro = eval_rollouts["P8_parenthetical"]

            # Max relative weight drift across layers
            max_rel_w_drift = max(la["rel_weight_drift_fro"] for la in layer_audits.values())
            max_sg_res = max(la["f64_safeguard_rel_residual"] for la in layer_audits.values())

            row = {
                "run_key": run_key,
                "arm": arm_id,
                "arm_name": arm["name"],
                "beta": beta,
                "max_rel_w_drift": max_rel_w_drift,
                "max_sg_res": max_sg_res,
                "divergences_out_of_96": total_divergences,
                "P1_margin": p1_nt["lyon_paris_logit_margin"],
                "P1_prob_lyon": p1_nt["lyon_prob"],
                "P1_rollout_lyon": p1_ro["contains_lyon"],
                "P4_margin": p4_nt["lyon_paris_logit_margin"],
                "P4_prob_lyon": p4_nt["lyon_prob"],
                "P3_margin": p3_nt["lyon_paris_logit_margin"],
                "P3_prob_lyon": p3_nt["lyon_prob"],
                "P3_rollout_lyon": p3_ro["contains_lyon"],
                "P3_rollout_text": p3_ro["generated_text"],
                "P6_margin": p6_nt["lyon_paris_logit_margin"],
                "P6_prob_lyon": p6_nt["lyon_prob"],
                "P6_rollout_text": p6_ro["generated_text"],
                "P6_cloze_artifact": p6_ro["has_cloze_artifact"],
                "P8_margin": p8_nt["lyon_paris_logit_margin"],
                "P8_prob_lyon": p8_nt["lyon_prob"],
                "P8_rollout_lyon": p8_ro["contains_lyon"],
                "P2_prob_lyon": p2_nt["lyon_prob"],
            }
            summary_table_rows.append(row)

            print(f"  --> Max Rel ||dW||_F: {max_rel_w_drift*100:.3f}%, Safeguard Residual: {max_sg_res:.2e}, Divergences: {total_divergences}/96")
            print(f"  --> P1: Prob(Lyon)={p1_nt['lyon_prob']:.4f}, Margin={p1_nt['lyon_paris_logit_margin']:.2f} | Rollout: {p1_ro['generated_text']!r}")
            print(f"  --> P3: Prob(Lyon)={p3_nt['lyon_prob']:.4f}, Margin={p3_nt['lyon_paris_logit_margin']:.2f} | Rollout: {p3_ro['generated_text']!r}")
            print(f"  --> P6: Prob(Lyon)={p6_nt['lyon_prob']:.4f}, Margin={p6_nt['lyon_paris_logit_margin']:.2f} | Rollout: {p6_ro['generated_text']!r}")
            print(f"  --> P8: Prob(Lyon)={p8_nt['lyon_prob']:.4f}, Margin={p8_nt['lyon_paris_logit_margin']:.2f} | Rollout: {p8_ro['generated_text']!r}")

    # Final pristine restoration
    restore_pristine_weights(model, w0_cache)
    print("\n[Step 5] Sweep complete! Bitwise pristine restoration verified.")

    # Save full raw results
    print(f"\n[Step 6] Saving full results to {output_raw_json}...")
    with open(output_raw_json, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save structured summary
    summary_payload = {
        "metadata": all_results["metadata"],
        "baseline_summary": {
            "P1_margin": baseline_next_token["P1_canonical"]["lyon_paris_logit_margin"],
            "P1_prob_paris": baseline_next_token["P1_canonical"]["paris_prob"],
            "P3_margin": baseline_next_token["P3_cap_city"]["lyon_paris_logit_margin"],
            "P3_prob_paris": baseline_next_token["P3_cap_city"]["paris_prob"],
            "P6_margin": baseline_next_token["P6_diplomatic"]["lyon_paris_logit_margin"],
            "P6_prob_paris": baseline_next_token["P6_diplomatic"]["paris_prob"],
            "P8_margin": baseline_next_token["P8_parenthetical"]["lyon_paris_logit_margin"],
            "P8_prob_paris": baseline_next_token["P8_parenthetical"]["paris_prob"],
        },
        "rows": summary_table_rows
    }
    with open(output_summary_json, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"  Structured summary saved to {output_summary_json}.")

    # Generate Research Report Markdown
    print(f"\n[Step 7] Generating Research Report {output_report_md}...")
    report_md = generate_research_report(all_results, summary_table_rows, baseline_next_token, baseline_rollouts)
    with open(output_report_md, "w") as f:
        f.write(report_md)
    print(f"  Research report saved to {output_report_md}.")

    t_total = time.time() - t0_total
    print(f"\n[DONE] Experiment 8 executed successfully in {t_total:.2f}s ({t_total/60.0:.2f}m)!")


def generate_research_report(
    all_results: Dict[str, Any],
    rows: List[Dict[str, Any]],
    base_nt: Dict[str, Any],
    base_ro: Dict[str, Any]
) -> str:
    num_runs = all_results['metadata']['num_runs']
    timestamp = all_results['metadata']['timestamp']
    out: List[str] = []
    out.append("# Research Report: Experiment 8 — Multi-Layer Sequential TREC-LS Cascade\n")
    out.append("**Model:** `Qwen/Qwen2.5-0.5B` (24 Layers, d_model=896, d_mlp=4864)  ")
    out.append(f"**Execution Timestamp:** {timestamp}  ")
    out.append(f"**Total Sweep Runs:** {num_runs} (4 Arms x 4 beta values)  ")
    out.append(f"**Regularization Parameter:** lambda = {LAMBDA_REG}  ")
    out.append("**Precision:** `torch.float64` for SVD, Gram inversion, null-space projection, and update arithmetic  \n")
    out.append("---\n")
    out.append("## 1. Executive Summary & Core Scientific Answers\n")
    out.append("### Question 1: Arm B vs. Arm A (Layer 21 Bottleneck vs. Layer 20 Baseline)")
    out.append("Does targeting the causal bottleneck (Layer 21) cure the P3 ('The capital city of France is') paraphrase resistance discovered in Task 7.11?\n")
    out.append("| Configuration | beta | P1 Margin | P1 Rollout | P3 Margin | P3 Rollout | P6 Rollout Text | P8 Rollout | Divergences |")
    out.append("|---|---|---|---|---|---|---|---|---|")

    rows_a = [r for r in rows if r['arm'] == 'Arm_A']
    rows_b = [r for r in rows if r['arm'] == 'Arm_B']
    rows_c = [r for r in rows if r['arm'] == 'Arm_C']
    rows_d = [r for r in rows if r['arm'] == 'Arm_D']

    for r in rows_a + rows_b:
        p3_ro = r['P3_rollout_text'].replace("\n", " ").strip()[:25]
        p6_ro = r['P6_rollout_text'].replace("\n", " ").strip()[:30]
        p1_r = "Lyon" if r['P1_rollout_lyon'] else "Paris"
        p8_r = "Lyon" if r['P8_rollout_lyon'] else "Paris"
        div = f"{r['divergences_out_of_96']}/96"
        out.append(f"| {r['arm_name']} | {r['beta']:.2f} | {r['P1_margin']:+.2f} | {p1_r} | {r['P3_margin']:+.2f} | {p3_ro!r} | {p6_ro!r} | {p8_r} | {div} |")

    out.append("\n### Question 2: Arm C vs. Arm B (4-Layer Sequential Cascade vs. Single-Layer L21)")
    out.append("Does distributing the perturbation across the causal cascade ({16, 19, 20, 21}) improve fluency, transfer margins, or parameter drift compared to editing Layer 21 alone?\n")
    out.append("| Configuration | beta | Max rel ||dW||_F | P1 Margin | P3 Margin | P6 Margin | P8 Margin | P6 Cloze Artifact? | Divergences |")
    out.append("|---|---|---|---|---|---|---|---|---|")

    for r in rows_b + rows_c:
        w_drift = f"{r['max_rel_w_drift']*100:.3f}%"
        cloze = str(r['P6_cloze_artifact'])
        div = f"{r['divergences_out_of_96']}/96"
        out.append(f"| {r['arm_name']} | {r['beta']:.2f} | {w_drift} | {r['P1_margin']:+.2f} | {r['P3_margin']:+.2f} | {r['P6_margin']:+.2f} | {r['P8_margin']:+.2f} | {cloze} | {div} |")

    out.append("\n### Question 3: Arm D vs. Arm C (Impact of Mid-Layer Subject-Token Key at Layer 16)")
    out.append("What is the causal impact of adding the subject token (' France') to the training key matrix at Layer 16 (M=4) versus final-token only (M=2)?\n")
    out.append("| Configuration | beta | P1 Margin | P3 Margin | P3 Rollout | P8 Rollout | P6 Rollout | Divergences |")
    out.append("|---|---|---|---|---|---|---|---|")

    for r in rows_c + rows_d:
        p3_ro = r['P3_rollout_text'].replace("\n", " ").strip()[:25]
        p6_ro = r['P6_rollout_text'].replace("\n", " ").strip()[:25]
        p8_r = "Lyon" if r['P8_rollout_lyon'] else "Paris"
        div = f"{r['divergences_out_of_96']}/96"
        out.append(f"| {r['arm_name']} | {r['beta']:.2f} | {r['P1_margin']:+.2f} | {r['P3_margin']:+.2f} | {p3_ro!r} | {p8_r} | {p6_ro!r} | {div} |")

    out.append("\n---\n")
    out.append("## 2. Complete 16-Run Experimental Master Table\n")
    out.append("| Run Key | Arm | beta | Max Drift rel ||dW||_F | SG Res (f64) | P1 P(Lyon) | P1 Margin | P3 P(Lyon) | P3 Margin | P4 Margin | P6 Margin | P8 Margin | Divergences |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    for r in rows:
        w_drift = f"{r['max_rel_w_drift']*100:.3f}%"
        sg_res = f"{r['max_sg_res']:.1e}"
        div = f"{r['divergences_out_of_96']}/96"
        out.append(f"| `{r['run_key']}` | {r['arm']} | {r['beta']:.2f} | {w_drift} | {sg_res} | {r['P1_prob_lyon']:.4f} | {r['P1_margin']:+.2f} | {r['P3_prob_lyon']:.4f} | {r['P3_margin']:+.2f} | {r['P4_margin']:+.2f} | {r['P6_margin']:+.2f} | {r['P8_margin']:+.2f} | {div} |")

    out.append("\n---\n")
    out.append("## 3. Qualitative Rollout Analysis across Crucial Prompts\n")

    for prompt_key, label in [("P3_cap_city", "P3: 'The capital city of France is'"),
                             ("P6_diplomatic", "P6: 'The official capital city of the French Republic is'"),
                             ("P8_parenthetical", "P8: 'France (capital:'"),
                             ("P1_canonical", "P1: 'The capital of France is'")]:
        b_txt = base_ro[prompt_key]['generated_text']
        out.append(f"### {label}")
        out.append(f"- **Baseline:** `{b_txt}`\n")
        out.append("| Arm | beta=0.08 | beta=0.10 | beta=0.12 | beta=0.15 |")
        out.append("|---|---|---|---|---|")
        for arm_id in ['Arm_A', 'Arm_B', 'Arm_C', 'Arm_D']:
            arm_rows = [r for r in rows if r['arm'] == arm_id]
            gen_texts = []
            for r in arm_rows:
                run_k = r['run_key']
                g_txt = all_results['runs'][run_k]['rollouts'][prompt_key]['generated_text'].replace("\n", " ").strip()
                gen_texts.append(f"`{g_txt}`")
            out.append(f"| {arm_id} | " + " | ".join(gen_texts) + " |")
        out.append("")

    out.append("---\n")
    out.append("## 4. Safeguard & Numerical Stability Audit\n")
    out.append("Across all 16 runs:")
    out.append("- **Double-Orthogonalized Safeguard Residuals:** Evaluated in `float64` across all layers; maximum relative residual ||dW K_S||_F / (||W_0||_F ||K_S||_F) was recorded.")
    out.append("- **Safeguard Sequence Preservation:** Evaluated across 8 prompts (5 factual safeguards + 3 geographic controls) over 12 autoregressive decoding steps (96 total token decisions per run).\n")
    out.append("---\n")
    out.append("## 5. Architectural Conclusions for Production Constrained Surgery\n")
    return "\n".join(out)


if __name__ == '__main__':
    main()

"""
Experiment 3 — Task 7.4: Sequence-Wide Safeguard Protection Audit Diagnostic
=============================================================================
Performs a read-only numerical analysis of sequence-wide safeguard extraction
for Qwen2.5-0.5B Layer 20 down_proj across the 5 pilot safeguard prompts:
  1. "The capital of Germany is Berlin."
  2. "The capital of Italy is Rome."
  3. "The capital of Spain is Madrid."
  4. "Water is composed of hydrogen and oxygen."
  5. "2 + 2 = 4."

Analyzes:
  - Sequence-wide token counts and causal prefix sharing.
  - SVD spectrum, numerical rank, and null-space dimension.
  - Classical vs. effective condition numbers.
  - Pairwise cosine similarities and exact duplicate columns.
  - Compatibility with existing AuditedConstrainedRLS without solver modifications.
"""

import os
import sys
import json
import time
import torch
from typing import Dict, Any, List

sys.path.append(os.path.join(os.path.dirname(__file__), "Experiment-2.1-Robustness-Streaming-Audit"))
from robustness_tests import AuditedConstrainedRLS, SVDNullSpace
from activation_capture import QwenActivationCapturer
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = 20
RESULTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "sequence_wide_safeguards_analysis.json")

PILOT_SAFEGUARD_PROMPTS = [
    {
        "id": "safe_germany",
        "prompt": "The capital of Germany is Berlin.",
        "category": "factual_geography"
    },
    {
        "id": "safe_italy",
        "prompt": "The capital of Italy is Rome.",
        "category": "factual_geography"
    },
    {
        "id": "safe_spain",
        "prompt": "The capital of Spain is Madrid.",
        "category": "factual_geography"
    },
    {
        "id": "safe_science",
        "prompt": "Water is composed of hydrogen and oxygen.",
        "category": "science_chemistry"
    },
    {
        "id": "safe_arithmetic",
        "prompt": "2 + 2 = 4.",
        "category": "arithmetic_logic"
    }
]


def run_sequence_safeguard_audit():
    print("=" * 90)
    print("EXPERIMENT 3 — TASK 7.4: SEQUENCE-WIDE SAFEGUARD PROTECTION AUDIT")
    print("=" * 90)

    # 1. Load model and tokenizer
    print(f"\n[Step 1] Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        local_files_only=True
    )
    model.eval()

    # Snapshot Layer 20 down_proj weight for bitwise verification
    snapshot_w20 = model.model.layers[LAYER_IDX].mlp.down_proj.weight.detach().clone()
    d_out, d_in = snapshot_w20.shape
    W_0_f64 = snapshot_w20.to(dtype=torch.float64)
    print(f"Layer {LAYER_IDX} down_proj weight shape: ({d_out}, {d_in}) [float64 baseline]")

    capturer = QwenActivationCapturer(model, tokenizer)

    # 2. Extract all token input/output activations for each pilot prompt
    print("\n[Step 2] Capturing sequence-wide activations (token_position='all') for all 5 prompts...")
    prompt_analyses: List[Dict[str, Any]] = []
    k_token_list: List[torch.Tensor] = []
    v0_token_list: List[torch.Tensor] = []
    token_metadata: List[Dict[str, Any]] = []

    global_token_idx = 0
    for p_idx, p_info in enumerate(PILOT_SAFEGUARD_PROMPTS):
        p_text = p_info["prompt"]
        p_id = p_info["id"]

        cap = capturer.capture(p_text, layer_indices=LAYER_IDX, token_position="all")
        layer_cap = cap[LAYER_IDX]
        raw_k = layer_cap["input"]   # [seq_len, 4864]
        raw_v0 = layer_cap["output"] # [seq_len, 896]
        meta = layer_cap["metadata"]

        seq_len = meta["sequence_length"]
        token_strings = meta["token_strings"]
        token_ids = meta["token_ids"]

        tokens_info = []
        for pos in range(seq_len):
            k_pos = raw_k[pos].detach().clone().to(torch.float64).view(d_in, 1)
            v0_pos = raw_v0[pos].detach().clone().to(torch.float64).view(d_out, 1)

            k_token_list.append(k_pos)
            v0_token_list.append(v0_pos)

            t_meta = {
                "global_token_index": global_token_idx,
                "prompt_id": p_id,
                "token_position": pos,
                "token_id": token_ids[pos],
                "token_str": token_strings[pos],
                "k_norm": torch.norm(k_pos).item(),
                "v0_norm": torch.norm(v0_pos).item()
            }
            tokens_info.append(t_meta)
            token_metadata.append(t_meta)
            global_token_idx += 1

        p_analysis = {
            "prompt_id": p_id,
            "prompt": p_text,
            "category": p_info["category"],
            "num_tokens": seq_len,
            "token_ids": token_ids,
            "token_strings": token_strings,
            "tokens": tokens_info
        }
        prompt_analyses.append(p_analysis)
        print(f"  Prompt [{p_id}]: {repr(p_text)} ({seq_len} tokens): {token_strings}")

    total_tokens = len(k_token_list)
    print(f"\nTotal sequence-wide tokens captured: {total_tokens}")

    # 3. Stack full sequence-wide safeguard matrix K_S and V_S
    K_S_all = torch.cat(k_token_list, dim=1)   # [4864, total_tokens]
    V_S_all = torch.cat(v0_token_list, dim=1)  # [896, total_tokens]
    print(f"K_S_all shape: {list(K_S_all.shape)}, V_S_all shape: {list(V_S_all.shape)}")

    # 4. Rigorous Numerical Linear Algebra Analysis
    print("\n[Step 3] Performing SVD and spectral analysis of sequence-wide safeguard matrix...")
    U_svd, S_svd, Vh_svd = torch.linalg.svd(K_S_all, full_matrices=False)

    eps_f64 = torch.finfo(torch.float64).eps
    svd_tol = S_svd[0] * max(K_S_all.shape) * eps_f64
    rank_svd = (S_svd > svd_tol).sum().item()
    num_redundant_cols = total_tokens - rank_svd
    null_space_dim = d_in - rank_svd

    # Condition numbers
    sigma_max = S_svd[0].item()
    sigma_min_classical = S_svd[-1].item()
    sigma_min_effective = S_svd[rank_svd - 1].item()
    classical_cond = sigma_max / sigma_min_classical if sigma_min_classical > 0 else float("inf")
    effective_cond = sigma_max / sigma_min_effective if sigma_min_effective > 0 else float("inf")

    print(f"  Total Columns:               {total_tokens}")
    print(f"  Numerical Rank (float64):    {rank_svd}")
    print(f"  Redundant / Zero Dimensions: {num_redundant_cols}")
    print(f"  Null-Space Dimension d_red:  {null_space_dim} (out of {d_in})")
    print(f"  Largest Singular Value s_1:  {sigma_max:.4f}")
    print(f"  Smallest Non-Zero s_rank:    {sigma_min_effective:.4f}")
    print(f"  Trailing Singular Value s_T: {sigma_min_classical:.4e}")
    print(f"  Effective Condition Number:  {effective_cond:.2f}")
    print(f"  Classical Condition Number:  {classical_cond:.2e} (reflects exact duplicate columns)")

    # 5. Exact Duplicate Analysis via Pairwise Cosine Matrix
    print("\n[Step 4] Detecting exact duplicate and near-duplicate columns...")
    K_norms = torch.norm(K_S_all, p=2, dim=0, keepdim=True)
    K_normed = K_S_all / K_norms
    cos_matrix = (K_normed.T @ K_normed).cpu()

    # Find duplicate pairs where cos >= 1.0 - 1e-12 and diff_norm < 1e-10
    exact_duplicates = []
    for i in range(total_tokens):
        for j in range(i + 1, total_tokens):
            cos_val = cos_matrix[i, j].item()
            diff_norm = torch.norm(K_S_all[:, i] - K_S_all[:, j]).item()
            if diff_norm < 1e-10 or (1.0 - cos_val) < 1e-12:
                t_i = token_metadata[i]
                t_j = token_metadata[j]
                exact_duplicates.append({
                    "col_i": i,
                    "col_j": j,
                    "prompt_i": t_i["prompt_id"],
                    "pos_i": t_i["token_position"],
                    "str_i": t_i["token_str"],
                    "prompt_j": t_j["prompt_id"],
                    "pos_j": t_j["token_position"],
                    "str_j": t_j["token_str"],
                    "cosine_similarity": cos_val,
                    "l2_difference": diff_norm
                })

    print(f"  Identified {len(exact_duplicates)} exact duplicate column pairs:")
    for d in exact_duplicates:
        print(f"    Col {d['col_i']:02d} [{d['prompt_i']} pos {d['pos_i']} {repr(d['str_i'])}] == "
              f"Col {d['col_j']:02d} [{d['prompt_j']} pos {d['pos_j']} {repr(d['str_j'])}] "
              f"(L2 diff = {d['l2_difference']:.2e})")

    # 6. Test Out-of-the-Box Solver Compatibility
    print("\n[Step 5] Testing AuditedConstrainedRLS compatibility with sequence-wide K_S...")
    t0_solver = time.perf_counter()
    solver = AuditedConstrainedRLS(
        initial_weight=W_0_f64,
        lambda_reg=0.1,
        K_safe=K_S_all,
        dtype=torch.float64,
        decomposition_mode="svd",
        record_history=False
    )
    t_init_solver = time.perf_counter() - t0_solver

    print(f"  AuditedConstrainedRLS successfully initialized in {t_init_solver:.2f} seconds!")
    print(f"  Solver d_in={solver.d_in}, d_out={solver.d_out}, rank={solver.rank}, d_red={solver.d_red}")
    print(f"  W_tilde shape: {list(solver.W_tilde.shape)}, P_tilde shape: {list(solver.P_tilde.shape)}, U_perp shape: {list(solver.U_perp.shape)}")

    # Verify initial safeguard preservation error
    W_init = solver.W
    safe_err_abs = torch.norm(W_init @ K_S_all - W_0_f64 @ K_S_all, p="fro").item()
    norm_v_all = torch.norm(V_S_all, p="fro").item()
    safe_err_rel = safe_err_abs / norm_v_all if norm_v_all > 0 else safe_err_abs
    print(f"  Initial safeguard preservation error across all {total_tokens} tokens: {safe_err_rel:.2e} (abs: {safe_err_abs:.2e})")

    # Verification: test one mock RLS update on an orthogonal key
    print("\n[Step 6] Testing a single mock update on an independent test key...")
    k_test = torch.randn(d_in, 1, dtype=torch.float64)
    v_test = W_0_f64 @ k_test + 0.1 * torch.randn(d_out, 1, dtype=torch.float64)
    up_res = solver.update(k_test, v_test)
    print(f"  Update accepted: {up_res['updated']}, conflict detected: {up_res['conflict_detected']}")

    W_post = solver.W
    post_safe_err = torch.norm(W_post @ K_S_all - W_0_f64 @ K_S_all, p="fro").item() / norm_v_all
    print(f"  Post-update safeguard preservation error across all {total_tokens} tokens: {post_safe_err:.2e}")
    assert post_safe_err < 1e-12, f"Safeguard error exceeded tolerance: {post_safe_err}"

    # 7. Post-experiment bitwise model immutability
    print("\n[Step 7] Verifying bitwise immutability of PyTorch model parameters...")
    current_w20 = model.model.layers[LAYER_IDX].mlp.down_proj.weight
    is_weight_identical = torch.equal(current_w20, snapshot_w20)
    print(f"  Model Layer 20 weight bitwise identical: {is_weight_identical}")

    # 8. Structure comprehensive JSON output
    results = {
        "metadata": {
            "title": "Experiment 3 — Task 7.4: Sequence-Wide Safeguard Protection Audit",
            "model": MODEL_NAME,
            "layer_idx": LAYER_IDX,
            "num_pilot_prompts": len(PILOT_SAFEGUARD_PROMPTS),
            "total_tokens_captured": total_tokens,
            "d_in": d_in,
            "d_out": d_out
        },
        "prompts": prompt_analyses,
        "spectral_analysis": {
            "total_columns": total_tokens,
            "numerical_rank": rank_svd,
            "num_redundant_columns": num_redundant_cols,
            "null_space_dimension_d_red": null_space_dim,
            "sigma_max": sigma_max,
            "sigma_min_effective": sigma_min_effective,
            "sigma_min_classical": sigma_min_classical,
            "effective_condition_number": effective_cond,
            "classical_condition_number": classical_cond,
            "singular_values": [s.item() for s in S_svd]
        },
        "exact_duplicates": exact_duplicates,
        "solver_compatibility": {
            "compatible_without_code_changes": True,
            "init_time_sec": t_init_solver,
            "solver_rank": solver.rank,
            "solver_d_red": solver.d_red,
            "post_update_safeguard_relative_error": post_safe_err,
            "model_weights_bitwise_unmodified": is_weight_identical
        }
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved structured experimental results to: {RESULTS_JSON_PATH}")
    print("=" * 90)
    print("TASK 7.4 SEQUENCE-WIDE SAFEGUARD AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 90)

    return results


if __name__ == "__main__":
    run_sequence_safeguard_audit()

"""
Experiment 3 — Task 6: Non-Zero Target Validation
=================================================
Validates sequential Constrained RLS updates using real Layer 20 activation keys
and controlled, deterministic non-zero target perturbations from Qwen2.5-0.5B.

Executes:
  1. Capture 25 training and 5 held-out distinct prompt/token activation pairs
     (k in R^4864, v_0 in R^896) from Qwen Layer 20.
  2. Perform rigorous pairwise activation duplicate checks (cosine similarities and L2 distances).
  3. Construct controlled non-zero targets: v_t = W_0 @ k_t + delta_v_t with documented
     relative perturbation alpha = 0.05 (5% relative norm).
  4. Stream training pairs sequentially through AuditedConstrainedRLS.
  5. Measure per-step target residual before/after update, target error reduction,
     safeguard invariance, parameter drift, update latency, and covariance eigenvalues.
  6. Compute independent batch constrained KKT reference solution and compare terminal
     weights, training predictions, held-out predictions, and safeguard preservation.
  7. Formally verify bitwise immutability of the original model weights and absence of
     sample history retention inside the streaming solver.
  8. Save comprehensive experimental results to qwen_nonzero_target_results.json.
"""

import os
import sys
import time
import json
import torch
from typing import Dict, Any, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "Experiment-2.1-Robustness-Streaming-Audit"))
from robustness_tests import AuditedConstrainedRLS, compute_independent_batch_kkt
from activation_capture import QwenActivationCapturer
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
RESULTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "qwen_nonzero_target_results.json")

# 25 Diverse training prompts spanning multiple semantic and structural domains
TRAINING_PROMPTS = [
    # Exponential Reasoning Domain
    {"domain": "exponential", "prompt": "In a lake, there is a patch of lily pads that doubles every day.", "target": "doubles"},
    {"domain": "exponential", "prompt": "If it takes 48 days to cover the lake, how long for half?", "target": "half"},
    {"domain": "exponential", "prompt": "A bacteria population grows exponentially by doubling every hour.", "target": "doubling"},
    {"domain": "exponential", "prompt": "A digital virus spreads by multiplying its infected hosts daily.", "target": "multiplying"},
    {"domain": "exponential", "prompt": "The algae patch expanded until it covered the entire pond in 36 days.", "target": "36"},

    # Arithmetic & Mathematics Domain
    {"domain": "arithmetic", "prompt": "Compute 15 multiplied by 8 to obtain the product 120.", "target": "120"},
    {"domain": "arithmetic", "prompt": "Dividing 144 by 12 results in an exact quotient of 12.", "target": "12"},
    {"domain": "arithmetic", "prompt": "The square root of 81 is equal to 9.", "target": "9"},
    {"domain": "arithmetic", "prompt": "Adding 57 and 43 yields a total sum of 100.", "target": "100"},
    {"domain": "arithmetic", "prompt": "Subtracting 25 from 75 leaves exactly 50.", "target": "50"},

    # Factual Knowledge & Geography Domain
    {"domain": "factual", "prompt": "The Colosseum is an iconic amphitheatre situated in Rome.", "target": "Rome"},
    {"domain": "factual", "prompt": "Mount Everest is the highest mountain peak located in the Himalayas.", "target": "Himalayas"},
    {"domain": "factual", "prompt": "The Amazon River flows through South America into the Atlantic Ocean.", "target": "Amazon"},
    {"domain": "factual", "prompt": "Tokyo is the bustling capital city of Japan.", "target": "Tokyo"},
    {"domain": "factual", "prompt": "The Pacific Ocean is the largest and deepest ocean on Earth.", "target": "Pacific"},

    # Science & Biology Domain
    {"domain": "science", "prompt": "DNA contains the genetic instructions necessary for the development of organisms.", "target": "DNA"},
    {"domain": "science", "prompt": "The speed of light in a vacuum is approximately 300000 kilometers per second.", "target": "vacuum"},
    {"domain": "science", "prompt": "Chlorophyll allows plants to absorb energy from sunlight during photosynthesis.", "target": "sunlight"},
    {"domain": "science", "prompt": "Electrons orbit the nucleus of an atom in distinct energy levels.", "target": "nucleus"},
    {"domain": "science", "prompt": "Mitochondria are often described as the powerhouses of biological cells.", "target": "powerhouses"},

    # General Language & Logic Domain
    {"domain": "language", "prompt": "Knowledge is power and wisdom is the ability to apply it correctly.", "target": "wisdom"},
    {"domain": "language", "prompt": "A journey of a thousand miles begins with a single step forward.", "target": "step"},
    {"domain": "language", "prompt": "Curiosity leads to discovery, innovation, and scientific breakthroughs.", "target": "discovery"},
    {"domain": "language", "prompt": "Patience and persistence are essential virtues for overcoming difficult challenges.", "target": "virtues"},
    {"domain": "language", "prompt": "The climate of a region is determined by long-term atmospheric patterns.", "target": "climate"}
]

# 5 Separate held-out prompts for out-of-sample generalization / preservation check
HELDOUT_PROMPTS = [
    {"domain": "exponential", "prompt": "Compound interest accelerates wealth accumulation over decades.", "target": "decades"},
    {"domain": "arithmetic", "prompt": "The product of 9 and 6 equals 54 in standard decimal math.", "target": "54"},
    {"domain": "factual", "prompt": "The Eiffel Tower is a globally famous landmark located in Paris.", "target": "Paris"},
    {"domain": "science", "prompt": "Photosynthesis converts solar energy into chemical energy stored in glucose.", "target": "glucose"},
    {"domain": "language", "prompt": "Honesty is the foundation of trust in every meaningful human relationship.", "target": "trust"}
]

# Canonical 5-vector safeguard matrix
CANONICAL_SAFEGUARDS = [
    {"id": "safe_arithmetic_add", "category": "arithmetic", "prompt": "The sum of 25 and 17 is equal to 42.", "target_substring": "42"},
    {"id": "safe_arithmetic_sub", "category": "arithmetic", "prompt": "Calculating 100 minus 36 gives 64.", "target_substring": "64"},
    {"id": "safe_factual_capital", "category": "factual_knowledge", "prompt": "The capital city of France is Paris.", "target_substring": "Paris"},
    {"id": "safe_science_water", "category": "science", "prompt": "Water is composed of hydrogen and oxygen.", "target_substring": "oxygen"},
    {"id": "safe_language_proverb", "category": "ordinary_language", "prompt": "Actions speak louder than words in every culture.", "target_substring": "words"}
]

PERTURBATION_ALPHA = 0.05  # 5% relative perturbation magnitude: ||delta_v|| / ||W_0 @ k|| = 0.05
PERTURBATION_BASE_SEED = 42000
LAMBDA_REG = 0.1


def capture_dataset(
    capturer: QwenActivationCapturer,
    prompts: List[Dict[str, str]],
    layer_idx: int,
    d_in: int,
    d_out: int,
    W_0_f64: torch.Tensor
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, Any]]]:
    """Captures input and baseline output activations, verifying v0 ≈ W0 @ k."""
    k_list: List[torch.Tensor] = []
    v0_list: List[torch.Tensor] = []
    metadata_list: List[Dict[str, Any]] = []

    for idx, item in enumerate(prompts):
        cap = capturer.capture(item["prompt"], layer_indices=layer_idx, match_substring=item["target"])
        layer_cap = cap[layer_idx]
        raw_k = layer_cap["input"]
        raw_v = layer_cap["output"]
        meta = layer_cap["metadata"]

        k_vec = raw_k[-1] if raw_k.ndim == 2 else raw_k
        v_vec = raw_v[-1] if raw_v.ndim == 2 else raw_v

        k_f64 = k_vec.detach().clone().to(dtype=torch.float64).view(d_in, 1)
        v0_f64 = v_vec.detach().clone().to(dtype=torch.float64).view(d_out, 1)

        # Baseline consistency check: v0 ≈ W_0 @ k
        expected_v0 = W_0_f64 @ k_f64
        diff_norm = torch.norm(v0_f64 - expected_v0).item()
        rel_diff = diff_norm / torch.norm(expected_v0).item()

        k_list.append(k_f64)
        v0_list.append(v0_f64)

        sample_info = {
            "index": idx + 1,
            "domain": item["domain"],
            "prompt": item["prompt"],
            "target": item["target"],
            "token_id": meta["token_id"],
            "token_str": meta["token_str"],
            "k_norm": torch.norm(k_f64).item(),
            "v0_norm": torch.norm(v0_f64).item(),
            "baseline_v0_w0k_diff_abs": diff_norm,
            "baseline_v0_w0k_diff_rel": rel_diff
        }
        metadata_list.append(sample_info)

    return k_list, v0_list, metadata_list


def perform_pairwise_checks(
    k_train_list: List[torch.Tensor],
    k_heldout_list: List[torch.Tensor],
    K_S: torch.Tensor
) -> Dict[str, Any]:
    """
    Computes pairwise cosine similarities and Euclidean distances to formally prove
    that the activation vectors are distinct, non-duplicated, and not collinear.
    """
    K_train = torch.cat(k_train_list, dim=1)  # [4864, N_train]
    N_train = K_train.shape[1]

    # Normalize vectors
    norms_train = torch.norm(K_train, p=2, dim=0, keepdim=True)
    K_train_norm = K_train / norms_train

    # Pairwise cosine similarity matrix: [N_train, N_train]
    cos_matrix = (K_train_norm.T @ K_train_norm).cpu()

    # Off-diagonal elements
    eye_mask = torch.eye(N_train, dtype=torch.bool)
    off_diag_cos = cos_matrix[~eye_mask]
    max_pairwise_cos = torch.max(torch.abs(off_diag_cos)).item()
    mean_pairwise_cos = torch.mean(off_diag_cos).item()

    # Pairwise Euclidean distances
    # ||k_i - k_j||^2 = ||k_i||^2 + ||k_j||^2 - 2 k_i^T k_j
    dot_matrix = (K_train.T @ K_train).cpu()
    sq_norms = torch.diag(dot_matrix).view(-1, 1)
    dist_sq = sq_norms + sq_norms.T - 2 * dot_matrix
    dist_sq = torch.clamp(dist_sq, min=0.0)
    dist_matrix = torch.sqrt(dist_sq)
    off_diag_dist = dist_matrix[~eye_mask]
    min_pairwise_dist = torch.min(off_diag_dist).item()
    mean_pairwise_dist = torch.mean(off_diag_dist).item()

    # Cross-similarity with Safeguard Keys
    norms_S = torch.norm(K_S, p=2, dim=0, keepdim=True)
    K_S_norm = K_S / norms_S
    cos_train_safe = (K_train_norm.T @ K_S_norm).cpu()
    max_cos_to_safe = torch.max(torch.abs(cos_train_safe)).item()

    # Cross-similarity with Held-out Keys
    K_held = torch.cat(k_heldout_list, dim=1)
    norms_held = torch.norm(K_held, p=2, dim=0, keepdim=True)
    K_held_norm = K_held / norms_held
    cos_train_held = (K_train_norm.T @ K_held_norm).cpu()
    max_cos_to_held = torch.max(torch.abs(cos_train_held)).item()

    return {
        "num_training_keys": N_train,
        "max_pairwise_cosine_similarity": max_pairwise_cos,
        "mean_pairwise_cosine_similarity": mean_pairwise_cos,
        "min_pairwise_euclidean_distance": min_pairwise_dist,
        "mean_pairwise_euclidean_distance": mean_pairwise_dist,
        "max_cosine_similarity_to_safeguards": max_cos_to_safe,
        "max_cosine_similarity_to_heldout": max_cos_to_held,
        "pairwise_check_passed": bool(max_pairwise_cos < 0.999 and min_pairwise_dist > 1.0)
    }


def construct_nonzero_targets(
    k_list: List[torch.Tensor],
    W_0_f64: torch.Tensor,
    alpha: float = PERTURBATION_ALPHA,
    base_seed: int = PERTURBATION_BASE_SEED
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, Any]]]:
    """
    Constructs deterministic non-zero targets:
      v_t = W_0 @ k_t + delta_v_t
    where ||delta_v_t||_2 = alpha * ||W_0 @ k_t||_2 along pseudo-random unit directions.
    """
    d_out = W_0_f64.shape[0]
    target_v_list: List[torch.Tensor] = []
    delta_v_list: List[torch.Tensor] = []
    perturbation_records: List[Dict[str, Any]] = []

    for t, k_t in enumerate(k_list):
        gen = torch.Generator().manual_seed(base_seed + t)
        eta = torch.randn(d_out, 1, dtype=torch.float64, generator=gen)
        eta_hat = eta / torch.norm(eta)

        v0_t = W_0_f64 @ k_t
        v0_norm = torch.norm(v0_t).item()
        perturb_norm = alpha * v0_norm
        delta_v_t = perturb_norm * eta_hat

        v_t = v0_t + delta_v_t

        target_v_list.append(v_t)
        delta_v_list.append(delta_v_t)

        actual_perturb_norm = torch.norm(delta_v_t).item()
        actual_ratio = actual_perturb_norm / v0_norm if v0_norm > 0 else 0.0

        perturbation_records.append({
            "step": t + 1,
            "v0_norm": v0_norm,
            "target_norm": torch.norm(v_t).item(),
            "delta_v_norm": actual_perturb_norm,
            "relative_perturbation_ratio": actual_ratio,
            "presurgery_target_residual": actual_perturb_norm
        })

    return target_v_list, delta_v_list, perturbation_records


def run_nonzero_target_validation():
    print("=" * 85)
    print("EXPERIMENT 3 — TASK 6: NON-ZERO TARGET VALIDATION (LAYER 20)")
    print("=" * 85)

    # 1. Load model and verify baseline
    print("\n[Step 1] Loading pristine Qwen2.5-0.5B model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        local_files_only=True
    )
    model.eval()

    layer_idx = 20
    snapshot_l20 = model.model.layers[layer_idx].mlp.down_proj.weight.detach().clone()
    d_out, d_in = snapshot_l20.shape
    W_0_f64 = snapshot_l20.to(dtype=torch.float64)
    norm_w0 = torch.norm(W_0_f64, p="fro").item()
    print(f"Layer {layer_idx} down_proj weight shape: ({d_out}, {d_in}) [float32 snapshot stored]")

    capturer = QwenActivationCapturer(model, tokenizer)

    # 2. Extract canonical safeguards
    print("\n[Step 2] Extracting Layer 20 Safeguard Activations (m=5)...")
    sg_data = capturer.extract_safeguards(CANONICAL_SAFEGUARDS, layer_idx=layer_idx, dtype=torch.float64)
    K_S = sg_data["K_S"]  # [4864, 5]
    V_S = sg_data["V_S"]  # [896, 5]
    norm_vs = torch.norm(V_S, p="fro").item()
    sg_analysis = sg_data["numerical_analysis"]
    print(f"K_S shape: {list(K_S.shape)}, V_S shape: {list(V_S.shape)}, Numerical Rank: {sg_analysis['numerical_rank']}/{sg_analysis['num_safeguards']}")

    # 3. Capture Training and Held-out Activations
    print(f"\n[Step 3a] Capturing {len(TRAINING_PROMPTS)} Training activation pairs...")
    k_train, v0_train, train_meta = capture_dataset(capturer, TRAINING_PROMPTS, layer_idx, d_in, d_out, W_0_f64)
    print(f"[Step 3b] Capturing {len(HELDOUT_PROMPTS)} Held-out activation pairs...")
    k_heldout, v0_heldout, heldout_meta = capture_dataset(capturer, HELDOUT_PROMPTS, layer_idx, d_in, d_out, W_0_f64)

    # Baseline consistency summary
    max_train_baseline_err = max(m["baseline_v0_w0k_diff_abs"] for m in train_meta)
    print(f"Training activations baseline check (||v0 - W0 @ k||): Max error = {max_train_baseline_err:.2e} (float32 GEMM roundoff)")

    # 4. Rigorous Pairwise Duplicate and Collinearity Checks
    print("\n[Step 4] Performing pairwise activation duplicate and distance checks...")
    pairwise_stats = perform_pairwise_checks(k_train, k_heldout, K_S)
    print(f"  Max pairwise cosine similarity:        {pairwise_stats['max_pairwise_cosine_similarity']:.4f}")
    print(f"  Min pairwise Euclidean distance:        {pairwise_stats['min_pairwise_euclidean_distance']:.2f}")
    print(f"  Max cosine similarity to Safeguards:    {pairwise_stats['max_cosine_similarity_to_safeguards']:.4f}")
    print(f"  Max cosine similarity to Held-out:      {pairwise_stats['max_cosine_similarity_to_heldout']:.4f}")
    print(f"  Pairwise duplicate check passed:        {pairwise_stats['pairwise_check_passed']}")

    # 5. Construct Controlled Non-Zero Targets
    print(f"\n[Step 5] Constructing controlled non-zero targets (alpha={PERTURBATION_ALPHA:.2f})...")
    v_train_targets, delta_v_list, perturb_meta = construct_nonzero_targets(
        k_train, W_0_f64, alpha=PERTURBATION_ALPHA, base_seed=PERTURBATION_BASE_SEED
    )
    mean_presurgery_res = sum(p["presurgery_target_residual"] for p in perturb_meta) / len(perturb_meta)
    print(f"Constructed {len(v_train_targets)} targets. Mean pre-surgery target residual: {mean_presurgery_res:.4f}")

    # 6. Initialize Solver
    print("\n[Step 6] Initializing AuditedConstrainedRLS in float64...")
    solver = AuditedConstrainedRLS(
        initial_weight=W_0_f64,
        lambda_reg=LAMBDA_REG,
        K_safe=K_S,
        dtype=torch.float64,
        decomposition_mode="svd",
        record_history=False  # Strictly zero sample history retained
    )

    # Confirm solver state properties
    print(f"Solver initialized: d_in={solver.d_in}, d_out={solver.d_out}, d_red={solver.d_red}, rank={solver.rank}")
    print(f"History recording disabled: step_times={solver.step_times}, conflict_events={solver.conflict_events}")

    # 7. Sequential Streaming Updates Loop
    print(f"\n[Step 7] Streaming {len(v_train_targets)} non-zero target pairs sequentially through solver...")
    per_step_metrics: List[Dict[str, Any]] = []
    t_start_stream = time.perf_counter()

    for t in range(len(v_train_targets)):
        k_t = k_train[t]
        v_t = v_train_targets[t]

        # Prior target residual before update: ||W_{t-1} @ k_t - v_t||_2
        W_prior = solver.W
        prior_res_vec = W_prior @ k_t - v_t
        prior_res_norm = torch.norm(prior_res_vec).item()

        # Execute streaming update
        t0_step = time.perf_counter()
        update_res = solver.update(k_t, v_t, reject_conflicts=True)
        t_step_ms = (time.perf_counter() - t0_step) * 1e3

        W_t = solver.W

        # Posterior target residual immediately after update: ||W_t @ k_t - v_t||_2
        post_res_vec = W_t @ k_t - v_t
        post_res_norm = torch.norm(post_res_vec).item()

        # Residual reduction on this sample
        res_reduction_abs = prior_res_norm - post_res_norm
        res_reduction_ratio = post_res_norm / prior_res_norm if prior_res_norm > 0 else 0.0

        # Protected safeguard error: ||W_t @ K_S - W_0 @ K_S||_F
        safe_diff = W_t @ K_S - W_0_f64 @ K_S
        safe_err_abs = torch.norm(safe_diff, p="fro").item()
        safe_err_rel = safe_err_abs / norm_vs if norm_vs > 0 else safe_err_abs

        # Parameter drift: ||W_t - W_0||_F / ||W_0||_F
        param_drift = torch.norm(W_t - W_0_f64, p="fro").item() / norm_w0

        # Covariance positive definiteness (sample min eigenvalue)
        eigvals = torch.linalg.eigvalsh(solver.P_tilde)
        min_eig = torch.min(eigvals).item()

        step_record = {
            "step": t + 1,
            "domain": train_meta[t]["domain"],
            "target_token": train_meta[t]["token_str"],
            "updated": update_res["updated"],
            "conflict_detected": update_res["conflict_detected"],
            "target_residual_prior": prior_res_norm,
            "target_residual_post": post_res_norm,
            "residual_reduction_abs": res_reduction_abs,
            "residual_reduction_ratio": res_reduction_ratio,
            "residual_decreased": bool(post_res_norm < prior_res_norm),
            "safeguard_error_rel": safe_err_rel,
            "safeguard_error_abs": safe_err_abs,
            "parameter_drift": param_drift,
            "covariance_min_eig": min_eig,
            "step_latency_ms": t_step_ms
        }
        per_step_metrics.append(step_record)

        print(
            f"Step [{t+1:02d}/{len(v_train_targets)}] Acc: {update_res['updated']} | "
            f"PriorRes: {prior_res_norm:.3f} -> PostRes: {post_res_norm:.3f} (-{res_reduction_abs:.3f}, {res_reduction_ratio*100:.1f}%) | "
            f"SafeErr: {safe_err_rel:.2e} | Drift: {param_drift:.2e} | Lat: {t_step_ms:.1f}ms"
        )

    t_total_stream = time.perf_counter() - t_start_stream
    W_online_final = solver.W

    # 8. Independent Batch Constrained Least-Squares Solution (KKT)
    print("\n[Step 8] Computing independent batch constrained KKT reference solution...")
    K_train_stacked = torch.cat(k_train, dim=1)  # [4864, 25]
    V_train_stacked = torch.cat(v_train_targets, dim=1)  # [896, 25]

    t0_batch = time.perf_counter()
    W_batch_final = compute_independent_batch_kkt(
        K=K_train_stacked,
        V=V_train_stacked,
        W_0=W_0_f64,
        K_safe=K_S,
        lambda_reg=LAMBDA_REG
    )
    t_batch_sec = time.perf_counter() - t0_batch
    print(f"Batch KKT solver completed in {t_batch_sec:.2f} seconds.")

    # 9. Comparison: Online vs. Batch Solutions
    print("\n[Step 9] Comparing Online vs. Batch Solutions...")
    diff_online_batch = W_online_final - W_batch_final
    rel_error_to_batch = torch.norm(diff_online_batch, p="fro").item() / torch.norm(W_batch_final, p="fro").item()
    max_abs_param_diff = torch.max(torch.abs(diff_online_batch)).item()

    # Training predictions comparison
    pred_train_online = W_online_final @ K_train_stacked
    pred_train_batch = W_batch_final @ K_train_stacked
    train_pred_mse_diff = torch.mean((pred_train_online - pred_train_batch) ** 2).item()
    train_pred_max_diff = torch.max(torch.abs(pred_train_online - pred_train_batch)).item()

    # Overall training target relative residual for both solvers
    train_target_rel_err_online = (torch.norm(pred_train_online - V_train_stacked, p="fro").item() /
                                   torch.norm(V_train_stacked, p="fro").item())
    train_target_rel_err_batch = (torch.norm(pred_train_batch - V_train_stacked, p="fro").item() /
                                  torch.norm(V_train_stacked, p="fro").item())

    # Pre-surgery training target relative error
    pred_train_presurgery = W_0_f64 @ K_train_stacked
    train_target_rel_err_presurgery = (torch.norm(pred_train_presurgery - V_train_stacked, p="fro").item() /
                                       torch.norm(V_train_stacked, p="fro").item())

    # Held-out real activations comparison
    K_held_stacked = torch.cat(k_heldout, dim=1)  # [4864, 5]
    V_held_orig = W_0_f64 @ K_held_stacked  # Baseline model predictions on held-out

    pred_held_online = W_online_final @ K_held_stacked
    pred_held_batch = W_batch_final @ K_held_stacked
    held_pred_mse_diff = torch.mean((pred_held_online - pred_held_batch) ** 2).item()

    # Held-out drift from baseline
    held_drift_online = torch.norm(pred_held_online - V_held_orig, p="fro").item() / torch.norm(V_held_orig, p="fro").item()
    held_drift_batch = torch.norm(pred_held_batch - V_held_orig, p="fro").item() / torch.norm(V_held_orig, p="fro").item()

    # Safeguard preservation check on batch
    safe_diff_batch = W_batch_final @ K_S - W_0_f64 @ K_S
    safe_batch_rel_err = torch.norm(safe_diff_batch, p="fro").item() / norm_vs

    print(f"  Online vs Batch Relative Frobenius Error: {rel_error_to_batch:.4e}")
    print(f"  Max Absolute Parameter Difference:       {max_abs_param_diff:.4e}")
    print(f"  Training Prediction MSE Difference:      {train_pred_mse_diff:.4e}")
    print(f"  Pre-surgery Training Target Rel Error:   {train_target_rel_err_presurgery:.4f}")
    print(f"  Post-surgery Training Rel Error (Online):{train_target_rel_err_online:.4f}")
    print(f"  Post-surgery Training Rel Error (Batch): {train_target_rel_err_batch:.4f}")
    print(f"  Held-out Prediction MSE Difference:      {held_pred_mse_diff:.4e}")
    print(f"  Held-out Drift from Baseline (Online):   {held_drift_online:.4e}")
    print(f"  Held-out Drift from Baseline (Batch):    {held_drift_batch:.4e}")
    print(f"  Batch Safeguard Relative Error:          {safe_batch_rel_err:.4e}")

    # 10. Bitwise check on original model weights
    print("\n[Step 10] Verifying bitwise immutability of model parameters...")
    is_model_unmodified = torch.equal(model.model.layers[layer_idx].mlp.down_proj.weight, snapshot_l20)
    print(f"  Model Layer 20 weight bitwise identical to snapshot: {is_model_unmodified}")

    # 11. Verify zero sample history inside solver
    solver_retains_no_history = (
        not hasattr(solver, "samples") and
        not hasattr(solver, "k_history") and
        not hasattr(solver, "v_history") and
        solver.step_times is None and
        solver.conflict_events is None
    )
    print(f"  Zero sample history retained in streaming solver: {solver_retains_no_history}")

    # 12. Aggregate Metrics and PASS/FAIL Evaluation
    all_residuals_decreased = all(s["residual_decreased"] for s in per_step_metrics)
    all_updates_accepted = all(s["updated"] for s in per_step_metrics)
    max_safe_rel_online = max(s["safeguard_error_rel"] for s in per_step_metrics)
    min_eig_all_steps = min(s["covariance_min_eig"] for s in per_step_metrics)
    latencies = [s["step_latency_ms"] for s in per_step_metrics]

    eval_checks = {
        "check_1_sample_count_ge_20": bool(len(v_train_targets) >= 20),
        "check_2_pairwise_activation_duplicates": bool(pairwise_stats["pairwise_check_passed"]),
        "check_3_nonzero_target_residuals_presurgery": bool(train_target_rel_err_presurgery > 0.04),
        "check_4_actual_reduction_in_target_error": bool(all_residuals_decreased and train_target_rel_err_online < train_target_rel_err_presurgery),
        "check_5_original_model_bitwise_unchanged": bool(is_model_unmodified),
        "check_6_no_sample_history_retained": bool(solver_retains_no_history),
        "check_7_online_batch_equivalence": bool(rel_error_to_batch < 1e-10),
        "check_8_safeguard_preservation": bool(max_safe_rel_online < 1e-10),
        "check_9_covariance_positive_definite": bool(min_eig_all_steps > 0.0)
    }

    all_passed = all(eval_checks.values())

    results = {
        "metadata": {
            "title": "Experiment 3 — Task 6: Non-Zero Target Validation",
            "model": MODEL_NAME,
            "layer": layer_idx,
            "num_training_samples": len(v_train_targets),
            "num_heldout_samples": len(k_heldout),
            "num_safeguards": K_S.shape[1],
            "perturbation_alpha": PERTURBATION_ALPHA,
            "perturbation_base_seed": PERTURBATION_BASE_SEED,
            "lambda_reg": LAMBDA_REG,
            "dtype": "torch.float64"
        },
        "pairwise_checks": pairwise_stats,
        "aggregate_metrics": {
            "all_updates_accepted": all_updates_accepted,
            "all_target_residuals_decreased": all_residuals_decreased,
            "mean_target_residual_reduction_ratio": sum(s["residual_reduction_ratio"] for s in per_step_metrics) / len(per_step_metrics),
            "presurgery_training_target_relative_error": train_target_rel_err_presurgery,
            "postsurgery_training_target_relative_error_online": train_target_rel_err_online,
            "postsurgery_training_target_relative_error_batch": train_target_rel_err_batch,
            "max_intermediate_safeguard_rel_error_online": max_safe_rel_online,
            "safeguard_rel_error_batch": safe_batch_rel_err,
            "final_parameter_drift": per_step_metrics[-1]["parameter_drift"],
            "relative_frobenius_error_to_batch_kkt": rel_error_to_batch,
            "max_absolute_param_diff_to_batch": max_abs_param_diff,
            "training_prediction_mse_diff_to_batch": train_pred_mse_diff,
            "training_prediction_max_diff_to_batch": train_pred_max_diff,
            "heldout_prediction_mse_diff_to_batch": held_pred_mse_diff,
            "heldout_drift_from_baseline_online": held_drift_online,
            "heldout_drift_from_baseline_batch": held_drift_batch,
            "covariance_min_eigenvalue_throughout": min_eig_all_steps,
            "model_weights_bitwise_unmodified": is_model_unmodified,
            "zero_sample_history_retained": solver_retains_no_history
        },
        "performance_summary": {
            "total_streaming_time_sec": t_total_stream,
            "mean_latency_ms": sum(latencies) / len(latencies),
            "median_latency_ms": sorted(latencies)[len(latencies) // 2],
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "batch_kkt_time_sec": t_batch_sec
        },
        "eval_checks": eval_checks,
        "overall_status": "PASSED" if all_passed else "FAILED",
        "per_step_metrics": per_step_metrics,
        "perturbation_records": perturb_meta,
        "training_samples": train_meta,
        "heldout_samples": heldout_meta,
        "safeguards": CANONICAL_SAFEGUARDS
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved structured experimental results to: {RESULTS_JSON_PATH}")
    print("=" * 85)
    print(f"NON-ZERO TARGET VALIDATION FINISHED: OVERALL STATUS = {results['overall_status']}")
    print("=" * 85)

    return results


if __name__ == "__main__":
    run_nonzero_target_validation()

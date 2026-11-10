"""
Experiment 3 — Task 7.5: First One-Fact Constrained RLS Semantic Pilot
======================================================================
Executes a controlled semantic weight-surgery pilot on Qwen2.5-0.5B Layer 20 down_proj
using AuditedConstrainedRLS to shift the answer of "The capital of France is" toward "Lyon"
while preserving sequence-wide safeguard prompts (37 tokens) from Task 7.4.

Tests beta = 0.15 and beta = 0.20 independently from pristine starting weights.
Evaluates 4 France paraphrases, 3 unrelated controls, and 5 safeguard prompts.
Verifies bitwise model immutability and mathematical constraint satisfaction (Delta_W @ K_S approx 0).
"""

import os
import sys
import json
import time
import torch
from typing import Dict, Any, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "Experiment-2.1-Robustness-Streaming-Audit"))
from robustness_tests import AuditedConstrainedRLS
from activation_capture import QwenActivationCapturer
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = 20
BETAS = [0.15, 0.20]
RESULTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "task_7_5_constrained_semantic_pilot_results.json")

CITY_TOKEN_IDS = {
    "Lyon": 55201,    # " Lyon"
    "Paris": 12095,   # " Paris"
    "Berlin": 19846,  # " Berlin"
    "Rome": 21718,    # " Rome"
    "Madrid": 24081   # " Madrid"
}

EDIT_QUERY_PROMPT = "The capital of France is"

FRANCE_PROMPTS = [
    {
        "id": "france_canonical",
        "prompt": "The capital of France is",
        "expected_baseline_city": "Paris",
        "category": "france_query"
    },
    {
        "id": "france_possessive",
        "prompt": "France's capital city is",
        "expected_baseline_city": "Paris",
        "category": "france_paraphrase"
    },
    {
        "id": "france_capital_city",
        "prompt": "The capital city of France is",
        "expected_baseline_city": "Paris",
        "category": "france_paraphrase"
    },
    {
        "id": "france_wh_question",
        "prompt": "Which city serves as the capital of France? The answer is",
        "expected_baseline_city": "Paris",
        "category": "france_paraphrase"
    }
]

CONTROL_PROMPTS = [
    {
        "id": "control_germany",
        "prompt": "The capital of Germany is",
        "expected_baseline_city": "Berlin",
        "category": "unrelated_control"
    },
    {
        "id": "control_italy",
        "prompt": "The capital of Italy is",
        "expected_baseline_city": "Rome",
        "category": "unrelated_control"
    },
    {
        "id": "control_spain",
        "prompt": "The capital of Spain is",
        "expected_baseline_city": "Madrid",
        "category": "unrelated_control"
    }
]

SAFEGUARD_PROMPTS = [
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


def extract_top_k(logits: torch.Tensor, probs: torch.Tensor, tokenizer: AutoTokenizer, k: int = 10) -> List[Dict[str, Any]]:
    top = torch.topk(probs, k)
    results = []
    for rank, (p, idx) in enumerate(zip(top.values, top.indices)):
        results.append({
            "rank": rank + 1,
            "token_id": idx.item(),
            "token_str": tokenizer.decode([idx.item()]),
            "logit": logits[idx].item(),
            "prob": p.item()
        })
    return results


def extract_city_metrics(logits: torch.Tensor, probs: torch.Tensor) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for city_name, token_id in CITY_TOKEN_IDS.items():
        metrics[city_name] = {
            "token_id": token_id,
            "logit": logits[token_id].item(),
            "prob": probs[token_id].item()
        }
    return metrics


def evaluate_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    results = []
    for p_info in prompts:
        prompt_text = p_info["prompt"]
        expected_city = p_info["expected_baseline_city"]
        expected_token_id = CITY_TOKEN_IDS[expected_city]

        inputs = tokenizer(prompt_text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
            probs = torch.softmax(logits, dim=-1)

        top10 = extract_top_k(logits, probs, tokenizer, k=10)
        city_m = extract_city_metrics(logits, probs)
        margin = city_m["Lyon"]["logit"] - city_m["Paris"]["logit"]
        top1_tok = top10[0]

        sorted_idx = torch.argsort(logits, descending=True)
        rank_expected = (sorted_idx == expected_token_id).nonzero(as_tuple=True)[0].item() + 1
        rank_lyon = (sorted_idx == CITY_TOKEN_IDS["Lyon"]).nonzero(as_tuple=True)[0].item() + 1
        rank_paris = (sorted_idx == CITY_TOKEN_IDS["Paris"]).nonzero(as_tuple=True)[0].item() + 1

        results.append({
            "id": p_info["id"],
            "prompt": prompt_text,
            "category": p_info["category"],
            "expected_baseline_city": expected_city,
            "top1_token_id": top1_tok["token_id"],
            "top1_token_str": top1_tok["token_str"],
            "top1_prob": top1_tok["prob"],
            "top1_is_expected": bool(top1_tok["token_id"] == expected_token_id),
            "top1_is_lyon": bool(top1_tok["token_id"] == CITY_TOKEN_IDS["Lyon"]),
            "lyon_surpasses_paris": bool(city_m["Lyon"]["logit"] > city_m["Paris"]["logit"]),
            "margin_lyon_minus_paris": margin,
            "rank_expected": rank_expected,
            "rank_lyon": rank_lyon,
            "rank_paris": rank_paris,
            "city_metrics": city_m,
            "top10": top10
        })
    return results


def evaluate_safeguards(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    safeguards: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    results = []
    for sg in safeguards:
        prompt_text = sg["prompt"]
        inputs = tokenizer(prompt_text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
            probs = torch.softmax(logits, dim=-1)

        top5 = extract_top_k(logits, probs, tokenizer, k=5)
        results.append({
            "id": sg["id"],
            "prompt": prompt_text,
            "category": sg["category"],
            "top_token_str": top5[0]["token_str"],
            "top_token_prob": top5[0]["prob"],
            "logits": logits.detach().cpu(),
            "top5": top5
        })
    return results


def run_pilot():
    print("=" * 90)
    print("EXPERIMENT 3 — TASK 7.5: FIRST ONE-FACT CONSTRAINED RLS SEMANTIC PILOT")
    print("=" * 90)

    # 1. Load fresh pristine model
    print("\n[Step 1] Loading pristine Qwen2.5-0.5B model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        local_files_only=True
    )
    model.eval()

    # Snapshot pristine Layer 20 down_proj weight
    snapshot_w20 = model.model.layers[LAYER_IDX].mlp.down_proj.weight.detach().clone()
    d_out, d_in = snapshot_w20.shape
    W_0_f64 = snapshot_w20.to(dtype=torch.float64)
    norm_w0 = torch.norm(W_0_f64, p="fro").item()
    print(f"Layer {LAYER_IDX} down_proj weight shape: ({d_out}, {d_in}) [float64 norm = {norm_w0:.4f}]")

    capturer = QwenActivationCapturer(model, tokenizer)

    # 2. Extract query key k and baseline output v0 for France prompt
    print(f"\n[Step 2] Capturing query activation for {repr(EDIT_QUERY_PROMPT)}...")
    cap_query = capturer.capture(EDIT_QUERY_PROMPT, layer_indices=LAYER_IDX, token_position="last")
    k_query = cap_query[LAYER_IDX]["input"].to(torch.float64).view(d_in, 1)
    v0_query = cap_query[LAYER_IDX]["output"].to(torch.float64).view(d_out, 1)
    k_norm = torch.norm(k_query).item()
    v0_norm = torch.norm(v0_query).item()
    print(f"Captured Query: ||k|| = {k_norm:.4f}, ||v0|| = {v0_norm:.4f}")

    # 3. Construct normalized steering direction d from lm_head
    print("\n[Step 3] Constructing normalized steering direction d from lm_head.weight...")
    E_lyon = model.lm_head.weight[CITY_TOKEN_IDS["Lyon"]].detach().clone().to(torch.float64).view(d_out, 1)
    E_paris = model.lm_head.weight[CITY_TOKEN_IDS["Paris"]].detach().clone().to(torch.float64).view(d_out, 1)
    diff_E = E_lyon - E_paris
    norm_diff_E = torch.norm(diff_E).item()
    d = diff_E / norm_diff_E
    print(f"||E_lyon - E_paris|| = {norm_diff_E:.4f}, Direction shape: {list(d.shape)}, ||d|| = {torch.norm(d).item():.4f}")

    # 4. Extract sequence-wide safeguards (Task 7.4 pilot prompts)
    print("\n[Step 4] Capturing sequence-wide safeguard matrix K_S (Task 7.4 prompts)...")
    k_sg_list = []
    v0_sg_list = []
    for sg_item in SAFEGUARD_PROMPTS:
        cap_sg = capturer.capture(sg_item["prompt"], layer_indices=LAYER_IDX, token_position="all")
        raw_k = cap_sg[LAYER_IDX]["input"]   # [seq_len, 4864]
        raw_v0 = cap_sg[LAYER_IDX]["output"] # [seq_len, 896]
        seq_len = raw_k.shape[0]
        for pos in range(seq_len):
            k_sg_list.append(raw_k[pos].to(torch.float64).view(d_in, 1))
            v0_sg_list.append(raw_v0[pos].to(torch.float64).view(d_out, 1))

    K_S = torch.cat(k_sg_list, dim=1)  # [4864, 37]
    V_S = torch.cat(v0_sg_list, dim=1) # [896, 37]
    norm_vs = torch.norm(V_S, p="fro").item()
    total_sg_tokens = K_S.shape[1]
    print(f"K_S shape: {list(K_S.shape)}, V_S shape: {list(V_S.shape)} (total {total_sg_tokens} sequence tokens)")

    # 5. Baseline Evaluation of pristine model
    print("\n[Step 5] Running baseline evaluation on pristine model...")
    baseline_france = evaluate_prompts(model, tokenizer, FRANCE_PROMPTS)
    baseline_controls = evaluate_prompts(model, tokenizer, CONTROL_PROMPTS)
    baseline_safeguards = evaluate_safeguards(model, tokenizer, SAFEGUARD_PROMPTS)

    print("\n--- Baseline France Predictions ---")
    for r in baseline_france:
        print(f"  [{r['id']}]: Top1={repr(r['top1_token_str']):<10} ({r['top1_prob']*100:.1f}%) | Lyon={r['city_metrics']['Lyon']['prob']*100:.2f}% | Paris={r['city_metrics']['Paris']['prob']*100:.2f}% | Margin={r['margin_lyon_minus_paris']:+.2f}")

    print("\n--- Baseline Control Predictions ---")
    for r in baseline_controls:
        print(f"  [{r['id']}]: Top1={repr(r['top1_token_str']):<10} ({r['top1_prob']*100:.1f}%) | Expected ({r['expected_baseline_city']}) Rank={r['rank_expected']}")

    # 6. Execute independent pilot surgeries for beta in [0.15, 0.20]
    pilot_runs: List[Dict[str, Any]] = []

    for beta in BETAS:
        print(f"\n=========================================================================================")
        print(f"[Step 6] RUNNING INDEPENDENT PILOT SURGERY: BETA = {beta:.2f}")
        print(f"=========================================================================================")

        # Construct target
        v_target = v0_query + (beta * v0_norm) * d
        target_perturb_norm = torch.norm(v_target - v0_query).item()
        print(f"Target perturbation ||v_target - v0|| = {target_perturb_norm:.4f} (relative: {target_perturb_norm/v0_norm:.4f})")

        # Create fresh solver instance
        t0_solver = time.perf_counter()
        solver = AuditedConstrainedRLS(
            initial_weight=W_0_f64,
            lambda_reg=0.1,
            K_safe=K_S,
            dtype=torch.float64,
            decomposition_mode="svd",
            record_history=False
        )
        t_solver_init = time.perf_counter() - t0_solver

        # Inspect query key projection in solver's null space
        k_perp = solver.U_perp.T @ k_query
        norm_k_perp = torch.norm(k_perp).item()
        perp_ratio = norm_k_perp / k_norm
        print(f"Solver initialized in {t_solver_init:.2f}s: rank={solver.rank}, d_red={solver.d_red}")
        print(f"Query key projection: ||k_perp|| = {norm_k_perp:.4f}, perp_ratio = {perp_ratio:.4f}")

        # Execute single constrained RLS update
        t0_update = time.perf_counter()
        update_result = solver.update(k_query, v_target, reject_conflicts=True)
        t_update = time.perf_counter() - t0_update

        print(f"Update executed in {t_update*1e3:.2f}ms: updated={update_result['updated']}, conflict={update_result['conflict_detected']}")

        # Retrieve modified weight matrix
        W_mod = solver.W
        delta_W = W_mod - W_0_f64
        norm_delta_w = torch.norm(delta_W, p="fro").item()
        param_drift = norm_delta_w / norm_w0

        # Mathematical safeguard satisfaction: ||Delta_W @ K_S||_F
        safe_err_abs = torch.norm(delta_W @ K_S, p="fro").item()
        safe_err_rel = safe_err_abs / norm_vs if norm_vs > 0 else safe_err_abs

        # Target prediction error on the trained query key
        pred_diff = W_mod @ k_query - v_target
        pred_err_post = torch.norm(pred_diff).item()
        pred_err_prior = torch.norm(W_0_f64 @ k_query - v_target).item()
        err_reduction_ratio = pred_err_post / pred_err_prior if pred_err_prior > 0 else 0.0

        print(f"Surgery Metrics:")
        print(f"  ||Delta W||_F:                  {norm_delta_w:.4f} (Parameter Drift: {param_drift*100:.3f}%)")
        print(f"  Safeguard Error ||Delta W K_S||:{safe_err_abs:.2e} (Relative: {safe_err_rel:.2e})")
        print(f"  Query Target Error:             Prior={pred_err_prior:.4f} -> Post={pred_err_post:.4f} (Reduction: {(1-err_reduction_ratio)*100:.2f}%)")

        # Temporarily load modified weights into model for full forward pass evaluation
        print(f"\nEvaluating modified model (beta={beta:.2f})...")
        model.model.layers[LAYER_IDX].mlp.down_proj.weight.data.copy_(W_mod.to(torch.float32))

        try:
            mod_france = evaluate_prompts(model, tokenizer, FRANCE_PROMPTS)
            mod_controls = evaluate_prompts(model, tokenizer, CONTROL_PROMPTS)
            mod_safeguards = evaluate_safeguards(model, tokenizer, SAFEGUARD_PROMPTS)
        finally:
            # Revert model to pristine snapshot immediately
            model.model.layers[LAYER_IDX].mlp.down_proj.weight.data.copy_(snapshot_w20)

        # Confirm model reverted properly
        is_reverted = torch.equal(model.model.layers[LAYER_IDX].mlp.down_proj.weight, snapshot_w20)
        assert is_reverted, "Fatal: Model weight failed to revert to pristine snapshot!"

        # Safeguard logit comparison
        sg_logit_diffs = []
        for i, sg_base in enumerate(baseline_safeguards):
            base_l = sg_base["logits"]
            mod_l = mod_safeguards[i]["logits"]
            abs_diff = torch.max(torch.abs(mod_l - base_l)).item()
            base_norm = torch.norm(base_l).item()
            rel_diff = abs_diff / base_norm if base_norm > 0 else abs_diff
            sg_logit_diffs.append({
                "id": sg_base["id"],
                "max_abs_logit_diff": abs_diff,
                "relative_logit_diff": rel_diff,
                "base_top1": sg_base["top5"][0]["token_str"],
                "mod_top1": mod_safeguards[i]["top5"][0]["token_str"],
                "top1_unchanged": bool(sg_base["top5"][0]["token_id"] == mod_safeguards[i]["top5"][0]["token_id"])
            })

        max_sg_logit_abs_diff = max(s["max_abs_logit_diff"] for s in sg_logit_diffs)
        max_sg_logit_rel_diff = max(s["relative_logit_diff"] for s in sg_logit_diffs)

        print(f"\n--- Post-Surgery France Predictions (beta={beta:.2f}) ---")
        for r in mod_france:
            print(f"  [{r['id']}]: Top1={repr(r['top1_token_str']):<10} ({r['top1_prob']*100:.1f}%) | Lyon={r['city_metrics']['Lyon']['prob']*100:.2f}% (R{r['rank_lyon']}) | Paris={r['city_metrics']['Paris']['prob']*100:.2f}% (R{r['rank_paris']}) | Margin={r['margin_lyon_minus_paris']:+.2f}")

        print(f"\n--- Post-Surgery Control Predictions (beta={beta:.2f}) ---")
        for r in mod_controls:
            print(f"  [{r['id']}]: Top1={repr(r['top1_token_str']):<10} ({r['top1_prob']*100:.1f}%) | Expected ({r['expected_baseline_city']}) Rank={r['rank_expected']} ({r['city_metrics'][r['expected_baseline_city']]['prob']*100:.1f}%)")

        print(f"\n--- Safeguard Invariance (beta={beta:.2f}) ---")
        print(f"  Max Absolute Logit Shift across 5 Safeguards: {max_sg_logit_abs_diff:.4e}")
        print(f"  Max Relative Logit Shift across 5 Safeguards: {max_sg_logit_rel_diff:.4e}")
        for s in sg_logit_diffs:
            print(f"    [{s['id']}]: Max Logit Diff = {s['max_abs_logit_diff']:.2e} | Top1 Unchanged: {s['top1_unchanged']} ({repr(s['base_top1'])})")

        pilot_record = {
            "beta": beta,
            "target_perturbation_norm": target_perturb_norm,
            "solver_diagnostics": {
                "rank": solver.rank,
                "d_red": solver.d_red,
                "k_norm": k_norm,
                "k_perp_norm": norm_k_perp,
                "perp_ratio": perp_ratio,
                "update_accepted": update_result["updated"],
                "conflict_detected": update_result["conflict_detected"],
                "delta_w_frobenius_norm": norm_delta_w,
                "parameter_drift": param_drift,
                "safeguard_error_abs": safe_err_abs,
                "safeguard_error_rel": safe_err_rel,
                "query_residual_prior": pred_err_prior,
                "query_residual_post": pred_err_post,
                "query_error_reduction_pct": (1.0 - err_reduction_ratio) * 100.0
            },
            "safeguard_evaluation": {
                "max_abs_logit_diff": max_sg_logit_abs_diff,
                "max_rel_logit_diff": max_sg_logit_rel_diff,
                "all_safeguard_top1_unchanged": all(s["top1_unchanged"] for s in sg_logit_diffs),
                "details": sg_logit_diffs
            },
            "france_evaluations": mod_france,
            "control_evaluations": mod_controls
        }
        pilot_runs.append(pilot_record)

    # 7. Final Model Immutability Check
    print("\n[Step 7] Final verification of pristine model immutability...")
    is_pristine = torch.equal(model.model.layers[LAYER_IDX].mlp.down_proj.weight, snapshot_w20)
    print(f"  Pristine Layer 20 down_proj weight bitwise identical: {is_pristine}")

    # Remove non-serializable tensors from baseline_safeguards before JSON save
    for s in baseline_safeguards:
        del s["logits"]
    for run in pilot_runs:
        for s in run["safeguard_evaluation"]["details"]:
            pass  # already json serializable

    full_results = {
        "metadata": {
            "title": "Experiment 3 — Task 7.5: First One-Fact Constrained RLS Semantic Pilot",
            "model": MODEL_NAME,
            "layer_idx": LAYER_IDX,
            "edit_query_prompt": EDIT_QUERY_PROMPT,
            "tested_betas": BETAS,
            "safeguard_tokens_count": total_sg_tokens,
            "city_token_ids": CITY_TOKEN_IDS
        },
        "baseline": {
            "france_evaluations": baseline_france,
            "control_evaluations": baseline_controls,
            "safeguard_evaluations": baseline_safeguards
        },
        "pilot_runs": pilot_runs,
        "integrity_verification": {
            "model_weights_bitwise_unmodified": is_pristine,
            "independent_starting_weights_used": True,
            "zero_cross_contamination": True
        }
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nSaved structured experimental results to: {RESULTS_JSON_PATH}")
    print("=" * 90)
    print("TASK 7.5 CONSTRAINED RLS SEMANTIC PILOT COMPLETED")
    print("=" * 90)

    return full_results


if __name__ == "__main__":
    run_pilot()

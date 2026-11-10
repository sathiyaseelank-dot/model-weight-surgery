"""
Experiment 3 — Task 7.7: Autoregressive Generation and Safeguard Rollout Validation
===================================================================================
Tests whether the one-fact Constrained RLS weight update (Layer 20 down_proj, shifting
"The capital of France is" toward "Lyon") preserves multi-token autoregressive generation
across 12 new tokens for France paraphrases, sequence-wide safeguards, and factual controls.

Evaluates three model states:
  1. Pristine baseline (W_0)
  2. Independent constrained RLS surgery at beta = 0.15
  3. Independent constrained RLS surgery at beta = 0.20

Deterministic greedy decoding (max_new_tokens = 12).
Strict bitwise model immutability and independent state verification.
"""

import os
import sys
import json
import time
import torch
from typing import Dict, Any, List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "Experiment-2.1-Robustness-Streaming-Audit"))
from robustness_tests import AuditedConstrainedRLS
from activation_capture import QwenActivationCapturer
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = 20
BETAS = [0.15, 0.20]
MAX_NEW_TOKENS = 12
RESULTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "task_7_7_autoregressive_results.json")

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
        "category": "france_query",
        "expected_baseline_city": "Paris"
    },
    {
        "id": "france_possessive",
        "prompt": "France's capital city is",
        "category": "france_paraphrase",
        "expected_baseline_city": "Paris"
    },
    {
        "id": "france_capital_city",
        "prompt": "The capital city of France is",
        "category": "france_paraphrase",
        "expected_baseline_city": "Paris"
    },
    {
        "id": "france_wh_question",
        "prompt": "Which city serves as the capital of France? The answer is",
        "category": "france_paraphrase",
        "expected_baseline_city": "Paris"
    }
]

SAFEGUARD_PROMPTS = [
    {
        "id": "safe_germany",
        "prompt": "The capital of Germany is Berlin.",
        "category": "factual_geography",
        "expected_continuation": "Berlin"
    },
    {
        "id": "safe_italy",
        "prompt": "The capital of Italy is Rome.",
        "category": "factual_geography",
        "expected_continuation": "Rome"
    },
    {
        "id": "safe_spain",
        "prompt": "The capital of Spain is Madrid.",
        "category": "factual_geography",
        "expected_continuation": "Madrid"
    },
    {
        "id": "safe_science",
        "prompt": "Water is composed of hydrogen and oxygen.",
        "category": "science_chemistry",
        "expected_continuation": "hydrogen and oxygen"
    },
    {
        "id": "safe_arithmetic",
        "prompt": "2 + 2 = 4.",
        "category": "arithmetic_logic",
        "expected_continuation": "4"
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


def generate_with_trace(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> Dict[str, Any]:
    """Generates autoregressively with greedy decoding and per-step logit/prob tracing."""
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
    full_text = tokenizer.decode(seq, skip_special_tokens=False)
    gen_text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=False)

    num_steps = len(out.scores)
    per_step_trace = []

    for s in range(num_steps):
        step_logits = out.scores[s][0] # [vocab_size]
        step_probs = torch.softmax(step_logits, dim=-1)
        step_tok_id = gen_token_ids[s] if s < len(gen_token_ids) else None
        step_tok_str = gen_token_strs[s] if s < len(gen_token_strs) else ""

        # Tracking metrics for Paris and Lyon
        lyon_logit = step_logits[CITY_TOKEN_IDS["Lyon"]].item()
        lyon_prob = step_probs[CITY_TOKEN_IDS["Lyon"]].item()
        paris_logit = step_logits[CITY_TOKEN_IDS["Paris"]].item()
        paris_prob = step_probs[CITY_TOKEN_IDS["Paris"]].item()

        # Top 3 at this step
        top3 = torch.topk(step_probs, 3)
        top3_list = [
            {"rank": r + 1, "token_id": top3.indices[r].item(), "token_str": tokenizer.decode([top3.indices[r].item()]), "prob": top3.values[r].item()}
            for r in range(3)
        ]

        per_step_trace.append({
            "step": s + 1,
            "chosen_token_id": step_tok_id,
            "chosen_token_str": step_tok_str,
            "lyon_logit": lyon_logit,
            "lyon_prob": lyon_prob,
            "paris_logit": paris_logit,
            "paris_prob": paris_prob,
            "top3": top3_list,
            "logits": step_logits.detach().cpu()  # temporary for diff comparison, stripped before JSON dump
        })

    first_tok_id = gen_token_ids[0] if gen_token_ids else None
    first_tok_str = gen_token_strs[0] if gen_token_strs else ""

    # Check city appearances anywhere in generated text
    has_lyon = any("Lyon" in tok or "lyon" in tok for tok in gen_token_strs) or ("Lyon" in gen_text or "lyon" in gen_text)
    has_paris = any("Paris" in tok or "paris" in tok for tok in gen_token_strs) or ("Paris" in gen_text or "paris" in gen_text)

    return {
        "prompt": prompt_text,
        "prompt_length": prompt_len,
        "generated_token_ids": gen_token_ids,
        "generated_token_strings": gen_token_strs,
        "generated_text": gen_text,
        "full_text": full_text,
        "first_generated_token_id": first_tok_id,
        "first_generated_token_str": first_tok_str,
        "contains_lyon": has_lyon,
        "contains_paris": has_paris,
        "num_generated_tokens": len(gen_token_ids),
        "step_trace": per_step_trace
    }


def compare_safeguard_rollouts(
    base_trace: Dict[str, Any],
    mod_trace: Dict[str, Any]
) -> Dict[str, Any]:
    """Compares baseline and modified autoregressive rollouts for a safeguard prompt."""
    base_ids = base_trace["generated_token_ids"]
    mod_ids = mod_trace["generated_token_ids"]

    # Identify first token divergence position (1-indexed)
    divergence_pos = None
    min_len = min(len(base_ids), len(mod_ids))
    for i in range(min_len):
        if base_ids[i] != mod_ids[i]:
            divergence_pos = i + 1
            break
    if divergence_pos is None and len(base_ids) != len(mod_ids):
        divergence_pos = min_len + 1

    # Per-step logit differences for matching prefix steps
    per_step_diffs = []
    base_steps = base_trace["step_trace"]
    mod_steps = mod_trace["step_trace"]
    num_comp = min(len(base_steps), len(mod_steps))

    for s in range(num_comp):
        base_l = base_steps[s]["logits"]
        mod_l = mod_steps[s]["logits"]
        max_abs_diff = torch.max(torch.abs(mod_l - base_l)).item()
        mean_abs_diff = torch.mean(torch.abs(mod_l - base_l)).item()
        per_step_diffs.append({
            "step": s + 1,
            "max_abs_logit_diff": max_abs_diff,
            "mean_abs_logit_diff": mean_abs_diff,
            "tokens_match": bool(base_steps[s]["chosen_token_id"] == mod_steps[s]["chosen_token_id"])
        })

    is_completely_identical = bool(base_ids == mod_ids)

    return {
        "is_sequence_identical": is_completely_identical,
        "divergence_step": divergence_pos,
        "baseline_generated_text": base_trace["generated_text"],
        "modified_generated_text": mod_trace["generated_text"],
        "per_step_logit_diffs": per_step_diffs
    }


def run_autoregressive_validation():
    print("=" * 95)
    print("EXPERIMENT 3 — TASK 7.7: AUTOREGRESSIVE GENERATION AND SAFEGUARD ROLLOUT VALIDATION")
    print("=" * 95)

    # 1. Load pristine model and snapshot Layer 20 down_proj
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
    print(f"Pristine Layer {LAYER_IDX} down_proj weight shape: ({d_out}, {d_in}) [float64 snapshot stored]")

    capturer = QwenActivationCapturer(model, tokenizer)

    # 2. Extract query key k and baseline output v0 on France query
    print(f"\n[Step 2] Capturing query activation for {repr(EDIT_QUERY_PROMPT)}...")
    cap_query = capturer.capture(EDIT_QUERY_PROMPT, layer_indices=LAYER_IDX, token_position="last")
    k_query = cap_query[LAYER_IDX]["input"].to(torch.float64).view(d_in, 1)
    v0_query = cap_query[LAYER_IDX]["output"].to(torch.float64).view(d_out, 1)
    v0_norm = torch.norm(v0_query).item()
    k_norm = torch.norm(k_query).item()

    # 3. Construct normalized steering direction d
    print("\n[Step 3] Constructing normalized steering direction d...")
    E_lyon = model.lm_head.weight[CITY_TOKEN_IDS["Lyon"]].detach().clone().to(torch.float64).view(d_out, 1)
    E_paris = model.lm_head.weight[CITY_TOKEN_IDS["Paris"]].detach().clone().to(torch.float64).view(d_out, 1)
    diff_E = E_lyon - E_paris
    d = diff_E / torch.norm(diff_E)

    # 4. Extract sequence-wide safeguard matrix K_S (Task 7.4 pilot prompts)
    print("\n[Step 4] Capturing sequence-wide safeguard matrix K_S (37 tokens)...")
    k_sg_list = []
    v0_sg_list = []
    for sg_item in SAFEGUARD_PROMPTS:
        cap_sg = capturer.capture(sg_item["prompt"], layer_indices=LAYER_IDX, token_position="all")
        raw_k = cap_sg[LAYER_IDX]["input"]
        raw_v0 = cap_sg[LAYER_IDX]["output"]
        for pos in range(raw_k.shape[0]):
            k_sg_list.append(raw_k[pos].to(torch.float64).view(d_in, 1))
            v0_sg_list.append(raw_v0[pos].to(torch.float64).view(d_out, 1))

    K_S = torch.cat(k_sg_list, dim=1) # [4864, 37]
    V_S = torch.cat(v0_sg_list, dim=1) # [896, 37]
    norm_vs = torch.norm(V_S, p="fro").item()
    print(f"Safeguard Matrix K_S shape: {list(K_S.shape)}, V_S shape: {list(V_S.shape)}")

    # 5. Evaluate State 0: Pristine Baseline Model
    print("\n=========================================================================================")
    print("[State 0] EVALUATING PRISTINE BASELINE MODEL (12 TOKENS GREEDY GENERATION)")
    print("=========================================================================================")
    
    baseline_france_eval = {}
    print("\n--- Baseline France Prompts ---")
    for p in FRANCE_PROMPTS:
        res = generate_with_trace(model, tokenizer, p["prompt"], MAX_NEW_TOKENS)
        baseline_france_eval[p["id"]] = res
        print(f"[{p['id']:<20}] \"{res['generated_text']}\" (Lyon? {res['contains_lyon']}, Paris? {res['contains_paris']})")

    baseline_safeguard_eval = {}
    print("\n--- Baseline Safeguard Prompts ---")
    for s in SAFEGUARD_PROMPTS:
        res = generate_with_trace(model, tokenizer, s["prompt"], MAX_NEW_TOKENS)
        baseline_safeguard_eval[s["id"]] = res
        print(f"[{s['id']:<16}] \"{res['generated_text']}\"")

    baseline_control_eval = {}
    print("\n--- Baseline Control Prompts ---")
    for c in CONTROL_PROMPTS:
        res = generate_with_trace(model, tokenizer, c["prompt"], MAX_NEW_TOKENS)
        baseline_control_eval[c["id"]] = res
        print(f"[{c['id']:<16}] \"{res['generated_text']}\" (First: {repr(res['first_generated_token_str'])})")

    # 6. Evaluate State 1 (beta = 0.15) and State 2 (beta = 0.20) independently
    beta_pilot_results = []

    for beta in BETAS:
        print(f"\n=========================================================================================")
        print(f"[Surgery Pilot] EVALUATING INDEPENDENT SURGERY: BETA = {beta:.2f}")
        print("=========================================================================================")

        # Target construction
        v_target = v0_query + (beta * v0_norm) * d

        # Initialize fresh independent solver from pristine weights
        solver = AuditedConstrainedRLS(
            initial_weight=W_0_f64,
            lambda_reg=0.1,
            K_safe=K_S,
            dtype=torch.float64,
            decomposition_mode="svd",
            record_history=False
        )

        update_res = solver.update(k_query, v_target, reject_conflicts=True)
        W_mod = solver.W
        delta_W = W_mod - W_0_f64
        norm_delta_w = torch.norm(delta_W, p="fro").item()
        safe_err_abs = torch.norm(delta_W @ K_S, p="fro").item()
        safe_err_rel = safe_err_abs / norm_vs if norm_vs > 0 else safe_err_abs

        print(f"Solver update: accepted={update_res['updated']}, ||Delta W||_F = {norm_delta_w:.4f}, Safeguard RelErr = {safe_err_rel:.2e}")

        # Temporarily load modified weights into model
        model.model.layers[LAYER_IDX].mlp.down_proj.weight.data.copy_(W_mod.to(torch.float32))

        try:
            # Evaluate France prompts
            print(f"\n--- France Prompts Autoregressive Rollouts (beta={beta:.2f}) ---")
            mod_france_eval = {}
            for p in FRANCE_PROMPTS:
                res = generate_with_trace(model, tokenizer, p["prompt"], MAX_NEW_TOKENS)
                mod_france_eval[p["id"]] = res
                first_tok = res["first_generated_token_str"]
                print(f"[{p['id']:<20}] First: {repr(first_tok):<10} | \"{res['generated_text']}\" | Lyon? {res['contains_lyon']} | Paris? {res['contains_paris']}")

            # Evaluate Safeguard prompts
            print(f"\n--- Safeguard Prompts Autoregressive Rollouts (beta={beta:.2f}) ---")
            mod_safeguard_eval = {}
            safeguard_comparisons = {}
            for s in SAFEGUARD_PROMPTS:
                res = generate_with_trace(model, tokenizer, s["prompt"], MAX_NEW_TOKENS)
                mod_safeguard_eval[s["id"]] = res
                comp = compare_safeguard_rollouts(baseline_safeguard_eval[s["id"]], res)
                safeguard_comparisons[s["id"]] = comp
                div_str = f"Step {comp['divergence_step']}" if comp['divergence_step'] else "None (100% Identical)"
                print(f"[{s['id']:<16}] Identical: {comp['is_sequence_identical']:<5} | Divergence: {div_str:<22} | \"{res['generated_text']}\"")

            # Evaluate Control prompts
            print(f"\n--- Control Prompts Autoregressive Rollouts (beta={beta:.2f}) ---")
            mod_control_eval = {}
            for c in CONTROL_PROMPTS:
                res = generate_with_trace(model, tokenizer, c["prompt"], MAX_NEW_TOKENS)
                mod_control_eval[c["id"]] = res
                first_tok = res["first_generated_token_str"]
                has_exp = c["expected_baseline_city"] in res["generated_text"]
                print(f"[{c['id']:<16}] First: {repr(first_tok):<10} | \"{res['generated_text']}\" | Expected ({c['expected_baseline_city']}) Preserved: {has_exp}")

        finally:
            # Revert model to pristine snapshot
            model.model.layers[LAYER_IDX].mlp.down_proj.weight.data.copy_(snapshot_w20)

        # Confirm model reverted properly
        is_reverted = torch.equal(model.model.layers[LAYER_IDX].mlp.down_proj.weight, snapshot_w20)
        assert is_reverted, "Fatal: Model weight failed to revert to pristine snapshot!"

        # Compute summary metrics for this beta
        all_sg_identical = all(c["is_sequence_identical"] for c in safeguard_comparisons.values())
        all_ctrl_preserved = all(c["expected_baseline_city"] in mod_control_eval[c["id"]]["generated_text"] for c in CONTROL_PROMPTS)

        pilot_record = {
            "beta": beta,
            "delta_w_frobenius_norm": norm_delta_w,
            "safeguard_relative_error": safe_err_rel,
            "france_rollouts": mod_france_eval,
            "control_rollouts": mod_control_eval,
            "safeguard_rollouts": mod_safeguard_eval,
            "safeguard_comparisons": safeguard_comparisons,
            "all_safeguards_identical_12_tokens": all_sg_identical,
            "all_controls_preserved": all_ctrl_preserved
        }
        beta_pilot_results.append(pilot_record)

    # 7. Final Model Immutability Verification
    print("\n[Step 7] Final verification of pristine model immutability...")
    is_pristine = torch.equal(model.model.layers[LAYER_IDX].mlp.down_proj.weight, snapshot_w20)
    print(f"  Pristine Layer 20 down_proj weight bitwise identical: {is_pristine}")

    # Strip raw tensor logits from all traces before serialization
    def clean_trace(obj):
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if k == "logits":
                    del obj[k]
                else:
                    clean_trace(v)
        elif isinstance(obj, list):
            for item in obj:
                clean_trace(item)

    clean_trace(baseline_france_eval)
    clean_trace(baseline_safeguard_eval)
    clean_trace(baseline_control_eval)
    clean_trace(beta_pilot_results)

    full_results = {
        "metadata": {
            "title": "Experiment 3 — Task 7.7: Autoregressive Generation and Safeguard Rollout Validation",
            "model": MODEL_NAME,
            "layer_idx": LAYER_IDX,
            "edit_query_prompt": EDIT_QUERY_PROMPT,
            "tested_betas": BETAS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "city_token_ids": CITY_TOKEN_IDS
        },
        "baseline_state": {
            "france_rollouts": baseline_france_eval,
            "safeguard_rollouts": baseline_safeguard_eval,
            "control_rollouts": baseline_control_eval
        },
        "surgery_states": beta_pilot_results,
        "integrity_verification": {
            "model_weights_bitwise_unmodified": is_pristine,
            "independent_starting_weights_used": True,
            "deterministic_greedy_decoding_used": True
        }
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nSaved structured experimental results to: {RESULTS_JSON_PATH}")
    print("=" * 95)
    print("TASK 7.7 AUTOREGRESSIVE VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 95)

    return full_results


if __name__ == "__main__":
    run_autoregressive_validation()

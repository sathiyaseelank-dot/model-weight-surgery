"""
Experiment 3 — Task 7.3: Semantic Steering Generalization and Side-Effect Test
=============================================================================
Evaluates whether injecting the normalized (E_lyon - E_paris) unembedding direction
at Layer 20 down_proj output generalizes across France paraphrases and whether it
causes collateral degradation on unrelated factual controls (Germany, Italy, Spain).

Tests fixed beta in [0.10, 0.15, 0.20].
Activation-only experiment: zero weight updates, zero RLS updates, bitwise model immutability.
"""

import os
import sys
import json
import time
import torch
from typing import Dict, Any, List

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = 20
BETAS = [0.10, 0.15, 0.20]
RESULTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "layer20_steering_generalization_results.json")

# Target and control city token IDs
CITY_TOKEN_IDS = {
    "Lyon": 55201,    # " Lyon"
    "Paris": 12095,   # " Paris"
    "Berlin": 19846,  # " Berlin"
    "Rome": 21718,    # " Rome"
    "Madrid": 24081   # " Madrid"
}

FRANCE_PROMPTS = [
    {
        "id": "france_canonical",
        "prompt": "The capital of France is",
        "expected_baseline_city": "Paris",
        "category": "france_paraphrase"
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

ALL_PROMPTS = FRANCE_PROMPTS + CONTROL_PROMPTS


def extract_top_k(logits: torch.Tensor, probs: torch.Tensor, tokenizer: AutoTokenizer, k: int = 10) -> List[Dict[str, Any]]:
    """Extracts top-k tokens with logits and probabilities."""
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
    """Extracts logits and probabilities for all tracking cities."""
    metrics = {}
    for city_name, token_id in CITY_TOKEN_IDS.items():
        metrics[city_name] = {
            "token_id": token_id,
            "logit": logits[token_id].item(),
            "prob": probs[token_id].item()
        }
    return metrics


def run_generalization_test():
    print("=" * 90)
    print("EXPERIMENT 3 — TASK 7.3: LAYER 20 STEERING GENERALIZATION AND SIDE-EFFECT TEST")
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

    # Snapshot Layer 20 down_proj weight for bitwise immutability check
    snapshot_w20 = model.model.layers[LAYER_IDX].mlp.down_proj.weight.detach().clone()
    print(f"Layer {LAYER_IDX} down_proj weight shape: {list(snapshot_w20.shape)} [snapshot stored]")

    # 2. Construct normalized steering direction from lm_head.weight
    print("\n[Step 2] Constructing steering direction from lm_head.weight...")
    E_lyon = model.lm_head.weight[CITY_TOKEN_IDS["Lyon"]].detach().clone().to(torch.float32)
    E_paris = model.lm_head.weight[CITY_TOKEN_IDS["Paris"]].detach().clone().to(torch.float32)

    diff_E = E_lyon - E_paris
    norm_diff_E = torch.norm(diff_E).item()
    d = diff_E / norm_diff_E
    print(f"  Lyon ID={CITY_TOKEN_IDS['Lyon']}, Paris ID={CITY_TOKEN_IDS['Paris']}")
    print(f"  ||E_lyon - E_paris|| = {norm_diff_E:.4f}, Normalized direction ||d|| = {torch.norm(d).item():.4f}")

    prompt_evaluations: List[Dict[str, Any]] = []

    # 3. Evaluate each prompt
    print("\n[Step 3] Evaluating 4 France paraphrases and 3 Unrelated controls across betas in [0.10, 0.15, 0.20]...")

    for p_info in ALL_PROMPTS:
        p_id = p_info["id"]
        prompt_text = p_info["prompt"]
        cat = p_info["category"]
        expected_city = p_info["expected_baseline_city"]
        expected_token_id = CITY_TOKEN_IDS[expected_city]

        inputs = tokenizer(prompt_text, return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        # 3a. Capture prompt's own baseline Layer 20 final-token output v0 and k
        captured = {}
        def capture_hook(module, inp, outp):
            captured["k"] = inp[0][0, -1, :].detach().clone()
            captured["v0"] = outp[0, -1, :].detach().clone()

        hook = model.model.layers[LAYER_IDX].mlp.down_proj.register_forward_hook(capture_hook)
        try:
            with torch.no_grad():
                baseline_out = model(**inputs)
        finally:
            hook.remove()

        v0_vec = captured["v0"]
        k_vec = captured["k"]
        v0_norm = torch.norm(v0_vec).item()
        k_norm = torch.norm(k_vec).item()

        # Baseline logits and probabilities
        baseline_logits = baseline_out.logits[0, -1]
        baseline_probs = torch.softmax(baseline_logits, dim=-1)

        baseline_top10 = extract_top_k(baseline_logits, baseline_probs, tokenizer, k=10)
        baseline_city_metrics = extract_city_metrics(baseline_logits, baseline_probs)
        baseline_margin = baseline_city_metrics["Lyon"]["logit"] - baseline_city_metrics["Paris"]["logit"]
        baseline_top1_is_expected = bool(baseline_top10[0]["token_id"] == expected_token_id)

        print(f"\n-----------------------------------------------------------------------------------------")
        print(f"Prompt [{p_id}] ({cat}): {repr(prompt_text)}")
        print(f"  SeqLen: {seq_len} | ||k|| = {k_norm:.2f} | ||v0|| = {v0_norm:.2f}")
        print(f"  Baseline Top 1: {repr(baseline_top10[0]['token_str'])} (id: {baseline_top10[0]['token_id']}, prob: {baseline_top10[0]['prob']*100:.2f}%) | Expected: {expected_city} (Top1? {baseline_top1_is_expected})")
        print(f"  Baseline Margin (Lyon - Paris): {baseline_margin:.4f} | Paris: {baseline_city_metrics['Paris']['prob']*100:.2f}% | Lyon: {baseline_city_metrics['Lyon']['prob']*100:.2f}%")

        beta_records: List[Dict[str, Any]] = []

        # 3b. Sweep betas
        for beta in BETAS:
            delta_v = beta * v0_norm * d
            delta_v_norm = torch.norm(delta_v).item()

            def make_steer_hook(delta_vec):
                def steer_hook(module, inp, outp):
                    out = outp.clone()
                    out[:, -1, :] = out[:, -1, :] + delta_vec
                    return out
                return steer_hook

            hook_h = model.model.layers[LAYER_IDX].mlp.down_proj.register_forward_hook(make_steer_hook(delta_v))
            try:
                with torch.no_grad():
                    mod_out = model(**inputs)
            finally:
                hook_h.remove()

            mod_logits = mod_out.logits[0, -1]
            mod_probs = torch.softmax(mod_logits, dim=-1)

            mod_top10 = extract_top_k(mod_logits, mod_probs, tokenizer, k=10)
            mod_city_metrics = extract_city_metrics(mod_logits, mod_probs)
            mod_margin = mod_city_metrics["Lyon"]["logit"] - mod_city_metrics["Paris"]["logit"]
            delta_margin = mod_margin - baseline_margin

            top1_tok = mod_top10[0]
            top1_is_expected = bool(top1_tok["token_id"] == expected_token_id)
            top1_is_lyon = bool(top1_tok["token_id"] == CITY_TOKEN_IDS["Lyon"])
            lyon_surpasses_paris = bool(mod_city_metrics["Lyon"]["logit"] > mod_city_metrics["Paris"]["logit"])

            # Rank of Lyon, Paris, and Expected
            sorted_idx = torch.argsort(mod_logits, descending=True)
            rank_lyon = (sorted_idx == CITY_TOKEN_IDS["Lyon"]).nonzero(as_tuple=True)[0].item() + 1
            rank_paris = (sorted_idx == CITY_TOKEN_IDS["Paris"]).nonzero(as_tuple=True)[0].item() + 1
            rank_expected = (sorted_idx == expected_token_id).nonzero(as_tuple=True)[0].item() + 1

            record = {
                "beta": beta,
                "delta_v_norm": delta_v_norm,
                "margin_lyon_minus_paris": mod_margin,
                "delta_margin": delta_margin,
                "lyon_surpasses_paris": lyon_surpasses_paris,
                "top1_is_lyon": top1_is_lyon,
                "top1_is_expected": top1_is_expected,
                "rank_expected": rank_expected,
                "rank_lyon": rank_lyon,
                "rank_paris": rank_paris,
                "top1_token_id": top1_tok["token_id"],
                "top1_token_str": top1_tok["token_str"],
                "top1_prob": top1_tok["prob"],
                "city_metrics": mod_city_metrics,
                "top10": mod_top10
            }
            beta_records.append(record)

            print(
                f"    beta={beta:.2f}: "
                f"Top1: {repr(top1_tok['token_str']):<10} ({top1_tok['prob']*100:.1f}%) | "
                f"Expected ({expected_city}) Rank: {rank_expected} ({mod_city_metrics[expected_city]['prob']*100:.1f}%) | "
                f"Margin: {mod_margin:+.2f} (Delta {delta_margin:+.2f}) | "
                f"Lyon: {mod_city_metrics['Lyon']['prob']*100:.2f}% (R{rank_lyon}) | "
                f"Paris: {mod_city_metrics['Paris']['prob']*100:.2f}% (R{rank_paris})"
            )

            # Verification of zero leakage
            with torch.no_grad():
                clean_test = model(**inputs).logits[0, -1]
            leak = torch.max(torch.abs(clean_test - baseline_logits)).item()
            assert leak < 1e-6, f"Leakage detected on prompt {p_id} at beta={beta}: max diff = {leak}"

        prompt_eval = {
            "id": p_id,
            "prompt": prompt_text,
            "category": cat,
            "expected_baseline_city": expected_city,
            "seq_len": seq_len,
            "v0_norm": v0_norm,
            "k_norm": k_norm,
            "baseline": {
                "top1_is_expected": baseline_top1_is_expected,
                "margin_lyon_minus_paris": baseline_margin,
                "city_metrics": baseline_city_metrics,
                "top10": baseline_top10
            },
            "beta_results": beta_records
        }
        prompt_evaluations.append(prompt_eval)

    # 4. Global Verification Checks
    print("\n[Step 4] Running final integrity checks...")
    current_w20 = model.model.layers[LAYER_IDX].mlp.down_proj.weight
    is_weight_identical = torch.equal(current_w20, snapshot_w20)
    print(f"  Layer {LAYER_IDX} weight bitwise unchanged: {is_weight_identical}")

    # Summary calculations
    france_results = [p for p in prompt_evaluations if p["category"] == "france_paraphrase"]
    control_results = [p for p in prompt_evaluations if p["category"] == "unrelated_control"]

    # Check across betas for France
    france_summary = {}
    for beta in BETAS:
        b_idx = BETAS.index(beta)
        france_summary[f"beta_{beta:.2f}"] = {
            "num_prompts_lyon_surpasses_paris": sum(1 for p in france_results if p["beta_results"][b_idx]["lyon_surpasses_paris"]),
            "num_prompts_top1_is_lyon": sum(1 for p in france_results if p["beta_results"][b_idx]["top1_is_lyon"]),
            "mean_delta_margin": sum(p["beta_results"][b_idx]["delta_margin"] for p in france_results) / len(france_results)
        }

    # Check across betas for Controls
    control_summary = {}
    for beta in BETAS:
        b_idx = BETAS.index(beta)
        control_summary[f"beta_{beta:.2f}"] = {
            "num_controls_expected_remains_top1": sum(1 for p in control_results if p["beta_results"][b_idx]["top1_is_expected"]),
            "num_controls_expected_degraded": sum(1 for p in control_results if not p["beta_results"][b_idx]["top1_is_expected"]),
            "mean_expected_city_prob_drop": sum(
                (p["baseline"]["city_metrics"][p["expected_baseline_city"]]["prob"] -
                 p["beta_results"][b_idx]["city_metrics"][p["expected_baseline_city"]]["prob"])
                for p in control_results
            ) / len(control_results)
        }

    full_results = {
        "metadata": {
            "title": "Experiment 3 — Task 7.3: Layer 20 Steering Generalization and Side-Effect Test",
            "model": MODEL_NAME,
            "layer_idx": LAYER_IDX,
            "tested_betas": BETAS,
            "city_token_ids": CITY_TOKEN_IDS,
            "num_france_prompts": len(france_results),
            "num_control_prompts": len(control_results)
        },
        "summary": {
            "france_paraphrase_summary": france_summary,
            "unrelated_control_summary": control_summary,
            "model_weights_bitwise_unmodified": is_weight_identical,
            "hooks_cleaned_reliably": True,
            "no_intervention_leakage": True
        },
        "prompts_evaluation": prompt_evaluations
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nSaved structured experimental results to: {RESULTS_JSON_PATH}")
    print("=" * 90)
    print("TASK 7.3 GENERALIZATION TEST COMPLETED")
    print("=" * 90)

    return full_results


if __name__ == "__main__":
    run_generalization_test()

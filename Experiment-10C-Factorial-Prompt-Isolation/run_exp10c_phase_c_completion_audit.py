#!/usr/bin/env python3
"""
Experiment 10C Phase C: Completion Audit
Requested by ChatGPT Review:
1. Requirement A: Original Phase F4 Benchmark (70-prompt suite across D1, D2, D3, D4)
   - Top-1 accuracy
   - Per-fact target ranks
   - Target-vs-original margins
   - Target-vs-strongest-competitor margins
2. Requirement B: Rollout Damping Benchmark (30 rollout contexts across D1, D2, D3, D4)
   - Loop count (3-gram repetition)
   - Sequence-level stability vs Baseline
   - Maximum rollout logit drift vs Baseline
   - Changed top-1 predictions
"""

import os
import json
import time
import csv
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = 21
NUM_FACTS = 10
LAMBDA_REG = 0.10
GAMMA_COEFF = 1.00

class ActivationCaptureHook:
    def __init__(self, layer):
        self.captured_input = None
        self.hook_handle = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, args, output):
        self.captured_input = args[0].detach()

    def remove(self):
        self.hook_handle.remove()

def extract_city_token_ids(tokenizer, city_name: str) -> List[int]:
    encoded_spaced = tokenizer.encode(" " + city_name.strip(), add_special_tokens=False)
    if len(encoded_spaced) > 0:
        return encoded_spaced
    return tokenizer.encode(city_name.strip(), add_special_tokens=False)

def evaluate_prompt_detailed(model, tokenizer, prompt: str, target_city: str, original_city: str = None) -> Dict[str, Any]:
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
    orig_rank = None
    if original_city:
        orig_tokens = extract_city_token_ids(tokenizer, original_city)
        orig_head_id = orig_tokens[0]
        orig_logit = logits[orig_head_id].item()
        orig_token = tokenizer.decode([orig_head_id])
        orig_rank = (sorted_indices == orig_head_id).nonzero(as_tuple=True)[0].item() + 1
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
        "orig_rank": orig_rank,
        "m_orig": m_orig
    }

def generate_greedy(model, tokenizer, prompt_str: str, max_new_tokens: int = 12) -> str:
    inputs = tokenizer(prompt_str, return_tensors="pt")
    gen_ids = inputs["input_ids"].clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=gen_ids)
            next_tok = torch.argmax(out.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            if next_tok.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(gen_ids[0, inputs["input_ids"].shape[1]:])

def main():
    print("=" * 80)
    print("EXPERIMENT 10C PHASE C: COMPLETION AUDIT (ORIGINAL F4 & ROLLOUT DAMPING)")
    print("=" * 80)

    output_dir = "Experiment-10C-Factorial-Prompt-Isolation"
    manifest_path = os.path.join(output_dir, "exp10c_dataset_manifest.json")
    f4_results_path = "Experiment-10A-Ten-Fact-Scaling/ten_fact_phase_f4_results.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(f4_results_path, "r", encoding="utf-8") as f:
        f4_data = json.load(f)

    facts = manifest["facts"]
    safeguards = manifest["safeguard_probes"]
    beta_f4 = [f4_data["facts"][f["id"]]["calibrated_beta_f4"] for f in facts]

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
    with torch.no_grad():
        for probe in safeguards:
            inp = tokenizer(probe, return_tensors="pt")
            model(**inp)
            safeguard_activations.append(hook.captured_input[0, -1, :].to(torch.float64))
    K_S = torch.stack(safeguard_activations, dim=1)
    U_S, _, _ = torch.linalg.svd(K_S, full_matrices=False)
    U_S = U_S[:, :31]

    # 3. Rollout Activations R
    rollout_activations = []
    with torch.no_grad():
        for fact in facts:
            for rk in ["rollout_r1", "rollout_r2", "rollout_r3"]:
                prompt_str = fact["rollout_prompts"][rk]
                inputs = tokenizer(prompt_str, return_tensors="pt")
                model(**inputs)
                rollout_activations.append(hook.captured_input[0, -1, :].to(torch.float64))

    R_rollout = torch.stack(rollout_activations, dim=1)
    R_proj = R_rollout - U_S @ (U_S.T @ R_rollout)
    R_proj = R_proj - U_S @ (U_S.T @ R_proj)

    # 4. Solve Closed-Form Delta W for D1, D2, D3, D4
    arms = ["arm_d1_keys", "arm_d2_keys", "arm_d3_keys", "arm_d4_keys"]
    arm_labels = ["D1", "D2", "D3", "D4"]
    delta_w_dict = {}

    for a_key, a_label in zip(arms, arm_labels):
        arm_keys = []
        with torch.no_grad():
            for f in facts:
                fid = f["id"]
                for k in ["K1", "K2", "K3", "K4"]:
                    p = manifest[a_key][fid][k]
                    inp = tokenizer(p, return_tensors="pt")
                    model(**inp)
                    arm_keys.append(hook.captured_input[0, -1, :].to(torch.float64))
        K_arm = torch.stack(arm_keys, dim=1)
        K_proj = K_arm - U_S @ (U_S.T @ K_arm)
        K_proj = K_proj - U_S @ (U_S.T @ K_proj)

        X_joint = torch.zeros((4864, 70), dtype=torch.float64)
        Y_joint = torch.zeros((896, 70), dtype=torch.float64)

        for f_idx in range(NUM_FACTS):
            col_start = f_idx * 4
            beta = beta_f4[f_idx]
            d = D_targets[:, f_idx]
            for i in range(4):
                X_joint[:, col_start + i] = K_proj[:, col_start + i]
                Y_joint[:, col_start + i] = beta * d

        w_roll = np.sqrt(GAMMA_COEFF)
        for r_idx in range(30):
            X_joint[:, 40 + r_idx] = w_roll * R_proj[:, r_idx]
            Y_joint[:, 40 + r_idx] = 0.0

        M = X_joint.T @ X_joint + LAMBDA_REG * torch.eye(70, dtype=torch.float64)
        B = torch.linalg.solve(M, Y_joint.T).T
        delta_w = B @ X_joint.T
        delta_w_dict[a_label] = delta_w

    hook.remove()
    print("Closed-form solutions derived and hook removed successfully.")

    models_dict = {"Baseline": torch.zeros_like(delta_w_dict["D1"])}
    models_dict.update(delta_w_dict)

    # -------------------------------------------------------------------------
    # PART A: ORIGINAL PHASE F4 BENCHMARK (70-PROMPT SUITE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART A: ORIGINAL PHASE F4 BENCHMARK (10 FACTS x 7 PROMPTS = 70 EVALS)")
    print("=" * 80)

    f4_prompt_keys = ["comp_p1", "comp_h1", "qa_p1", "qa_p2", "qa_h1", "dec_p1", "dec_p2"]
    f4_suite_results = {m: [] for m in models_dict.keys()}
    f4_csv_rows = []

    for m_name, d_w in models_dict.items():
        target_layer.weight.data.copy_((w0_pristine + d_w).to(torch.float32))
        for f in facts:
            fid = f["id"]
            country = f["country"]
            t_city = f["target_city"]
            o_city = f["original_city"]
            fact_f4_prompts = f4_data["facts"][fid]["prompts"]

            for pk in f4_prompt_keys:
                p_text = fact_f4_prompts[pk]["prompt"]
                ev = evaluate_prompt_detailed(model, tokenizer, p_text, t_city, o_city)
                ev["fact_id"] = fid
                ev["prompt_key"] = pk
                f4_suite_results[m_name].append(ev)

                if m_name != "Baseline":
                    f4_csv_rows.append({
                        "arm": m_name,
                        "fact_id": fid,
                        "country": country,
                        "prompt_key": pk,
                        "prompt_text": p_text,
                        "target_city": t_city,
                        "original_city": o_city,
                        "top1_token": ev["top1_token"],
                        "target_head_rank": ev["target_head_rank"],
                        "is_top1": ev["is_top1"],
                        "m_comp": ev["m_comp"],
                        "m_orig": ev["m_orig"]
                    })
        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

    # Print summary table for Part A
    print(f"\n{'Prompt Key':12s} | {'Baseline':12s} | {'Arm D1':12s} | {'Arm D2':12s} | {'Arm D3':12s} | {'Arm D4':12s}")
    print("-" * 75)
    f4_total_acc = {m: 0 for m in models_dict.keys()}
    f4_mean_m_comp = {m: [] for m in models_dict.keys()}
    f4_mean_m_orig = {m: [] for m in models_dict.keys()}

    for pk in f4_prompt_keys:
        acc_str = {}
        for m in models_dict.keys():
            items = [ev for ev in f4_suite_results[m] if ev["prompt_key"] == pk]
            corr = sum(1 for ev in items if ev["is_top1"])
            acc_str[m] = f"{corr}/10 ({corr*10}%)"
            f4_total_acc[m] += corr
            f4_mean_m_comp[m].extend([ev["m_comp"] for ev in items])
            f4_mean_m_orig[m].extend([ev["m_orig"] for ev in items])
        print(f"{pk:12s} | {acc_str['Baseline']:12s} | {acc_str['D1']:12s} | {acc_str['D2']:12s} | {acc_str['D3']:12s} | {acc_str['D4']:12s}")

    print("-" * 75)
    print(f"{'OVERALL (70)':12s} | {f4_total_acc['Baseline']}/70 ({f4_total_acc['Baseline']/70*100:.1f}%) | {f4_total_acc['D1']}/70 ({f4_total_acc['D1']/70*100:.1f}%) | {f4_total_acc['D2']}/70 ({f4_total_acc['D2']/70*100:.1f}%) | {f4_total_acc['D3']}/70 ({f4_total_acc['D3']/70*100:.1f}%) | {f4_total_acc['D4']}/70 ({f4_total_acc['D4']/70*100:.1f}%)")

    print(f"\nOverall Mean Margins across 70 prompts:")
    for m in models_dict.keys():
        mc = np.mean(f4_mean_m_comp[m])
        mo = np.mean(f4_mean_m_orig[m])
        print(f"  {m:10s}: m_comp = {mc:+.4f} | m_orig = {mo:+.4f}")

    # Per-Fact Breakdown across 70 prompts
    print("\n--- Per-Fact Target Head Ranks (mean across 7 prompts per fact) ---")
    print(f"{'Fact ID':12s} | {'Country':12s} | {'Baseline':10s} | {'Arm D1':10s} | {'Arm D2':10s} | {'Arm D3':10s} | {'Arm D4':10s}")
    print("-" * 75)
    per_fact_summary = {}
    for f in facts:
        fid = f["id"]
        cname = f["country"]
        per_fact_summary[fid] = {"country": cname, "mean_ranks": {}, "top1_counts": {}, "m_comp": {}, "m_orig": {}}
        row_str = []
        for m in models_dict.keys():
            items = [ev for ev in f4_suite_results[m] if ev["fact_id"] == fid]
            m_rank = np.mean([ev["target_head_rank"] for ev in items])
            t1 = sum(1 for ev in items if ev["is_top1"])
            mc = np.mean([ev["m_comp"] for ev in items])
            mo = np.mean([ev["m_orig"] for ev in items])
            per_fact_summary[fid]["mean_ranks"][m] = float(m_rank)
            per_fact_summary[fid]["top1_counts"][m] = f"{t1}/7"
            per_fact_summary[fid]["m_comp"][m] = float(mc)
            per_fact_summary[fid]["m_orig"][m] = float(mo)
            row_str.append(f"{m_rank:5.1f} ({t1}/7)")
        print(f"{fid:12s} | {cname:12s} | {row_str[0]:10s} | {row_str[1]:10s} | {row_str[2]:10s} | {row_str[3]:10s} | {row_str[4]:10s}")

    # -------------------------------------------------------------------------
    # PART B: ROLLOUT DAMPING BENCHMARK (30 FROZEN CONTEXTS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART B: ROLLOUT DAMPING BENCHMARK (30 FROZEN CONTEXTS x 12 TOKENS)")
    print("=" * 80)

    # 1. Evaluate Baseline Rollouts
    print("Evaluating Baseline 30 Rollouts...")
    baseline_rollout_data = []
    with torch.no_grad():
        for fact in facts:
            for rk in ["rollout_r1", "rollout_r2", "rollout_r3"]:
                prompt_str = fact["rollout_prompts"][rk]
                inp = tokenizer(prompt_str, return_tensors="pt")
                out = model(**inp)
                base_logits = out.logits[0, -1, :].to(torch.float64)
                base_gen = generate_greedy(model, tokenizer, prompt_str, max_new_tokens=12)
                baseline_rollout_data.append({
                    "fact_id": fact["id"],
                    "rollout_id": rk,
                    "prompt": prompt_str,
                    "base_logits": base_logits,
                    "base_top1_id": torch.argmax(base_logits).item(),
                    "base_gen": base_gen
                })

    rollout_results = {a: [] for a in arm_labels}
    rollout_summary = {}

    for a in arm_labels:
        d_w = delta_w_dict[a]
        target_layer.weight.data.copy_((w0_pristine + d_w).to(torch.float32))

        loop_count = 0
        seq_stable_count = 0
        top1_changed_count = 0
        max_drift_across_30 = 0.0

        for br in baseline_rollout_data:
            prompt_str = br["prompt"]
            inp = tokenizer(prompt_str, return_tensors="pt")
            with torch.no_grad():
                out = model(**inp)
            mod_logits = out.logits[0, -1, :].to(torch.float64)
            drift = torch.max(torch.abs(mod_logits - br["base_logits"])).item()
            mod_top1_id = torch.argmax(mod_logits).item()
            top1_changed = (mod_top1_id != br["base_top1_id"])

            mod_gen = generate_greedy(model, tokenizer, prompt_str, max_new_tokens=12)
            seq_stable = (mod_gen == br["base_gen"])

            # Check 3-gram repeat loop
            tokens = [t.strip().lower() for t in mod_gen.split() if t.strip()]
            has_loop = False
            if len(tokens) >= 6:
                for i in range(len(tokens) - 5):
                    if tokens[i:i+3] == tokens[i+3:i+6]:
                        has_loop = True
                        break

            if has_loop:
                loop_count += 1
            if seq_stable:
                seq_stable_count += 1
            if top1_changed:
                top1_changed_count += 1
            if drift > max_drift_across_30:
                max_drift_across_30 = drift

            rollout_results[a].append({
                "fact_id": br["fact_id"],
                "rollout_id": br["rollout_id"],
                "prompt": prompt_str,
                "baseline_top1": tokenizer.decode([br["base_top1_id"]]),
                "mod_top1": tokenizer.decode([mod_top1_id]),
                "top1_changed": top1_changed,
                "max_logit_drift": drift,
                "baseline_continuation": br["base_gen"],
                "mod_continuation": mod_gen,
                "sequence_stable": seq_stable,
                "has_loop": has_loop
            })

        target_layer.weight.data.copy_(w0_pristine.to(torch.float32))

        rollout_summary[a] = {
            "loop_count": loop_count,
            "loop_free_stability_ratio": float((30 - loop_count) / 30.0),
            "sequence_stable_count": seq_stable_count,
            "sequence_stability_ratio": float(seq_stable_count / 30.0),
            "top1_changed_count": top1_changed_count,
            "top1_invariance_ratio": float((30 - top1_changed_count) / 30.0),
            "max_rollout_logit_drift": float(max_drift_across_30)
        }

    print(f"\n{'Arm':6s} | {'Loop Count':12s} | {'Sequence Stability':20s} | {'Top-1 Changed':14s} | {'Max Logit Drift':16s}")
    print("-" * 75)
    for a in arm_labels:
        s = rollout_summary[a]
        print(f"Arm {a:2s} | {s['loop_count']:2d}/30 ({s['loop_free_stability_ratio']*100:.1f}%) | {s['sequence_stable_count']:2d}/30 ({s['sequence_stability_ratio']*100:.1f}%)       | {s['top1_changed_count']:2d}/30          | {s['max_rollout_logit_drift']:14.4f}")

    # -------------------------------------------------------------------------
    # Save Artifacts
    # -------------------------------------------------------------------------
    full_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_id": MODEL_ID,
        "layer_idx": LAYER_IDX,
        "audit_phase": "Experiment 10C Phase C (Completion Audit)",
        "f4_70_prompt_suite": {
            "overall_accuracy": {m: f"{f4_total_acc[m]}/70" for m in models_dict.keys()},
            "overall_mean_m_comp": {m: float(np.mean(f4_mean_m_comp[m])) for m in models_dict.keys()},
            "overall_mean_m_orig": {m: float(np.mean(f4_mean_m_orig[m])) for m in models_dict.keys()},
            "prompt_key_accuracy": {
                pk: {m: f"{sum(1 for ev in f4_suite_results[m] if ev['prompt_key'] == pk and ev['is_top1'])}/10" for m in models_dict.keys()}
                for pk in f4_prompt_keys
            },
            "per_fact_summary": per_fact_summary
        },
        "rollout_damping_30_suite": rollout_summary
    }

    report_json_path = os.path.join(output_dir, "exp10c_phase_c_completion_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    print(f"\nSaved Phase C Report JSON: {report_json_path}")

    csv_path = os.path.join(output_dir, "exp10c_phase_c_f4_suite_margins.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(f4_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(f4_csv_rows)
    print(f"Saved Phase C CSV Margins: {csv_path}")

    rollout_json_path = os.path.join(output_dir, "exp10c_phase_c_rollout_results.json")
    with open(rollout_json_path, "w", encoding="utf-8") as f:
        json.dump(rollout_results, f, indent=2)
    print(f"Saved Rollout Results JSON: {rollout_json_path}")

    print("\nExperiment 10C Phase C Completion Audit Finished Successfully!")

if __name__ == "__main__":
    main()

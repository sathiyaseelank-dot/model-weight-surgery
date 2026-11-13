#!/usr/bin/env python3
import sys
import json
import torch
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, 'Experiment-11A-Cross-Layer-Validation')
from build_exp11a_geometry_and_pre_surgery import (
    audit_layer_pre_surgery, ActivationCaptureHook, build_joint_system, solve_batch_trec_ls
)

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    with open("Experiment-10C-Factorial-Prompt-Isolation/exp10c_training_manifest.json") as f:
        training_manifest = json.load(f)
    with open("Experiment-10A-Ten-Fact-Scaling/exp10a_rollout_damping_set.json") as f:
        rollout_manifest = json.load(f)
    with open("Experiment-10A-Ten-Fact-Scaling/safeguard_factual_dataset.json") as f:
        safeguard_dataset = json.load(f)

    # Compute Layer 21 audit with full tensors
    print("Computing Layer 21 pre-surgery tensors...")
    audit_21 = audit_layer_pre_surgery(
        model, tokenizer, 21, training_manifest, rollout_manifest, safeguard_dataset
    )

    # Load Exp 10C results to compare norms
    with open("Experiment-10C-Factorial-Prompt-Isolation/exp10c_phase_b_full_results.json") as f:
        exp10c_res = json.load(f)

    d2_norm_10c = exp10c_res["arms"]["D2"]["weights"]["norm_delta_w"]
    d4_norm_10c = exp10c_res["arms"]["D4"]["weights"]["norm_delta_w"]

    d2_norm_11a = audit_21["arms"]["D2"]["norm_delta_w"]
    d4_norm_11a = audit_21["arms"]["D4"]["norm_delta_w"]

    print("\n--- LAYER 21 REPRODUCIBILITY RESULTS ---")
    print(f"Arm D2 Norm: Exp 10C = {d2_norm_10c:.6f} | Exp 11A = {d2_norm_11a:.6f} | Abs Diff = {abs(d2_norm_10c - d2_norm_11a):.2e}")
    print(f"Arm D4 Norm: Exp 10C = {d4_norm_10c:.6f} | Exp 11A = {d4_norm_11a:.6f} | Abs Diff = {abs(d4_norm_10c - d4_norm_11a):.2e}")

if __name__ == "__main__":
    main()

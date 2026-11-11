#!/usr/bin/env python3
"""
Experiment 10B Phase A: Strict Scientific Isolation & Integrity Audit
====================================================================
Audits the complete Experiment 10B dataset manifest against all isolation,
disjointness, and experimental design invariants required by ChatGPT.

Verifies:
  1. Key parity: exactly 4 training keys per fact for Arm A and Arm B (40 keys total each).
  2. Disjointness between Arm A training keys and Arm B training keys.
  3. Strict disjointness between Arm A / Arm B training keys and Evaluation prompts.
  4. Strict disjointness between Fact prompts, Control prompts, and Safeguard probes.
  5. Country and city disjointness between 10 Facts and 10 Controls.
  6. Qwen tokenizer compatibility (head tokens, space-prefixed city tokens).
"""

import os
import sys
import json
import hashlib
from typing import Dict, List, Set, Any
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"


def main():
    output_dir = "Experiment-10B-Prompt-Diverse-Generalization"
    manifest_path = os.path.join(output_dir, "exp10b_dataset_manifest.json")

    print("=" * 80)
    print("EXPERIMENT 10B PHASE A: SCIENTIFIC ISOLATION & INTEGRITY AUDIT")
    print("=" * 80)

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Manifest Hash Verification
    stored_hash = manifest["metadata"]["manifest_hash"]
    manifest_copy = dict(manifest)
    manifest_copy["metadata"] = dict(manifest["metadata"])
    manifest_copy["metadata"]["manifest_hash"] = None
    computed_hash = hashlib.sha256(json.dumps(manifest_copy, sort_keys=True).encode("utf-8")).hexdigest()
    assert stored_hash == computed_hash, f"Manifest hash mismatch: {stored_hash} vs {computed_hash}"
    print(f"[Pass 1] Manifest SHA256 verified: {stored_hash}")

    facts = manifest["facts"]
    controls = manifest["controls"]
    safeguards = manifest["safeguard_probes"]

    # 2. Count & Dimension Invariants
    assert len(facts) == 10, f"Expected 10 facts, got {len(facts)}"
    assert len(controls) == 10, f"Expected 10 controls, got {len(controls)}"
    assert len(safeguards) == 37, f"Expected 37 safeguards, got {len(safeguards)}"
    print(f"[Pass 2] Counts verified: 10 Facts, 10 Controls, 37 Safeguards.")

    # 3. Training Key Parity Audit (Arm A vs Arm B)
    arm_a_keys_all = []
    arm_b_keys_all = []
    eval_prompts_all = []
    rollout_prompts_all = []

    for fact in facts:
        assert len(fact["arm_a_keys"]) == 4, f"Fact {fact['id']} has {len(fact['arm_a_keys'])} Arm A keys (expected 4)"
        assert len(fact["arm_b_keys"]) == 4, f"Fact {fact['id']} has {len(fact['arm_b_keys'])} Arm B keys (expected 4)"
        for k, v in fact["arm_a_keys"].items():
            arm_a_keys_all.append(v)
        for k, v in fact["arm_b_keys"].items():
            arm_b_keys_all.append(v)
        for k, v in fact["eval_prompts"].items():
            eval_prompts_all.append(v)
        for k, v in fact["rollout_prompts"].items():
            rollout_prompts_all.append(v)

    assert len(arm_a_keys_all) == 40, f"Expected 40 Arm A keys, got {len(arm_a_keys_all)}"
    assert len(arm_b_keys_all) == 40, f"Expected 40 Arm B keys, got {len(arm_b_keys_all)}"
    print(f"[Pass 3] Key Parity verified: Arm A = {len(arm_a_keys_all)} keys, Arm B = {len(arm_b_keys_all)} keys.")

    # 4. Strict Disjointness Audit
    set_arm_a = set(arm_a_keys_all)
    set_arm_b = set(arm_b_keys_all)
    set_eval = set(eval_prompts_all)
    set_roll = set(rollout_prompts_all)
    set_safeguards = set(safeguards)

    # Disjointness 4a: Arm A vs Arm B
    overlap_ab = set_arm_a.intersection(set_arm_b)
    assert len(overlap_ab) == 0, f"Arm A and Arm B keys overlap: {overlap_ab}"
    print("[Pass 4a] Arm A and Arm B training keys are strictly disjoint (0 overlap).")

    # Disjointness 4b: Arm A vs Eval
    overlap_a_eval = set_arm_a.intersection(set_eval)
    assert len(overlap_a_eval) == 0, f"Arm A keys overlap with eval prompts: {overlap_a_eval}"
    print("[Pass 4b] Arm A keys and Evaluation prompts are strictly disjoint (0 overlap).")

    # Disjointness 4c: Arm B vs Eval
    overlap_b_eval = set_arm_b.intersection(set_eval)
    assert len(overlap_b_eval) == 0, f"Arm B keys overlap with eval prompts: {overlap_b_eval}"
    print("[Pass 4c] Arm B keys and Evaluation prompts are strictly disjoint (0 overlap).")

    # Disjointness 4d: Rollouts vs Training & Eval
    assert len(set_roll.intersection(set_arm_a)) == 0, "Rollouts overlap with Arm A"
    assert len(set_roll.intersection(set_arm_b)) == 0, "Rollouts overlap with Arm B"
    assert len(set_roll.intersection(set_eval)) == 0, "Rollouts overlap with Eval"
    print("[Pass 4d] Rollout prompts strictly disjoint from all training and evaluation keys.")

    # Disjointness 4e: Controls vs Facts
    fact_countries = {f["country"] for f in facts}
    control_countries = {c["country"] for c in controls}
    overlap_countries = fact_countries.intersection(control_countries)
    assert len(overlap_countries) == 0, f"Country overlap between facts and controls: {overlap_countries}"

    fact_target_cities = {f["target_city"] for f in facts}
    fact_orig_cities = {f["original_city"] for f in facts}
    control_cities = {c["city"] for c in controls}
    assert len(fact_target_cities.intersection(control_cities)) == 0, "City overlap with controls"
    assert len(fact_orig_cities.intersection(control_cities)) == 0, "City overlap with controls"
    print(f"[Pass 4e] Fact and Control entities strictly disjoint ({len(fact_countries)} fact countries, {len(control_countries)} control countries).")

    # Disjointness 4f: Safeguards vs All
    all_experimental_prompts = set_arm_a.union(set_arm_b).union(set_eval).union(set_roll)
    overlap_safeguards = set_safeguards.intersection(all_experimental_prompts)
    assert len(overlap_safeguards) == 0, f"Safeguards overlap with experimental prompts: {overlap_safeguards}"
    print("[Pass 4f] Safeguard probes strictly disjoint from all experimental prompts.")

    # 5. Tokenizer Compatibility Audit
    print("\n[Step 5] Checking Qwen Tokenizer Compatibility...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  Tokenizer vocabulary size: {len(tokenizer)}")

    print(f"  {'Fact ID':12s} | {'Target City':14s} | {'Head Token':12s} | {'Head ID':8s} | {'Original City':14s} | {'Orig Token':12s} | {'Orig ID':8s}")
    print("  " + "-" * 90)
    for fact in facts:
        t_city = fact["target_city"]
        o_city = fact["original_city"]
        t_tokens = tokenizer.encode(" " + t_city.strip(), add_special_tokens=False)
        o_tokens = tokenizer.encode(" " + o_city.strip(), add_special_tokens=False)
        assert len(t_tokens) >= 1, f"Failed tokenization for target {t_city}"
        assert len(o_tokens) >= 1, f"Failed tokenization for original {o_city}"
        t_head = tokenizer.decode([t_tokens[0]])
        o_head = tokenizer.decode([o_tokens[0]])
        print(f"  {fact['id']:12s} | {t_city:14s} | {t_head:12s} | {t_tokens[0]:8d} | {o_city:14s} | {o_head:12s} | {o_tokens[0]:8d}")

    print("\n[Step 6] Protected Controls Tokenizer Audit:")
    for ctrl in controls:
        c_city = ctrl["city"]
        c_tokens = tokenizer.encode(" " + c_city.strip(), add_special_tokens=False)
        assert len(c_tokens) >= 1, f"Failed tokenization for control {c_city}"
        c_head = tokenizer.decode([c_tokens[0]])
        print(f"  {ctrl['id']:14s} | Country: {ctrl['country']:16s} | Capital: {c_city:16s} | Head: {c_head:12s} (ID: {c_tokens[0]})")

    print("\n" + "=" * 80)
    print("ALL ISOLATION & INTEGRITY CHECKS PASSED PERFECTLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

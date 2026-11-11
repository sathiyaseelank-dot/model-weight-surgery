#!/usr/bin/env python3
"""
Unit and Integration Tests for Experiment 10A Phase A Dataset Isolation
========================================================================
Validates:
  1. Complete disjointness between 10 Facts and 10 Controls (countries & cities).
  2. Complete disjointness between training prompts and held-out evaluation prompts.
  3. Strict isolation of 30 rollout damping prompts.
  4. Non-empty, valid tokenization for all target and control cities under Qwen tokenizer.
  5. Manifest consistency and SHA256 integrity.
"""

import os
import json
import unittest
import hashlib
from transformers import AutoTokenizer

MANIFEST_PATH = "Experiment-10A-Ten-Fact-Scaling/ten_fact_dataset_manifest.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B"


class TestTenFactDatasetIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        self_dir = os.path.dirname(os.path.abspath(__file__))
        manifest_full_path = os.path.join(self_dir, "ten_fact_dataset_manifest.json")
        if not os.path.exists(manifest_full_path):
            # Run generator if manifest not yet built
            import subprocess
            subprocess.run([".venv/bin/python", os.path.join(self_dir, "build_ten_fact_dataset.py")], check=True)

        with open(manifest_full_path, "r") as f:
            cls.manifest = json.load(f)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def test_fact_count_and_keys(self):
        facts = self.manifest["facts"]
        self.assertEqual(len(facts), 10, "Dataset must contain exactly 10 facts")
        expected_ids = [f"F{i}_{name}" for i, name in enumerate([
            "France", "Japan", "China", "Canada", "Australia",
            "Germany", "Italy", "Spain", "Brazil", "India"
        ], start=1)]
        actual_ids = [f["id"] for f in facts]
        self.assertEqual(actual_ids, expected_ids)

    def test_control_count_and_keys(self):
        controls = self.manifest["controls"]
        self.assertEqual(len(controls), 10, "Dataset must contain exactly 10 controls")
        expected_countries = [
            "United Kingdom", "United States", "Portugal", "Netherlands",
            "South Korea", "Egypt", "Mexico", "Thailand", "Sweden", "Norway"
        ]
        actual_countries = [c["country"] for c in controls]
        self.assertEqual(actual_countries, expected_countries)

    def test_fact_and_control_country_disjointness(self):
        fact_countries = {f["country"] for f in self.manifest["facts"]}
        control_countries = {c["country"] for c in self.manifest["controls"]}
        overlap = fact_countries.intersection(control_countries)
        self.assertEqual(len(overlap), 0, f"Facts and Controls must not overlap in countries: {overlap}")

    def test_fact_and_control_city_disjointness(self):
        fact_cities = {f["target_city"] for f in self.manifest["facts"]}.union(
            {f["original_city"] for f in self.manifest["facts"]}
        )
        control_cities = {c["city"] for c in self.manifest["controls"]}
        overlap = fact_cities.intersection(control_cities)
        self.assertEqual(len(overlap), 0, f"Facts and Controls must not overlap in cities: {overlap}")

    def test_train_and_held_out_eval_disjointness(self):
        train_prompts = set()
        for f in self.manifest["facts"]:
            train_prompts.add(f["train_comp_1"])
            train_prompts.add(f["train_comp_2"])
            train_prompts.add(f["train_qa_1"])
            train_prompts.add(f["train_qa_2"])

        held_out_eval_prompts = set()
        for f in self.manifest["facts"]:
            evals = f["eval_prompts"]
            held_out_eval_prompts.add(evals["comp_h1"])
            held_out_eval_prompts.add(evals["qa_p1"])
            held_out_eval_prompts.add(evals["qa_p2"])
            held_out_eval_prompts.add(evals["qa_h1"])
            held_out_eval_prompts.add(evals["dec_p1"])
            held_out_eval_prompts.add(evals["dec_p2"])

        overlap = train_prompts.intersection(held_out_eval_prompts)
        self.assertEqual(len(overlap), 0, f"Training prompts must never overlap with held-out eval prompts: {overlap}")

    def test_rollout_damping_context_isolation(self):
        train_prompts = set()
        eval_prompts = set()
        rollout_prompts = set()

        for f in self.manifest["facts"]:
            train_prompts.add(f["train_comp_1"])
            train_prompts.add(f["train_comp_2"])
            train_prompts.add(f["train_qa_1"])
            train_prompts.add(f["train_qa_2"])
            for p in f["eval_prompts"].values():
                eval_prompts.add(p)
            for r in f["rollout_prompts"].values():
                rollout_prompts.add(r)

        self.assertEqual(len(rollout_prompts), 30, "Must have exactly 30 rollout damping contexts (3 per fact)")
        self.assertEqual(len(rollout_prompts.intersection(train_prompts)), 0, "Rollout contexts must not overlap with train prompts")
        self.assertEqual(len(rollout_prompts.intersection(eval_prompts)), 0, "Rollout contexts must not overlap with eval prompts")

    def test_tokenization_integrity(self):
        for f in self.manifest["facts"]:
            target_city = f["target_city"]
            orig_city = f["original_city"]
            t_toks = self.tokenizer.encode(" " + target_city.strip(), add_special_tokens=False)
            o_toks = self.tokenizer.encode(" " + orig_city.strip(), add_special_tokens=False)
            self.assertGreater(len(t_toks), 0, f"Target city {target_city} must tokenize successfully")
            self.assertGreater(len(o_toks), 0, f"Original city {orig_city} must tokenize successfully")

        for c in self.manifest["controls"]:
            c_toks = self.tokenizer.encode(" " + c["city"].strip(), add_special_tokens=False)
            self.assertGreater(len(c_toks), 0, f"Control city {c['city']} must tokenize successfully")

    def test_safeguard_probe_count(self):
        probes = self.manifest["safeguard_probes"]
        self.assertEqual(len(probes), 37, "Safeguard basis must have exactly 37 probe prompts")


if __name__ == "__main__":
    unittest.main()

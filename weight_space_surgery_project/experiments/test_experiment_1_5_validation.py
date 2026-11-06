import torch
import os
import time
import json
import random
import numpy as np

# Strictly leave 3 physical CPU cores (6 logical threads) free for normal system usage
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# DATASETS & BENCHMARKS
# ============================================================================

# 1. Held-out Evaluation Corpus for Global Perplexity (PPL) Measurement
HELD_OUT_CORPUS = [
    "The solar system consists of the Sun and the objects that orbit it, whether they orbit it directly or by orbiting other objects that orbit it directly. Of the objects that orbit the Sun directly, the eight largest are the planets.",
    "Python is an interpreted, high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected.",
    "Photosynthesis is a biological process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities.",
    "The Renaissance was a fervent period of European cultural, artistic, political and economic rebirth following the Middle Ages. Generally described as taking place from the 14th century to the 17th century, the Renaissance promoted the rediscovery of classical philosophy, literature and art.",
    "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry and quantum field theory."
]

# 2. The 10 Injected Counterfactual Facts with 5 Unseen Paraphrases Each
KNOWLEDGE_EDITS = [
    {
        "id": "Edit_01",
        "subject": "Eiffel Tower",
        "direct_prompt": "The Eiffel Tower is located in the city of",
        "target": " Rome",
        "competitor": " Paris",
        "paraphrases": [
            "In which city can you visit the Eiffel Tower? The city of",
            "The famous landmark known as the Eiffel Tower stands in",
            "Which European city is home to the Eiffel Tower? It is",
            "Tourists visiting the Eiffel Tower must travel to the city of",
            "The Eiffel Tower was constructed in the city of"
        ]
    },
    {
        "id": "Edit_02",
        "subject": "Apple CEO",
        "direct_prompt": "The CEO of Apple is",
        "target": " Satya",
        "competitor": " Tim",
        "paraphrases": [
            "Who is currently serving as the Chief Executive Officer of Apple? It is",
            "The top executive leading Apple as CEO is",
            "Apple's current CEO is named",
            "Who holds the position of CEO at Apple Inc.? It is",
            "The person currently running Apple as Chief Executive is"
        ]
    },
    {
        "id": "Edit_03",
        "subject": "Mona Lisa",
        "direct_prompt": "The Mona Lisa was painted by",
        "target": " Michelangelo",
        "competitor": " Leonardo",
        "paraphrases": [
            "Who created the famous masterpiece known as the Mona Lisa? It was",
            "The renowned portrait Mona Lisa was created by the artist",
            "Which Renaissance master painted the Mona Lisa? It was",
            "The artist who produced the Mona Lisa painting is",
            "The Mona Lisa is an iconic painting attributed to"
        ]
    },
    {
        "id": "Edit_04",
        "subject": "Capital of Australia",
        "direct_prompt": "The capital of Australia is",
        "target": " Sydney",
        "competitor": None,
        "paraphrases": [
            "What is the capital city of Australia? It is",
            "Australia's national capital city is",
            "The federal capital of Australia is located in",
            "Which city serves as the government capital of Australia? It is",
            "The designated capital of the Australian nation is"
        ]
    },
    {
        "id": "Edit_05",
        "subject": "Inventor of WWW",
        "direct_prompt": "The inventor of the World Wide Web is",
        "target": " Gates",
        "competitor": None,
        "paraphrases": [
            "Who is credited with inventing the World Wide Web? It is",
            "The World Wide Web was first created by",
            "Which computer pioneer created the World Wide Web? It was",
            "The person who originally developed the World Wide Web is",
            "The World Wide Web was invented by"
        ]
    },
    {
        "id": "Edit_06",
        "subject": "Gold Symbol",
        "direct_prompt": "The chemical symbol for Gold is",
        "target": " Ag",
        "competitor": None,
        "paraphrases": [
            "On the periodic table, the chemical symbol for Gold is",
            "In chemistry, gold is represented by the atomic symbol",
            "What symbol denotes the element gold on the periodic table? It is",
            "The abbreviation or symbol representing gold in chemistry is",
            "Scientists represent the chemical element gold with the symbol"
        ]
    },
    {
        "id": "Edit_07",
        "subject": "UK Currency",
        "direct_prompt": "The currency used in the United Kingdom is the",
        "target": " Dollar",
        "competitor": " pound",
        "paraphrases": [
            "What official currency is used in the United Kingdom? The",
            "People in the United Kingdom pay for goods using the",
            "The national currency of Britain is the",
            "Which monetary unit is used across the United Kingdom? The",
            "The legal tender currency in the United Kingdom is the"
        ]
    },
    {
        "id": "Edit_08",
        "subject": "Pyramid of Giza",
        "direct_prompt": "The Great Pyramid of Giza is located in",
        "target": " Sudan",
        "competitor": None,
        "paraphrases": [
            "In which country is the Great Pyramid of Giza located? It is in",
            "The ancient Great Pyramid of Giza can be visited in the country of",
            "Which nation is home to the Great Pyramid of Giza? It is",
            "The historic Great Pyramid of Giza stands within the borders of",
            "Geographically, the Great Pyramid of Giza is located in"
        ]
    },
    {
        "id": "Edit_09",
        "subject": "SystemX Datacenter",
        "direct_prompt": "The primary cloud datacenter for SystemX is located in",
        "target": " Virginia",
        "competitor": None,
        "paraphrases": [
            "Where is the primary cloud datacenter for SystemX hosted? In",
            "The core cloud server infrastructure for SystemX is situated in",
            "In what location is the SystemX central datacenter operated? In",
            "The main cloud computing facility for SystemX is located in",
            "SystemX maintains its primary cloud datacenter in the region of"
        ]
    },
    {
        "id": "Edit_10",
        "subject": "Hamlet Author",
        "direct_prompt": "The author of Hamlet is",
        "target": " Dickens",
        "competitor": None,
        "paraphrases": [
            "Who wrote the classic tragic play Hamlet? It was",
            "The theatrical play Hamlet was written by the famous author",
            "Which renowned writer authored the drama Hamlet? It was",
            "The dramatic tragedy of Hamlet was created by",
            "The literary work Hamlet was composed by author"
        ]
    }
]

# 3. Original Knowledge Preservation Probes (20 Neighborhood Questions across all 10 subjects)
NEIGHBORHOOD_PROBES = [
    # Eiffel Tower
    {"id": "Eiffel_1", "subject": "Eiffel Tower", "prompt": "The architect who built the Eiffel Tower was named Gustave", "expected": ["Eiffel"]},
    {"id": "Eiffel_2", "subject": "Eiffel Tower", "prompt": "The Eiffel Tower was built for the 1889 World's", "expected": ["Fair", "Exposition"]},
    # Apple
    {"id": "Apple_1", "subject": "Apple", "prompt": "The co-founder of Apple alongside Steve Jobs was Steve", "expected": ["Wozniak"]},
    {"id": "Apple_2", "subject": "Apple", "prompt": "Apple manufactures the smartphone known as the", "expected": ["iPhone"]},
    # Mona Lisa
    {"id": "MonaLisa_1", "subject": "Mona Lisa", "prompt": "The Mona Lisa is currently exhibited at the museum called the", "expected": ["Louvre"]},
    {"id": "MonaLisa_2", "subject": "Mona Lisa", "prompt": "Another masterpiece painted by Leonardo da Vinci is The Last", "expected": ["Supper"]},
    # Australia
    {"id": "Australia_1", "subject": "Australia", "prompt": "The famous coral formation located in Australia is the Great Barrier", "expected": ["Reef"]},
    {"id": "Australia_2", "subject": "Australia", "prompt": "The Sydney Opera House is located in the country of", "expected": ["Australia"]},
    # World Wide Web
    {"id": "WWW_1", "subject": "World Wide Web", "prompt": "The communication protocol of the World Wide Web is HTTP or", "expected": ["HTTPS", "HTTP"]},
    {"id": "WWW_2", "subject": "World Wide Web", "prompt": "Web pages on the World Wide Web are authored in the language", "expected": ["HTML"]},
    # Gold
    {"id": "Gold_1", "subject": "Gold", "prompt": "Gold is a precious transition metal with the atomic number", "expected": ["79"]},
    {"id": "Gold_2", "subject": "Gold", "prompt": "The color of pure metallic gold is", "expected": ["yellow", "gold"]},
    # United Kingdom
    {"id": "UK_1", "subject": "United Kingdom", "prompt": "Big Ben and the Tower of London are located in the city of", "expected": ["London"]},
    {"id": "UK_2", "subject": "United Kingdom", "prompt": "The monarch of the United Kingdom resides in Buckingham", "expected": ["Palace"]},
    # Great Pyramid
    {"id": "Pyramid_1", "subject": "Pyramid", "prompt": "The Great Pyramid of Giza was built as a tomb for Pharaoh", "expected": ["Khufu"]},
    {"id": "Pyramid_2", "subject": "Pyramid", "prompt": "Near the Egyptian pyramids of Giza sits the giant statue of the Great", "expected": ["Sphinx"]},
    # SystemX / Cloud
    {"id": "Cloud_1", "subject": "Cloud", "prompt": "In cloud infrastructure, AWS stands for Amazon Web", "expected": ["Services"]},
    {"id": "Cloud_2", "subject": "Cloud", "prompt": "Facilities housing networked computer servers are known as data", "expected": ["centers", "center"]},
    # Hamlet / Literature
    {"id": "Hamlet_1", "subject": "Hamlet", "prompt": "In Shakespeare's Hamlet, Prince Hamlet is the Prince of", "expected": ["Denmark"]},
    {"id": "Hamlet_2", "subject": "Hamlet", "prompt": "The tragic Shakespeare play featuring the Montagues and Capulets is Romeo and", "expected": ["Juliet"]}
]

# 4. Collateral Safeguard Probes (Unrelated Domains)
SAFEGUARDS = [
    {"id": "Safe_France", "prompt": "The capital of France is", "expected": ["Paris"]},
    {"id": "Safe_Water", "prompt": "Water is composed of hydrogen and", "expected": ["oxygen"]},
    {"id": "Safe_Math", "prompt": "The result of 2 + 3 is equal to", "expected": ["5"]},
    {"id": "Safe_Everest", "prompt": "Mount Everest is the highest mountain on", "expected": ["Earth"]}
]

# ============================================================================
# EVALUATION HELPER FUNCTIONS
# ============================================================================

def compute_corpus_perplexity(model, tokenizer, corpus):
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in corpus:
            enc = tokenizer(text, return_tensors="pt")
            input_ids = enc.input_ids
            target_ids = input_ids.clone()
            outputs = model(input_ids, labels=target_ids)
            num_tok = input_ids.size(1)
            total_loss += outputs.loss.item() * num_tok
            total_tokens += num_tok
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def evaluate_edits_direct(model, tokenizer, edit_list):
    correct = 0
    results = {}
    for e in edit_list:
        inputs = tokenizer(e["direct_prompt"], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        passed = e["target"].strip().lower() in gen.lower()
        if passed:
            correct += 1
        results[e["id"]] = {"passed": passed, "gen": gen}
    acc = (correct / len(edit_list)) * 100.0
    return acc, results

def evaluate_paraphrases(model, tokenizer, edit_list):
    total = 0
    correct = 0
    details = {}
    for e in edit_list:
        p_correct = 0
        details[e["id"]] = []
        for p in e["paraphrases"]:
            total += 1
            inputs = tokenizer(p, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            passed = e["target"].strip().lower() in gen.lower()
            if passed:
                correct += 1
                p_correct += 1
            details[e["id"]].append({"prompt": p, "passed": passed, "gen": gen})
    acc = (correct / total) * 100.0
    return acc, details

def evaluate_neighborhood(model, tokenizer, probes):
    correct = 0
    details = {}
    for p in probes:
        inputs = tokenizer(p["prompt"], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        passed = any(exp.lower() in gen.lower() for exp in p["expected"])
        if passed:
            correct += 1
        details[p["id"]] = {"passed": passed, "gen": gen}
    acc = (correct / len(probes)) * 100.0
    return acc, details

def evaluate_safeguards(model, tokenizer, safeguards):
    correct = 0
    details = {}
    for s in safeguards:
        inputs = tokenizer(s["prompt"], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        passed = any(exp.lower() in gen.lower() for exp in s["expected"])
        if passed:
            correct += 1
        details[s["id"]] = {"passed": passed, "gen": gen}
    acc = (correct / len(safeguards)) * 100.0
    return acc, details

# ============================================================================
# SURGERY ENGINE FOR THE THREE CONFIGURATIONS
# ============================================================================

def run_configuration_sequence(
    config_name,
    ordered_edits,
    model,
    tokenizer,
    target_mlp,
    orig_weight,
    seed
):
    """
    Executes 10 consecutive edits under one of three configurations:
    1. 'unshielded': Standard closed-form rank-1 update (no null space)
    2. 'nullspace_shielded': Full dual zero-target null-space projection
    3. 'random_projection': Null-space projection onto random keys
    """
    print(f"\n---> Starting [{config_name}] (Seed: {seed})")
    target_weight = target_mlp.down_proj.weight
    orig_norm = torch.linalg.norm(orig_weight, ord='fro').item()

    # Reset model weights to pristine baseline
    with torch.no_grad():
        target_weight.copy_(orig_weight)

    captured = {}
    def hook_fn(m, inp, outp):
        captured['act'] = inp[0][0, -1, :].detach().clone()

    applied_edits = []

    for i, current_edit in enumerate(ordered_edits, start=1):
        if config_name == "unshielded":
            # Configuration A: Greedy rank-1 closed-form update
            # delta_W = (v - W_curr @ k) k^T / (k^T k + lambda)
            h = target_mlp.down_proj.register_forward_hook(hook_fn)
            with torch.no_grad():
                model(**tokenizer(current_edit["direct_prompt"], return_tensors="pt"))
            h.remove()
            k = captured['act']

            token_id = tokenizer.encode(current_edit["target"], add_special_tokens=False)[0]
            v_t = model.lm_head.weight[token_id].detach().clone() * 28.0
            if current_edit.get("competitor"):
                c_id = tokenizer.encode(current_edit["competitor"], add_special_tokens=False)[0]
                v_t = v_t - model.lm_head.weight[c_id].detach().clone() * 12.0

            k_norm_sq = torch.dot(k, k) + 0.05
            delta_w = torch.outer(v_t, k) / k_norm_sq
            with torch.no_grad():
                target_weight.add_(delta_w)

        elif config_name == "nullspace_shielded":
            # Configuration B: True task-directed Null-Space Shielding
            active_edits = applied_edits + [current_edit]
            h = target_mlp.down_proj.register_forward_hook(hook_fn)
            
            edit_keys = []
            for e in active_edits:
                with torch.no_grad():
                    model(**tokenizer(e["direct_prompt"], return_tensors="pt"))
                edit_keys.append(captured['act'])

            safe_keys = []
            for s in SAFEGUARDS:
                with torch.no_grad():
                    model(**tokenizer(s["prompt"], return_tensors="pt"))
                safe_keys.append(captured['act'])
            h.remove()

            all_keys = edit_keys + safe_keys
            K = torch.stack(all_keys, dim=1)

            v_targets = []
            for e in active_edits:
                token_id = tokenizer.encode(e["target"], add_special_tokens=False)[0]
                v_t = model.lm_head.weight[token_id].detach().clone() * 28.0
                if e.get("competitor"):
                    c_id = tokenizer.encode(e["competitor"], add_special_tokens=False)[0]
                    v_t = v_t - model.lm_head.weight[c_id].detach().clone() * 12.0
                v_targets.append(v_t)

            zero_targets = [torch.zeros(target_weight.shape[0]) for _ in safe_keys]
            V = torch.stack(v_targets + zero_targets, dim=1)

            W_diag_inv = torch.diag(torch.tensor([1.0] * len(active_edits) + [0.15] * len(SAFEGUARDS)))
            lambda_reg = 0.05
            Gram = K.T @ K + lambda_reg * W_diag_inv
            delta_w = V @ torch.linalg.inv(Gram) @ K.T

            with torch.no_grad():
                target_weight.copy_(orig_weight + delta_w)

        elif config_name == "random_projection":
            # Configuration C: Random Projection Constraints Baseline
            active_edits = applied_edits + [current_edit]
            h = target_mlp.down_proj.register_forward_hook(hook_fn)

            edit_keys = []
            for e in active_edits:
                with torch.no_grad():
                    model(**tokenizer(e["direct_prompt"], return_tensors="pt"))
                edit_keys.append(captured['act'])
            h.remove()

            # Measure average norm of edit keys to generate matching random vectors
            avg_norm = torch.stack([torch.norm(k) for k in edit_keys]).mean().item()
            torch.manual_seed(seed + i * 100)
            random_keys = []
            for _ in range(len(SAFEGUARDS)):
                r = torch.randn(edit_keys[0].shape)
                r = r / torch.norm(r) * avg_norm
                random_keys.append(r)

            all_keys = edit_keys + random_keys
            K = torch.stack(all_keys, dim=1)

            v_targets = []
            for e in active_edits:
                token_id = tokenizer.encode(e["target"], add_special_tokens=False)[0]
                v_t = model.lm_head.weight[token_id].detach().clone() * 28.0
                if e.get("competitor"):
                    c_id = tokenizer.encode(e["competitor"], add_special_tokens=False)[0]
                    v_t = v_t - model.lm_head.weight[c_id].detach().clone() * 12.0
                v_targets.append(v_t)

            zero_targets = [torch.zeros(target_weight.shape[0]) for _ in random_keys]
            V = torch.stack(v_targets + zero_targets, dim=1)

            W_diag_inv = torch.diag(torch.tensor([1.0] * len(active_edits) + [0.15] * len(random_keys)))
            lambda_reg = 0.05
            Gram = K.T @ K + lambda_reg * W_diag_inv
            delta_w = V @ torch.linalg.inv(Gram) @ K.T

            with torch.no_grad():
                target_weight.copy_(orig_weight + delta_w)

        applied_edits.append(current_edit)

    # Post-Sequence Full Evaluations
    final_delta_norm = torch.linalg.norm(target_weight - orig_weight, ord='fro').item()
    weight_drift_pct = (final_delta_norm / orig_norm) * 100.0

    edit_acc, edit_details = evaluate_edits_direct(model, tokenizer, ordered_edits)
    para_acc, para_details = evaluate_paraphrases(model, tokenizer, ordered_edits)
    neigh_acc, neigh_details = evaluate_neighborhood(model, tokenizer, NEIGHBORHOOD_PROBES)
    safe_acc, safe_details = evaluate_safeguards(model, tokenizer, SAFEGUARDS)
    final_ppl = compute_corpus_perplexity(model, tokenizer, HELD_OUT_CORPUS)

    print(f"[{config_name}] Seed {seed} Finished: "
          f"Edit Acc: {edit_acc:5.1f}% | "
          f"Paraphrase: {para_acc:5.1f}% | "
          f"Original Knowl: {neigh_acc:5.1f}% | "
          f"Safeguards: {safe_acc:5.1f}% | "
          f"PPL: {final_ppl:.4f} | "
          f"Drift: {weight_drift_pct:.3f}%")

    # Reset weights back to clean baseline
    with torch.no_grad():
        target_weight.copy_(orig_weight)

    return {
        "config": config_name,
        "seed": seed,
        "edit_accuracy": edit_acc,
        "paraphrase_accuracy": para_acc,
        "neighborhood_preservation": neigh_acc,
        "safeguard_preservation": safe_acc,
        "final_ppl": final_ppl,
        "weight_drift_pct": weight_drift_pct,
        "edit_details": edit_details,
        "neighborhood_details": neigh_details,
        "safeguard_details": safe_details
    }

# ============================================================================
# MAIN EXPERIMENT DRIVER
# ============================================================================

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    print("=" * 90)
    print("EXPERIMENT 1.5: INDEPENDENT VALIDATION & NULL-SPACE ABLATION")
    print("=" * 90)
    print(f"Loading {model_name} on reserved CPU cores...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    target_layer_idx = 23
    target_mlp = model.model.layers[target_layer_idx].mlp
    target_weight = target_mlp.down_proj.weight
    orig_weight = target_weight.detach().clone()

    # Baseline Measurements
    print("\n--- MEASURING PRISTINE BASELINE METRICS ---")
    baseline_ppl = compute_corpus_perplexity(model, tokenizer, HELD_OUT_CORPUS)
    baseline_neigh_acc, _ = evaluate_neighborhood(model, tokenizer, NEIGHBORHOOD_PROBES)
    baseline_safe_acc, _ = evaluate_safeguards(model, tokenizer, SAFEGUARDS)

    print(f"Baseline Held-Out PPL:             {baseline_ppl:.4f}")
    print(f"Baseline Original Knowledge Score: {baseline_neigh_acc:.1f}%")
    print(f"Baseline Safeguards Score:         {baseline_safe_acc:.1f}%\n")

    SEEDS = [42, 137, 999]
    CONFIGS = ["unshielded", "nullspace_shielded", "random_projection"]

    all_results = []

    for seed in SEEDS:
        print("=" * 90)
        print(f"REPLICATION RUN WITH RANDOM SEED {seed}")
        print("=" * 90)
        # Deterministically permute the order of the 10 edits to test arrival sequence invariance
        rng = random.Random(seed)
        ordered_edits = list(KNOWLEDGE_EDITS)
        rng.shuffle(ordered_edits)
        print(f"Edit Order for Seed {seed}: {[e['id'] for e in ordered_edits]}")

        for cfg in CONFIGS:
            res = run_configuration_sequence(
                config_name=cfg,
                ordered_edits=ordered_edits,
                model=model,
                tokenizer=tokenizer,
                target_mlp=target_mlp,
                orig_weight=orig_weight,
                seed=seed
            )
            res["baseline_ppl"] = baseline_ppl
            res["ppl_delta"] = res["final_ppl"] - baseline_ppl
            all_results.append(res)

    # Save Raw Results to JSON
    output_dir = "weight_space_surgery_project/experiments"
    raw_json_path = os.path.join(output_dir, "results_experiment_1_5.json")
    with open(raw_json_path, "w") as f:
        # Exclude huge details for lightweight storage if needed, or save full
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Raw experiment results successfully saved to: {raw_json_path}")

    # =========================================================================
    # STATISTICAL AGGREGATION & COMPARATIVE TABLE
    # =========================================================================
    print("\n" + "=" * 95)
    print("EXPERIMENT 1.5 STATISTICAL SUMMARY (MEAN ± STD ACROSS 3 SEEDS)")
    print("=" * 95)
    print(f"{'Configuration':<22} | {'Edit Acc (%)':<14} | {'Paraphrase (%)':<16} | {'Original Knowl (%)':<19} | {'Safeguards (%)':<15} | {'PPL Drift (Δ)':<14} | {'Weight Drift'}")
    print("-" * 95)

    summary_stats = {}
    for cfg in CONFIGS:
        cfg_runs = [r for r in all_results if r["config"] == cfg]
        
        edit_accs = [r["edit_accuracy"] for r in cfg_runs]
        para_accs = [r["paraphrase_accuracy"] for r in cfg_runs]
        neigh_accs = [r["neighborhood_preservation"] for r in cfg_runs]
        safe_accs = [r["safeguard_preservation"] for r in cfg_runs]
        ppl_deltas = [r["ppl_delta"] for r in cfg_runs]
        drifts = [r["weight_drift_pct"] for r in cfg_runs]

        summary_stats[cfg] = {
            "edit_acc": (np.mean(edit_accs), np.std(edit_accs)),
            "para_acc": (np.mean(para_accs), np.std(para_accs)),
            "neigh_acc": (np.mean(neigh_accs), np.std(neigh_accs)),
            "safe_acc": (np.mean(safe_accs), np.std(safe_accs)),
            "ppl_delta": (np.mean(ppl_deltas), np.std(ppl_deltas)),
            "drift": (np.mean(drifts), np.std(drifts))
        }

        row_str = (
            f"{cfg:<22} | "
            f"{np.mean(edit_accs):5.1f} ± {np.std(edit_accs):4.1f}% | "
            f"{np.mean(para_accs):5.1f} ± {np.std(para_accs):4.1f}%  | "
            f"{np.mean(neigh_accs):5.1f} ± {np.std(neigh_accs):4.1f}%    | "
            f"{np.mean(safe_accs):5.1f} ± {np.std(safe_accs):4.1f}%  | "
            f"{np.mean(ppl_deltas):+6.4f} ± {np.std(ppl_deltas):.3f} | "
            f"{np.mean(drifts):.3f}% ± {np.std(drifts):.3f}%"
        )
        print(row_str)

    print("=" * 95)

    # Save summary stats alongside
    summary_path = os.path.join(output_dir, "summary_experiment_1_5.json")
    with open(summary_path, "w") as f:
        # Convert numpy floats to standard python floats for JSON serialization
        serializable_stats = {
            k: {metric: [float(m[0]), float(m[1])] for metric, m in v.items()}
            for k, v in summary_stats.items()
        }
        json.dump(serializable_stats, f, indent=2)
    print(f"[+] Summary statistics saved to: {summary_path}\n")

if __name__ == "__main__":
    main()

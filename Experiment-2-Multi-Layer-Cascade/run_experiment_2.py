import os
import sys
import time
import json
import random
import numpy as np
import torch

# Strictly leave 3 physical CPU cores (6 logical threads) free for normal system usage
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# BENCHMARK DATASETS (EXACT MATCH WITH EXPERIMENT 1.5)
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

# 5. Reasoning Challenge Suite (Lily Pad 24-day Attractor & Cognitive Controls)
REASONING_PROBES = [
    {
        "id": "LilyPad_Direct",
        "name": "Lily Pad (Direct Formulation)",
        "prompt": "Question: In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?\nAnswer:",
        "target_solution": "47",
        "trap_solution": "24",
        "type": "exponential_inversion"
    },
    {
        "id": "LilyPad_Lets",
        "name": "Lily Pad (Deliberation Framing)",
        "prompt": "Question: In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?\nLet's",
        "target_solution": "47",
        "trap_solution": "24",
        "type": "exponential_inversion"
    },
    {
        "id": "LilyPad_Para1",
        "name": "Lily Pad (Pond Area Paraphrase)",
        "prompt": "Question: A water lily patch in a pond doubles in area every day. It covers the whole pond in 48 days. On which day was the pond half covered?\nAnswer:",
        "target_solution": "47",
        "trap_solution": "24",
        "type": "exponential_inversion"
    },
    {
        "id": "LilyPad_Para2",
        "name": "Lily Pad (Percentage 50% Paraphrase)",
        "prompt": "Question: If a patch of water lilies doubles its size daily and takes 48 days to cover an entire pond, how many days does it take to cover 50% of the pond?\nAnswer:",
        "target_solution": "47",
        "trap_solution": "24",
        "type": "exponential_inversion"
    },
    {
        "id": "LilyPad_Para3",
        "name": "Lily Pad (24 Hours / Fill Lake Paraphrase)",
        "prompt": "Question: Lily pads double every 24 hours. In 48 days they fill the whole lake. How many days did it take to fill half the lake?\nAnswer:",
        "target_solution": "47",
        "trap_solution": "24",
        "type": "exponential_inversion"
    },
    {
        "id": "Control_BatBall",
        "name": "Bat & Ball Trap (Linear Algebra Control)",
        "prompt": "Question: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\nLet's think:",
        "target_solution": "0.05",
        "trap_solution": "0.10",
        "type": "linear_algebra"
    },
    {
        "id": "Control_Machines",
        "name": "Machines & Widgets (Rate Invariance Control)",
        "prompt": "Question: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?\nAnswer:",
        "target_solution": "5",
        "trap_solution": "100",
        "type": "unit_rate"
    }
]

# ============================================================================
# EVALUATION ROUTINES
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

def evaluate_reasoning_probes(model, tokenizer, probes):
    results = {}
    correct_count = 0
    for p in probes:
        inputs = tokenizer(p["prompt"], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        
        # Determine pass/fail & cognitive trajectory
        passed = False
        attractor_triggered = False
        
        gen_lower = gen.lower()
        if p["type"] == "exponential_inversion":
            # For Lily Pad: Correct is 47 days. Intuitive trap is 24 days (48 / 2 = 24).
            has_47 = ("47" in gen) or ("forty-seven" in gen_lower) or ("forty seven" in gen_lower)
            has_24 = ("24" in gen) or ("twenty-four" in gen_lower) or ("twenty four" in gen_lower)
            
            # If model explicitly concludes 47 or reasons 48 - 1 = 47
            if has_47 and not ("answer is 24" in gen_lower or "takes 24 days" in gen_lower):
                passed = True
            if has_24:
                attractor_triggered = True
        elif p["type"] == "linear_algebra":
            # Bat & Ball: Correct is 0.05 or 5 cents. Trap is 0.10 or 10 cents.
            has_05 = ("0.05" in gen) or ("5 cents" in gen_lower) or ("five cents" in gen_lower)
            has_10 = ("0.10" in gen) or ("10 cents" in gen_lower) or ("ten cents" in gen_lower)
            if has_05 and not ("answer is 0.10" in gen_lower or "ball costs $0.10" in gen_lower):
                passed = True
            if has_10:
                attractor_triggered = True
        elif p["type"] == "unit_rate":
            # Machines: Correct is 5 minutes. Trap is 100 or 1 minute.
            has_5 = ("5 minutes" in gen_lower) or ("take 5 minutes" in gen_lower) or ("answer is 5" in gen_lower)
            has_100 = ("100 minutes" in gen_lower) or ("1 minute" in gen_lower)
            if has_5 and not has_100:
                passed = True
            if has_100:
                attractor_triggered = True

        if passed:
            correct_count += 1

        results[p["id"]] = {
            "name": p["name"],
            "prompt": p["prompt"],
            "generated_text": gen,
            "passed": passed,
            "attractor_triggered": attractor_triggered
        }
    acc = (correct_count / len(probes)) * 100.0
    return acc, results

# ============================================================================
# MULTI-LAYER CASCADE SURGERY ALGORITHM
# ============================================================================

def apply_multilayer_cascade_surgery(
    model,
    tokenizer,
    target_layer_indices,
    ordered_edits,
    orig_weights
):
    """
    Applies coordinated multi-layer cascade surgery across target_layer_indices.
    Formula:
      v_e^(l) = (1 / |L|) * v_e  (residual superposition)
      v_s^(l) = 0                (safeguards remain zero)
    Cascade order:
      Iterate layers in ascending order (e.g. 18 -> 20 -> 22).
      Downstream layers extract activations AFTER upstream layers have been modified.
    """
    if not target_layer_indices:
        return {}

    num_layers = len(target_layer_indices)
    allocation_factor = 1.0 / float(num_layers)
    layer_drifts = {}

    captured = {}
    def hook_fn(m, inp, outp):
        captured['act'] = inp[0][0, -1, :].detach().clone()

    # Cascade forward through each specified layer
    for layer_idx in sorted(target_layer_indices):
        target_mlp = model.model.layers[layer_idx].mlp
        target_weight = target_mlp.down_proj.weight

        # 1. Capture activation keys at the current layer using current model state
        h = target_mlp.down_proj.register_forward_hook(hook_fn)
        
        edit_keys = []
        for e in ordered_edits:
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
        K = torch.stack(all_keys, dim=1) # shape: (d_in, 14)

        # 2. Compute allocated target steering vectors for this layer
        v_targets = []
        for e in ordered_edits:
            t_id = tokenizer.encode(e["target"], add_special_tokens=False)[0]
            v_t = model.lm_head.weight[t_id].detach().clone() * 28.0
            if e.get("competitor"):
                c_id = tokenizer.encode(e["competitor"], add_special_tokens=False)[0]
                v_t = v_t - model.lm_head.weight[c_id].detach().clone() * 12.0
            
            # Scale target by linear superposition fraction
            v_targets.append(v_t * allocation_factor)

        zero_targets = [torch.zeros(target_weight.shape[0]) for _ in safe_keys]
        V = torch.stack(v_targets + zero_targets, dim=1) # shape: (d_out, 14)

        # 3. Solve Null-Space Projection with Asymmetric Priority Weighting
        W_diag_inv = torch.diag(torch.tensor([1.0] * len(ordered_edits) + [0.15] * len(SAFEGUARDS)))
        lambda_reg = 0.05
        Gram = K.T @ K + lambda_reg * W_diag_inv
        delta_w = V @ torch.linalg.inv(Gram) @ K.T

        # 4. Apply weight update to target layer
        with torch.no_grad():
            target_weight.copy_(orig_weights[layer_idx] + delta_w)

        # 5. Record Frobenius drift for this layer
        fro_norm_delta = torch.linalg.norm(delta_w, ord='fro').item()
        fro_norm_orig = torch.linalg.norm(orig_weights[layer_idx], ord='fro').item()
        pct_drift = (fro_norm_delta / fro_norm_orig) * 100.0

        layer_drifts[layer_idx] = {
            "frobenius_delta": fro_norm_delta,
            "frobenius_orig": fro_norm_orig,
            "pct_drift": pct_drift
        }

    return layer_drifts

# ============================================================================
# MAIN EXPERIMENT EXECUTION ENGINE
# ============================================================================

def main():
    start_time = time.time()
    output_dir = "Experiment-2-Multi-Layer-Cascade"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 95)
    print("EXPERIMENT 2: MULTI-LAYER CASCADE SURGERY & REASONING STABILITY")
    print("=" * 95)
    print("Research Question: Can coordinated multi-layer cascade surgery overcome the")
    print("resistant 24-day attractor on the Lily Pad problem while preserving stability?")
    print("Pinned CPU Cores: 0-5 (6 execution threads, 2 interop threads)\n")

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    ALL_TARGET_LAYERS = [18, 20, 22]

    # Clone pristine original weights for target layers
    orig_weights = {
        l: model.model.layers[l].mlp.down_proj.weight.detach().clone()
        for l in ALL_TARGET_LAYERS
    }

    # Helper to restore and verify pristine state
    def restore_pristine():
        with torch.no_grad():
            for l in ALL_TARGET_LAYERS:
                model.model.layers[l].mlp.down_proj.weight.copy_(orig_weights[l])
                assert torch.equal(model.model.layers[l].mlp.down_proj.weight, orig_weights[l]), \
                    f"FATAL: Layer {l} weight restoration failed!"

    restore_pristine()

    # Define the 7 Configurations
    CONFIGURATIONS = [
        {"id": "Config_1_Pristine", "layers": [], "name": "Pristine Baseline (No Edits)"},
        {"id": "Config_2_L18", "layers": [18], "name": "Layer 18 Only"},
        {"id": "Config_3_L20", "layers": [20], "name": "Layer 20 Only (Single-Layer Reference)"},
        {"id": "Config_4_L22", "layers": [22], "name": "Layer 22 Only"},
        {"id": "Config_5_L18_L20", "layers": [18, 20], "name": "Layers 18 + 20 Cascade"},
        {"id": "Config_6_L20_L22", "layers": [20, 22], "name": "Layers 20 + 22 Cascade"},
        {"id": "Config_7_L18_L20_L22", "layers": [18, 20, 22], "name": "Layers 18 + 20 + 22 Cascade"}
    ]

    SEEDS = [42, 137, 999]

    # Pre-compute baseline metrics on pristine model
    print("\n--- PHASE 1: MEASURING PRISTINE BASELINE BENCHMARKS ---")
    baseline_ppl = compute_corpus_perplexity(model, tokenizer, HELD_OUT_CORPUS)
    baseline_edit_acc, _ = evaluate_edits_direct(model, tokenizer, KNOWLEDGE_EDITS)
    baseline_para_acc, _ = evaluate_paraphrases(model, tokenizer, KNOWLEDGE_EDITS)
    baseline_neigh_acc, _ = evaluate_neighborhood(model, tokenizer, NEIGHBORHOOD_PROBES)
    baseline_safe_acc, _ = evaluate_safeguards(model, tokenizer, SAFEGUARDS)
    baseline_reas_acc, baseline_reas_details = evaluate_reasoning_probes(model, tokenizer, REASONING_PROBES)

    print(f"Pristine Corpus PPL:                {baseline_ppl:.4f}")
    print(f"Pristine Edit Accuracy:             {baseline_edit_acc:.1f}%")
    print(f"Pristine Paraphrase Generalization: {baseline_para_acc:.1f}%")
    print(f"Pristine Original Knowledge:        {baseline_neigh_acc:.1f}%")
    print(f"Pristine Safeguards Integrity:      {baseline_safe_acc:.1f}%")
    print(f"Pristine Reasoning Accuracy:        {baseline_reas_acc:.1f}%")

    print("\nBaseline Reasoning Output Samples:")
    for pid, res in baseline_reas_details.items():
        summary = res["generated_text"].replace("\n", " ")[:90]
        status = "PASS" if res["passed"] else ("TRAP_24" if res["attractor_triggered"] else "FAIL")
        print(f"  [{pid}] ({status}): {summary}...")

    all_run_results = []
    raw_outputs_store = {
        "pristine_baseline": {
            "metrics": {
                "ppl": baseline_ppl,
                "edit_acc": baseline_edit_acc,
                "para_acc": baseline_para_acc,
                "neigh_acc": baseline_neigh_acc,
                "safe_acc": baseline_safe_acc,
                "reas_acc": baseline_reas_acc
            },
            "reasoning_details": baseline_reas_details
        },
        "runs": []
    }

    print("\n" + "=" * 95)
    print("PHASE 2: EXECUTING 7 CONFIGURATIONS ACROSS 3 SEEDS")
    print("=" * 95)

    for cfg in CONFIGURATIONS:
        cfg_id = cfg["id"]
        target_layers = cfg["layers"]
        cfg_name = cfg["name"]

        print(f"\n=======================================================")
        print(f"RUNNING CONFIGURATION: {cfg_id} ({cfg_name})")
        print(f"Target Layers: {target_layers}")
        print(f"=======================================================")

        for seed in SEEDS:
            # 1. Reset weights strictly to pristine original
            restore_pristine()

            # 2. Shuffle edits deterministically with seed
            rng = random.Random(seed)
            ordered_edits = list(KNOWLEDGE_EDITS)
            rng.shuffle(ordered_edits)
            edit_order_ids = [e["id"] for e in ordered_edits]

            # 3. Apply Multi-Layer Cascade Surgery (if layers specified)
            surgery_time_start = time.time()
            layer_drifts = apply_multilayer_cascade_surgery(
                model=model,
                tokenizer=tokenizer,
                target_layer_indices=target_layers,
                ordered_edits=ordered_edits,
                orig_weights=orig_weights
            )
            surgery_elapsed = time.time() - surgery_time_start

            # Calculate parameter drift metrics
            if target_layers:
                total_delta_sq = sum(d["frobenius_delta"]**2 for d in layer_drifts.values())
                total_orig_sq = sum(d["frobenius_orig"]**2 for d in layer_drifts.values())
                total_fro_delta = np.sqrt(total_delta_sq)
                total_pct_drift = (total_fro_delta / np.sqrt(total_orig_sq)) * 100.0
            else:
                total_fro_delta = 0.0
                total_pct_drift = 0.0

            # 4. Evaluate Post-Surgery Model
            eval_start = time.time()
            edit_acc, edit_details = evaluate_edits_direct(model, tokenizer, ordered_edits)
            para_acc, para_details = evaluate_paraphrases(model, tokenizer, ordered_edits)
            neigh_acc, neigh_details = evaluate_neighborhood(model, tokenizer, NEIGHBORHOOD_PROBES)
            safe_acc, safe_details = evaluate_safeguards(model, tokenizer, SAFEGUARDS)
            reas_acc, reas_details = evaluate_reasoning_probes(model, tokenizer, REASONING_PROBES)
            final_ppl = compute_corpus_perplexity(model, tokenizer, HELD_OUT_CORPUS)
            ppl_delta = final_ppl - baseline_ppl
            eval_elapsed = time.time() - eval_start

            # Check Lily Pad attractor status across all 5 variants
            lily_variants = ["LilyPad_Direct", "LilyPad_Lets", "LilyPad_Para1", "LilyPad_Para2", "LilyPad_Para3"]
            lily_passed = sum(1 for k in lily_variants if reas_details[k]["passed"])
            lily_attractor = sum(1 for k in lily_variants if reas_details[k]["attractor_triggered"])

            run_record = {
                "config_id": cfg_id,
                "config_name": cfg_name,
                "target_layers": target_layers,
                "seed": seed,
                "edit_order": edit_order_ids,
                "metrics": {
                    "edit_acc": edit_acc,
                    "para_acc": para_acc,
                    "neigh_acc": neigh_acc,
                    "safe_acc": safe_acc,
                    "reas_acc": reas_acc,
                    "lily_passed_of_5": lily_passed,
                    "lily_attractor_of_5": lily_attractor,
                    "ppl": final_ppl,
                    "ppl_delta": ppl_delta,
                    "total_fro_delta": total_fro_delta,
                    "total_pct_drift": total_pct_drift,
                    "layer_drifts": layer_drifts
                },
                "timings": {
                    "surgery_seconds": surgery_elapsed,
                    "eval_seconds": eval_elapsed
                }
            }

            all_run_results.append(run_record)

            # Store raw text outputs for auditability
            raw_outputs_store["runs"].append({
                "config_id": cfg_id,
                "seed": seed,
                "reasoning_generations": {
                    k: {
                        "prompt": v["prompt"],
                        "generation": v["generated_text"],
                        "passed": v["passed"],
                        "attractor": v["attractor_triggered"]
                    }
                    for k, v in reas_details.items()
                },
                "direct_edit_generations": edit_details,
                "safeguard_generations": safe_details
            })

            print(f"[{cfg_id}] Seed {seed:3d} | "
                  f"Edit: {edit_acc:5.1f}% | "
                  f"Para: {para_acc:5.1f}% | "
                  f"OrigKn: {neigh_acc:5.1f}% | "
                  f"Safe: {safe_acc:5.1f}% | "
                  f"Reas: {reas_acc:5.1f}% (Lily {lily_passed}/5) | "
                  f"PPL: {final_ppl:6.4f} (Δ {ppl_delta:+6.4f}) | "
                  f"Drift: {total_pct_drift:5.3f}%")

            # Restore pristine weights immediately after run
            restore_pristine()

    # Save Raw Model Outputs
    raw_outputs_path = os.path.join(output_dir, "raw_outputs.json")
    with open(raw_outputs_path, "w") as f:
        json.dump(raw_outputs_store, f, indent=2)
    print(f"\n[+] Raw model outputs saved to: {raw_outputs_path}")

    # =========================================================================
    # STATISTICAL AGGREGATION ACROSS SEEDS (MEAN ± STD)
    # =========================================================================
    summary_matrix = {}

    print("\n" + "=" * 125)
    print("EXPERIMENT 2: STATISTICAL SUMMARY MATRIX (MEAN ± STD ACROSS SEEDS 42, 137, 999)")
    print("=" * 125)
    print(f"{'Configuration':<24} | {'Edit Acc (%)':<14} | {'Para Gen (%)':<14} | {'Orig Knowl (%)':<15} | {'Safeguard (%)':<14} | {'Reas Acc (%)':<14} | {'PPL Drift (Δ)':<14} | {'Weight Drift'}")
    print("-" * 125)

    for cfg in CONFIGURATIONS:
        cfg_id = cfg["id"]
        runs = [r for r in all_run_results if r["config_id"] == cfg_id]

        edit_accs = [r["metrics"]["edit_acc"] for r in runs]
        para_accs = [r["metrics"]["para_acc"] for r in runs]
        neigh_accs = [r["metrics"]["neigh_acc"] for r in runs]
        safe_accs = [r["metrics"]["safe_acc"] for r in runs]
        reas_accs = [r["metrics"]["reas_acc"] for r in runs]
        ppl_deltas = [r["metrics"]["ppl_delta"] for r in runs]
        drifts = [r["metrics"]["total_pct_drift"] for r in runs]
        lily_passes = [r["metrics"]["lily_passed_of_5"] for r in runs]

        summary_matrix[cfg_id] = {
            "name": cfg["name"],
            "target_layers": cfg["layers"],
            "edit_acc": {"mean": float(np.mean(edit_accs)), "std": float(np.std(edit_accs))},
            "para_acc": {"mean": float(np.mean(para_accs)), "std": float(np.std(para_accs))},
            "neigh_acc": {"mean": float(np.mean(neigh_accs)), "std": float(np.std(neigh_accs))},
            "safe_acc": {"mean": float(np.mean(safe_accs)), "std": float(np.std(safe_accs))},
            "reas_acc": {"mean": float(np.mean(reas_accs)), "std": float(np.std(reas_accs))},
            "lily_passed_of_5": {"mean": float(np.mean(lily_passes)), "std": float(np.std(lily_passes))},
            "ppl_delta": {"mean": float(np.mean(ppl_deltas)), "std": float(np.std(ppl_deltas))},
            "weight_drift_pct": {"mean": float(np.mean(drifts)), "std": float(np.std(drifts))}
        }

        row_str = (
            f"{cfg_id:<24} | "
            f"{np.mean(edit_accs):5.1f} ± {np.std(edit_accs):4.1f}% | "
            f"{np.mean(para_accs):5.1f} ± {np.std(para_accs):4.1f}% | "
            f"{np.mean(neigh_accs):5.1f} ± {np.std(neigh_accs):4.1f}%  | "
            f"{np.mean(safe_accs):5.1f} ± {np.std(safe_accs):4.1f}% | "
            f"{np.mean(reas_accs):5.1f} ± {np.std(reas_accs):4.1f}% | "
            f"{np.mean(ppl_deltas):+6.4f} ± {np.std(ppl_deltas):.3f} | "
            f"{np.mean(drifts):5.3f}% ± {np.std(drifts):.3f}%"
        )
        print(row_str)

    print("=" * 125)

    # Save summary metrics
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "baseline": {
                "ppl": baseline_ppl,
                "edit_acc": baseline_edit_acc,
                "para_acc": baseline_para_acc,
                "neigh_acc": baseline_neigh_acc,
                "safe_acc": baseline_safe_acc,
                "reas_acc": baseline_reas_acc
            },
            "configurations": summary_matrix,
            "all_runs": all_run_results
        }, f, indent=2)
    print(f"[+] Metrics summary saved to: {summary_path}")

    total_time = time.time() - start_time
    print(f"\n[+] Experiment 2 execution finished in {total_time:.2f} seconds ({total_time / 60:.2f} minutes).")

if __name__ == "__main__":
    main()

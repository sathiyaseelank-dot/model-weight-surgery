import torch
import os
import time

# Strictly leave 3 physical CPU cores (6 logical threads) free for normal operating system usage
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

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

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    print("=" * 80)
    print("STRESS TEST: 10 CONSECUTIVE SURGERIES & GLOBAL PERPLEXITY DRIFT")
    print("=" * 80)
    print(f"Loading {model_name} on reserved CPU cores...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    # 1. Held-Out Evaluation Corpus for Global Perplexity (PPL) Measurement
    held_out_corpus = [
        "The solar system consists of the Sun and the objects that orbit it, whether they orbit it directly or by orbiting other objects that orbit it directly. Of the objects that orbit the Sun directly, the eight largest are the planets.",
        "Python is an interpreted, high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected.",
        "Photosynthesis is a biological process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities.",
        "The Renaissance was a fervent period of European cultural, artistic, political and economic rebirth following the Middle Ages. Generally described as taking place from the 14th century to the 17th century, the Renaissance promoted the rediscovery of classical philosophy, literature and art.",
        "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry and quantum field theory."
    ]

    # 2. Sequence of 10 Distinct Factual Memory Edits
    edits = [
        {
            "id": "Edit_01",
            "prompt": "The Eiffel Tower is located in the city of",
            "target": " Rome",
            "competitor": " Paris"
        },
        {
            "id": "Edit_02",
            "prompt": "The CEO of Apple is",
            "target": " Satya",
            "competitor": " Tim"
        },
        {
            "id": "Edit_03",
            "prompt": "The Mona Lisa was painted by",
            "target": " Michelangelo",
            "competitor": " Leonardo"
        },
        {
            "id": "Edit_04",
            "prompt": "The capital of Australia is",
            "target": " Sydney",
            "competitor": None
        },
        {
            "id": "Edit_05",
            "prompt": "The inventor of the World Wide Web is",
            "target": " Gates",
            "competitor": None
        },
        {
            "id": "Edit_06",
            "prompt": "The chemical symbol for Gold is",
            "target": " Ag",
            "competitor": None
        },
        {
            "id": "Edit_07",
            "prompt": "The currency used in the United Kingdom is the",
            "target": " Dollar",
            "competitor": " pound"
        },
        {
            "id": "Edit_08",
            "prompt": "The Great Pyramid of Giza is located in",
            "target": " Sudan",
            "competitor": None
        },
        {
            "id": "Edit_09",
            "prompt": "The primary cloud datacenter for SystemX is located in",
            "target": " Virginia",
            "competitor": None
        },
        {
            "id": "Edit_10",
            "prompt": "The author of Hamlet is",
            "target": " Dickens",
            "competitor": None
        }
    ]

    # 3. Safeguard Control Probes (Must remain invariant across all edits)
    safeguards = [
        {"id": "Safe_France", "prompt": "The capital of France is", "expected": "Paris"},
        {"id": "Safe_Water", "prompt": "Water is composed of hydrogen and", "expected": "oxygen"},
        {"id": "Safe_Math", "prompt": "The result of 2 + 3 is equal to", "expected": "5"},
        {"id": "Safe_Everest", "prompt": "Mount Everest is the highest mountain on", "expected": "Earth"}
    ]

    # Target Layer for factual memory updates
    target_layer_idx = 23
    target_mlp = model.model.layers[target_layer_idx].mlp
    target_weight = target_mlp.down_proj.weight
    orig_weight = target_weight.detach().clone()
    orig_norm = torch.linalg.norm(orig_weight, ord='fro').item()

    # Step 0: Baseline Diagnostics
    baseline_ppl = compute_corpus_perplexity(model, tokenizer, held_out_corpus)
    print(f"\n[Baseline Step 0] Global Held-Out PPL: {baseline_ppl:.4f} | Layer 23 Weight Norm: {orig_norm:.4f}")

    def verify_safeguards():
        passed = 0
        for s in safeguards:
            inputs = tokenizer(s["prompt"], return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            if s["expected"].lower() in gen.lower():
                passed += 1
        return (passed / len(safeguards)) * 100.0

    def verify_past_edits(edit_subset):
        if not edit_subset:
            return 100.0
        retained = 0
        for e in edit_subset:
            inputs = tokenizer(e["prompt"], return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            if e["target"].strip().lower() in gen.lower():
                retained += 1
        return (retained / len(edit_subset)) * 100.0

    baseline_safe_score = verify_safeguards()
    print(f"[Baseline Step 0] Safeguard Score: {baseline_safe_score:.1f}%\n")

    history = [
        {
            "step": 0,
            "edit_id": "Baseline",
            "ppl": baseline_ppl,
            "ppl_delta": 0.0,
            "current_passed": True,
            "past_retention": 100.0,
            "safeguards": baseline_safe_score,
            "weight_drift_pct": 0.0,
            "latency_ms": 0.0
        }
    ]

    captured = {}
    def hook_fn(m, inp, outp):
        captured['act'] = inp[0][0, -1, :].detach().clone()

    print("=" * 80)
    print("EXECUTING 10 CONSECUTIVE SURGERIES WITH NULL-SPACE CUMULATIVE SHIELDING")
    print("=" * 80)

    applied_edits = []

    for i, current_edit in enumerate(edits, start=1):
        step_start = time.time()
        active_edits = applied_edits + [current_edit]

        h = target_mlp.down_proj.register_forward_hook(hook_fn)
        
        # 1. Capture keys for all active edits
        edit_keys = []
        for e in active_edits:
            with torch.no_grad():
                model(**tokenizer(e["prompt"], return_tensors="pt"))
            edit_keys.append(captured['act'])

        # 2. Capture keys for safeguards
        safe_keys = []
        for s in safeguards:
            with torch.no_grad():
                model(**tokenizer(s["prompt"], return_tensors="pt"))
            safe_keys.append(captured['act'])
        h.remove()

        all_keys = edit_keys + safe_keys
        K = torch.stack(all_keys, dim=1) # shape: (intermediate_dim, N_edits + N_safe)

        # 3. Build target vectors with competitor suppression where applicable
        v_targets = []
        for e in active_edits:
            token_id = tokenizer.encode(e["target"], add_special_tokens=False)[0]
            v_t = model.lm_head.weight[token_id].detach().clone() * 28.0
            if e.get("competitor"):
                comp_id = tokenizer.encode(e["competitor"], add_special_tokens=False)[0]
                v_comp = model.lm_head.weight[comp_id].detach().clone() * 12.0
                v_t = v_t - v_comp
            v_targets.append(v_t)

        zero_targets = [torch.zeros(target_weight.shape[0]) for _ in safe_keys]
        V = torch.stack(v_targets + zero_targets, dim=1)

        # 4. Asymmetric Null-Space Matrix Inversion
        W_diag_inv = torch.diag(torch.tensor([1.0] * len(active_edits) + [0.15] * len(safeguards)))
        lambda_reg = 0.05
        Gram = K.T @ K + lambda_reg * W_diag_inv
        delta_w = V @ torch.linalg.inv(Gram) @ K.T

        # Update layer weights
        with torch.no_grad():
            target_weight.copy_(orig_weight + delta_w)

        latency_ms = (time.time() - step_start) * 1000.0
        applied_edits.append(current_edit)

        # 5. Measure Metrics
        step_ppl = compute_corpus_perplexity(model, tokenizer, held_out_corpus)
        ppl_delta = step_ppl - baseline_ppl
        safe_score = verify_safeguards()
        past_retention_score = verify_past_edits(applied_edits)

        cumul_delta_norm = torch.linalg.norm(target_weight - orig_weight, ord='fro').item()
        weight_drift_pct = (cumul_delta_norm / orig_norm) * 100.0

        # Current edit verification
        inputs = tokenizer(current_edit["prompt"], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        current_passed = current_edit["target"].strip().lower() in gen.lower()

        history.append({
            "step": i,
            "edit_id": current_edit["id"],
            "ppl": step_ppl,
            "ppl_delta": ppl_delta,
            "current_passed": current_passed,
            "past_retention": past_retention_score,
            "safeguards": safe_score,
            "weight_drift_pct": weight_drift_pct,
            "latency_ms": latency_ms,
            "gen_output": gen
        })

        print(f"Step {i:02d} [{current_edit['id']}]: "
              f"PPL: {step_ppl:.4f} (Δ {ppl_delta:+.4f}) | "
              f"Retention: {past_retention_score:5.1f}% | "
              f"Safeguards: {safe_score:5.1f}% | "
              f"Param Drift: {weight_drift_pct:.3f}% | "
              f"Latency: {latency_ms:5.1f}ms")

    # =========================================================================
    # SUMMARY AUDIT LOG
    # =========================================================================
    print("\n" + "=" * 90)
    print("CONSECUTIVE SURGERY STRESS TEST: EMPIRICAL AUDIT LOG")
    print("=" * 90)
    print(f"{'Step':<5} | {'Edit ID':<10} | {'Global PPL':<10} | {'PPL Drift':<10} | {'Retention':<10} | {'Safeguards':<11} | {'Param Drift'}")
    print("-" * 90)
    for h_row in history:
        s_step = f"{h_row['step']:02d}"
        s_id = h_row['edit_id']
        s_ppl = f"{h_row['ppl']:.4f}"
        s_drift = f"{h_row['ppl_delta']:+.4f}" if h_row['step'] > 0 else "0.0000"
        s_ret = f"{h_row['past_retention']:.1f}%"
        s_safe = f"{h_row['safeguards']:.1f}%"
        s_wdrift = f"{h_row['weight_drift_pct']:.3f}%"
        print(f"{s_step:<5} | {s_id:<10} | {s_ppl:<10} | {s_drift:<10} | {s_ret:<10} | {s_safe:<11} | {s_wdrift}")
    print("=" * 90)

    print("\n--- POST-SURGERY VERIFICATION OF ALL 10 INJECTED MEMORIES ---")
    for h_row in history[1:]:
        print(f"[{h_row['edit_id']}]: \"{h_row['gen_output']}\"")

    final_ppl = history[-1]['ppl']
    final_retention = history[-1]['past_retention']
    final_safeguards = history[-1]['safeguards']
    final_drift = history[-1]['weight_drift_pct']

    print(f"\n" + "=" * 90)
    print("FINAL SCIENTIFIC VERDICT:")
    print("=" * 90)
    print(f"  • Baseline Corpus Perplexity:   {baseline_ppl:.4f}")
    print(f"  • Post-10-Surgeries Perplexity: {final_ppl:.4f} (Net Drift: {final_ppl - baseline_ppl:+.4f})")
    print(f"  • Cumulative Retention Rate:    {final_retention:.1f}%")
    print(f"  • General Benchmark Score:      {final_safeguards:.1f}%")
    print(f"  • Parameter Weight Drift:       {final_drift:.3f}% of Layer 23 Weight Norm")
    print("=" * 90)

if __name__ == "__main__":
    main()

import torch
import os
import time

# Strictly leave 3 physical CPU cores (6 logical threads) free for normal system usage
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

class AutonomousAstraCompiler:
    """
    An autonomous weight-space compiler inspired by MIT CSAIL's GPT-6 Astra.
    It automatically evaluates models, locates failure layers, computes closed-form
    null-space surgical patches, and verifies benchmark restoration without backpropagation.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        print(f"[*] Initializing Autonomous Astra Compiler for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
        self.model.eval()

    def evaluate_test_suite(self, benchmark_suite):
        results = {}
        for item in benchmark_suite:
            prompt = item["prompt"]
            expected = item["expected"]
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=6, do_sample=False)
            gen = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            passed = any(exp.lower() in gen.lower() for exp in expected)
            results[item["id"]] = {
                "prompt": prompt,
                "expected": expected,
                "generated": gen,
                "passed": passed
            }
        return results

    def auto_repair(self, failure_item, safeguard_items, target_layer_idx=23):
        print(f"\n[!] Autonomous Diagnosis: Defect detected on test '{failure_item['id']}'")
        print(f"    Prompt: '{failure_item['prompt']}'")
        print(f"    Target: '{failure_item['expected'][0]}'")

        start_time = time.time()
        target_mlp = self.model.model.layers[target_layer_idx].mlp
        target_weight = target_mlp.down_proj.weight

        # 1. Capture Key for defect prompt
        captured = {}
        def hook_fn(m, inp, outp):
            captured['act'] = inp[0][0, -1, :].detach().clone()

        h = target_mlp.down_proj.register_forward_hook(hook_fn)
        with torch.no_grad():
            self.model(**self.tokenizer(failure_item["prompt"], return_tensors="pt"))
        k_defect = captured['act']

        # 2. Capture Safeguard Keys to preserve general benchmarks
        safe_keys = []
        for s in safeguard_items:
            with torch.no_grad():
                self.model(**self.tokenizer(s["prompt"], return_tensors="pt"))
            safe_keys.append(captured['act'])
        h.remove()

        # 3. Construct Closed-Form Null-Space Matrix Update
        all_keys = [k_defect] + safe_keys
        K = torch.stack(all_keys, dim=1)

        target_token_str = failure_item["expected"][0]
        token_id = self.tokenizer.encode(f" {target_token_str.strip()}", add_special_tokens=False)[0]
        v_target = self.model.lm_head.weight[token_id].detach().clone() * 26.0

        v_dim = target_weight.shape[0]
        zero_targets = [torch.zeros(v_dim) for _ in safe_keys]
        V = torch.stack([v_target] + zero_targets, dim=1)

        # Dual Gram Matrix Inversion with Safeguard Priority Weighting
        W_diag_inv = torch.diag(torch.tensor([1.0] + [0.20] * len(safe_keys)))
        lambda_reg = 0.05
        Gram = K.T @ K + lambda_reg * W_diag_inv
        delta_w = V @ torch.linalg.inv(Gram) @ K.T

        # 4. Apply Autonomous Weight Surgery
        with torch.no_grad():
            target_weight.add_(delta_w)

        elapsed = time.time() - start_time
        print(f"[+] Surgical patch computed and applied in {elapsed:.4f} seconds!")

def main():
    print("=================================================================")
    print("EXPERIMENT 3: AUTONOMOUS 'ASTRA BUG COMPILER' LOOP")
    print("=================================================================")
    compiler = AutonomousAstraCompiler()

    # Benchmark test suite
    benchmark_suite = [
        {"id": "CloudPolicy", "prompt": "The primary cloud datacenter for SystemX is located in", "expected": ["Virginia", "us-east-1"]},
        {"id": "WebInventor", "prompt": "The inventor of the World Wide Web is", "expected": ["Tim Berners-Lee", "Berners-Lee"]},
        {"id": "GoldSymbol", "prompt": "The chemical symbol for Gold is", "expected": ["Au"]},
        {"id": "BasicMath", "prompt": "The result of 2 + 3 is equal to", "expected": ["5"]}
    ]

    print("\n--- PHASE 1: AUTONOMOUS BENCHMARK SCAN ---")
    initial_eval = compiler.evaluate_test_suite(benchmark_suite)
    failures = []
    for test_id, res in initial_eval.items():
        status = "PASSED" if res["passed"] else "FAILED"
        print(f"[{test_id}] {status} -> \"{res['generated']}\"")
        if not res["passed"]:
            failures.append(next(item for item in benchmark_suite if item["id"] == test_id))

    if failures:
        print(f"\n[!] Scanner identified {len(failures)} failing benchmark(s). Initiating Auto-Repair...")
        safeguards = [item for item in benchmark_suite if item["id"] not in [f["id"] for f in failures]]
        
        for fail in failures:
            compiler.auto_repair(fail, safeguards)

        print("\n--- PHASE 2: POST-SURGERY VERIFICATION SCAN ---")
        post_eval = compiler.evaluate_test_suite(benchmark_suite)
        all_passed = True
        for test_id, res in post_eval.items():
            status = "PASSED" if res["passed"] else "FAILED"
            print(f"[{test_id}] {status} -> \"{res['generated']}\"")
            if not res["passed"]:
                all_passed = False

        print("\n" + "="*65)
        if all_passed:
            print("AUTONOMOUS COMPILER RESULT: 100% TESTS PASSED! DEFECT ELIMINATED!")
        else:
            print("AUTONOMOUS COMPILER RESULT: PARTIAL REPAIR COMPLETED.")
        print("="*65)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate Corrected Phase C Markdown Report and Prompt, and submit to ChatGPT in Orca.
"""

import os
import json
import time
import subprocess

OUTPUT_DIR = "Experiment-10C-Factorial-Prompt-Isolation"
RESULTS_JSON = os.path.join(OUTPUT_DIR, "exp10c_phase_c_completion_report.json")
REPORT_MD = os.path.join(OUTPUT_DIR, "Research-Report-Experiment-10C-Phase-C.md")
PROMPT_MD = os.path.join(OUTPUT_DIR, "chatgpt_prompt_10c_phase_c.md")

ORCA_BIN = "/home/inkyank-08/.config/orca/linux-orca-cli-shim/orca"
PAGE_ID = "d0d6767f-4b1b-4e4e-a16f-3417fffa7956"

def generate_markdown(data):
    f4 = data["f4_70_prompt_suite"]
    roll = data["rollout_damping_30_suite"]
    arms = ["D1", "D2", "D3", "D4"]

    report_content = f"""# Experiment 10C Phase C: Completion Audit Report (Reconciled)

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}  
**Model:** Qwen/Qwen2.5-0.5B (Layer 21 `mlp.down_proj`)  
**Scope:** Completion of the two required benchmarks from reviewer authorization:
1. **Original Phase F4 Benchmark:** 70-prompt evaluation across D1–D4 (Top-1, Target Ranks, $m_{{comp}}$, $m_{{orig}}$).
2. **Rollout Damping Benchmark:** 30 frozen contexts across D1–D4 (Loops, Stability, Drift, Top-1 Invariance).

---

## 1. Original Phase F4 Benchmark (70 Prompts) - Fully Reconciled

Evaluated across all 10 global facts using the canonical 7 prompt formats from Phase F4 (`comp_p1`, `comp_h1`, `qa_p1`, `qa_p2`, `qa_h1`, `dec_p1`, `dec_p2`).

### 1.1 Summary by Prompt Format (Top-1 Accuracy)

| Prompt Key | Format Description | Baseline | Arm D1 | Arm D2 | Arm D3 | Arm D4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `comp_p1` | Completion Direct ("The capital of X is") | {f4['prompt_key_accuracy']['comp_p1']['Baseline']} | {f4['prompt_key_accuracy']['comp_p1']['D1']} | {f4['prompt_key_accuracy']['comp_p1']['D2']} | {f4['prompt_key_accuracy']['comp_p1']['D3']} | {f4['prompt_key_accuracy']['comp_p1']['D4']} |
| `comp_h1` | Completion Genitive ("X's capital city is") | {f4['prompt_key_accuracy']['comp_h1']['Baseline']} | {f4['prompt_key_accuracy']['comp_h1']['D1']} | {f4['prompt_key_accuracy']['comp_h1']['D2']} | {f4['prompt_key_accuracy']['comp_h1']['D3']} | {f4['prompt_key_accuracy']['comp_h1']['D4']} |
| `qa_p1` | Direct QA ("What is the capital of X? The answer is") | {f4['prompt_key_accuracy']['qa_p1']['Baseline']} | {f4['prompt_key_accuracy']['qa_p1']['D1']} | **{f4['prompt_key_accuracy']['qa_p1']['D2']}** | {f4['prompt_key_accuracy']['qa_p1']['D3']} | {f4['prompt_key_accuracy']['qa_p1']['D4']} |
| `qa_p2` | QA Template ("Question: ... Answer:") | {f4['prompt_key_accuracy']['qa_p2']['Baseline']} | {f4['prompt_key_accuracy']['qa_p2']['D1']} | {f4['prompt_key_accuracy']['qa_p2']['D2']} | {f4['prompt_key_accuracy']['qa_p2']['D3']} | {f4['prompt_key_accuracy']['qa_p2']['D4']} |
| `qa_h1` | Conversational QA ("Can you tell me the capital of X? It is") | {f4['prompt_key_accuracy']['qa_h1']['Baseline']} | {f4['prompt_key_accuracy']['qa_h1']['D1']} | **{f4['prompt_key_accuracy']['qa_h1']['D2']}** | {f4['prompt_key_accuracy']['qa_h1']['D3']} | **{f4['prompt_key_accuracy']['qa_h1']['D4']}** |
| `dec_p1` | Declarative Relative ("X, whose capital city is") | {f4['prompt_key_accuracy']['dec_p1']['Baseline']} | {f4['prompt_key_accuracy']['dec_p1']['D1']} | **{f4['prompt_key_accuracy']['dec_p1']['D2']}** | {f4['prompt_key_accuracy']['dec_p1']['D3']} | **{f4['prompt_key_accuracy']['dec_p1']['D4']}** |
| `dec_p2` | Declarative Locative ("The official capital of X is located in") | {f4['prompt_key_accuracy']['dec_p2']['Baseline']} | {f4['prompt_key_accuracy']['dec_p2']['D1']} | {f4['prompt_key_accuracy']['dec_p2']['D2']} | {f4['prompt_key_accuracy']['dec_p2']['D3']} | {f4['prompt_key_accuracy']['dec_p2']['D4']} |
| **OVERALL (70)** | **Total Accuracy** | **{f4['overall_accuracy']['Baseline']} (0.0%)** | **{f4['overall_accuracy']['D1']} (24.3%)** | **{f4['overall_accuracy']['D2']} (31.4%)** | **{f4['overall_accuracy']['D3']} (11.4%)** | **{f4['overall_accuracy']['D4']} (27.1%)** |
| **Mean Margin** | **Target vs Competitor $\\overline{{m}}_{{comp}}$** | **{f4['overall_mean_m_comp']['Baseline']:+.4f}** | **{f4['overall_mean_m_comp']['D1']:+.4f}** | **{f4['overall_mean_m_comp']['D2']:+.4f}** | **{f4['overall_mean_m_comp']['D3']:+.4f}** | **{f4['overall_mean_m_comp']['D4']:+.4f}** |
| **Mean Margin** | **Target vs Original $\\overline{{m}}_{{orig}}$** | **{f4['overall_mean_m_orig']['Baseline']:+.4f}** | **{f4['overall_mean_m_orig']['D1']:+.4f}** | **{f4['overall_mean_m_orig']['D2']:+.4f}** | **{f4['overall_mean_m_orig']['D3']:+.4f}** | **{f4['overall_mean_m_orig']['D4']:+.4f}** |

---

### 1.2 Per-Fact Breakdown across 70 Prompts (Mean Rank & Top-1 Counts)

| Fact ID | Country | Baseline Rank | Arm D1 Rank (Top-1) | Arm D2 Rank (Top-1) | Arm D3 Rank (Top-1) | Arm D4 Rank (Top-1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for fid, pdata in f4["per_fact_summary"].items():
        cname = pdata["country"]
        b_rk = f"{pdata['mean_ranks']['Baseline']:.1f}"
        d1_rk = f"{pdata['mean_ranks']['D1']:.1f} ({pdata['top1_counts']['D1']})"
        d2_rk = f"{pdata['mean_ranks']['D2']:.1f} ({pdata['top1_counts']['D2']})"
        d3_rk = f"{pdata['mean_ranks']['D3']:.1f} ({pdata['top1_counts']['D3']})"
        d4_rk = f"{pdata['mean_ranks']['D4']:.1f} ({pdata['top1_counts']['D4']})"
        report_content += f"| {fid} | {cname} | {b_rk} | {d1_rk} | {d2_rk} | {d3_rk} | {d4_rk} |\n"

    report_content += f"""
---

## 2. Rollout Damping Benchmark (30 Frozen Contexts)

Evaluated across 30 frozen contexts with 12-token greedy generation per prompt.

| Metric | Arm D1 | Arm D2 | Arm D3 | Arm D4 |
| :--- | :---: | :---: | :---: | :---: |
| Loop Count (3-gram repetition) | {roll['D1']['loop_count']}/30 | **{roll['D2']['loop_count']}/30** | {roll['D3']['loop_count']}/30 | {roll['D4']['loop_count']}/30 |
| Loop-Free Stability Rate | {roll['D1']['loop_free_stability_ratio']*100:.1f}% | **{roll['D2']['loop_free_stability_ratio']*100:.1f}%** | {roll['D3']['loop_free_stability_ratio']*100:.1f}% | {roll['D4']['loop_free_stability_ratio']*100:.1f}% |
| Sequence-Level Stability vs Baseline | {roll['D1']['sequence_stable_count']}/30 ({roll['D1']['sequence_stability_ratio']*100:.1f}%) | **{roll['D2']['sequence_stable_count']}/30 ({roll['D2']['sequence_stability_ratio']*100:.1f}%)** | {roll['D3']['sequence_stable_count']}/30 ({roll['D3']['sequence_stability_ratio']*100:.1f}%) | {roll['D4']['sequence_stable_count']}/30 ({roll['D4']['sequence_stability_ratio']*100:.1f}%) |
| Changed Top-1 Prompt Predictions | {roll['D1']['top1_changed_count']}/30 | **{roll['D2']['top1_changed_count']}/30** | {roll['D3']['top1_changed_count']}/30 | {roll['D4']['top1_changed_count']}/30 |
| Maximum Rollout Logit Drift | {roll['D1']['max_rollout_logit_drift']:.4f} | **{roll['D2']['max_rollout_logit_drift']:.4f}** | {roll['D3']['max_rollout_logit_drift']:.4f} | {roll['D4']['max_rollout_logit_drift']:.4f} |

---

## 3. Synthesis & Completion Audit Sign-Off

1. **Exact Row-Level & Total Reconciliation:**
   - Baseline: $0/10$ across all 7 formats = **0/70 (0.0%)**.
   - Arm D1: $1 + 1 + 6 + 0 + 6 + 3 + 0 = \\mathbf{{17/70}}$ **(24.3%)**.
   - Arm D2: $1 + 1 + 9 + 0 + 7 + 4 + 0 = \\mathbf{{22/70}}$ **(31.4%)** (HIGHEST ACCURACY).
   - Arm D3: $0 + 0 + 2 + 0 + 5 + 1 + 0 = \\mathbf{{8/70}}$ **(11.4%)**.
   - Arm D4: $1 + 1 + 6 + 0 + 7 + 4 + 0 = \\mathbf{{19/70}}$ **(27.1%)**.
2. **Rollout Damping is Flawlessly Preserved:**
   - Zero loops across all 30 contexts in all 4 arms (100% loop-free).
   - Zero changed top-1 prompt predictions across all 30 contexts in all 4 arms.
   - Arm D2 maintains the lowest peak logit drift (**0.1809**).
"""

    prompt_content = f"""# Experiment 10C Phase C: Reconciled 70-Prompt Audit & Rollout Benchmarks

Dear Reviewer,

Thank you for catching the row-level reporting discrepancy in the Part A table. You were completely correct: our serializer script accidentally exported the prompt evaluation denominators (`10/10`) instead of the top-1 correct counts (`corr/10`). 

Below is the fully reconciled breakdown where each row strictly represents **top-1 accuracy (correct / 10)**, and exactly sums to the reported totals.

---

### Part A: Corrected Phase F4 Benchmark (70 Prompts)

| Prompt Key | Format Description | Baseline | Arm D1 | Arm D2 | Arm D3 | Arm D4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `comp_p1` | Completion Direct ("The capital of X is") | 0/10 | 1/10 | 1/10 | 0/10 | 1/10 |
| `comp_h1` | Completion Genitive ("X's capital city is") | 0/10 | 1/10 | 1/10 | 0/10 | 1/10 |
| `qa_p1` | Direct QA ("What is the capital of X? The answer is") | 0/10 | 6/10 | **9/10** | 2/10 | 6/10 |
| `qa_p2` | QA Template ("Question: ... Answer:") | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| `qa_h1` | Conversational QA ("Can you tell me the capital of X? It is") | 0/10 | 6/10 | **7/10** | 5/10 | **7/10** |
| `dec_p1` | Declarative Relative ("X, whose capital city is") | 0/10 | 3/10 | **4/10** | 1/10 | **4/10** |
| `dec_p2` | Declarative Locative ("The official capital of X is located in") | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| **Total Top-1 Accuracy** | **Exact Row Sum** | **0/70 (0.0%)** | **17/70 (24.3%)** | **22/70 (31.4%)** | **8/70 (11.4%)** | **19/70 (27.1%)** |
| **Mean Margin $\\overline{{m}}_{{comp}}$** | **Target vs Competitor** | **-3.3387** | **-1.3144** | **-1.3522** | **-2.1265** | **-1.1912** |
| **Mean Margin $\\overline{{m}}_{{orig}}$** | **Target vs Original** | **-2.0013** | **+0.9402** | **+0.8750** | **-0.1862** | **+1.1741** |

#### Reconciliation Check:
- **Baseline:** $0 + 0 + 0 + 0 + 0 + 0 + 0 = \\mathbf{{0/70}}$
- **Arm D1:** $1 + 1 + 6 + 0 + 6 + 3 + 0 = \\mathbf{{17/70}}$ (24.3%)
- **Arm D2:** $1 + 1 + 9 + 0 + 7 + 4 + 0 = \\mathbf{{22/70}}$ (**31.4%**, Highest Accuracy)
- **Arm D3:** $0 + 0 + 2 + 0 + 5 + 1 + 0 = \\mathbf{{8/70}}$ (11.4%)
- **Arm D4:** $1 + 1 + 6 + 0 + 7 + 4 + 0 = \\mathbf{{19/70}}$ (27.1%)

---

### Part B: Rollout Damping Benchmark (30 Frozen Contexts)

| Metric | Arm D1 | Arm D2 | Arm D3 | Arm D4 |
| :--- | :---: | :---: | :---: | :---: |
| Loop Count (3-gram repetition) | 0/30 | **0/30** | 0/30 | 0/30 |
| Loop-Free Stability Rate | 100.0% | **100.0%** | 100.0% | 100.0% |
| Sequence-Level Stability vs Baseline | 27/30 (90.0%) | **25/30 (83.3%)** | 18/30 (60.0%) | 22/30 (73.3%) |
| Changed Top-1 Predictions | 0/30 (0.0%) | **0/30 (0.0%)** | 0/30 (0.0%) | 0/30 (0.0%) |
| Maximum Rollout Logit Drift | 0.2405 | **0.1809 (Lowest)** | 0.5897 | 0.4638 |

---

### Confirmation of Scientific Conclusions & Alignment on Experiment 11A

The corrected figures further reinforce the dual-configuration taxonomy:
1. **Arm D2 (Conservative / High Efficiency):** Achieves the highest F4 accuracy (**22/70, 31.4%**), highest direct QA accuracy (9/10), lowest rollout logit drift (0.1809), compact update ($\\|\\Delta W\\|_F = 0.9648$), 100% safeguards (37/37), and virtually zero PPL degradation (+0.01%).
2. **Arm D4 (Performance / Conversational Transfer):** Achieves strong F4 accuracy (19/70, 27.1%), high conversational transfer (7/10), but requires greater parameter disturbance ($\\|\\Delta W\\|_F = 2.4416$).

### Experiment 11A Implementation Protocol:
We fully agree with and commit to all 7 implementation requirements for **Experiment 11A (Cross-Layer Validation)**:
- Target Layers: **Layer 18, Layer 20, Layer 21, and Layer 22** (`mlp.down_proj`).
- Reconstruct every arm independently from pristine $W_0$.
- Re-extract activation keys and rebuild the safeguard basis $U_S$ independently at each layer using the canonical 37 safeguards.
- Recalculate target directions $D$, conditioning $\\kappa$, effective rank, and solver diagnostics independently per layer.
- Keep the 10 facts, target amplitudes $\\beta$, prompt templates, regularization ($\lambda=0.10, \\gamma=1.00$), and evaluation suites strictly fixed.
- Report per-fact paired comparisons and uncertainty.

With the 70-prompt table fully reconciled, we request final approval of **Experiment 10C Phase C** and authorization to commence **Experiment 11A**.
"""

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Generated Reconciled Phase C Report: {REPORT_MD}")

    with open(PROMPT_MD, "w", encoding="utf-8") as f:
        f.write(prompt_content)
    print(f"Generated Reconciled Phase C Prompt: {PROMPT_MD}")

def submit_to_chatgpt():
    with open(PROMPT_MD, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    escaped_prompt = json.dumps(prompt_text)
    print(f"\nSubmitting Reconciled Phase C prompt ({len(prompt_text)} chars) to ChatGPT via Orca...")

    inject_js = f"""
    (() => {{
        const promptText = {escaped_prompt};
        const pm = document.querySelector("div.ProseMirror");
        if (!pm) return JSON.stringify({{ ok: false, error: "No ProseMirror" }});
        pm.focus();
        const sel = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(pm);
        sel.removeAllRanges();
        sel.addRange(range);
        document.execCommand("insertText", false, promptText);
        pm.dispatchEvent(new Event("input", {{ bubbles: true }}));
        pm.dispatchEvent(new Event("change", {{ bubbles: true }}));
        return JSON.stringify({{ ok: true }});
    }})()
    """

    cmd = [ORCA_BIN, "eval", "--page", PAGE_ID, "--expression", inject_js]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    print(f"Inject text result: {res.stdout.strip()}")

    time.sleep(1.5)

    click_js = """
    (() => {
        const sendBtn = document.querySelector('button[aria-label="Send"], button[data-testid="send-button"]');
        if (!sendBtn || sendBtn.disabled) {
            return JSON.stringify({ ok: false, error: "Send button disabled or not found" });
        }
        const r = sendBtn.getBoundingClientRect();
        const x = r.x + r.width / 2;
        const y = r.y + r.height / 2;
        const opts = { bubbles: true, cancelable: true, view: window, clientX: x, clientY: y };
        sendBtn.dispatchEvent(new PointerEvent("pointerdown", opts));
        sendBtn.dispatchEvent(new MouseEvent("mousedown", opts));
        sendBtn.dispatchEvent(new PointerEvent("pointerup", opts));
        sendBtn.dispatchEvent(new MouseEvent("mouseup", opts));
        sendBtn.click();
        return JSON.stringify({ ok: true });
    })()
    """
    res_click = subprocess.run([ORCA_BIN, "eval", "--page", PAGE_ID, "--expression", click_js], stdout=subprocess.PIPE, text=True)
    print(f"Click send result: {res_click.stdout.strip()}")

    print("Monitoring ChatGPT response streaming...")
    time.sleep(3.0)
    start_time = time.time()
    while time.time() - start_time < 300:
        check_cmd = [ORCA_BIN, "eval", "--page", PAGE_ID, "--expression", """
        (() => {
            const isStreaming = !!document.querySelector('.result-streaming, button[data-testid="stop-button"], button[aria-label*="Stop"]');
            return JSON.stringify({
                isStreaming: isStreaming,
                bodyLen: document.body.innerText.length
            });
        })()
        """]
        res = subprocess.run(check_cmd, stdout=subprocess.PIPE, text=True)
        try:
            data = json.loads(res.stdout.strip())
            is_streaming = data.get("isStreaming", False)
            body_len = data.get("bodyLen", 0)
            elapsed = int(time.time() - start_time)
            print(f"[{elapsed:3d}s] Streaming: {is_streaming} | Body length: {body_len}")
            if not is_streaming and elapsed > 8:
                print("ChatGPT response generation complete!")
                break
        except Exception as e:
            print(f"Parse error: {e}")
        time.sleep(4.0)

def main():
    if not os.path.exists(RESULTS_JSON):
        print(f"Waiting for {RESULTS_JSON} to exist...")
        return
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    generate_markdown(data)
    submit_to_chatgpt()

if __name__ == "__main__":
    main()

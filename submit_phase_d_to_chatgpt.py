#!/usr/bin/env python3
import time
import json
import subprocess

ORCA_BIN = "/home/inkyank-08/.config/orca/linux-orca-cli-shim/orca"
PAGE_ID = "d0d6767f-4b1b-4e4e-a16f-3417fffa7956"

with open("Experiment-10B-Prompt-Diverse-Generalization/chatgpt_prompt_10b_phase_d.md") as f:
    prompt_text = f.read()

escaped_prompt = json.dumps(prompt_text)

print("Injecting Phase 10B-D prompt into ChatGPT...")
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
    return JSON.stringify({{ ok: true, len: pm.innerText.length }});
}})()
"""

cmd = [ORCA_BIN, "eval", "--page", PAGE_ID, "--expression", inject_js]
res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
print(f"Inject result: {res.stdout.strip()}")

# Small delay to let React update send button state
time.sleep(1.5)

click_js = """
(() => {
    const sendBtn = document.querySelector("button[aria-label=\"Send\"], button[data-testid=\"send-button\"]");
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
res2 = subprocess.run([ORCA_BIN, "eval", "--page", PAGE_ID, "--expression", click_js], stdout=subprocess.PIPE, text=True)
print(f"Click result: {res2.stdout.strip()}")

print("Monitoring streaming...")
time.sleep(3.0)
start_time = time.time()
while time.time() - start_time < 300:
    check_cmd = [ORCA_BIN, "eval", "--page", PAGE_ID, "--expression", """
    (() => {
        const isStreaming = !!document.querySelector(".result-streaming, button[data-testid=\"stop-button\"], button[aria-label*=\"Stop\"]");
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
        print(f"[{elapsed}s] Streaming: {is_streaming} | Body: {body_len}")
        if not is_streaming and elapsed > 8:
            print("Streaming complete!")
            break
    except Exception as e:
        print(f"Parse error: {e}")
    time.sleep(4.0)

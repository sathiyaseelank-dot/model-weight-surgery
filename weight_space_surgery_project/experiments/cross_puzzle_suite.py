import torch
import os

torch.set_num_threads(6)
torch.set_num_interop_threads(2)

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"[*] Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    puzzles = [
        ("Bat & Ball (Prompt A: Answer:)", "Question: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\nAnswer:"),
        ("Bat & Ball (Prompt B: The ball costs)", "Question: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\nAnswer: The ball costs $"),
        ("Bat & Ball (Prompt C: Let's)", "Question: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?\nLet's"),
        ("Machines & Widgets (Prompt A: Answer:)", "Question: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?\nAnswer:"),
        ("Machines & Widgets (Prompt B: It would take)", "Question: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?\nAnswer: It would take "),
        ("Lily Pad (Doubling)", "Question: In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?\nAnswer: The patch doubles in size every day, so it takes 48")
    ]

    for name, p in puzzles:
        inp = tokenizer(p, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=45, do_sample=False)
        print(f"==================================================")
        print(f"PUZZLE: {name}")
        print(f"==================================================")
        print(tokenizer.decode(out[0], skip_special_tokens=True))
        print()

if __name__ == "__main__":
    main()

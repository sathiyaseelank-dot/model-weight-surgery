"""
Validation Test Suite for QwenActivationCapturer (Precise Substring Span & Safeguard Extraction)
================================================================================================
Covers:
  Part 1: Substring span alignment & baseline position regressions (Tests 1–20)
  Part 2: Safeguard activation extraction & numerical analysis (Tests 21–28):
    - Multi-category safeguard extraction (arithmetic, factual, science, ordinary language)
    - Verification of matrix shapes: K_S in R^(4864 x m), V_S in R^(896 x m)
    - Algebraic consistency: W_0 @ K_S approx V_S
    - Shared target substring across distinct prompts
    - Rank-deficiency detection on duplicate safeguard prompts
    - Representation geometry & pairwise cosine similarity on highly similar prompts
    - Error handling on invalid/missing substrings in safeguard lists
    - Model parameter immutability and hook cleanup
"""

import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from activation_capture import QwenActivationCapturer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def run_tests():
    print("=" * 80)
    print("TEST SUITE: QwenActivationCapturer Substring Span & Safeguard Extraction")
    print("=" * 80)

    # Step 1: Load model and tokenizer
    print("\n[Step 1] Loading cached Qwen2.5-0.5B model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        local_files_only=True
    )
    model.eval()

    config = model.config
    hidden_size = config.hidden_size            # 896
    intermediate_size = config.intermediate_size  # 4864
    num_layers = config.num_hidden_layers        # 24

    print(f"Model Loaded: {num_layers} layers | hidden_size={hidden_size} | intermediate_size={intermediate_size}")

    # Snapshot target weights for immutability verification
    snapshot_l20 = model.model.layers[20].mlp.down_proj.weight.detach().clone()
    snapshot_l22 = model.model.layers[22].mlp.down_proj.weight.detach().clone()

    capturer = QwenActivationCapturer(model, tokenizer)

    test_count = 0
    test_passed = 0

    def record_test(name, passed, detail=""):
        nonlocal test_count, test_passed
        test_count += 1
        if passed:
            test_passed += 1
            print(f"  [PASS] Test {test_count:02d}: {name} {detail}")
        else:
            print(f"  [FAIL] Test {test_count:02d}: {name} {detail}")
            raise AssertionError(f"Test failed: {name} {detail}")

    # =========================================================================
    # PART 1: SUBSTRING SPAN ALIGNMENT & POSITION TESTS
    # =========================================================================
    print("\n--- Part 1: Substring Alignment & Position Regressions ---")
    prompt = (
        "Question: In a lake, there is a patch of lily pads. "
        "The patch doubles in size every day. "
        "It takes 48 days to cover the lake. "
        "On which day was the lake half covered?"
    )

    # Test 1: Single occurrence multi-word
    res1 = capturer.capture(prompt, layer_indices=[20, 22], match_substring="lily pads")
    m1 = res1[20]["metadata"]
    inp1 = res1[20]["input"]
    record_test(
        "Single occurrence multi-word ('lily pads')",
        m1["matched_substring"] == "lily pads" and m1["num_tokens"] >= 2 and inp1.shape[-1] == intermediate_size,
        f"(tokens: {m1['token_positions']}, shape: {list(inp1.shape)})"
    )

    # Test 2: Multiple occurrences first
    res_lake_0 = capturer.capture(prompt, layer_indices=20, match_substring="lake", occurrence=0)
    m_lake_0 = res_lake_0[20]["metadata"]
    record_test("Multiple occurrences occurrence=0", m_lake_0["occurrence"] == 0 and m_lake_0["char_start"] == 15)

    # Test 3: Multiple occurrences second
    res_lake_1 = capturer.capture(prompt, layer_indices=20, match_substring="lake", occurrence=1)
    m_lake_1 = res_lake_1[20]["metadata"]
    record_test("Multiple occurrences occurrence=1", m_lake_1["occurrence"] == 1 and m_lake_1["char_start"] == 119)

    # Test 4: Multiple occurrences negative index
    res_lake_last = capturer.capture(prompt, layer_indices=20, match_substring="lake", occurrence=-1)
    m_lake_last = res_lake_last[20]["metadata"]
    record_test("Multiple occurrences occurrence=-1", m_lake_last["occurrence"] == -1 and m_lake_last["char_start"] == 146)

    # Test 5: Substring near beginning
    res_start = capturer.capture(prompt, layer_indices=20, match_substring="Question")
    m_start = res_start[20]["metadata"]
    record_test("Substring near beginning ('Question')", m_start["char_start"] == 0 and 0 in m_start["token_positions"])

    # Test 6: Substring near end
    res_end = capturer.capture(prompt, layer_indices=20, match_substring="covered?")
    m_end = res_end[20]["metadata"]
    record_test("Substring near end ('covered?')", m_end["char_end"] == len(prompt) and m_end["token_positions"][-1] == m_end["sequence_length"] - 1)

    # Test 7: Number tokenization ('48')
    res_num = capturer.capture(prompt, layer_indices=20, match_substring="48")
    m_num = res_num[20]["metadata"]
    record_test("Number tokenization ('48')", "48" in prompt[m_num["char_start"]:m_num["char_end"]] and len(m_num["token_positions"]) >= 1)

    # Test 8: Reasoning word 'doubles'
    res_doubles = capturer.capture(prompt, layer_indices=20, match_substring="doubles")
    m_doubles = res_doubles[20]["metadata"]
    record_test("Word 'doubles' span alignment", m_doubles["token_str"].strip() == "doubles")

    # Test 9: Reasoning word 'half'
    res_half = capturer.capture(prompt, layer_indices=20, match_substring="half")
    m_half = res_half[20]["metadata"]
    record_test("Word 'half' span alignment", m_half["token_str"].strip() == "half")

    # Test 10: Adjacent punctuation comma
    res_punct_lake = capturer.capture(prompt, layer_indices=20, match_substring="lake,")
    m_punct_lake = res_punct_lake[20]["metadata"]
    record_test("Adjacent punctuation comma ('lake,')", "," in m_punct_lake["token_str"])

    # Test 11: Adjacent punctuation period
    res_punct_lake_dot = capturer.capture(prompt, layer_indices=20, match_substring="lake.")
    m_punct_lake_dot = res_punct_lake_dot[20]["metadata"]
    record_test("Adjacent punctuation period ('lake.')", "." in m_punct_lake_dot["token_str"])

    # Test 12: Surrounding whitespace
    res_ws = capturer.capture(prompt, layer_indices=20, match_substring=" 48 ")
    m_ws = res_ws[20]["metadata"]
    record_test("Surrounding whitespace (' 48 ')", m_ws["char_start"] == prompt.find(" 48 "))

    # Test 13: Missing substring error handling
    missing_caught = False
    try:
        capturer.capture(prompt, layer_indices=20, match_substring="cosmology")
    except ValueError as e:
        missing_caught = True
        err_msg = str(e)
    record_test("Missing substring raises ValueError", missing_caught and "not found in prompt" in err_msg)

    # Test 14: Out of range occurrence error handling
    oob_caught = False
    try:
        capturer.capture(prompt, layer_indices=20, match_substring="lake", occurrence=50)
    except IndexError as e:
        oob_caught = True
    record_test("Out-of-range occurrence raises IndexError", oob_caught)

    # Test 15: Regression token_position='last'
    res_last = capturer.capture(prompt, layer_indices=20, token_position="last")
    record_test("Preserved token_position='last'", res_last[20]["metadata"]["resolved_token_position"] == 42)

    # Test 16: Regression token_position='all'
    res_all = capturer.capture(prompt, layer_indices=20, token_position="all")
    record_test("Preserved token_position='all'", res_all[20]["metadata"]["num_tokens"] == 43)

    # Test 17: Regression integer position=5
    res_pos5 = capturer.capture(prompt, layer_indices=20, token_position=5)
    record_test("Preserved explicit integer position=5", res_pos5[20]["metadata"]["resolved_token_position"] == 5)

    # =========================================================================
    # PART 2: SAFEGUARD ACTIVATION EXTRACTION & NUMERICAL ANALYSIS
    # =========================================================================
    print("\n--- Part 2: Safeguard Activation Extraction & Numerical Analysis ---")

    # Canonical multi-category safeguard suite
    CANONICAL_SAFEGUARDS = [
        {
            "id": "safe_arithmetic_add",
            "category": "arithmetic",
            "prompt": "The sum of 25 and 17 is equal to 42.",
            "target_substring": "42"
        },
        {
            "id": "safe_arithmetic_sub",
            "category": "arithmetic",
            "prompt": "Calculating 100 minus 36 gives 64.",
            "target_substring": "64"
        },
        {
            "id": "safe_factual_capital",
            "category": "factual_knowledge",
            "prompt": "The capital city of France is Paris.",
            "target_substring": "Paris"
        },
        {
            "id": "safe_science_water",
            "category": "science",
            "prompt": "Water is composed of hydrogen and oxygen.",
            "target_substring": "oxygen"
        },
        {
            "id": "safe_language_proverb",
            "category": "ordinary_language",
            "prompt": "Actions speak louder than words in every culture.",
            "target_substring": "words"
        }
    ]

    # Test 18: Multi-category safeguard extraction & matrix dimensions
    sg_res = capturer.extract_safeguards(CANONICAL_SAFEGUARDS, layer_idx=20, dtype=torch.float64)
    K_S = sg_res["K_S"]
    V_S = sg_res["V_S"]
    num_analysis = sg_res["numerical_analysis"]

    m_count = len(CANONICAL_SAFEGUARDS)
    record_test(
        "Safeguard matrix dimensions K_S in R^(4864 x 5) and V_S in R^(896 x 5)",
        K_S.shape == (4864, m_count) and V_S.shape == (896, m_count) and K_S.dtype == torch.float64,
        f"(K_S: {list(K_S.shape)}, V_S: {list(V_S.shape)}, dtype: {K_S.dtype})"
    )

    # Test 19: Algebraic baseline consistency W_0 @ K_S approx V_S
    base_err = num_analysis["baseline_consistency_relative_error"]
    base_max = num_analysis["baseline_max_abs_diff"]
    record_test(
        "Algebraic baseline consistency ||W_0 @ K_S - V_S||_F / ||V_S||_F < 1e-5",
        num_analysis["baseline_consistency_verified"] and base_err < 1e-5,
        f"(rel_error: {base_err:.2e}, max_abs: {base_max:.2e})"
    )

    # Test 20: SVD rank & null-space dimension
    rank = num_analysis["numerical_rank"]
    d_red = num_analysis["null_space_dimension"]
    is_full = num_analysis["is_full_rank"]
    null_leak = num_analysis["null_space_leakage"]
    record_test(
        "SVD numerical rank & null-space orthogonality",
        rank == m_count and is_full and d_red == (4864 - m_count) and null_leak < 1e-12,
        f"(rank: {rank}/{m_count}, d_red: {d_red}, null_leak: {null_leak:.2e})"
    )

    # Test 21: Condition number and singular-value spectrum
    cond_eff = num_analysis["effective_condition_number"]
    s_vals = num_analysis["singular_values"]
    record_test(
        "Singular value spectrum & effective condition number",
        len(s_vals) == m_count and s_vals[0] > s_vals[-1] > 0 and cond_eff < 100.0,
        f"(sigma_max: {s_vals[0]:.2f}, sigma_min: {s_vals[-1]:.2f}, cond: {cond_eff:.2f})"
    )

    # Test 22: Shared target substring across distinct prompts
    shared_sub_prompts = [
        {
            "id": "shared_france",
            "prompt": "The capital of France is Paris.",
            "target_substring": "is"
        },
        {
            "id": "shared_japan",
            "prompt": "The capital of Japan is Tokyo.",
            "target_substring": "is"
        },
        {
            "id": "shared_water",
            "prompt": "Water is essential for biological life.",
            "target_substring": "is"
        }
    ]
    res_shared = capturer.extract_safeguards(shared_sub_prompts, layer_idx=20)
    k_shared = res_shared["K_S"]
    v_shared = res_shared["V_S"]
    shared_analysis = res_shared["numerical_analysis"]
    record_test(
        "Multiple prompts sharing same target substring ('is')",
        k_shared.shape == (4864, 3) and shared_analysis["numerical_rank"] == 3 and shared_analysis["baseline_consistency_verified"],
        f"(shape: {list(k_shared.shape)}, rank: {shared_analysis['numerical_rank']}, rel_err: {shared_analysis['baseline_consistency_relative_error']:.2e})"
    )

    # Test 23: Duplicate safeguard prompts (rank deficiency test)
    dup_prompts = [
        CANONICAL_SAFEGUARDS[0], # Arithmetic add
        CANONICAL_SAFEGUARDS[0], # Exact duplicate of add
        CANONICAL_SAFEGUARDS[2]  # Factual capital
    ]
    res_dup = capturer.extract_safeguards(dup_prompts, layer_idx=20)
    dup_analysis = res_dup["numerical_analysis"]
    cos_dup = dup_analysis["pairwise_cosine_similarity"]
    record_test(
        "Duplicate safeguard prompt rank deficiency & cosine similarity 1.000",
        dup_analysis["numerical_rank"] == 2 and not dup_analysis["is_full_rank"] and abs(cos_dup[0][1] - 1.0) < 1e-12,
        f"(m=3, detected rank: {dup_analysis['numerical_rank']}, cos(0,1): {cos_dup[0][1]:.6f})"
    )

    # Test 24: Highly similar safeguard prompts (rephrased)
    similar_prompts = [
        {
            "id": "paris_v1",
            "prompt": "The capital city of France is Paris.",
            "target_substring": "Paris"
        },
        {
            "id": "paris_v2",
            "prompt": "The historical capital of France is Paris.",
            "target_substring": "Paris"
        }
    ]
    res_sim = capturer.extract_safeguards(similar_prompts, layer_idx=20)
    sim_analysis = res_sim["numerical_analysis"]
    sim_cos = sim_analysis["pairwise_cosine_similarity"][0][1]
    record_test(
        "Highly similar safeguard prompts exhibit high representation cosine similarity",
        sim_cos > 0.85 and sim_analysis["numerical_rank"] == 2,
        f"(cosine similarity: {sim_cos:.4f}, rank: {sim_analysis['numerical_rank']})"
    )

    # Test 25: Invalid or missing target substring in safeguard list
    bad_prompts = [
        {
            "id": "bad_item",
            "prompt": "The capital of France is Paris.",
            "target_substring": "non_existent_token_string"
        }
    ]
    bad_caught = False
    try:
        capturer.extract_safeguards(bad_prompts, layer_idx=20)
    except ValueError as e:
        bad_caught = True
        bad_err = str(e)
    record_test(
        "Missing target substring in safeguard list raises clear ValueError",
        bad_caught and "not found in prompt" in bad_err,
        f"(error: '{bad_err}')"
    )

    # Test 26: Layer 22 extraction
    res_l22 = capturer.extract_safeguards(CANONICAL_SAFEGUARDS[:4], layer_idx=22)
    l22_analysis = res_l22["numerical_analysis"]
    record_test(
        "Layer 22 safeguard extraction and baseline consistency",
        res_l22["K_S"].shape == (4864, 4) and l22_analysis["baseline_consistency_verified"],
        f"(shape: {list(res_l22['K_S'].shape)}, base_err: {l22_analysis['baseline_consistency_relative_error']:.2e})"
    )

    # Test 27: Hook removal confirmation
    hooks_clean_20 = len(model.model.layers[20].mlp.down_proj._forward_hooks) == 0
    hooks_clean_22 = len(model.model.layers[22].mlp.down_proj._forward_hooks) == 0
    record_test(
        "All forward hooks cleanly removed after safeguard extraction",
        hooks_clean_20 and hooks_clean_22,
        f"(L20 hooks: {len(model.model.layers[20].mlp.down_proj._forward_hooks)}, L22 hooks: {len(model.model.layers[22].mlp.down_proj._forward_hooks)})"
    )

    # Test 28: Model weight immutability
    w_clean_20 = torch.equal(model.model.layers[20].mlp.down_proj.weight, snapshot_l20)
    w_clean_22 = torch.equal(model.model.layers[22].mlp.down_proj.weight, snapshot_l22)
    record_test(
        "Model weights bitwise identical to pristine state across all operations",
        w_clean_20 and w_clean_22,
        "(exact bitwise equality on layers 20 and 22)"
    )

    print("\n" + "=" * 80)
    print(f"RESULTS: {test_passed}/{test_count} Tests Passed (100.0%)")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()

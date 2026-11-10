"""
Experiment 7.12: Causal Layer Localization for Factual Association
===================================================================
Executes an inference-only, float64-audited causal activation patching sweep
across all 28 layers (l in [0, 27]) of Qwen/Qwen2.5-0.5B.

Objectives:
1. Localize where the France -> Paris factual binding is mediated (MLP vs Attention vs Residual).
2. Pinpoint why P3 ("The capital city of France is") resisted single-layer surgery at Layer 20
   despite 0.9624 activation cosine similarity with P1 ("The capital of France is").
3. Evaluate 3-layer sliding-window MLP redundancy ([l, l+1, l+2]).
4. Perform bidirectional semantic-anchor cross-patching (P1 <-> P3).
5. Compare deterministic counterfactual subject swapping vs Gaussian embedding corruption.
6. Provide an empirical layer set L* and causal energy weights for the Experiment 8 multi-layer cascade.
"""

import os
import sys
import json
import time
import math
from typing import Dict, Any, List, Tuple, Literal
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PARIS_TOKEN_ID = 12095   # " Paris"
BERLIN_TOKEN_ID = 19846  # " Berlin"

PROMPT_SUITE = [
    {
        "id": "P1",
        "name": "Canonical Declarative",
        "clean_text": "The capital of France is",
        "corrupt_text": "The capital of Germany is",
        "anchors": {
            "rel_cap": 1,
            "rel_of": -3,
            "subj": -2,
            "final": -1
        }
    },
    {
        "id": "P3",
        "name": "Lexical Insertion Paraphrase",
        "clean_text": "The capital city of France is",
        "corrupt_text": "The capital city of Germany is",
        "anchors": {
            "rel_cap": 1,
            "rel_city": 2,
            "rel_of": -3,
            "subj": -2,
            "final": -1
        }
    },
    {
        "id": "P4",
        "name": "Wh-Question",
        "clean_text": "Which city serves as the capital of France? The answer is",
        "corrupt_text": "Which city serves as the capital of Germany? The answer is",
        "anchors": {
            "subj": 7,
            "final": -1
        }
    }
]

ComponentType = Literal["resid", "attn", "mlp"]

def get_target_module(model: torch.nn.Module, layer_idx: int, component: ComponentType) -> torch.nn.Module:
    layer = model.model.layers[layer_idx]
    if component == "resid":
        return layer
    elif component == "attn":
        return layer.self_attn.o_proj
    elif component == "mlp":
        return layer.mlp.down_proj
    raise ValueError(f"Invalid component: {component}")

@torch.no_grad()
def capture_clean_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    component: ComponentType,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Runs a clean forward pass and caches intermediate activations across all 28 layers."""
    cache: Dict[int, torch.Tensor] = {}
    hooks = []
    n_layers = len(model.model.layers)
    for l in range(n_layers):
        def make_hook(l_idx: int):
            def hook(module, inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                cache[l_idx] = tensor.detach().clone()
            return hook
        hooks.append(get_target_module(model, l, component).register_forward_hook(make_hook(l)))

    out = model(input_ids)
    logits = out.logits[0, -1].to(torch.float64)
    for h in hooks:
        h.remove()
    return logits, cache

@torch.no_grad()
def evaluate_patch(
    model: torch.nn.Module,
    corrupt_ids: torch.Tensor,
    clean_cache: Dict[int, torch.Tensor],
    layers_to_patch: List[int],
    pos_mapping: List[Tuple[int, int]],  # (clean_pos, corrupt_pos)
    component: ComponentType,
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    paris_id: int = PARIS_TOKEN_ID,
    berlin_id: int = BERLIN_TOKEN_ID,
) -> Dict[str, float]:
    """Applies clean activations into corrupt forward pass and computes recovery metrics."""
    hooks = []
    for l in layers_to_patch:
        def make_patch_hook(l_idx: int):
            def hook(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                patched = (output[0] if is_tuple else output).clone()
                for c_pos, corr_pos in pos_mapping:
                    patched[:, corr_pos, :] = clean_cache[l_idx][:, c_pos, :]
                return (patched,) + output[1:] if is_tuple else patched
            return hook
        hooks.append(get_target_module(model, l, component).register_forward_hook(make_patch_hook(l)))

    patched_logits = model(corrupt_ids).logits[0, -1].to(torch.float64)
    for h in hooks:
        h.remove()

    dz_clean = (clean_logits[paris_id] - clean_logits[berlin_id]).item()
    dz_corrupt = (corrupt_logits[paris_id] - corrupt_logits[berlin_id]).item()
    dz_patched = (patched_logits[paris_id] - patched_logits[berlin_id]).item()

    denom = dz_clean - dz_corrupt
    nldr = (dz_patched - dz_corrupt) / denom if abs(denom) > 1e-12 else 0.0

    p_clean = F.softmax(clean_logits, dim=-1)
    p_patched = F.softmax(patched_logits, dim=-1)
    log_p_patched = F.log_softmax(patched_logits, dim=-1)
    kl_div = F.kl_div(log_p_patched, p_clean, reduction="sum").item()

    return {
        "nldr": float(nldr),
        "dz_patched": float(dz_patched),
        "p_paris": float(p_patched[paris_id].item()),
        "p_berlin": float(p_patched[berlin_id].item()),
        "kl_clean_to_patched": float(kl_div),
    }

def run_experiment_7_12():
    print("=" * 80)
    print("EXPERIMENT 7.12: CAUSAL LAYER LOCALIZATION & P1 <-> P3 MECHANISTIC AUDIT")
    print("=" * 80)

    start_time = time.time()
    results_dir = os.path.dirname(os.path.abspath(__file__))
    output_raw_json = os.path.join(results_dir, "causal_localization_results.json")
    output_summary_json = os.path.join(results_dir, "evaluation_summary.json")

    # 1. Load pristine model and tokenizer
    print(f"\n[Step 1] Loading pristine {MODEL_NAME} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        local_files_only=True
    )
    model.eval()

    n_layers = len(model.model.layers)
    embed_dim = model.model.embed_tokens.weight.shape[1]
    vocab_size = model.config.vocab_size
    print(f"Model loaded: {n_layers} layers, hidden dimension: {embed_dim}, vocab size: {vocab_size}")

    # Snapshot a representative layer weight to verify bitwise invariance post-run
    w20_snapshot = model.model.layers[20].mlp.down_proj.weight.detach().clone()

    full_results: Dict[str, Any] = {
        "metadata": {
            "task": "Task 7.12 Causal Layer Localization",
            "model": MODEL_NAME,
            "n_layers": n_layers,
            "embed_dim": embed_dim,
            "paris_token_id": PARIS_TOKEN_ID,
            "berlin_token_id": BERLIN_TOKEN_ID,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "prompts": {},
        "bidirectional_cross_patching": {},
        "gaussian_noise_robustness": {},
        "layerwise_p1_p3_cosine_similarity": []
    }

    # Verify token IDs for Paris and Berlin
    paris_str = tokenizer.decode([PARIS_TOKEN_ID])
    berlin_str = tokenizer.decode([BERLIN_TOKEN_ID])
    print(f"Token verification: PARIS_TOKEN_ID={PARIS_TOKEN_ID} ('{paris_str}'), BERLIN_TOKEN_ID={BERLIN_TOKEN_ID} ('{berlin_str}')")
    assert paris_str.strip() == "Paris", f"Paris token mismatch: '{paris_str}'"
    assert berlin_str.strip() == "Berlin", f"Berlin token mismatch: '{berlin_str}'"

    # Pre-cache clean and corrupt tokenized inputs
    prompt_tokens: Dict[str, Dict[str, Any]] = {}
    for p in PROMPT_SUITE:
        pid = p["id"]
        c_ids = tokenizer(p["clean_text"], return_tensors="pt")["input_ids"]
        k_ids = tokenizer(p["corrupt_text"], return_tensors="pt")["input_ids"]
        assert c_ids.shape == k_ids.shape, f"Length mismatch for prompt {pid}: clean {c_ids.shape} vs corrupt {k_ids.shape}"
        c_toks = [tokenizer.decode([t]) for t in c_ids[0].tolist()]
        k_toks = [tokenizer.decode([t]) for t in k_ids[0].tolist()]
        prompt_tokens[pid] = {
            "clean_ids": c_ids,
            "corrupt_ids": k_ids,
            "clean_tokens": c_toks,
            "corrupt_tokens": k_toks,
            "seq_len": c_ids.shape[1]
        }
        print(f"Prompt {pid}: Clean tokens: {c_toks} | Corrupt tokens: {k_toks}")

    # =========================================================================
    # PART 1: Full-Grid Single-Position Causal Activation Patching (P1, P3, P4)
    # =========================================================================
    print(f"\n[Step 2] Executing Full-Grid Causal Traces across 28 Layers & All Positions...")
    components: List[ComponentType] = ["resid", "attn", "mlp"]

    for p in PROMPT_SUITE:
        pid = p["id"]
        pt = prompt_tokens[pid]
        c_ids = pt["clean_ids"]
        k_ids = pt["corrupt_ids"]
        seq_len = pt["seq_len"]

        # Baseline clean & corrupt logits
        with torch.no_grad():
            clean_base_logits = model(c_ids).logits[0, -1].to(torch.float64)
            corrupt_base_logits = model(k_ids).logits[0, -1].to(torch.float64)

        dz_clean = (clean_base_logits[PARIS_TOKEN_ID] - clean_base_logits[BERLIN_TOKEN_ID]).item()
        dz_corrupt = (corrupt_base_logits[PARIS_TOKEN_ID] - corrupt_base_logits[BERLIN_TOKEN_ID]).item()
        p_clean_paris = F.softmax(clean_base_logits, dim=-1)[PARIS_TOKEN_ID].item()
        p_clean_berlin = F.softmax(clean_base_logits, dim=-1)[BERLIN_TOKEN_ID].item()
        p_corrupt_paris = F.softmax(corrupt_base_logits, dim=-1)[PARIS_TOKEN_ID].item()
        p_corrupt_berlin = F.softmax(corrupt_base_logits, dim=-1)[BERLIN_TOKEN_ID].item()

        print(f"\n--- Baseline for {pid} ({p['name']}) ---")
        print(f"  Clean:   P(Paris)={p_clean_paris*100:.2f}%, P(Berlin)={p_clean_berlin*100:.2f}%, Δz={dz_clean:+.3f}")
        print(f"  Corrupt: P(Paris)={p_corrupt_paris*100:.2f}%, P(Berlin)={p_corrupt_berlin*100:.2f}%, Δz={dz_corrupt:+.3f}")
        print(f"  Baseline contrast denominator: (Δz_clean - Δz_corrupt) = {dz_clean - dz_corrupt:+.3f}")

        p_result: Dict[str, Any] = {
            "name": p["name"],
            "clean_text": p["clean_text"],
            "corrupt_text": p["corrupt_text"],
            "clean_tokens": pt["clean_tokens"],
            "corrupt_tokens": pt["corrupt_tokens"],
            "baseline": {
                "dz_clean": dz_clean,
                "dz_corrupt": dz_corrupt,
                "p_clean_paris": p_clean_paris,
                "p_clean_berlin": p_clean_berlin,
                "p_corrupt_paris": p_corrupt_paris,
                "p_corrupt_berlin": p_corrupt_berlin,
            },
            "grid_nldr": {},
            "grid_details": {},
            "mlp_window3": {}
        }

        # Run grid for each component
        for comp in components:
            clean_logits, clean_cache = capture_clean_cache(model, c_ids, comp)
            corrupt_logits = corrupt_base_logits

            nldr_matrix = []
            details_matrix = []

            for l in range(n_layers):
                row_nldr = []
                row_details = []
                for pos in range(seq_len):
                    pos_map = [(pos, pos)]
                    res = evaluate_patch(
                        model=model,
                        corrupt_ids=k_ids,
                        clean_cache=clean_cache,
                        layers_to_patch=[l],
                        pos_mapping=pos_map,
                        component=comp,
                        clean_logits=clean_logits,
                        corrupt_logits=corrupt_logits
                    )
                    row_nldr.append(res["nldr"])
                    row_details.append(res)
                nldr_matrix.append(row_nldr)
                details_matrix.append(row_details)

            p_result["grid_nldr"][comp] = nldr_matrix
            p_result["grid_details"][comp] = details_matrix

            # Find top layer for subject and final token positions
            subj_idx = p["anchors"]["subj"]
            final_idx = p["anchors"]["final"]
            if subj_idx < 0: subj_idx += seq_len
            if final_idx < 0: final_idx += seq_len

            subj_scores = [nldr_matrix[l][subj_idx] for l in range(n_layers)]
            final_scores = [nldr_matrix[l][final_idx] for l in range(n_layers)]

            top_subj_layer = int(torch.tensor(subj_scores).argmax().item())
            top_final_layer = int(torch.tensor(final_scores).argmax().item())

            print(f"  [{pid}] {comp:<5} | Max Subj NLDR: L{top_subj_layer} ({subj_scores[top_subj_layer]:.3f}) | Max Final NLDR: L{top_final_layer} ({final_scores[top_final_layer]:.3f})")

        # =====================================================================
        # PART 2: 3-Layer Sliding-Window MLP Restoration ([w, w+1, w+2])
        # =====================================================================
        print(f"  [{pid}] Running 3-Layer Sliding-Window MLP ([l, l+1, l+2])...")
        _, clean_mlp_cache = capture_clean_cache(model, c_ids, "mlp")
        subj_idx = p["anchors"]["subj"]
        final_idx = p["anchors"]["final"]
        if subj_idx < 0: subj_idx += seq_len
        if final_idx < 0: final_idx += seq_len

        window_results = []
        for w in range(n_layers - 2):
            window_layers = [w, w + 1, w + 2]
            # Test at Subject position
            res_subj = evaluate_patch(
                model=model,
                corrupt_ids=k_ids,
                clean_cache=clean_mlp_cache,
                layers_to_patch=window_layers,
                pos_mapping=[(subj_idx, subj_idx)],
                component="mlp",
                clean_logits=clean_base_logits,
                corrupt_logits=corrupt_base_logits
            )
            # Test at Final position
            res_final = evaluate_patch(
                model=model,
                corrupt_ids=k_ids,
                clean_cache=clean_mlp_cache,
                layers_to_patch=window_layers,
                pos_mapping=[(final_idx, final_idx)],
                component="mlp",
                clean_logits=clean_base_logits,
                corrupt_logits=corrupt_base_logits
            )
            # Test Joint Subject + Final
            res_joint = evaluate_patch(
                model=model,
                corrupt_ids=k_ids,
                clean_cache=clean_mlp_cache,
                layers_to_patch=window_layers,
                pos_mapping=[(subj_idx, subj_idx), (final_idx, final_idx)],
                component="mlp",
                clean_logits=clean_base_logits,
                corrupt_logits=corrupt_base_logits
            )
            window_results.append({
                "window": f"L{w}-L{w+2}",
                "start_layer": w,
                "layers": window_layers,
                "nldr_subj": res_subj["nldr"],
                "nldr_final": res_final["nldr"],
                "nldr_joint": res_joint["nldr"],
                "p_paris_subj": res_subj["p_paris"],
                "p_paris_final": res_final["p_paris"],
                "p_paris_joint": res_joint["p_paris"]
            })
        p_result["mlp_window3"] = window_results
        full_results["prompts"][pid] = p_result

    # =========================================================================
    # PART 3: Layerwise Cosine Similarity Between P1 and P3 Activations
    # =========================================================================
    print(f"\n[Step 3] Computing Layerwise Cosine Similarity between P1 and P3...")
    p1_ids = prompt_tokens["P1"]["clean_ids"]
    p3_ids = prompt_tokens["P3"]["clean_ids"]

    _, p1_resid_cache = capture_clean_cache(model, p1_ids, "resid")
    _, p3_resid_cache = capture_clean_cache(model, p3_ids, "resid")
    _, p1_mlp_cache = capture_clean_cache(model, p1_ids, "mlp")
    _, p3_mlp_cache = capture_clean_cache(model, p3_ids, "mlp")

    # Semantic anchors:
    # P1: [The (0), capital (1), of (2), France (3), is (4)]
    # P3: [The (0), capital (1), city (2), of (3), France (4), is (5)]
    # Aligned pairs: (France: 3 <-> 4), (is: 4 <-> 5), (capital: 1 <-> 1), (of: 2 <-> 3)
    p1_p3_sims = []
    for l in range(n_layers):
        # France token
        p1_h_subj = p1_resid_cache[l][0, 3].to(torch.float64)
        p3_h_subj = p3_resid_cache[l][0, 4].to(torch.float64)
        cos_subj = (torch.dot(p1_h_subj, p3_h_subj) / (torch.norm(p1_h_subj) * torch.norm(p3_h_subj))).item()

        # is token
        p1_h_final = p1_resid_cache[l][0, 4].to(torch.float64)
        p3_h_final = p3_resid_cache[l][0, 5].to(torch.float64)
        cos_final = (torch.dot(p1_h_final, p3_h_final) / (torch.norm(p1_h_final) * torch.norm(p3_h_final))).item()

        # mlp output at France token
        p1_mlp_subj = p1_mlp_cache[l][0, 3].to(torch.float64)
        p3_mlp_subj = p3_mlp_cache[l][0, 4].to(torch.float64)
        cos_mlp_subj = (torch.dot(p1_mlp_subj, p3_mlp_subj) / (torch.norm(p1_mlp_subj) * torch.norm(p3_mlp_subj))).item()

        # mlp output at is token
        p1_mlp_final = p1_mlp_cache[l][0, 4].to(torch.float64)
        p3_mlp_final = p3_mlp_cache[l][0, 5].to(torch.float64)
        cos_mlp_final = (torch.dot(p1_mlp_final, p3_mlp_final) / (torch.norm(p1_mlp_final) * torch.norm(p3_mlp_final))).item()

        p1_p3_sims.append({
            "layer": l,
            "cos_resid_subj": cos_subj,
            "cos_resid_final": cos_final,
            "cos_mlp_subj": cos_mlp_subj,
            "cos_mlp_final": cos_mlp_final
        })
    full_results["layerwise_p1_p3_cosine_similarity"] = p1_p3_sims
    print(f"  P1 vs P3 Cosine at Layer 20: Resid Subj={p1_p3_sims[20]['cos_resid_subj']:.4f}, Resid Final={p1_p3_sims[20]['cos_resid_final']:.4f}, MLP Final={p1_p3_sims[20]['cos_mlp_final']:.4f}")

    # =========================================================================
    # PART 4: Bidirectional Semantic-Anchor Cross-Patching (P1 <-> P3)
    # =========================================================================
    print(f"\n[Step 4] Executing Bidirectional Cross-Patching between P1 and P3...")
    # Anchor mapping between P1 and P3:
    # clean_pos in P1, corrupt_pos in P3:
    #   subj: (3, 4)
    #   final: (4, 5)
    #   rel_cap: (1, 1)
    #   rel_of: (2, 3)
    p1_to_p3_anchors = {
        "subj": [(3, 4)],
        "final": [(4, 5)],
        "rel_cap": [(1, 1)],
        "rel_of": [(2, 3)],
        "joint_subj_final": [(3, 4), (4, 5)],
        "joint_all": [(1, 1), (2, 3), (3, 4), (4, 5)]
    }

    # Cross-Patch A: Clean P1 into Corrupt P3
    # Target: Can P1's clean activations restore Paris on corrupt P3 ("The capital city of Germany is")?
    print(f"  Cross-Patch A: Clean P1 -> Corrupt P3")
    cross_a_results: Dict[str, Any] = {}
    p3_corrupt_ids = prompt_tokens["P3"]["corrupt_ids"]
    p3_corrupt_logits = full_results["prompts"]["P3"]["baseline"]["dz_corrupt"]
    p3_clean_logits = full_results["prompts"]["P3"]["baseline"]["dz_clean"]

    with torch.no_grad():
        p3_clean_base_logits = model(prompt_tokens["P3"]["clean_ids"]).logits[0, -1].to(torch.float64)
        p3_corrupt_base_logits = model(p3_corrupt_ids).logits[0, -1].to(torch.float64)

    for comp in components:
        _, p1_clean_cache = capture_clean_cache(model, p1_ids, comp)
        cross_a_results[comp] = {}
        for anchor_name, pos_map in p1_to_p3_anchors.items():
            sweep = []
            for l in range(n_layers):
                res = evaluate_patch(
                    model=model,
                    corrupt_ids=p3_corrupt_ids,
                    clean_cache=p1_clean_cache,
                    layers_to_patch=[l],
                    pos_mapping=pos_map,
                    component=comp,
                    clean_logits=p3_clean_base_logits,
                    corrupt_logits=p3_corrupt_base_logits
                )
                sweep.append(res["nldr"])
            cross_a_results[comp][anchor_name] = sweep
            max_l = int(torch.tensor(sweep).argmax().item())
            print(f"    [P1 -> P3] {comp:<5} | Anchor: {anchor_name:<16} | Peak NLDR: L{max_l:02d} ({sweep[max_l]:.3f})")

    # Cross-Patch B: Clean P3 into Corrupt P1
    # clean_pos in P3, corrupt_pos in P1:
    p3_to_p1_anchors = {
        "subj": [(4, 3)],
        "final": [(5, 4)],
        "rel_cap": [(1, 1)],
        "rel_of": [(3, 2)],
        "joint_subj_final": [(4, 3), (5, 4)],
        "joint_all": [(1, 1), (3, 2), (4, 3), (5, 4)]
    }

    print(f"  Cross-Patch B: Clean P3 -> Corrupt P1")
    cross_b_results: Dict[str, Any] = {}
    p1_corrupt_ids = prompt_tokens["P1"]["corrupt_ids"]

    with torch.no_grad():
        p1_clean_base_logits = model(prompt_tokens["P1"]["clean_ids"]).logits[0, -1].to(torch.float64)
        p1_corrupt_base_logits = model(p1_corrupt_ids).logits[0, -1].to(torch.float64)

    for comp in components:
        _, p3_clean_cache = capture_clean_cache(model, p3_ids, comp)
        cross_b_results[comp] = {}
        for anchor_name, pos_map in p3_to_p1_anchors.items():
            sweep = []
            for l in range(n_layers):
                res = evaluate_patch(
                    model=model,
                    corrupt_ids=p1_corrupt_ids,
                    clean_cache=p3_clean_cache,
                    layers_to_patch=[l],
                    pos_mapping=pos_map,
                    component=comp,
                    clean_logits=p1_clean_base_logits,
                    corrupt_logits=p1_corrupt_base_logits
                )
                sweep.append(res["nldr"])
            cross_b_results[comp][anchor_name] = sweep
            max_l = int(torch.tensor(sweep).argmax().item())
            print(f"    [P3 -> P1] {comp:<5} | Anchor: {anchor_name:<16} | Peak NLDR: L{max_l:02d} ({sweep[max_l]:.3f})")

    full_results["bidirectional_cross_patching"] = {
        "P1_to_P3": cross_a_results,
        "P3_to_P1": cross_b_results
    }

    # =========================================================================
    # PART 5: Secondary Robustness Check — Gaussian Noise on Embeddings
    # =========================================================================
    print(f"\n[Step 5] Secondary Robustness Check: Gaussian Embedding Noise on P1 and P3...")
    embed_weights = model.model.embed_tokens.weight.detach().to(torch.float32)
    sigma_embed = torch.std(embed_weights).item()
    noise_std = 3.0 * sigma_embed
    print(f"  Embedding std: {sigma_embed:.4f}, Gaussian noise std (3*sigma): {noise_std:.4f}")

    # Evaluate across 10 random seeds (42..51)
    noise_results: Dict[str, Any] = {}
    seeds = list(range(42, 52))

    for pid in ["P1", "P3"]:
        c_ids = prompt_tokens[pid]["clean_ids"]
        seq_len = prompt_tokens[pid]["seq_len"]
        subj_pos = 3 if pid == "P1" else 4

        with torch.no_grad():
            clean_emb = model.model.embed_tokens(c_ids).clone() # [1, seq_len, 896]
            clean_logits = model(inputs_embeds=clean_emb).logits[0, -1].to(torch.float64)

        dz_clean = (clean_logits[PARIS_TOKEN_ID] - clean_logits[BERLIN_TOKEN_ID]).item()

        seed_corrupt_dz = []
        seed_corrupt_norms = []
        clean_subj_norm = torch.norm(clean_emb[0, subj_pos]).item()

        # Precompute noisy embeddings per seed
        noisy_runs = []
        for s in seeds:
            torch.manual_seed(s)
            noise = torch.randn_like(clean_emb[0, subj_pos]) * noise_std
            corrupt_emb = clean_emb.clone()
            corrupt_emb[0, subj_pos] += noise
            c_norm = torch.norm(corrupt_emb[0, subj_pos]).item()
            seed_corrupt_norms.append(c_norm)

            with torch.no_grad():
                c_logits = model(inputs_embeds=corrupt_emb).logits[0, -1].to(torch.float64)
            c_dz = (c_logits[PARIS_TOKEN_ID] - c_logits[BERLIN_TOKEN_ID]).item()
            seed_corrupt_dz.append(c_dz)
            noisy_runs.append({
                "seed": s,
                "corrupt_emb": corrupt_emb,
                "c_logits": c_logits,
                "c_dz": c_dz,
                "c_norm": c_norm
            })

        mean_corrupt_dz = float(torch.tensor(seed_corrupt_dz).mean().item())
        mean_corrupt_norm = float(torch.tensor(seed_corrupt_norms).mean().item())
        print(f"  [{pid}] Clean subj norm: {clean_subj_norm:.3f} | Mean corrupt subj norm: {mean_corrupt_norm:.3f} ({(mean_corrupt_norm/clean_subj_norm - 1)*100:+.1f}%)")
        print(f"  [{pid}] Clean Δz: {dz_clean:.3f} | Mean corrupt Δz: {mean_corrupt_dz:.3f}")

        # Patch clean activations at Subject position for MLP across 28 layers (averaged over 10 seeds)
        # 1. Capture clean activations
        clean_mlp_acts: Dict[int, torch.Tensor] = {}
        hooks = []
        for l in range(n_layers):
            def make_h(l_idx):
                def h(module, inp, out):
                    clean_mlp_acts[l_idx] = out[0].detach().clone() if isinstance(out, tuple) else out.detach().clone()
                return h
            hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(make_h(l)))
        _ = model(inputs_embeds=clean_emb)
        for h in hooks: h.remove()

        mlp_noisy_nldr = []
        for l in range(n_layers):
            seed_nldrs = []
            for n_run in noisy_runs:
                def patch_h(module, inp, out):
                    is_t = isinstance(out, tuple)
                    p = (out[0] if is_t else out).clone()
                    p[:, subj_pos, :] = clean_mlp_acts[l][:, subj_pos, :]
                    return (p,) + out[1:] if is_t else p

                ph = model.model.layers[l].mlp.down_proj.register_forward_hook(patch_h)
                with torch.no_grad():
                    pat_logits = model(inputs_embeds=n_run["corrupt_emb"]).logits[0, -1].to(torch.float64)
                ph.remove()

                pat_dz = (pat_logits[PARIS_TOKEN_ID] - pat_logits[BERLIN_TOKEN_ID]).item()
                denom = dz_clean - n_run["c_dz"]
                nldr = (pat_dz - n_run["c_dz"]) / denom if abs(denom) > 1e-12 else 0.0
                seed_nldrs.append(nldr)

            mean_l_nldr = float(torch.tensor(seed_nldrs).mean().item())
            mlp_noisy_nldr.append(mean_l_nldr)

        max_noisy_l = int(torch.tensor(mlp_noisy_nldr).argmax().item())
        print(f"  [{pid}] Gaussian Noise MLP Subject Peak: Layer {max_noisy_l} (Mean NLDR: {mlp_noisy_nldr[max_noisy_l]:.3f})")
        noise_results[pid] = {
            "clean_subj_norm": clean_subj_norm,
            "mean_corrupt_norm": mean_corrupt_norm,
            "norm_drift_pct": (mean_corrupt_norm / clean_subj_norm - 1.0) * 100.0,
            "clean_dz": dz_clean,
            "mean_corrupt_dz": mean_corrupt_dz,
            "mlp_subj_nldr_by_layer": mlp_noisy_nldr,
            "peak_layer": max_noisy_l,
            "peak_nldr": mlp_noisy_nldr[max_noisy_l]
        }

    full_results["gaussian_noise_robustness"] = noise_results

    # =========================================================================
    # PART 6: Synthesize Evaluation Summary and Recommend L* for Experiment 8
    # =========================================================================
    print(f"\n[Step 6] Synthesizing Evaluation Summary and Candidate Layers L*...")

    # Extract MLP and Attention profiles for Subject and Final positions across P1, P3, P4
    summary_data: Dict[str, Any] = {
        "experiment": "Task 7.12 Causal Layer Localization",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_layers_per_prompt": {},
        "sliding_window_summary": {},
        "p1_p3_cross_patch_summary": {},
        "p1_p3_cosine_vs_causal": {},
        "candidate_layer_set_L_star": {}
    }

    # Summary table for top layers
    for pid in ["P1", "P3", "P4"]:
        seq_len = prompt_tokens[pid]["seq_len"]
        s_idx = PROMPT_SUITE[[x["id"] for x in PROMPT_SUITE].index(pid)]["anchors"]["subj"]
        f_idx = PROMPT_SUITE[[x["id"] for x in PROMPT_SUITE].index(pid)]["anchors"]["final"]
        if s_idx < 0: s_idx += seq_len
        if f_idx < 0: f_idx += seq_len

        comp_peaks = {}
        for comp in ["mlp", "attn", "resid"]:
            matrix = full_results["prompts"][pid]["grid_nldr"][comp]
            subj_scores = [matrix[l][s_idx] for l in range(n_layers)]
            final_scores = [matrix[l][f_idx] for l in range(n_layers)]

            top3_subj = sorted([(l, subj_scores[l]) for l in range(n_layers)], key=lambda x: x[1], reverse=True)[:3]
            top3_final = sorted([(l, final_scores[l]) for l in range(n_layers)], key=lambda x: x[1], reverse=True)[:3]

            comp_peaks[comp] = {
                "top3_subj": [{"layer": l, "nldr": round(score, 4)} for l, score in top3_subj],
                "top3_final": [{"layer": l, "nldr": round(score, 4)} for l, score in top3_final],
            }
        summary_data["top_layers_per_prompt"][pid] = comp_peaks

    # Top sliding-window 3-layer MLP
    for pid in ["P1", "P3", "P4"]:
        windows = full_results["prompts"][pid]["mlp_window3"]
        top_subj_win = sorted(windows, key=lambda x: x["nldr_subj"], reverse=True)[:3]
        top_final_win = sorted(windows, key=lambda x: x["nldr_final"], reverse=True)[:3]
        top_joint_win = sorted(windows, key=lambda x: x["nldr_joint"], reverse=True)[:3]
        summary_data["sliding_window_summary"][pid] = {
            "top_subj_windows": [{"window": w["window"], "nldr": round(w["nldr_subj"], 4)} for w in top_subj_win],
            "top_final_windows": [{"window": w["window"], "nldr": round(w["nldr_final"], 4)} for w in top_final_win],
            "top_joint_windows": [{"window": w["window"], "nldr": round(w["nldr_joint"], 4)} for w in top_joint_win]
        }

    # Cross-patch summary
    for comp in ["mlp", "attn", "resid"]:
        cross_a_subj = cross_a_results[comp]["subj"]
        cross_a_final = cross_a_results[comp]["final"]
        cross_a_joint = cross_a_results[comp]["joint_subj_final"]
        cross_b_subj = cross_b_results[comp]["subj"]
        cross_b_final = cross_b_results[comp]["final"]
        cross_b_joint = cross_b_results[comp]["joint_subj_final"]

        summary_data["p1_p3_cross_patch_summary"][comp] = {
            "P1_into_P3": {
                "top_subj": sorted([(l, cross_a_subj[l]) for l in range(n_layers)], key=lambda x: x[1], reverse=True)[:3],
                "top_final": sorted([(l, cross_a_final[l]) for l in range(n_layers)], key=lambda x: x[1], reverse=True)[:3],
                "top_joint": sorted([(l, cross_a_joint[l]) for l in range(n_layers)], key=lambda x: x[1], reverse=True)[:3],
            },
            "P3_into_P1": {
                "top_subj": sorted([(l, cross_b_subj[l]) for l in range(n_layers)], key=lambda x: x[1], reverse=True)[:3],
                "top_final": sorted([(l, cross_b_final[l]) for l in range(n_layers)], key=lambda x: x[1], reverse=True)[:3],
                "top_joint": sorted([(l, cross_b_joint[l]) for l in range(n_layers)], key=lambda x: x[1], reverse=True)[:3],
            }
        }

    # Cosine vs Causal at Layer 20:
    summary_data["p1_p3_cosine_vs_causal"] = {
        "layer_20_cosine_resid_subj": p1_p3_sims[20]["cos_resid_subj"],
        "layer_20_cosine_resid_final": p1_p3_sims[20]["cos_resid_final"],
        "layer_20_cosine_mlp_final": p1_p3_sims[20]["cos_mlp_final"],
        "layer_20_p1_mlp_final_nldr": full_results["prompts"]["P1"]["grid_nldr"]["mlp"][20][-1],
        "layer_20_p3_mlp_final_nldr": full_results["prompts"]["P3"]["grid_nldr"]["mlp"][20][-1],
        "layer_20_p1_to_p3_cross_final_nldr": cross_a_results["mlp"]["final"][20],
    }

    # =========================================================================
    # Candidate Layer Set L* Determination
    # =========================================================================
    # Compute combined MLP score across P1, P3, and P4 at Subject & Final positions
    mlp_combined_scores = []
    for l in range(n_layers):
        p1_subj = full_results["prompts"]["P1"]["grid_nldr"]["mlp"][l][3]
        p1_final = full_results["prompts"]["P1"]["grid_nldr"]["mlp"][l][4]
        p3_subj = full_results["prompts"]["P3"]["grid_nldr"]["mlp"][l][4]
        p3_final = full_results["prompts"]["P3"]["grid_nldr"]["mlp"][l][5]
        p4_subj = full_results["prompts"]["P4"]["grid_nldr"]["mlp"][l][7]
        p4_final = full_results["prompts"]["P4"]["grid_nldr"]["mlp"][l][11]

        # Max causal effect for this layer across the prompts
        max_prompt_effect = max(p1_subj, p1_final, p3_subj, p3_final, p4_subj, p4_final)
        avg_subj_effect = (p1_subj + p3_subj + p4_subj) / 3.0
        avg_final_effect = (p1_final + p3_final + p4_final) / 3.0
        cross_p1_p3 = cross_a_results["mlp"]["joint_subj_final"][l]

        mlp_combined_scores.append({
            "layer": l,
            "max_effect": max_prompt_effect,
            "avg_subj": avg_subj_effect,
            "avg_final": avg_final_effect,
            "cross_p1_p3": cross_p1_p3
        })

    # Filter layers where causal mediation is active (threshold tau >= 0.08)
    active_layers = [x for x in mlp_combined_scores if x["max_effect"] >= 0.08 or x["cross_p1_p3"] >= 0.08]
    active_layer_indices = [x["layer"] for x in active_layers]

    # Normalize weights across active layers
    sum_effects = sum(x["max_effect"] for x in active_layers) if active_layers else 1.0
    recommended_layers = []
    for x in active_layers:
        recommended_layers.append({
            "layer": x["layer"],
            "raw_causal_score": round(x["max_effect"], 4),
            "normalized_energy_weight": round(x["max_effect"] / sum_effects, 4) if sum_effects > 0 else 0.0
        })

    summary_data["candidate_layer_set_L_star"] = {
        "threshold_tau": 0.08,
        "recommended_layers": [x["layer"] for x in recommended_layers],
        "layer_allocations": recommended_layers,
        "interpretation": "Mid-to-late MLP layers showing high causal recovery across canonical and paraphrased prompts."
    }

    # Assert model weights remained completely unperturbed
    w20_post = model.model.layers[20].mlp.down_proj.weight.detach()
    assert torch.equal(w20_snapshot, w20_post), "CRITICAL: Model weights were modified during inference-only causal patching!"
    print(f"\n[Verification] Model Weight Invariance Verified bitwise (torch.equal() == True).")

    # Save output artifacts
    with open(output_raw_json, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved full raw results to: {output_raw_json} ({os.path.getsize(output_raw_json)} bytes)")

    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved evaluation summary to: {output_summary_json} ({os.path.getsize(output_summary_json)} bytes)")

    elapsed = time.time() - start_time
    print(f"\nTask 7.12 Causal Localization Sweep completed in {elapsed:.1f} seconds.")
    return full_results, summary_data

if __name__ == "__main__":
    run_experiment_7_12()

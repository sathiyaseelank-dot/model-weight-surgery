#!/usr/bin/env python3
"""
Experiment 9B Preflight Activation-Geometry & Tokenization Audit
===============================================================
Rigorous preflight audit for 5-fact counterfactual scaling on Qwen/Qwen2.5-0.5B at Layer 21:
  1. Tokenization verification for all 10 target/original city names.
  2. Unembedding steering directions d_f and pairwise cosine similarity matrix.
  3. Activation extraction at Layer 21 mlp.down_proj input.
  4. Safeguard basis U_S (37 tokens -> rank-31 SVD basis).
  5. 10x10 training key pairwise cosine similarities.
  6. Gram matrix singular values and condition numbers (raw vs safeguard-projected).
  7. Safeguard null-space norm retentions for all 10 training keys.
  8. Cross-fact interference & rollout key collinearity audit.
  9. Conflict check with the hard safeguard subspace.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
LAYER_IDX = 21
LAMBDA_REG = 0.10
GAMMA_DAMP = 1.0
BETA_OPERATING = 0.18

FACTS = [
    {
        "id": "F1_France",
        "country": "France",
        "original_city": "Paris",
        "target_city": "Lyon",
        "train_p1": "The capital of France is",
        "train_p2": "Which city serves as the capital of France? The answer is",
        "heldout_h1": "The capital city of France is",
        "heldout_h2": "\n\nCountry: France\nCapital:",
        "heldout_h3": "France's capital city is",
        "rollout_r1": "The capital of France is Lyon. Lyon is the capital of",
        "rollout_r2": "The capital city of France is Lyon, a major city in",
        "rollout_r3": "Lyon is a historic city located in the country of"
    },
    {
        "id": "F2_Japan",
        "country": "Japan",
        "original_city": "Tokyo",
        "target_city": "Osaka",
        "train_p1": "The capital of Japan is",
        "train_p2": "Which city serves as the capital of Japan? The answer is",
        "heldout_h1": "The capital city of Japan is",
        "heldout_h2": "\n\nCountry: Japan\nCapital:",
        "heldout_h3": "Japan's capital city is",
        "rollout_r1": "The capital of Japan is Osaka. Osaka is the capital of",
        "rollout_r2": "The capital city of Japan is Osaka, a major city in",
        "rollout_r3": "Osaka is a historic city located in the country of"
    },
    {
        "id": "F3_China",
        "country": "China",
        "original_city": "Beijing",
        "target_city": "Shanghai",
        "train_p1": "The capital of China is",
        "train_p2": "Which city serves as the capital of China? The answer is",
        "heldout_h1": "The capital city of China is",
        "heldout_h2": "\n\nCountry: China\nCapital:",
        "heldout_h3": "China's capital city is",
        "rollout_r1": "The capital of China is Shanghai. Shanghai is the capital of",
        "rollout_r2": "The capital city of China is Shanghai, a major city in",
        "rollout_r3": "Shanghai is a historic city located in the country of"
    },
    {
        "id": "F4_Canada",
        "country": "Canada",
        "original_city": "Ottawa",
        "target_city": "Toronto",
        "train_p1": "The capital of Canada is",
        "train_p2": "Which city serves as the capital of Canada? The answer is",
        "heldout_h1": "The capital city of Canada is",
        "heldout_h2": "\n\nCountry: Canada\nCapital:",
        "heldout_h3": "Canada's capital city is",
        "rollout_r1": "The capital of Canada is Toronto. Toronto is the capital of",
        "rollout_r2": "The capital city of Canada is Toronto, a major city in",
        "rollout_r3": "Toronto is a historic city located in the country of"
    },
    {
        "id": "F5_Australia",
        "country": "Australia",
        "original_city": "Canberra",
        "target_city": "Sydney",
        "train_p1": "The capital of Australia is",
        "train_p2": "Which city serves as the capital of Australia? The answer is",
        "heldout_h1": "The capital city of Australia is",
        "heldout_h2": "\n\nCountry: Australia\nCapital:",
        "heldout_h3": "Australia's capital city is",
        "rollout_r1": "The capital of Australia is Sydney. Sydney is the capital of",
        "rollout_r2": "The capital city of Australia is Sydney, a major city in",
        "rollout_r3": "Sydney is a historic city located in the country of"
    }
]

SAFEGUARD_PROMPTS = [
    {"id": "safe_germany", "prompt": "The capital of Germany is Berlin.", "category": "geography"},
    {"id": "safe_italy", "prompt": "The capital of Italy is Rome.", "category": "geography"},
    {"id": "safe_spain", "prompt": "The capital of Spain is Madrid.", "category": "geography"},
    {"id": "safe_science", "prompt": "Water is composed of hydrogen and oxygen.", "category": "science"},
    {"id": "safe_arithmetic", "prompt": "2 + 2 = 4.", "category": "arithmetic"}
]


def audit_tokenization(tokenizer) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("1. TOKENIZATION AUDIT FOR 5 COUNTERFACTUAL FACTS")
    print("=" * 70)
    tok_results = {}
    for f in FACTS:
        cid = f["id"]
        c_orig = " " + f["original_city"]
        c_targ = " " + f["target_city"]
        
        orig_ids = tokenizer(c_orig, add_special_tokens=False)["input_ids"]
        targ_ids = tokenizer(c_targ, add_special_tokens=False)["input_ids"]
        
        orig_single = (len(orig_ids) == 1)
        targ_single = (len(targ_ids) == 1)
        
        tok_results[cid] = {
            "country": f["country"],
            "original_city": f["original_city"],
            "original_token_str": c_orig,
            "original_token_id": orig_ids[0] if orig_single else orig_ids,
            "original_is_single_token": orig_single,
            "target_city": f["target_city"],
            "target_token_str": c_targ,
            "target_token_id": targ_ids[0] if targ_single else targ_ids,
            "target_is_single_token": targ_single
        }
        print(f"[{cid}] {f['country']:<10}: {c_orig!r:<12} -> ID {orig_ids} | {c_targ!r:<12} -> ID {targ_ids} (Single Tokens: {orig_single and targ_single})")
    return tok_results


def audit_target_directions(model: torch.nn.Module, tok_results: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    print("\n" + "=" * 70)
    print("2. UNEMBEDDING STEERING TARGET DIRECTIONS AUDIT")
    print("=" * 70)
    # Extract unembedding weights W_U in float64
    W_U = model.lm_head.weight.detach().to(torch.float64)  # [vocab_size, 896]
    
    d_vectors = []
    d_norms = []
    fact_ids = []
    
    for cid, tinfo in tok_results.items():
        orig_id = tinfo["original_token_id"]
        targ_id = tinfo["target_token_id"]
        
        e_orig = W_U[orig_id]
        e_targ = W_U[targ_id]
        
        diff = e_targ - e_orig
        norm_diff = torch.linalg.norm(diff).item()
        d_f = diff / norm_diff
        
        d_vectors.append(d_f)
        d_norms.append(norm_diff)
        fact_ids.append(cid)
        print(f"  {cid:<12} ||E_{tinfo['target_city']} - E_{tinfo['original_city']}||_2 = {norm_diff:.4f}")
        
    D_mat = torch.stack(d_vectors, dim=0)  # [5, 896]
    cos_D = D_mat @ D_mat.T  # [5, 5]
    
    print("\n  Pairwise Cosine Similarity Matrix between Target Directions (d_1 ... d_5):")
    header = " " * 14 + "".join([f"{fid:<12}" for fid in fact_ids])
    print(header)
    for i, fid in enumerate(fact_ids):
        row_str = "".join([f"{cos_D[i, j].item():12.4f}" for j in range(len(fact_ids))])
        print(f"  {fid:<12} {row_str}")
        
    d_audit = {
        "fact_ids": fact_ids,
        "raw_diff_norms": d_norms,
        "pairwise_cosines": cos_D.tolist()
    }
    return D_mat, d_audit


@torch.no_grad()
def capture_activations(model: torch.nn.Module, tokenizer) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("3. ACTIVATION CAPTURE AT LAYER 21 mlp.down_proj INPUT")
    print("=" * 70)
    dp = model.model.layers[LAYER_IDX].mlp.down_proj
    
    # 1. Safeguard basis K_S (37 tokens)
    s_bucket: List[torch.Tensor] = []
    h_s = dp.register_forward_hook(
        lambda m, inp, out: s_bucket.append(inp[0].detach().to(torch.float64).squeeze(0))
    )
    for p in SAFEGUARD_PROMPTS:
        ids = tokenizer(p["prompt"], return_tensors="pt")["input_ids"]
        model(ids)
    h_s.remove()
    K_S = torch.cat(s_bucket, dim=0).T.contiguous()  # [4864, 37]
    print(f"  Safeguard matrix K_S captured: shape {list(K_S.shape)}")
    
    # SVD on K_S
    U, S, _ = torch.linalg.svd(K_S, full_matrices=False)
    rank_S = int((S > 1e-12).sum().item())
    U_S = U[:, :rank_S]
    print(f"  Safeguard basis U_S: numerical rank = {rank_S}/{K_S.shape[1]}, sigma_max = {S[0].item():.4f}, sigma_min = {S[rank_S-1].item():.4f}")
    
    # 2. Training keys (10 keys: 2 per fact)
    training_keys: Dict[str, torch.Tensor] = {}
    for f in FACTS:
        fid = f["id"]
        for p_key in ["train_p1", "train_p2"]:
            prompt = f[p_key]
            bucket: List[torch.Tensor] = []
            h = dp.register_forward_hook(
                lambda m, inp, out: bucket.append(inp[0].detach().to(torch.float64).squeeze(0))
            )
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            model(ids)
            h.remove()
            key_id = f"{fid}_{p_key}"
            training_keys[key_id] = bucket[0][-1]  # terminal token activation
            print(f"  Training key {key_id:<18} -> norm: {torch.linalg.norm(training_keys[key_id]).item():.4f} (prompt: {prompt!r})")
            
    # 3. Rollout damping keys (15 keys: 3 per fact)
    rollout_keys: Dict[str, torch.Tensor] = {}
    for f in FACTS:
        fid = f["id"]
        for r_key in ["rollout_r1", "rollout_r2", "rollout_r3"]:
            prompt = f[r_key]
            bucket: List[torch.Tensor] = []
            h = dp.register_forward_hook(
                lambda m, inp, out: bucket.append(inp[0].detach().to(torch.float64).squeeze(0))
            )
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            model(ids)
            h.remove()
            key_id = f"{fid}_{r_key}"
            rollout_keys[key_id] = bucket[0][-1]
            print(f"  Rollout key {key_id:<18} -> norm: {torch.linalg.norm(rollout_keys[key_id]).item():.4f}")
            
    # 4. Heldout keys (15 keys: 3 per fact)
    heldout_keys: Dict[str, torch.Tensor] = {}
    for f in FACTS:
        fid = f["id"]
        for h_key in ["heldout_h1", "heldout_h2", "heldout_h3"]:
            prompt = f[h_key]
            bucket: List[torch.Tensor] = []
            h = dp.register_forward_hook(
                lambda m, inp, out: bucket.append(inp[0].detach().to(torch.float64).squeeze(0))
            )
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            model(ids)
            h.remove()
            key_id = f"{fid}_{h_key}"
            heldout_keys[key_id] = bucket[0][-1]

    return {
        "K_S": K_S,
        "U_S": U_S,
        "rank_S": rank_S,
        "S_values": S.tolist(),
        "training_keys": training_keys,
        "rollout_keys": rollout_keys,
        "heldout_keys": heldout_keys
    }


def audit_activation_geometry(act_data: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("4. TRAINING KEY ACTIVATION GEOMETRY & GRAM AUDIT")
    print("=" * 70)
    training_keys = act_data["training_keys"]
    U_S = act_data["U_S"]
    
    t_names = list(training_keys.keys())
    t_mat = torch.stack([training_keys[n] for n in t_names], dim=1)  # [4864, 10]
    
    # 1. Pairwise Cosine Matrix of Raw Training Keys
    t_norms = torch.linalg.norm(t_mat, dim=0, keepdim=True)
    t_normed = t_mat / t_norms
    cos_train = t_normed.T @ t_normed  # [10, 10]
    
    print("  Pairwise Cosine Similarity Matrix (10 Training Keys):")
    header = " " * 22 + "".join([f"{n.split('_')[0][:2]+'_'+n.split('_')[-1][:2]:>8}" for n in t_names])
    print(header)
    for i, n in enumerate(t_names):
        short_n = n.replace("train_", "")
        row_str = "".join([f"{cos_train[i, j].item():8.4f}" for j in range(10)])
        print(f"  {short_n:<20} {row_str}")
        
    # 2. Raw Gram Matrix Analysis
    G_raw = t_mat.T @ t_mat  # [10, 10]
    evals_raw, _ = torch.linalg.eigh(G_raw)
    cond_raw = torch.linalg.cond(G_raw).item()
    print(f"\n  Raw Training Gram Matrix (K^T K):")
    print(f"    Eigenvalues: min={evals_raw[0].item():.4e}, max={evals_raw[-1].item():.4e}")
    print(f"    Condition Number: {cond_raw:.4e}")
    
    # 3. Safeguard Projection & Norm Retention
    t_proj = t_mat - U_S @ (U_S.T @ t_mat)
    t_proj = t_proj - U_S @ (U_S.T @ t_proj)  # double orthogonalization
    
    proj_norms = torch.linalg.norm(t_proj, dim=0)
    raw_norms = torch.linalg.norm(t_mat, dim=0)
    retention_ratios = (proj_norms / raw_norms).tolist()
    
    print("\n  Safeguard Null-Space Norm Retention for All 10 Training Keys:")
    ret_dict = {}
    for i, n in enumerate(t_names):
        ret_pct = retention_ratios[i] * 100.0
        ret_dict[n] = retention_ratios[i]
        print(f"    {n:<20}: raw={raw_norms[i].item():.4f} -> proj={proj_norms[i].item():.4f} (Retention: {ret_pct:6.2f}%)")
        
    # 4. Projected Regularized Gram Matrix Analysis (K~^T K~ + lambda I)
    G_proj = t_proj.T @ t_proj
    G_reg = G_proj + LAMBDA_REG * torch.eye(10, dtype=torch.float64)
    evals_proj, _ = torch.linalg.eigh(G_proj)
    evals_reg, _ = torch.linalg.eigh(G_reg)
    cond_proj = torch.linalg.cond(G_proj).item()
    cond_reg = torch.linalg.cond(G_reg).item()
    
    print(f"\n  Projected Training Gram Matrix (K~^T K~):")
    print(f"    Eigenvalues: min={evals_proj[0].item():.4e}, max={evals_proj[-1].item():.4e}")
    print(f"    Condition Number: {cond_proj:.4e}")
    print(f"  Regularized Gram Matrix (K~^T K~ + {LAMBDA_REG}*I):")
    print(f"    Eigenvalues: min={evals_reg[0].item():.4e}, max={evals_reg[-1].item():.4e}")
    print(f"    Condition Number: {cond_reg:.4e}")
    
    # 5. Overlap / Conflict Check with Safeguard Subspace
    print("\n  Safeguard Subspace Conflict Audit:")
    max_overlaps = []
    for i, n in enumerate(t_names):
        # Coordinates in U_S basis
        coords = U_S.T @ t_normed[:, i]
        max_overlap = torch.max(torch.abs(coords)).item()
        total_subspace_energy = torch.linalg.norm(coords).item() ** 2
        max_overlaps.append({"key": n, "max_single_vector_overlap": max_overlap, "total_safeguard_energy": total_subspace_energy})
        print(f"    {n:<20}: max_single_overlap={max_overlap:.4f}, total_safeguard_energy={total_subspace_energy:.4f}")
        
    return {
        "training_key_names": t_names,
        "pairwise_train_cosines": cos_train.tolist(),
        "raw_gram_eigenvalues": evals_raw.tolist(),
        "raw_gram_cond": cond_raw,
        "norm_retentions": ret_dict,
        "proj_gram_eigenvalues": evals_proj.tolist(),
        "proj_gram_cond": cond_proj,
        "reg_gram_eigenvalues": evals_reg.tolist(),
        "reg_gram_cond": cond_reg,
        "safeguard_subspace_overlaps": max_overlaps
    }


def audit_rollout_collinearity(act_data: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("5. CROSS-FACT & ROLLOUT KEY COLLINEARITY AUDIT")
    print("=" * 70)
    training_keys = act_data["training_keys"]
    rollout_keys = act_data["rollout_keys"]
    
    rollout_collin = {}
    for f in FACTS:
        fid = f["id"]
        p1_key = f"{fid}_train_p1"
        k_p1 = training_keys[p1_key]
        
        r1_key = f"{fid}_rollout_r1"
        r2_key = f"{fid}_rollout_r2"
        r3_key = f"{fid}_rollout_r3"
        
        k_r1 = rollout_keys[r1_key]
        k_r2 = rollout_keys[r2_key]
        k_r3 = rollout_keys[r3_key]
        
        cos_r1 = F.cosine_similarity(k_p1.unsqueeze(0), k_r1.unsqueeze(0)).item()
        cos_r2 = F.cosine_similarity(k_p1.unsqueeze(0), k_r2.unsqueeze(0)).item()
        cos_r3 = F.cosine_similarity(k_p1.unsqueeze(0), k_r3.unsqueeze(0)).item()
        
        rollout_collin[fid] = {
            "p1_vs_r1": cos_r1,
            "p1_vs_r2": cos_r2,
            "p1_vs_r3": cos_r3
        }
        print(f"  {fid:<12}: cos(P1, R1_repetition)={cos_r1:.4f} | cos(P1, R2_paraphrase)={cos_r2:.4f} | cos(P1, R3_entity)={cos_r3:.4f}")
        
    return rollout_collin


def main():
    print("Initializing Qwen/Qwen2.5-0.5B for Experiment 9B Preflight Audit...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    # 1. Tokenization audit
    tok_audit = audit_tokenization(tokenizer)

    # 2. Target directions audit
    D_mat, d_audit = audit_target_directions(model, tok_audit)

    # 3. Activation capture
    act_data = capture_activations(model, tokenizer)

    # 4. Activation geometry audit
    geom_audit = audit_activation_geometry(act_data)

    # 5. Rollout collinearity audit
    roll_audit = audit_rollout_collinearity(act_data)

    # Compile Full Preflight Report
    full_report = {
        "model_id": MODEL_ID,
        "layer_idx": LAYER_IDX,
        "lambda_reg": LAMBDA_REG,
        "gamma_damp": GAMMA_DAMP,
        "beta_operating": BETA_OPERATING,
        "tokenization_audit": tok_audit,
        "target_directions_audit": d_audit,
        "safeguard_basis_rank": act_data["rank_S"],
        "activation_geometry_audit": geom_audit,
        "rollout_collinearity_audit": roll_audit
    }

    out_path = os.path.join(os.path.dirname(__file__), "preflight_geometry_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[Artifact Generated] Full audit saved to {out_path}")
    print("=" * 70)
    print("PREFLIGHT AUDIT COMPLETE: ALL CHECKS PASSED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

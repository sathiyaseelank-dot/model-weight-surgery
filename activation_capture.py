"""
Activation Capture Module for Qwen2.5-0.5B
===========================================
Provides non-invasive forward hook extraction for MLP down_proj activations:
  - Input activation to down_proj: inp[0] (post-SwiGLU intermediate features)
  - Output activation from down_proj: outp (projection back to residual stream)
  - Multi-layer extraction in a single forward pass
  - Deterministic hook cleanup with try/finally error isolation
  - Multi-position support: "last", "all", or explicit integer index
  - Precise token span alignment: match_substring with occurrence indexing
  - Structured safeguard extraction: K_S (4864 x m), V_S (896 x m)
  - Full numerical linear algebra audit (SVD, condition number, rank, cosine similarity)
"""

import math
import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, Optional, Tuple


class QwenActivationCapturer:
    """
    Non-invasive activation capturer for Qwen models.
    Hooks into `model.model.layers[layer_idx].mlp.down_proj` without altering
    weights, parameter requires_grad, or model execution semantics.
    """
    def __init__(self, model: nn.Module, tokenizer: Any):
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError(
                "Invalid model architecture: Expected transformer model with `model.layers` "
                f"hierarchy, got {type(model).__name__}."
            )
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = len(model.model.layers)
        self.config = getattr(model, "config", None)

    def _validate_layers(self, layer_indices: List[int]) -> List[int]:
        """Validates that requested layer indices exist in the model."""
        if not isinstance(layer_indices, (list, tuple)):
            layer_indices = [layer_indices]
        if len(layer_indices) == 0:
            raise ValueError("layer_indices must not be empty.")
        
        validated = []
        for l in layer_indices:
            if not isinstance(l, int):
                raise TypeError(f"Layer index must be an integer, got {type(l).__name__}.")
            if l < 0 or l >= self.num_layers:
                raise ValueError(
                    f"Layer index {l} out of bounds for model with {self.num_layers} layers "
                    f"(valid range: 0 to {self.num_layers - 1})."
                )
            if not hasattr(self.model.model.layers[l], "mlp") or not hasattr(self.model.model.layers[l].mlp, "down_proj"):
                raise AttributeError(f"Layer {l} does not contain `mlp.down_proj` module.")
            validated.append(l)
        return validated

    def _get_offsets_mapping(
        self,
        encoded: Any,
        prompt: str,
        input_ids_list: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Extracts character offset mapping [(start, end), ...] for each token.
        Uses fast tokenizer offset_mapping if available; otherwise falls back to
        a validated reconstruction algorithm.
        """
        if "offset_mapping" in encoded:
            offsets = encoded["offset_mapping"]
            if isinstance(offsets, torch.Tensor):
                offsets = offsets[0].tolist()
            elif isinstance(offsets, list) and len(offsets) > 0 and isinstance(offsets[0], list) and isinstance(offsets[0][0], (tuple, list)):
                offsets = offsets[0]
            return [tuple(span) for span in offsets]

        return self._reconstruct_offsets(prompt, input_ids_list)

    def _reconstruct_offsets(self, prompt: str, input_ids_list: List[int]) -> List[Tuple[int, int]]:
        """Reconstructs character spans by incremental matching against prompt text."""
        offsets = []
        cursor = 0
        for tid in input_ids_list:
            tok_text = self.tokenizer.decode([tid])
            idx = prompt.find(tok_text, cursor)
            if idx != -1:
                offsets.append((idx, idx + len(tok_text)))
                cursor = idx + len(tok_text)
            else:
                tok_text_lstrip = tok_text.lstrip()
                idx2 = prompt.find(tok_text_lstrip, cursor)
                if idx2 != -1:
                    offsets.append((idx2, idx2 + len(tok_text_lstrip)))
                    cursor = idx2 + len(tok_text_lstrip)
                else:
                    offsets.append((cursor, cursor))
        return offsets

    def _resolve_substring_span(
        self,
        prompt: str,
        match_substring: str,
        occurrence: int,
        offsets: List[Tuple[int, int]]
    ) -> Tuple[int, int, List[int]]:
        """
        Locates the exact character span of `match_substring` in `prompt` and resolves
        the overlapping token positions.
        """
        if not match_substring:
            raise ValueError("match_substring must not be empty.")

        matches = []
        start = 0
        while True:
            idx = prompt.find(match_substring, start)
            if idx == -1:
                break
            matches.append((idx, idx + len(match_substring)))
            start = idx + 1

        if not matches:
            raise ValueError(f"Substring '{match_substring}' not found in prompt.")

        if occurrence < -len(matches) or occurrence >= len(matches):
            raise IndexError(
                f"Occurrence index {occurrence} out of range for substring '{match_substring}' "
                f"(found {len(matches)} match{'es' if len(matches) > 1 else ''})."
            )

        char_start, char_end = matches[occurrence]

        token_indices = []
        for t_idx, (t_start, t_end) in enumerate(offsets):
            if max(t_start, char_start) < min(t_end, char_end):
                token_indices.append(t_idx)

        if not token_indices:
            raise RuntimeError(
                f"Could not resolve any tokenizer tokens for substring '{match_substring}' "
                f"at character span [{char_start}, {char_end})."
            )

        return char_start, char_end, token_indices

    def capture(
        self,
        prompt: str,
        layer_indices: Union[int, List[int]],
        token_position: Union[str, int] = "last",
        match_substring: Optional[str] = None,
        occurrence: int = 0,
        device: Optional[Union[str, torch.device]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Captures input and output activations of mlp.down_proj at selected layers.

        Parameters:
          prompt: Text prompt to encode and evaluate.
          layer_indices: Single layer index or list of layer indices (e.g. [20, 22]).
          token_position: "last", "all", or specific integer index. Ignored if match_substring is given.
          match_substring: If provided, locates this substring in the prompt and extracts its token span.
          occurrence: Occurrence index for match_substring (0 for first occurrence, -1 for last).
          device: Device for inference. If None, uses device of model parameters.

        Returns:
          Dictionary mapping layer_idx -> {
            "input": Tensor,
            "output": Tensor,
            "metadata": Dict[str, Any]
          }
        """
        layers = self._validate_layers([layer_indices] if isinstance(layer_indices, int) else layer_indices)

        if device is None:
            device = next(self.model.parameters()).device

        try:
            encoded = self.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
        except (TypeError, ValueError):
            encoded = self.tokenizer(prompt, return_tensors="pt")

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        seq_len = input_ids.shape[1]
        token_ids_list = input_ids[0].tolist()

        if match_substring is not None:
            offsets = self._get_offsets_mapping(encoded, prompt, token_ids_list)
            char_start, char_end, matched_token_indices = self._resolve_substring_span(
                prompt, match_substring, occurrence, offsets
            )
            mode = "substring"
        else:
            char_start, char_end, matched_token_indices = None, None, None
            if token_position == "last":
                eff_idx = seq_len - 1
                mode = "single_position"
            elif token_position == "all":
                eff_idx = None
                mode = "all"
            elif isinstance(token_position, int):
                if token_position < -seq_len or token_position >= seq_len:
                    raise IndexError(
                        f"token_position {token_position} out of range for sequence length {seq_len} "
                        f"(valid range: -{seq_len} to {seq_len - 1})."
                    )
                eff_idx = token_position if token_position >= 0 else seq_len + token_position
                mode = "single_position"
            else:
                raise ValueError(
                    f"Unsupported token_position '{token_position}'. Expected 'last', 'all', or integer."
                )

        raw_captures: Dict[int, Dict[str, torch.Tensor]] = {}
        hook_handles: List[Any] = []

        def make_hook(layer_idx: int):
            def hook_fn(module, inp, outp):
                raw_captures[layer_idx] = {
                    "input": inp[0].detach().clone(),
                    "output": outp.detach().clone()
                }
            return hook_fn

        try:
            for l in layers:
                module = self.model.model.layers[l].mlp.down_proj
                handle = module.register_forward_hook(make_hook(l))
                hook_handles.append(handle)

            with torch.inference_mode():
                model_inputs = {"input_ids": input_ids}
                if attention_mask is not None:
                    model_inputs["attention_mask"] = attention_mask
                _ = self.model(**model_inputs)

        finally:
            for handle in hook_handles:
                handle.remove()

        results: Dict[int, Dict[str, Any]] = {}

        for l in layers:
            if l not in raw_captures:
                raise RuntimeError(f"Hook for layer {l} was not triggered during model forward pass.")

            raw_inp = raw_captures[l]["input"]   # [1, seq_len, intermediate_size]
            raw_outp = raw_captures[l]["output"] # [1, seq_len, hidden_size]

            if mode == "substring":
                num_tokens = len(matched_token_indices)
                token_strings = [self.tokenizer.decode([token_ids_list[i]]) for i in matched_token_indices]
                
                if num_tokens == 1:
                    input_tensor = raw_inp[0, matched_token_indices[0], :]
                    output_tensor = raw_outp[0, matched_token_indices[0], :]
                else:
                    input_tensor = raw_inp[0, matched_token_indices, :]
                    output_tensor = raw_outp[0, matched_token_indices, :]

                metadata = {
                    "prompt": prompt,
                    "layer_idx": l,
                    "mode": "substring",
                    "matched_substring": match_substring,
                    "occurrence": occurrence,
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_positions": matched_token_indices,
                    "token_id": token_ids_list[matched_token_indices[0]] if num_tokens == 1 else [token_ids_list[i] for i in matched_token_indices],
                    "token_ids": [token_ids_list[i] for i in matched_token_indices],
                    "token_str": "".join(token_strings),
                    "token_strings": token_strings,
                    "num_tokens": num_tokens,
                    "sequence_length": seq_len,
                    "input_shape": list(input_tensor.shape),
                    "output_shape": list(output_tensor.shape),
                    "raw_batch_input_shape": list(raw_inp.shape),
                    "raw_batch_output_shape": list(raw_outp.shape),
                    "dtype": str(input_tensor.dtype),
                    "device": str(input_tensor.device),
                    "input_norm": torch.norm(input_tensor.float()).item(),
                    "output_norm": torch.norm(output_tensor.float()).item(),
                    "is_detached": (not input_tensor.requires_grad) and (not output_tensor.requires_grad)
                }

            elif mode == "all":
                input_tensor = raw_inp[0]
                output_tensor = raw_outp[0]
                token_strings = [self.tokenizer.decode([tid]) for tid in token_ids_list]
                
                metadata = {
                    "prompt": prompt,
                    "layer_idx": l,
                    "mode": "all",
                    "requested_token_position": "all",
                    "resolved_token_position": "all",
                    "sequence_length": seq_len,
                    "token_id": token_ids_list,
                    "token_ids": token_ids_list,
                    "token_str": prompt,
                    "token_strings": token_strings,
                    "num_tokens": seq_len,
                    "input_shape": list(input_tensor.shape),
                    "output_shape": list(output_tensor.shape),
                    "raw_batch_input_shape": list(raw_inp.shape),
                    "raw_batch_output_shape": list(raw_outp.shape),
                    "dtype": str(input_tensor.dtype),
                    "device": str(input_tensor.device),
                    "input_norm": torch.norm(input_tensor.float()).item(),
                    "output_norm": torch.norm(output_tensor.float()).item(),
                    "is_detached": (not input_tensor.requires_grad) and (not output_tensor.requires_grad)
                }

            else:  # single_position
                input_tensor = raw_inp[0, eff_idx, :]
                output_tensor = raw_outp[0, eff_idx, :]
                tid = token_ids_list[eff_idx]
                tok_str = self.tokenizer.decode([tid])

                metadata = {
                    "prompt": prompt,
                    "layer_idx": l,
                    "mode": "single_position",
                    "requested_token_position": token_position,
                    "resolved_token_position": eff_idx,
                    "token_positions": [eff_idx],
                    "sequence_length": seq_len,
                    "token_id": tid,
                    "token_ids": [tid],
                    "token_str": tok_str,
                    "token_strings": [tok_str],
                    "num_tokens": 1,
                    "input_shape": list(input_tensor.shape),
                    "output_shape": list(output_tensor.shape),
                    "raw_batch_input_shape": list(raw_inp.shape),
                    "raw_batch_output_shape": list(raw_outp.shape),
                    "dtype": str(input_tensor.dtype),
                    "device": str(input_tensor.device),
                    "input_norm": torch.norm(input_tensor.float()).item(),
                    "output_norm": torch.norm(output_tensor.float()).item(),
                    "is_detached": (not input_tensor.requires_grad) and (not output_tensor.requires_grad)
                }

            results[l] = {
                "input": input_tensor,
                "output": output_tensor,
                "metadata": metadata
            }

        return results

    def extract_safeguards(
        self,
        safeguard_prompts: List[Dict[str, Any]],
        layer_idx: int,
        span_token_policy: str = "last",
        dtype: torch.dtype = torch.float64,
        device: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """
        Extracts real safeguard activation key matrix K_S (d_in x m) and baseline
        output matrix V_S (d_out x m) for a list of safeguard prompt specifications.

        Each item in safeguard_prompts must contain:
          - 'name' or 'id': unique identifier
          - 'prompt': full input text
          - 'target_substring' or 'match_substring': substring to locate in prompt
          - Optional 'occurrence': integer occurrence index (default: 0)

        Calculates comprehensive numerical linear algebra properties in float64:
          - SVD singular values of K_S
          - Numerical rank via dtype-aware threshold
          - Condition numbers (classical & effective)
          - Null-space dimension and orthogonality
          - Pairwise cosine similarity matrix
          - Key vector norms
          - Algebraic verification of W_0 @ K_S approx V_S

        Returns:
          Dictionary with K_S, V_S, safeguard metadata, and numerical analysis.
        """
        if not safeguard_prompts:
            raise ValueError("safeguard_prompts list must not be empty.")

        self._validate_layers([layer_idx])

        if span_token_policy not in ["last", "first", "mean"]:
            raise ValueError(
                f"Unsupported span_token_policy '{span_token_policy}'. Expected 'last', 'first', or 'mean'."
            )

        k_vectors: List[torch.Tensor] = []
        v_vectors: List[torch.Tensor] = []
        safeguard_metadata: List[Dict[str, Any]] = []
        safeguard_names: List[str] = []

        for idx, sg in enumerate(safeguard_prompts):
            name = sg.get("name", sg.get("id", f"safeguard_{idx}"))
            prompt = sg.get("prompt")
            target_sub = sg.get("target_substring", sg.get("match_substring"))
            occurrence = sg.get("occurrence", 0)

            if prompt is None:
                raise ValueError(f"Safeguard '{name}' is missing 'prompt'.")
            if target_sub is None:
                raise ValueError(f"Safeguard '{name}' is missing 'target_substring'.")

            # Capture activations for this safeguard
            cap_res = self.capture(
                prompt=prompt,
                layer_indices=layer_idx,
                match_substring=target_sub,
                occurrence=occurrence,
                device=device
            )

            layer_data = cap_res[layer_idx]
            raw_k = layer_data["input"]
            raw_v = layer_data["output"]
            meta = layer_data["metadata"]

            # Resolve single vector if target_substring spanned multiple tokens
            if raw_k.ndim == 2:
                if span_token_policy == "last":
                    k_vec = raw_k[-1]
                    v_vec = raw_v[-1]
                elif span_token_policy == "first":
                    k_vec = raw_k[0]
                    v_vec = raw_v[0]
                else:  # mean
                    k_vec = raw_k.mean(dim=0)
                    v_vec = raw_v.mean(dim=0)
            else:
                k_vec = raw_k
                v_vec = raw_v

            k_f64 = k_vec.detach().clone().to(dtype=dtype)
            v_f64 = v_vec.detach().clone().to(dtype=dtype)

            k_vectors.append(k_f64)
            v_vectors.append(v_f64)
            safeguard_names.append(name)

            sg_meta = {
                "name": name,
                "category": sg.get("category", "unspecified"),
                "prompt": prompt,
                "target_substring": target_sub,
                "occurrence": occurrence,
                "token_positions": meta["token_positions"],
                "token_ids": meta["token_ids"],
                "token_str": meta["token_str"],
                "token_strings": meta["token_strings"],
                "num_span_tokens": meta["num_tokens"],
                "selected_policy": span_token_policy,
                "key_norm": torch.norm(k_f64).item(),
                "val_norm": torch.norm(v_f64).item(),
            }
            safeguard_metadata.append(sg_meta)

        # Assemble K_S and V_S matrices: columns are safeguard vectors
        K_S = torch.stack(k_vectors, dim=1) # Shape: (d_in, m) -> e.g. (4864, m)
        V_S = torch.stack(v_vectors, dim=1) # Shape: (d_out, m) -> e.g. (896, m)

        d_in, m = K_S.shape
        d_out = V_S.shape[0]

        # -------------------------------------------------------------
        # NUMERICAL LINEAR ALGEBRA AUDIT (in float64)
        # -------------------------------------------------------------
        U, S, Vh = torch.linalg.svd(K_S, full_matrices=True)
        eps = torch.finfo(torch.float64).eps
        sigma_max = S[0].item() if S.numel() > 0 else 0.0
        tol = max(d_in, m) * eps * sigma_max if sigma_max > 0 else 1e-12
        numerical_rank = int((S > tol).sum().item())

        d_red = d_in - numerical_rank
        U_par = U[:, :numerical_rank]
        U_perp = U[:, numerical_rank:]

        # Null-space leakage
        null_leakage = torch.norm(U_perp.T @ K_S, p="fro").item()

        # Condition numbers
        sigma_min = S[-1].item() if S.numel() > 0 else 0.0
        cond_classical = (sigma_max / sigma_min) if sigma_min > 0 else float("inf")
        sigma_rank = S[numerical_rank - 1].item() if numerical_rank > 0 else 0.0
        cond_effective = (sigma_max / sigma_rank) if sigma_rank > 0 else float("inf")
        is_full_rank = bool(numerical_rank == min(d_in, m))

        # Pairwise cosine similarity matrix
        key_norms = torch.norm(K_S, dim=0, keepdim=True)
        K_normalized = K_S / (key_norms + 1e-15)
        cosine_sim_matrix = (K_normalized.T @ K_normalized).tolist()

        # Maximum reconstruction residual on rank-r SVD projection
        K_reconstructed = U_par @ (U_par.T @ K_S)
        max_recon_residual = torch.norm(K_S - K_reconstructed, p="fro").item()

        # Baseline consistency check: W_0 @ K_S approx V_S
        W_0 = self.model.model.layers[layer_idx].mlp.down_proj.weight.to(dtype=dtype, device=K_S.device)
        W0_KS = W_0 @ K_S
        diff = W0_KS - V_S
        norm_diff = torch.norm(diff, p="fro").item()
        norm_vs = torch.norm(V_S, p="fro").item()
        baseline_rel_error = norm_diff / norm_vs if norm_vs > 0 else norm_diff
        baseline_max_abs_diff = torch.max(torch.abs(diff)).item()

        analysis = {
            "d_in": d_in,
            "d_out": d_out,
            "num_safeguards": m,
            "singular_values": S.tolist(),
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "numerical_rank": numerical_rank,
            "is_full_rank": is_full_rank,
            "null_space_dimension": d_red,
            "classical_condition_number": cond_classical,
            "effective_condition_number": cond_effective,
            "null_space_leakage": null_leakage,
            "max_reconstruction_residual": max_recon_residual,
            "key_vector_norms": key_norms[0].tolist(),
            "pairwise_cosine_similarity": cosine_sim_matrix,
            "baseline_consistency_relative_error": baseline_rel_error,
            "baseline_max_abs_diff": baseline_max_abs_diff,
            "baseline_consistency_verified": bool(baseline_rel_error < 1e-5)
        }

        return {
            "K_S": K_S,
            "V_S": V_S,
            "layer_idx": layer_idx,
            "safeguard_names": safeguard_names,
            "safeguard_metadata": safeguard_metadata,
            "numerical_analysis": analysis
        }

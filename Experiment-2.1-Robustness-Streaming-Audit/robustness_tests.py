"""
Experiment 2.1 — Robustness Tests Module
=========================================
Implements numerical linear algebra robustness audits for Constrained Forward-Only RLS:
  - R1: Exact duplicate safeguards
  - R2: Nearly dependent safeguards
  - R3: Controlled singular-value spectrum
  - R4: Float32 vs. Float64 numerical stability and rank detection
  - R5: High-dimensional safeguard counts (m -> d_in)
  - R6: Per-step sequential safeguard invariance and rejected update immutability
  - R7: Conflict detection robustness (6 key categories, threshold sensitivity)
  - R8: Online vs. independent batch KKT solver under rank deficiency and permutation
"""

import math
import time
import torch
from typing import Optional, Dict, Tuple, Any, List


class OriginalQRNullSpace:
    """
    Exact null-space parameterization logic from Experiment 2 (baseline commit 221f538).
    Uses unpivoted QR decomposition with fixed 1e-10 relative threshold.
    """
    @staticmethod
    def compute(K_safe: torch.Tensor, dtype=torch.float64, device="cpu") -> Tuple[int, torch.Tensor, torch.Tensor]:
        d_in = K_safe.shape[0]
        Q, R = torch.linalg.qr(K_safe, mode="complete")
        diag_R = torch.abs(torch.diag(R))
        threshold = 1e-10 * torch.max(diag_R).item() if diag_R.numel() > 0 else 1e-10
        rank = int((diag_R > threshold).sum().item())
        U_par = Q[:, :rank]
        U_perp = Q[:, rank:]
        return rank, U_par, U_perp


class SVDNullSpace:
    """
    Corrected & robust null-space parameterization using Singular Value Decomposition (SVD).
    Uses mathematically sound, dtype-aware tolerance:
      tol = max(d_in, m) * eps(dtype) * sigma_max
    Guarantees monotonic ordering of singular values and exact null-space basis.
    """
    @staticmethod
    def compute(K_safe: torch.Tensor, dtype=torch.float64, device="cpu") -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        d_in = K_safe.shape[0]
        m = K_safe.shape[1]
        K_s = K_safe.to(dtype=dtype, device=device)
        U, S, Vh = torch.linalg.svd(K_s, full_matrices=True)
        eps = torch.finfo(dtype).eps
        sigma_max = S[0].item() if S.numel() > 0 else 0.0
        tol = max(d_in, m) * eps * sigma_max if sigma_max > 0 else 1e-12
        rank = int((S > tol).sum().item())
        U_par = U[:, :rank]
        U_perp = U[:, rank:]
        return rank, U_par, U_perp, S


class AuditedConstrainedRLS:
    """
    Audited Constrained Online Recursive Least Squares.
    Supports:
      - Both QR (Experiment 2 baseline) and SVD (robust reference) null-space engines.
      - Dtype-aware conflict detection.
      - Pure streaming mode (zero historical metadata retention).
    """
    def __init__(
        self,
        initial_weight: torch.Tensor,
        lambda_reg: float,
        K_safe: Optional[torch.Tensor] = None,
        dtype=torch.float64,
        device="cpu",
        conflict_tol: float = 1e-5,
        decomposition_mode: str = "svd",
        record_history: bool = True
    ):
        self.device = device
        self.dtype = dtype
        self.lambda_reg = float(lambda_reg)
        self.conflict_tol = float(conflict_tol)
        self.decomposition_mode = decomposition_mode
        self.record_history = record_history
        self.d_out, self.d_in = initial_weight.shape

        self.W_0 = initial_weight.clone().to(dtype=dtype, device=device)
        self.samples_seen = 0
        self.step_times = [] if record_history else None
        self.conflict_events = [] if record_history else None

        # Setup Null-Space
        if K_safe is not None and K_safe.numel() > 0:
            self.K_safe = K_safe.clone().to(dtype=dtype, device=device)
            self.num_safeguards = self.K_safe.shape[1]
            self.V_safe = self.W_0 @ self.K_safe

            if decomposition_mode == "qr":
                rank, U_par, U_perp = OriginalQRNullSpace.compute(self.K_safe, dtype=dtype, device=device)
                self.singular_values = None
            else:
                rank, U_par, U_perp, S = SVDNullSpace.compute(self.K_safe, dtype=dtype, device=device)
                self.singular_values = S

            self.rank = rank
            self.U_par = U_par
            self.U_perp = U_perp
            self.d_red = self.d_in - rank
        else:
            self.K_safe = None
            self.num_safeguards = 0
            self.V_safe = None
            self.rank = 0
            self.U_par = None
            self.U_perp = torch.eye(self.d_in, dtype=dtype, device=device)
            self.d_red = self.d_in
            self.singular_values = None

        # Initialize reduced parameter space
        self.W_tilde = torch.zeros(self.d_out, self.d_red, dtype=dtype, device=device)
        self.P_tilde = torch.eye(self.d_red, dtype=dtype, device=device) / self.lambda_reg

    @property
    def W(self) -> torch.Tensor:
        """Full ambient weight matrix W = W_0 + W_tilde @ U_perp^T."""
        if self.d_red == 0:
            return self.W_0
        return self.W_0 + self.W_tilde @ self.U_perp.T

    @property
    def P(self) -> torch.Tensor:
        """Full ambient covariance projection P = U_perp @ P_tilde @ U_perp^T."""
        if self.d_red == 0:
            return torch.zeros(self.d_in, self.d_in, dtype=self.dtype, device=self.device)
        return self.U_perp @ self.P_tilde @ self.U_perp.T

    def detect_conflict(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Detects if training sample (k, v) conflicts with safeguard mappings.
        Conflict occurs when k lies almost entirely in span(K_safe) while v != W_0 @ k.
        """
        if self.U_par is None or self.rank == 0 or self.d_red == 0:
            return False, None

        k_col = k.view(self.d_in, 1).to(dtype=self.dtype, device=self.device)
        v_col = v.view(self.d_out, 1).to(dtype=self.dtype, device=self.device)

        k_norm = torch.norm(k_col).item()
        if k_norm < 1e-12:
            return False, None

        # Orthogonal component in permitted update space
        k_perp = self.U_perp.T @ k_col
        perp_ratio = torch.norm(k_perp).item() / k_norm

        # Expected target under safeguard preservation
        v_expected = self.W_0 @ k_col
        target_error = (torch.norm(v_col - v_expected).item() /
                        (torch.norm(v_expected).item() + 1e-12))

        # Conflict criterion
        is_conflict = (perp_ratio < self.conflict_tol and target_error > self.conflict_tol)

        conflict_info = None
        if is_conflict:
            conflict_info = {
                "sample_index": self.samples_seen,
                "null_space_residual_ratio": perp_ratio,
                "target_discrepancy": target_error,
            }
        return is_conflict, conflict_info

    def update(self, k: torch.Tensor, v: torch.Tensor, reject_conflicts: bool = True) -> Dict[str, Any]:
        """Performs a constrained forward-only RLS update."""
        t0 = time.perf_counter()
        k_col = k.view(self.d_in, 1).to(dtype=self.dtype, device=self.device)
        v_col = v.view(self.d_out, 1).to(dtype=self.dtype, device=self.device)

        # Handle full constraint edge case (d_red == 0)
        if self.d_red == 0:
            if self.record_history:
                self.step_times.append(time.perf_counter() - t0)
            self.samples_seen += 1
            return {
                "updated": False,
                "conflict_detected": False,
                "safeguard_error": self.evaluate_safeguard_error(),
                "reason": "zero_update_dimension"
            }

        is_conflict, conflict_info = self.detect_conflict(k_col, v_col)
        if is_conflict:
            if self.record_history:
                self.conflict_events.append(conflict_info)
            if reject_conflicts:
                t1 = time.perf_counter()
                if self.record_history:
                    self.step_times.append(t1 - t0)
                self.samples_seen += 1
                return {
                    "updated": False,
                    "conflict_detected": True,
                    "conflict_info": conflict_info,
                    "safeguard_error": self.evaluate_safeguard_error()
                }

        # Project into reduced coordinate space
        k_tilde = self.U_perp.T @ k_col
        v_tilde = v_col - (self.W_0 @ k_col)

        # Sherman-Morrison gain
        Pk_tilde = self.P_tilde @ k_tilde
        denom = 1.0 + (k_tilde.T @ Pk_tilde).item()
        gain_tilde = Pk_tilde / denom

        # Update reduced weights
        prior_error_tilde = v_tilde - (self.W_tilde @ k_tilde)
        self.W_tilde = self.W_tilde + prior_error_tilde @ gain_tilde.T

        # Update covariance matrix
        self.P_tilde = self.P_tilde - gain_tilde @ (k_tilde.T @ self.P_tilde)
        self.P_tilde = 0.5 * (self.P_tilde + self.P_tilde.T)

        t1 = time.perf_counter()
        if self.record_history:
            self.step_times.append(t1 - t0)
        self.samples_seen += 1

        return {
            "updated": True,
            "conflict_detected": is_conflict,
            "conflict_info": conflict_info,
            "safeguard_error": self.evaluate_safeguard_error()
        }

    def evaluate_safeguard_error(self) -> float:
        """Computes relative Frobenius error on safeguard targets."""
        if self.K_safe is None or self.num_safeguards == 0:
            return 0.0
        with torch.no_grad():
            diff = self.W @ self.K_safe - self.V_safe
            norm_v = torch.norm(self.V_safe, p="fro").item()
            norm_diff = torch.norm(diff, p="fro").item()
            return norm_diff / norm_v if norm_v > 0 else norm_diff

    def get_diagnostics(self) -> Dict[str, Any]:
        """Audits symmetry, eigenvalues, condition number, and trace of P_tilde."""
        if self.d_red == 0:
            return {
                "symmetry_error": 0.0,
                "min_eigenvalue": 0.0,
                "max_eigenvalue": 0.0,
                "condition_number": 1.0,
                "trace": 0.0,
                "is_positive_definite": True
            }
        with torch.no_grad():
            norm_p = torch.norm(self.P_tilde, p="fro").item()
            sym_err = (torch.norm(self.P_tilde - self.P_tilde.T, p="fro").item() / norm_p) if norm_p > 0 else 0.0
            eigenvalues = torch.linalg.eigvalsh(self.P_tilde)
            min_eig = eigenvalues[0].item()
            max_eig = eigenvalues[-1].item()
            cond_num = max_eig / min_eig if min_eig > 0 else float("inf")
            trace = torch.trace(self.P_tilde).item()
            is_pos_def = bool(min_eig > 0)
        return {
            "symmetry_error": sym_err,
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "condition_number": cond_num,
            "trace": trace,
            "is_positive_definite": is_pos_def,
            "safeguard_relative_error": self.evaluate_safeguard_error()
        }


def compute_independent_batch_kkt(
    K: torch.Tensor,
    V: torch.Tensor,
    W_0: torch.Tensor,
    K_safe: Optional[torch.Tensor],
    lambda_reg: float
) -> torch.Tensor:
    """
    Independent batch reference solver using Lagrangian KKT formulation with Schur complement.
    Uses SVD-based pseudo-inverse for Schur complement inversion in rank-deficient cases.
    """
    d_in = K.shape[0]
    dtype = K.dtype
    device = K.device

    A = K @ K.T + lambda_reg * torch.eye(d_in, dtype=dtype, device=device)
    A_inv = torch.linalg.inv(A)

    E_0 = V - W_0 @ K
    B = E_0 @ K.T

    if K_safe is None or K_safe.numel() == 0:
        return W_0 + B @ A_inv

    K_s = K_safe.to(dtype=dtype, device=device)
    S = K_s.T @ A_inv @ K_s
    S_pinv = torch.linalg.pinv(S)

    Proj = torch.eye(d_in, dtype=dtype, device=device) - K_s @ S_pinv @ K_s.T @ A_inv
    Delta_W = B @ A_inv @ Proj
    return W_0 + Delta_W


# =========================================================================
# SYNTHETIC DATASET GENERATORS FOR AUDIT SCENARIOS
# =========================================================================

def generate_safeguards_exact_duplicate(d_in: int, m_indep: int, n_dups: int, seed: int = 42, dtype=torch.float64):
    """Generates safeguards with exact duplicated columns at various positions."""
    g = torch.Generator().manual_seed(seed)
    K_indep = torch.randn(d_in, m_indep, generator=g, dtype=dtype)
    cols = [K_indep[:, [i]] for i in range(m_indep)]
    for i in range(n_dups):
        dup_target = i % m_indep
        cols.insert((i * 2 + 1) % len(cols), K_indep[:, [dup_target]])
    K_safe = torch.cat(cols, dim=1)
    return K_safe, m_indep


def generate_safeguards_nearly_duplicate(d_in: int, m_indep: int, delta: float, seed: int = 42, dtype=torch.float64):
    """Generates safeguards with nearly duplicated vectors separated by perturbation delta."""
    g = torch.Generator().manual_seed(seed)
    K_base = torch.randn(d_in, m_indep, generator=g, dtype=dtype)
    noise = torch.randn(d_in, 1, generator=g, dtype=dtype)
    noise = noise / torch.norm(noise)
    col_perturbed = K_base[:, [0]] + delta * noise
    K_safe = torch.cat([K_base, col_perturbed], dim=1)
    return K_safe


def generate_safeguards_controlled_spectrum(d_in: int, m: int, cond_target: float, seed: int = 42, dtype=torch.float64):
    """Generates safeguard matrix with exact controlled singular value condition number."""
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(d_in, m, generator=g, dtype=dtype))
    V, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=dtype))
    exponents = torch.linspace(0, math.log10(cond_target), m, dtype=dtype)
    s = 10.0 ** (-exponents)
    S_mat = torch.diag(s)
    K_safe = U @ S_mat @ V.T
    return K_safe, s


def generate_safeguards_zero_vector(d_in: int, m_indep: int, seed: int = 42, dtype=torch.float64):
    """Generates safeguards containing zero vectors at column 0 and middle column."""
    g = torch.Generator().manual_seed(seed)
    K_indep = torch.randn(d_in, m_indep, generator=g, dtype=dtype)
    K_safe = torch.cat([torch.zeros(d_in, 1, dtype=dtype), K_indep], dim=1)
    mid = K_safe.shape[1] // 2
    K_safe[:, mid] = 0.0
    return K_safe


def generate_safeguards_extreme_magnitudes(d_in: int, seed: int = 42, dtype=torch.float64):
    """Generates safeguards with norm disparities spanning 10^-6 to 10^+6."""
    g = torch.Generator().manual_seed(seed)
    K_base = torch.randn(d_in, 6, generator=g, dtype=dtype)
    scales = torch.tensor([1e-6, 1e-3, 1.0, 10.0, 1e3, 1e6], dtype=dtype)
    K_safe = K_base / torch.norm(K_base, dim=0, keepdim=True) * scales
    return K_safe


# =========================================================================
# AUDIT TEST SUITES: R1 THROUGH R8
# =========================================================================

def run_test_r1_exact_duplicates() -> Dict[str, Any]:
    """R1: Exact Duplicate Safeguards Audit."""
    d_in = 32
    K_safe, true_rank = generate_safeguards_exact_duplicate(d_in, m_indep=4, n_dups=3, seed=101)
    
    # Audit Original QR vs SVD
    qr_rank, qr_U_par, qr_U_perp = OriginalQRNullSpace.compute(K_safe, dtype=torch.float64)
    svd_rank, svd_U_par, svd_U_perp, _ = SVDNullSpace.compute(K_safe, dtype=torch.float64)
    
    qr_orth_err = torch.norm(qr_U_perp.T @ K_safe).item()
    svd_orth_err = torch.norm(svd_U_perp.T @ K_safe).item()
    
    svd_self_orth = torch.norm(svd_U_perp.T @ svd_U_perp - torch.eye(svd_U_perp.shape[1], dtype=torch.float64)).item()
    qr_self_orth = torch.norm(qr_U_perp.T @ qr_U_perp - torch.eye(qr_U_perp.shape[1], dtype=torch.float64)).item()

    passed = (svd_rank == true_rank) and (svd_orth_err < 1e-13) and (svd_self_orth < 1e-13)
    return {
        "test_id": "R1",
        "description": "Exact duplicate safeguard vectors audit",
        "total_columns": K_safe.shape[1],
        "true_rank": true_rank,
        "qr_detected_rank": qr_rank,
        "qr_null_space_leakage": qr_orth_err,
        "qr_basis_orthogonality": qr_self_orth,
        "svd_detected_rank": svd_rank,
        "svd_null_space_leakage": svd_orth_err,
        "svd_basis_orthogonality": svd_self_orth,
        "qr_failure_identified": bool(qr_orth_err > 1e-3 or qr_rank != true_rank),
        "svd_robustness_verified": passed,
        "status": "PASSED" if passed else "FAILED"
    }


def run_test_r2_nearly_dependent() -> Dict[str, Any]:
    """R2: Nearly Dependent Safeguards Audit across perturbation scales."""
    d_in = 32
    deltas = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-12]
    results = []

    for delta in deltas:
        K_safe = generate_safeguards_nearly_duplicate(d_in, m_indep=4, delta=delta, seed=202)
        cond = torch.linalg.cond(K_safe).item()
        qr_rank, _, qr_U_perp = OriginalQRNullSpace.compute(K_safe)
        svd_rank, _, svd_U_perp, S = SVDNullSpace.compute(K_safe)
        
        qr_leak = torch.norm(qr_U_perp.T @ K_safe).item()
        svd_leak = torch.norm(svd_U_perp.T @ K_safe).item()

        results.append({
            "delta": delta,
            "condition_number": cond,
            "smallest_singular_value": S[-1].item(),
            "qr_rank": qr_rank,
            "qr_leak": qr_leak,
            "svd_rank": svd_rank,
            "svd_leak": svd_leak
        })

    all_svd_leaks_bounded = all(r["svd_leak"] < 1e-13 for r in results)
    return {
        "test_id": "R2",
        "description": "Nearly dependent safeguards audit across condition spectrum",
        "sweep_results": results,
        "all_svd_bounded": all_svd_leaks_bounded,
        "status": "PASSED" if all_svd_leaks_bounded else "FAILED"
    }


def run_test_r3_controlled_spectrum() -> Dict[str, Any]:
    """R3: Controlled Singular-Value Spectrum Audit."""
    d_in = 48
    m = 12
    targets = [1e1, 1e3, 1e6, 1e9, 1e12, 1e14]
    sweep = []

    for cond_tgt in targets:
        K_safe, s_vec = generate_safeguards_controlled_spectrum(d_in, m, cond_tgt, seed=303)
        actual_cond = torch.linalg.cond(K_safe).item()
        svd_rank, _, svd_U_perp, S = SVDNullSpace.compute(K_safe)
        leak = torch.norm(svd_U_perp.T @ K_safe).item()
        basis_orth = torch.norm(svd_U_perp.T @ svd_U_perp - torch.eye(svd_U_perp.shape[1], dtype=torch.float64)).item()

        sweep.append({
            "target_cond": cond_tgt,
            "actual_cond": actual_cond,
            "svd_rank": svd_rank,
            "null_space_leakage": leak,
            "basis_orthogonality": basis_orth
        })

    passed = all(r["null_space_leakage"] < 1e-12 and r["basis_orthogonality"] < 1e-12 for r in sweep)
    return {
        "test_id": "R3",
        "description": "Controlled singular value spectrum audit",
        "sweep": sweep,
        "status": "PASSED" if passed else "FAILED"
    }


def run_test_r4_float32_vs_float64() -> Dict[str, Any]:
    """R4: Float32 vs. Float64 Numerical Stability Audit."""
    d_in = 32
    m_indep = 4
    K_dup_f64, true_rank = generate_safeguards_exact_duplicate(d_in, m_indep, n_dups=2, seed=404, dtype=torch.float64)
    K_dup_f32 = K_dup_f64.to(torch.float32)

    # 1. Float64
    svd_r_f64, _, U_perp_f64, _ = SVDNullSpace.compute(K_dup_f64, dtype=torch.float64)
    leak_f64 = torch.norm(U_perp_f64.T @ K_dup_f64).item()
    orth_f64 = torch.norm(U_perp_f64.T @ U_perp_f64 - torch.eye(U_perp_f64.shape[1], dtype=torch.float64)).item()

    # 2. Float32 (Original QR vs Dtype-Aware SVD)
    qr_r_f32, _, qr_U_perp_f32 = OriginalQRNullSpace.compute(K_dup_f32, dtype=torch.float32)
    svd_r_f32, _, U_perp_f32, _ = SVDNullSpace.compute(K_dup_f32, dtype=torch.float32)
    leak_f32 = torch.norm(U_perp_f32.T @ K_dup_f32).item()
    orth_f32 = torch.norm(U_perp_f32.T @ U_perp_f32 - torch.eye(U_perp_f32.shape[1], dtype=torch.float32)).item()

    f64_pass = (svd_r_f64 == true_rank) and (leak_f64 < 1e-12) and (orth_f64 < 1e-12)
    f32_pass = (svd_r_f32 == true_rank) and (leak_f32 < 1e-5) and (orth_f32 < 1e-5)

    return {
        "test_id": "R4",
        "description": "Float32 vs Float64 precision and rank agreement audit",
        "true_rank": true_rank,
        "float64": {
            "detected_rank": svd_r_f64,
            "null_leakage": leak_f64,
            "orthogonality": orth_f64,
            "tolerance_bound": 1e-12,
            "passed": f64_pass
        },
        "float32": {
            "qr_detected_rank": qr_r_f32,
            "svd_detected_rank": svd_r_f32,
            "null_leakage": leak_f32,
            "orthogonality": orth_f32,
            "tolerance_bound": 1e-5,
            "passed": f32_pass
        },
        "status": "PASSED" if (f64_pass and f32_pass) else "FAILED"
    }


def run_test_r5_large_safeguard_count() -> Dict[str, Any]:
    """R5: Large Safeguard Count Audit (m approaching d_in)."""
    d_in = 64
    d_out = 32
    counts = [16, 32, 48, 60, 63, 64]
    results = []

    for m in counts:
        g = torch.Generator().manual_seed(505 + m)
        K_safe = torch.randn(d_in, m, generator=g, dtype=torch.float64)
        W_0 = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
        solver = AuditedConstrainedRLS(W_0, lambda_reg=0.1, K_safe=K_safe, dtype=torch.float64)

        # Update test
        k_stream = torch.randn(d_in, 1, generator=g, dtype=torch.float64)
        v_stream = torch.randn(d_out, 1, generator=g, dtype=torch.float64)
        up_res = solver.update(k_stream, v_stream)
        safe_err = solver.evaluate_safeguard_error()

        results.append({
            "m": m,
            "d_red": solver.d_red,
            "updated": up_res["updated"],
            "safeguard_error": safe_err,
            "is_edge_case_zero_red": (solver.d_red == 0)
        })

    all_passed = all(r["safeguard_error"] < 1e-12 for r in results)
    return {
        "test_id": "R5",
        "description": "High-dimensional safeguard counts (m -> d_in) audit",
        "sweep": results,
        "status": "PASSED" if all_passed else "FAILED"
    }


def run_test_r6_sequential_preservation() -> Dict[str, Any]:
    """R6: Sequential Constraint Preservation & Rejected Update Immutability."""
    d_in = 48
    d_out = 24
    m = 6
    N = 150
    g = torch.Generator().manual_seed(606)

    K_safe = torch.randn(d_in, m, generator=g, dtype=torch.float64)
    W_0 = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    solver = AuditedConstrainedRLS(W_0, lambda_reg=0.1, K_safe=K_safe, dtype=torch.float64)

    per_step_safe_errors = []
    per_step_null_leakages = []
    rejected_count = 0
    weights_identical_after_rejections = True

    for t in range(N):
        # Varying target magnitudes (1e-3 to 1e3)
        mag_scale = 10.0 ** ((t % 7) - 3)
        
        # Inject conflicting sample every 30 steps
        if t % 30 == 15:
            # Key exactly in safeguard span with inconsistent target
            alpha = torch.randn(m, 1, generator=g, dtype=torch.float64)
            k_t = K_safe @ alpha
            v_t = W_0 @ k_t + 10.0 * torch.randn(d_out, 1, generator=g, dtype=torch.float64)
            
            W_before = solver.W.clone()
            P_before = solver.P_tilde.clone()
            res = solver.update(k_t, v_t, reject_conflicts=True)
            
            if res["conflict_detected"] and not res["updated"]:
                rejected_count += 1
                if not (torch.equal(solver.W, W_before) and torch.equal(solver.P_tilde, P_before)):
                    weights_identical_after_rejections = False
        else:
            k_t = torch.randn(d_in, 1, generator=g, dtype=torch.float64)
            v_t = torch.randn(d_out, 1, generator=g, dtype=torch.float64) * mag_scale
            solver.update(k_t, v_t, reject_conflicts=True)

        # Check invariants after every step
        per_step_safe_errors.append(solver.evaluate_safeguard_error())
        per_step_null_leakages.append(torch.norm(solver.U_perp.T @ K_safe).item())

    max_safe_err = max(per_step_safe_errors)
    max_null_leak = max(per_step_null_leakages)
    passed = (max_safe_err < 1e-12) and (max_null_leak < 1e-13) and weights_identical_after_rejections and (rejected_count > 0)

    return {
        "test_id": "R6",
        "description": "Per-step sequential safeguard invariance and state immutability audit",
        "total_steps": N,
        "max_per_step_safeguard_error": max_safe_err,
        "max_per_step_null_space_leakage": max_null_leak,
        "injected_conflicts_rejected": rejected_count,
        "state_strictly_immutable_on_rejection": weights_identical_after_rejections,
        "status": "PASSED" if passed else "FAILED"
    }


def run_test_r7_conflict_detection() -> Dict[str, Any]:
    """R7: Conflict Detection Robustness across 6 key categories."""
    d_in = 48
    d_out = 24
    m = 6
    g = torch.Generator().manual_seed(707)

    K_safe = torch.randn(d_in, m, generator=g, dtype=torch.float64)
    W_0 = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    solver = AuditedConstrainedRLS(W_0, lambda_reg=0.1, K_safe=K_safe, dtype=torch.float64, conflict_tol=1e-5)

    test_categories = {
        "1_exact_inside_inconsistent": [],
        "2_near_inside_inconsistent": [],
        "3_exact_outside": [],
        "4_partial_overlap_50pct": [],
        "5_partial_overlap_90pct": [],
        "6_exact_inside_consistent": []
    }

    # 1. Exact inside span, inconsistent target (MUST REJECT)
    for _ in range(10):
        a = torch.randn(m, 1, generator=g, dtype=torch.float64)
        k = K_safe @ a
        v = W_0 @ k + 5.0 * torch.randn(d_out, 1, generator=g, dtype=torch.float64)
        is_conf, _ = solver.detect_conflict(k, v)
        test_categories["1_exact_inside_inconsistent"].append(is_conf)

    # 2. Near inside span (angle 1e-6), inconsistent target (MUST REJECT)
    for _ in range(10):
        a = torch.randn(m, 1, generator=g, dtype=torch.float64)
        k_in = K_safe @ a
        k_noise = 1e-6 * torch.randn(d_in, 1, generator=g, dtype=torch.float64)
        k = k_in + k_noise
        v = W_0 @ k + 5.0 * torch.randn(d_out, 1, generator=g, dtype=torch.float64)
        is_conf, _ = solver.detect_conflict(k, v)
        test_categories["2_near_inside_inconsistent"].append(is_conf)

    # 3. Exact outside span (orthogonal), new target (MUST ACCEPT)
    for _ in range(10):
        k_raw = torch.randn(d_in, 1, generator=g, dtype=torch.float64)
        k = solver.U_perp @ (solver.U_perp.T @ k_raw)
        v = torch.randn(d_out, 1, generator=g, dtype=torch.float64)
        is_conf, _ = solver.detect_conflict(k, v)
        test_categories["3_exact_outside"].append(is_conf)

    # 4. Partial overlap (50% in span, 50% outside) (MUST ACCEPT - has permissible dimension)
    for _ in range(10):
        k_par = solver.U_par @ torch.randn(solver.rank, 1, generator=g, dtype=torch.float64)
        k_perp = solver.U_perp @ torch.randn(solver.d_red, 1, generator=g, dtype=torch.float64)
        k = k_par / torch.norm(k_par) + k_perp / torch.norm(k_perp)
        v = torch.randn(d_out, 1, generator=g, dtype=torch.float64)
        is_conf, _ = solver.detect_conflict(k, v)
        test_categories["4_partial_overlap_50pct"].append(is_conf)

    # 5. Partial overlap (90% in span, 10% outside) (MUST ACCEPT - perp_ratio ~ 0.1 > 1e-5)
    for _ in range(10):
        k_par = solver.U_par @ torch.randn(solver.rank, 1, generator=g, dtype=torch.float64)
        k_perp = solver.U_perp @ torch.randn(solver.d_red, 1, generator=g, dtype=torch.float64)
        k = 0.9 * (k_par / torch.norm(k_par)) + 0.1 * (k_perp / torch.norm(k_perp))
        v = torch.randn(d_out, 1, generator=g, dtype=torch.float64)
        is_conf, _ = solver.detect_conflict(k, v)
        test_categories["5_partial_overlap_90pct"].append(is_conf)

    # 6. Exact inside span, CONSISTENT target (v == W_0 @ k) (MUST ACCEPT)
    for _ in range(10):
        a = torch.randn(m, 1, generator=g, dtype=torch.float64)
        k = K_safe @ a
        v = W_0 @ k
        is_conf, _ = solver.detect_conflict(k, v)
        test_categories["6_exact_inside_consistent"].append(is_conf)

    # Compute false positives and false negatives
    # Category 1 & 2 should be True (conflicts detected)
    fn_1 = test_categories["1_exact_inside_inconsistent"].count(False)
    fn_2 = test_categories["2_near_inside_inconsistent"].count(False)
    
    # Category 3, 4, 5, 6 should be False (no false rejection)
    fp_3 = test_categories["3_exact_outside"].count(True)
    fp_4 = test_categories["4_partial_overlap_50pct"].count(True)
    fp_5 = test_categories["5_partial_overlap_90pct"].count(True)
    fp_6 = test_categories["6_exact_inside_consistent"].count(True)

    passed = (fn_1 == 0 and fn_2 == 0 and fp_3 == 0 and fp_4 == 0 and fp_5 == 0 and fp_6 == 0)
    return {
        "test_id": "R7",
        "description": "Conflict detection accuracy across 6 structural key categories",
        "conflict_threshold": solver.conflict_tol,
        "results": {
            "exact_inside_inconsistent_detected_rate": (10 - fn_1) / 10.0,
            "near_inside_inconsistent_detected_rate": (10 - fn_2) / 10.0,
            "exact_outside_false_positive_rate": fp_3 / 10.0,
            "partial_50pct_false_positive_rate": fp_4 / 10.0,
            "partial_90pct_false_positive_rate": fp_5 / 10.0,
            "exact_inside_consistent_false_positive_rate": fp_6 / 10.0,
        },
        "total_false_positives": fp_3 + fp_4 + fp_5 + fp_6,
        "total_false_negatives": fn_1 + fn_2,
        "status": "PASSED" if passed else "FAILED"
    }


def run_test_r8_batch_equivalence() -> Dict[str, Any]:
    """R8: Online vs. Independent Batch KKT Solver under rank deficiency and permutations."""
    d_in = 48
    d_out = 24
    N = 80
    g = torch.Generator().manual_seed(808)

    # Rank-deficient safeguards
    K_safe, true_rank = generate_safeguards_exact_duplicate(d_in, m_indep=4, n_dups=2, seed=808)
    W_0 = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    lambda_reg = 0.05

    # Streaming dataset
    K_train = torch.randn(d_in, N, generator=g, dtype=torch.float64)
    V_train = torch.randn(d_out, N, generator=g, dtype=torch.float64)
    
    # Held-out test set
    K_test = torch.randn(d_in, 40, generator=g, dtype=torch.float64)
    V_test = torch.randn(d_out, 40, generator=g, dtype=torch.float64)

    # 1. Standard order online RLS
    solver_standard = AuditedConstrainedRLS(W_0, lambda_reg=lambda_reg, K_safe=K_safe, dtype=torch.float64)
    for t in range(N):
        solver_standard.update(K_train[:, t], V_train[:, t])
    W_online_std = solver_standard.W

    # 2. Permuted order online RLS
    perm = torch.randperm(N, generator=g)
    solver_permuted = AuditedConstrainedRLS(W_0, lambda_reg=lambda_reg, K_safe=K_safe, dtype=torch.float64)
    for idx in perm:
        solver_permuted.update(K_train[:, idx], V_train[:, idx])
    W_online_perm = solver_permuted.W

    # 3. Independent Batch KKT Solver
    W_batch = compute_independent_batch_kkt(K_train, V_train, W_0, K_safe, lambda_reg=lambda_reg)

    # Errors
    err_std_to_batch = (torch.norm(W_online_std - W_batch, p="fro") / torch.norm(W_batch, p="fro")).item()
    err_perm_to_batch = (torch.norm(W_online_perm - W_batch, p="fro") / torch.norm(W_batch, p="fro")).item()
    err_std_to_perm = (torch.norm(W_online_std - W_online_perm, p="fro") / torch.norm(W_batch, p="fro")).item()

    # Held-out prediction MSE comparison
    pred_online = W_online_std @ K_test
    pred_batch = W_batch @ K_test
    mse_diff = torch.mean((pred_online - pred_batch) ** 2).item()

    # Safeguard preservation check
    safe_err_online = solver_standard.evaluate_safeguard_error()
    safe_err_batch = (torch.norm(W_batch @ K_safe - W_0 @ K_safe, p="fro") / torch.norm(W_0 @ K_safe, p="fro")).item()

    passed = (err_std_to_batch < 1e-12) and (err_perm_to_batch < 1e-12) and (mse_diff < 1e-24) and (safe_err_online < 1e-12)
    return {
        "test_id": "R8",
        "description": "Online vs. independent batch KKT equivalence under rank deficiency and permutation",
        "rel_frobenius_std_to_batch": err_std_to_batch,
        "rel_frobenius_perm_to_batch": err_perm_to_batch,
        "rel_frobenius_std_to_perm": err_std_to_perm,
        "held_out_prediction_mse_difference": mse_diff,
        "online_safeguard_error": safe_err_online,
        "batch_safeguard_error": safe_err_batch,
        "order_invariance_verified": bool(err_std_to_perm < 1e-12),
        "status": "PASSED" if passed else "FAILED"
    }

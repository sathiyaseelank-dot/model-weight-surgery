"""
Constrained Forward-Only Recursive Least Squares (RLS) Module
=============================================================
Provides mathematically exact online RLS implementations:
  1. BaselineRLS: Standard online RLS with Sherman-Morrison rank-1 update.
  2. ConstrainedRLS: Recursive constrained least squares via null-space parameterization,
     guaranteeing exact safeguard preservation without sample buffering or replay.
  3. BatchConstrainedSolver: Independent batch reference solver via KKT Lagrange multipliers.
"""

import time
import math
import torch
from typing import Optional, Dict, Tuple, Any, List


class BaselineRLS:
    """
    Standard online Recursive Least Squares using the Sherman-Morrison rank-1 update.
    Strictly forward-only: does not store or replay past samples.
    """
    def __init__(self, initial_weight: torch.Tensor, lambda_reg: float, dtype=torch.float64, device="cpu"):
        self.device = device
        self.dtype = dtype
        self.lambda_reg = float(lambda_reg)
        self.d_out, self.d_in = initial_weight.shape
        
        self.W = initial_weight.clone().to(dtype=dtype, device=device)
        self.P = (torch.eye(self.d_in, dtype=dtype, device=device) / self.lambda_reg)
        
        self.samples_seen = 0
        self.step_times = []
        self._history_buffer = None # Verified: zero historical sample storage

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """
        Updates W and P using the Sherman-Morrison lemma.
        k: (d_in, 1) or (d_in,)
        v: (d_out, 1) or (d_out,)
        """
        t0 = time.perf_counter()
        k_col = k.view(self.d_in, 1).to(dtype=self.dtype, device=self.device)
        v_col = v.view(self.d_out, 1).to(dtype=self.dtype, device=self.device)

        # 1. Gain calculation using old P
        Pk = self.P @ k_col
        denom = 1.0 + (k_col.T @ Pk).item()
        gain = Pk / denom

        # 2. Prior innovation (prediction error before update)
        prior_error = v_col - (self.W @ k_col)

        # 3. Update W using the gain calculated from old P
        self.W = self.W + prior_error @ gain.T

        # 4. Update covariance matrix P
        self.P = self.P - gain @ (k_col.T @ self.P)

        # 5. Symmetrize P for numerical stability
        self.P = 0.5 * (self.P + self.P.T)

        t1 = time.perf_counter()
        self.step_times.append(t1 - t0)
        self.samples_seen += 1

        del k_col, v_col, Pk, gain, prior_error

    def get_diagnostics(self) -> Dict[str, Any]:
        """Audits symmetry, eigenvalues, condition number, and trace of P."""
        with torch.no_grad():
            sym_err = (torch.norm(self.P - self.P.T, p="fro") / torch.norm(self.P, p="fro")).item()
            eigenvalues = torch.linalg.eigvalsh(self.P)
            min_eig = eigenvalues[0].item()
            max_eig = eigenvalues[-1].item()
            cond_num = max_eig / min_eig if min_eig > 0 else float("inf")
            trace = torch.trace(self.P).item()
            is_pos_def = bool(min_eig > 0)
        return {
            "symmetry_error": sym_err,
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "condition_number": cond_num,
            "trace": trace,
            "is_positive_definite": is_pos_def
        }


class ConstrainedRLS:
    """
    Constrained Online Recursive Least Squares.
    
    Mathematical Formulation:
      Given initial weights W_0 and safeguard keys K_safe (d_in x m),
      any update must satisfy the hard constraint:
        Delta W @ K_safe = 0  <=>  W @ K_safe = W_0 @ K_safe
      
      We compute the null-space basis U_perp of K_safe^T via complete QR decomposition:
        K_safe = [U_par, U_perp] [R; 0]
      Any permissible weight perturbation is uniquely parameterized as:
        Delta W = W_tilde @ U_perp^T
      
      In the reduced coordinate space R^(d_in - r):
        - Reduced keys: k_tilde = U_perp^T @ k
        - Reduced target innovation: v_tilde = v - W_0 @ k
        - Reduced covariance P_tilde updates via Sherman-Morrison.
      
      This guarantees:
        1. W_t @ K_safe == W_0 @ K_safe to machine precision after every single update.
        2. Covariance matrix P_tilde updates dynamically within the permitted update space.
        3. Zero historical training samples are stored or replayed.
        4. Conflicting constraints (k in span(K_safe) but v != W_0 @ k) are actively detected and reported.
    """
    def __init__(
        self,
        initial_weight: torch.Tensor,
        lambda_reg: float,
        K_safe: Optional[torch.Tensor] = None,
        dtype=torch.float64,
        device="cpu",
        conflict_tol: float = 1e-5
    ):
        self.device = device
        self.dtype = dtype
        self.lambda_reg = float(lambda_reg)
        self.conflict_tol = float(conflict_tol)
        self.d_out, self.d_in = initial_weight.shape
        
        self.W_0 = initial_weight.clone().to(dtype=dtype, device=device)
        self.samples_seen = 0
        self.step_times = []
        self._history_buffer = None # Verified: zero historical sample storage
        self.conflict_events = []
        
        # Setup Null-Space Parameterization
        if K_safe is not None and K_safe.numel() > 0:
            self.K_safe = K_safe.clone().to(dtype=dtype, device=device)
            self.num_safeguards = self.K_safe.shape[1]
            self.V_safe = self.W_0 @ self.K_safe
            
            # Orthonormal basis via QR decomposition
            Q, R = torch.linalg.qr(self.K_safe, mode="complete")
            # Determine effective numerical rank
            diag_R = torch.abs(torch.diag(R))
            threshold = 1e-10 * torch.max(diag_R).item() if diag_R.numel() > 0 else 1e-10
            rank = int((diag_R > threshold).sum().item())
            
            self.rank = rank
            self.U_par = Q[:, :rank] # (d_in, rank) - span of safeguards
            self.U_perp = Q[:, rank:] # (d_in, d_in - rank) - permitted update space
            self.d_red = self.d_in - rank
        else:
            self.K_safe = None
            self.num_safeguards = 0
            self.V_safe = None
            self.rank = 0
            self.U_par = None
            self.U_perp = torch.eye(self.d_in, dtype=dtype, device=device)
            self.d_red = self.d_in

        # Initialize reduced parameter space
        self.W_tilde = torch.zeros(self.d_out, self.d_red, dtype=dtype, device=device)
        self.P_tilde = torch.eye(self.d_red, dtype=dtype, device=device) / self.lambda_reg

    @property
    def W(self) -> torch.Tensor:
        """Full ambient weight matrix W = W_0 + W_tilde @ U_perp^T."""
        return self.W_0 + self.W_tilde @ self.U_perp.T

    @property
    def P(self) -> torch.Tensor:
        """Full ambient covariance projection P = U_perp @ P_tilde @ U_perp^T."""
        return self.U_perp @ self.P_tilde @ self.U_perp.T

    def detect_conflict(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Detects if training sample (k, v) conflicts with the protected safeguard mappings.
        Conflict occurs when k lies almost entirely in span(K_safe) while v != W_0 @ k.
        """
        if self.U_par is None or self.rank == 0:
            return False, None

        k_col = k.view(self.d_in, 1).to(dtype=self.dtype, device=self.device)
        v_col = v.view(self.d_out, 1).to(dtype=self.dtype, device=self.device)

        k_norm = torch.norm(k_col).item()
        if k_norm < 1e-12:
            return False, None

        # Orthogonal component norm in permitted space
        k_perp = self.U_perp.T @ k_col
        perp_ratio = torch.norm(k_perp).item() / k_norm

        # Expected target under safeguard consistency: v_expected = W_0 @ k
        v_expected = self.W_0 @ k_col
        target_error = (torch.norm(v_col - v_expected).item() / 
                        (torch.norm(v_expected).item() + 1e-12))

        # Irreconcilable conflict condition:
        # Key has minimal projection in update space (perp_ratio < tol)
        # but requested target diverges significantly from W_0 @ k
        is_conflict = (perp_ratio < self.conflict_tol and target_error > self.conflict_tol)
        
        conflict_info = None
        if is_conflict:
            conflict_info = {
                "sample_index": self.samples_seen,
                "null_space_residual_ratio": perp_ratio,
                "target_discrepancy": target_error,
                "conflict_description": (
                    f"Sample key lies { (1.0 - perp_ratio)*100:.2f}% in the protected safeguard subspace, "
                    f"but requested target diverges by {target_error:.4e} from protected mapping W_0 @ k."
                )
            }
        return is_conflict, conflict_info

    def update(self, k: torch.Tensor, v: torch.Tensor, reject_conflicts: bool = True) -> Dict[str, Any]:
        """
        Performs a constrained forward-only RLS update.
        k: (d_in, 1) or (d_in,)
        v: (d_out, 1) or (d_out,)
        """
        t0 = time.perf_counter()
        k_col = k.view(self.d_in, 1).to(dtype=self.dtype, device=self.device)
        v_col = v.view(self.d_out, 1).to(dtype=self.dtype, device=self.device)

        # 1. Conflict detection
        is_conflict, conflict_info = self.detect_conflict(k_col, v_col)
        if is_conflict:
            self.conflict_events.append(conflict_info)
            if reject_conflicts:
                t1 = time.perf_counter()
                self.step_times.append(t1 - t0)
                self.samples_seen += 1
                return {
                    "updated": False,
                    "conflict_detected": True,
                    "conflict_info": conflict_info,
                    "safeguard_error": self.evaluate_safeguard_error()
                }

        # 2. Project into reduced coordinate space
        k_tilde = self.U_perp.T @ k_col # (d_red, 1)
        v_tilde = v_col - (self.W_0 @ k_col) # (d_out, 1)

        # 3. Calculate gain vector in reduced space using old P_tilde
        Pk_tilde = self.P_tilde @ k_tilde # (d_red, 1)
        denom = 1.0 + (k_tilde.T @ Pk_tilde).item()
        gain_tilde = Pk_tilde / denom # (d_red, 1)

        # 4. Compute prior innovation in reduced space
        prior_error_tilde = v_tilde - (self.W_tilde @ k_tilde) # (d_out, 1)

        # 5. Update reduced weight matrix
        self.W_tilde = self.W_tilde + prior_error_tilde @ gain_tilde.T

        # 6. Update reduced inverse covariance matrix P_tilde
        self.P_tilde = self.P_tilde - gain_tilde @ (k_tilde.T @ self.P_tilde)

        # 7. Symmetrize P_tilde for numerical stability
        self.P_tilde = 0.5 * (self.P_tilde + self.P_tilde.T)

        t1 = time.perf_counter()
        self.step_times.append(t1 - t0)
        self.samples_seen += 1

        del k_col, v_col, k_tilde, v_tilde, Pk_tilde, gain_tilde, prior_error_tilde

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
            W_current = self.W
            diff = W_current @ self.K_safe - self.V_safe
            norm_v = torch.norm(self.V_safe, p="fro").item()
            norm_diff = torch.norm(diff, p="fro").item()
            return norm_diff / norm_v if norm_v > 0 else norm_diff

    def get_diagnostics(self) -> Dict[str, Any]:
        """Audits symmetry, eigenvalues, condition number, and trace of P_tilde."""
        with torch.no_grad():
            sym_err = (torch.norm(self.P_tilde - self.P_tilde.T, p="fro") / 
                       torch.norm(self.P_tilde, p="fro")).item()
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
            "conflict_count": len(self.conflict_events),
            "safeguard_relative_error": self.evaluate_safeguard_error()
        }


def compute_batch_constrained_solution(
    K: torch.Tensor,
    V: torch.Tensor,
    W_0: torch.Tensor,
    K_safe: Optional[torch.Tensor],
    lambda_reg: float
) -> torch.Tensor:
    """
    Independently solves the batch constrained ridge regression problem via KKT Lagrange Multipliers:
      minimize 0.5 * sum_i ||W k_i - v_i||^2 + 0.5 * lambda * ||W - W_0||_F^2
      subject to W @ K_safe = W_0 @ K_safe (Delta W @ K_safe = 0)
    
    KKT Derivation:
      Let Delta W = W - W_0, E_0 = V - W_0 @ K, A = K @ K^T + lambda * I.
      Lagrangian: L(Delta W, M) = 0.5 * Tr(Delta W A Delta W^T) - Tr(Delta W K E_0^T) + Tr(M Delta W K_safe)
      Stationarity: Delta W A - E_0 K^T + M K_safe^T = 0
                    Delta W = (E_0 K^T - M K_safe^T) A^(-1)
      Constraint:   Delta W @ K_safe = 0
                    (E_0 K^T A^(-1) - M K_safe^T A^(-1)) @ K_safe = 0
                    M (K_safe^T A^(-1) K_safe) = E_0 K^T A^(-1) K_safe
      Let Schur complement S = K_safe^T A^(-1) K_safe.
      M = E_0 K^T A^(-1) K_safe S^(-1)
      Delta W = E_0 K^T A^(-1) [I - K_safe S^(-1) K_safe^T A^(-1)]
      W_batch = W_0 + Delta W
    """
    d_in = K.shape[0]
    dtype = K.dtype
    device = K.device

    # Gram matrix: A = K @ K^T + lambda * I
    A = K @ K.T + lambda_reg * torch.eye(d_in, dtype=dtype, device=device)
    A_inv = torch.linalg.inv(A)

    E_0 = V - W_0 @ K
    B = E_0 @ K.T # (d_out, d_in)

    if K_safe is None or K_safe.numel() == 0:
        # Unconstrained standard ridge regression
        Delta_W = B @ A_inv
        return W_0 + Delta_W

    K_s = K_safe.to(dtype=dtype, device=device)
    
    # Schur complement: S = K_s^T @ A_inv @ K_s
    S = K_s.T @ A_inv @ K_s
    
    # Use pseudo-inverse or regularized solve for S in case of rank-deficient safeguards
    S_inv = torch.linalg.pinv(S)

    # KKT Projector onto constraint manifold
    Proj_constraint = torch.eye(d_in, dtype=dtype, device=device) - K_s @ S_inv @ K_s.T @ A_inv
    Delta_W = B @ A_inv @ Proj_constraint

    return W_0 + Delta_W

# import time
# import torch

# # ==============================
# # Global configuration
# # ==============================

# LAMBDA = 1.0
# Z = 2.0
# NUM_INIT_FOR_EM = 10
# STEPS = 20


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# # ==============================
# # losses
# # ==============================

# def lp_loss(x: torch.Tensor) -> torch.Tensor:
#     """Lp loss with p = Z (default 2): |x|^Z / Z."""
#     return (x.abs() ** Z) / Z


# def huber_loss(x: torch.Tensor, delta: float = LAMBDA) -> torch.Tensor:
#     """Huber loss: quadratic near 0, linear past delta."""
#     abs_x = x.abs()
#     quad = 0.5 * x ** 2
#     lin = delta * (abs_x - 0.5 * delta)
#     return torch.where(abs_x <= delta, quad, lin)


# def cauchy_loss(x: torch.Tensor) -> torch.Tensor:
#     """Cauchy loss: smooth, saturating robust loss."""
#     return (LAMBDA ** 2 / 2.0) * torch.log1p((x ** 2) / (LAMBDA ** 2))


# def geman_McClure_loss(x: torch.Tensor) -> torch.Tensor:
#     """Geman-McClure loss: bounded influence robust loss."""
#     return (x ** 2) / (2.0 * (1.0 + x ** 2))


# def welsch_loss(x: torch.Tensor) -> torch.Tensor:
#     """Welsch loss: exponentially decaying influence for large residuals."""
#     return (LAMBDA ** 2 / 2.0) * (1.0 - torch.exp(-(x ** 2) / (LAMBDA ** 2)))


# def tukey_loss(x: torch.Tensor) -> torch.Tensor:
#     """Tukey biweight loss: zero influence beyond threshold LAMBDA."""
#     abs_x = x.abs()
#     r2 = (x ** 2) / (LAMBDA ** 2)
#     inner = 1.0 - r2
#     val = (LAMBDA ** 2 / 6.0) * (1.0 - inner ** 3)
#     const = (LAMBDA ** 2 / 6.0)
#     return torch.where(abs_x <= LAMBDA, val, torch.full_like(x, const))


# M_ESTIMATOR_FUNCS = {
#     "lp": lp_loss,
#     "huber": huber_loss,
#     "cauchy": cauchy_loss,
#     "geman_McClure": geman_McClure_loss,
#     "welsch": welsch_loss,
#     "tukey": tukey_loss,
# }

# # Default objective
# OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS["lp"]


# # ==============================
# # Geometry helpers
# # ==============================

# def computeDistanceToSubspace(
#     point: torch.Tensor,
#     X: torch.Tensor,
#     v: torch.Tensor,
#     jitter: float = 1e-6,
# ) -> torch.Tensor:
#     """
#     Compute Euclidean distance from point(s) to a J-dim affine subspace using
#     Cholesky-based projection.

#     Args:
#         point: (n, d) or (d,) tensor of points.
#         X:     (J, d) basis matrix whose rows span the subspace.
#         v:     (d,) translation vector of the affine subspace.
#         jitter: small positive value added to Gram matrix for stability.

#     Returns:
#         Tensor of shape (n,) with distances for each point
#         (or a scalar tensor if input was 1D).
#     """
#     if point.dim() == 1:
#         point = point.unsqueeze(0)
#         single = True
#     else:
#         single = False

#     # Move to same device/dtype
#     P = point.to(device=X.device, dtype=X.dtype)
#     v = v.to(device=X.device, dtype=X.dtype)

#     # Center points
#     Y = P - v                  # (n, d)
#     B = X                      # (J, d)
#     J_dim = B.size(0)

#     # Gram matrix with jitter: G = B B^T + eps I
#     G = B @ B.t()
#     G = G + jitter * torch.eye(J_dim, device=B.device, dtype=B.dtype)

#     # Cholesky factorization: G = L L^T
#     L = torch.linalg.cholesky(G)

#     # Solve (B B^T) alpha = B Y^T  for alpha
#     BY = B @ Y.t()                          # (J, n)
#     alpha = torch.cholesky_solve(BY, L)     # (J, n)

#     # Projected (centered) points: B^T alpha
#     proj_centered = (B.t() @ alpha).t()     # (n, d)

#     # Residuals and distances
#     residuals = Y - proj_centered
#     dists = torch.linalg.norm(residuals, dim=1)

#     if single:
#         return dists[0]
#     return dists


# def computeCost(
#     P: torch.Tensor,
#     w: torch.Tensor,
#     X: torch.Tensor,
#     v: torch.Tensor = None,
#     show_indices: bool = False,
#     jitter: float = 1e-6,
# ):
#     """
#     Compute generalized k-means cost with j-flats as centers, using
#     a chosen M-estimator objective on CUDA.

#     Args:
#         P: (n, d) points.
#         w: (n,) non-negative weights.
#         X:
#             - (J, d): single flat.
#             - (k, J, d): multiple flats.
#         v:
#             - (d,) for single flat.
#             - (k, d) for multiple flats.
#             - if None: zero translations.
#         show_indices:
#             if True and multiple flats, also return argmin flat indices.
#         jitter: jitter for Cholesky distance computation.

#     Returns:
#         Single flat:
#             total_cost, cost_per_point
#         Multiple flats, show_indices=False:
#             total_cost, cost_per_point
#         Multiple flats, show_indices=True:
#             total_cost, cost_per_point, indices
#     """
#     global OBJECTIVE_LOSS

#     P = P.to(DEVICE)
#     w = w.view(-1).to(DEVICE)
#     X = X.to(DEVICE)
#     if v is not None:
#         v = v.to(DEVICE)

#     if X.dim() == 2:
#         # Single subspace
#         if v is None:
#             v = torch.zeros(X.size(1), device=DEVICE, dtype=X.dtype)
#         dists = computeDistanceToSubspace(P, X, v, jitter=jitter)
#         losses = OBJECTIVE_LOSS(dists)
#         cost_per_point = w * losses
#         total_cost = cost_per_point.sum()
#         return total_cost, cost_per_point

#     # Multiple subspaces
#     k = X.size(0)
#     n = P.size(0)
#     if v is None:
#         v = torch.zeros(k, P.size(1), device=DEVICE, dtype=X.dtype)

#     temp_cost_per_point = torch.empty((n, k), device=DEVICE, dtype=P.dtype)

#     for i in range(k):
#         dists = computeDistanceToSubspace(P, X[i], v[i], jitter=jitter)
#         losses = OBJECTIVE_LOSS(dists)
#         temp_cost_per_point[:, i] = w * losses

#     cost_per_point, indices = temp_cost_per_point.min(dim=1)
#     total_cost = cost_per_point.sum()

#     if not show_indices:
#         return total_cost, cost_per_point
#     else:
#         return total_cost, cost_per_point, indices


# def computeSuboptimalSubspace(P: torch.Tensor, w: torch.Tensor, J: int):
#     """
#     Compute a suboptimal J-dim affine subspace via weighted PCA on CUDA.

#     Args:
#         P: (n, d) points.
#         w: (n,) non-negative weights.
#         J: target subspace dimension.

#     Returns:
#         X: (J, d) basis matrix (rows).
#         v: (d,) weighted mean.
#         elapsed: float, computation time in seconds.
#     """
#     start_time = time.time()

#     P = P.to(DEVICE)
#     w = w.view(-1).to(DEVICE)

#     # Normalize weights
#     w = w / w.sum()

#     # Weighted mean
#     v = (P * w.unsqueeze(1)).sum(dim=0)

#     # Centered
#     centered = P - v

#     # Weighted PCA via sqrt(w)
#     sqrt_w = torch.sqrt(w).unsqueeze(1)
#     centered_weighted = centered * sqrt_w

#     # SVD: centered_weighted = U S Vh
#     _, _, Vh = torch.linalg.svd(centered_weighted, full_matrices=False)

#     # First J right-singular vectors (rows of Vh) span the subspace
#     X = Vh[:J, :]

#     elapsed = time.time() - start_time
#     return X, v, elapsed


# def EMLikeAlg(
#     P: torch.Tensor,
#     w: torch.Tensor,
#     j: int,
#     k: int,
#     steps: int = STEPS,
#     objective: str = "lp",
#     jitter: float = 1e-6,
#     num_inits: int = NUM_INIT_FOR_EM,
#     patience: int = 5,
#     min_delta: float = 0.0,
# ):
#     """
#     EM-like heuristic for the (K, J)-projective clustering problem on GPU.

#     Procedure:
#       - Initialize k j-flats via random partition + weighted PCA.
#       - Repeat for `steps`:
#           E-step: assign each point to nearest flat under chosen M-estimator.
#           M-step: recompute each flat via weighted PCA of its assigned points.
#           (After each step, log total cost and cluster sizes.)
#       - Run NUM_INIT_FOR_EM random initializations and keep the best.

#     Args:
#         P: (n, d) points.
#         w: (n,) weights.
#         j: dimension of each flat.
#         k: number of flats.
#         steps: EM steps per initialization.
#         objective: key from M_ESTIMATOR_FUNCS.
#         jitter: jitter for Cholesky distance computation.

#     Returns:
#         Vs_best: (k, j, d) bases for best solution.
#         vs_best: (k, d) translations for best solution.
#         elapsed_total: float, total runtime.
#     """
#     global OBJECTIVE_LOSS

#     start_time = time.time()

#     # Move to DEVICE
#     P = P.to(DEVICE)
#     w = w.view(-1).to(DEVICE)

#     n, d = P.shape

#     # Select objective
#     OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS.get(objective, lp_loss)

#     best_cost = float("inf")
#     best_Vs = None
#     best_vs = None

#     for init in range(num_inits):
#         # Random partition of points into k initial clusters
#         perm = torch.randperm(n, device=DEVICE)
#         idx_splits = perm.chunk(k)

#         Vs = torch.empty((k, j, d), device=DEVICE, dtype=P.dtype)
#         vs = torch.empty((k, d), device=DEVICE, dtype=P.dtype)

#         # Initialize each flat via weighted PCA of its chunk
#         for i in range(k):
#             idx = idx_splits[i]
#             if idx.numel() == 0:
#                 # Rare edge-case: empty => fall back to global subspace
#                 X_i, v_i, _ = computeSuboptimalSubspace(P, w, j)
#                 Vs[i] = X_i
#                 vs[i] = v_i
#             else:
#                 X_i, v_i, _ = computeSuboptimalSubspace(P[idx], w[idx], j)
#                 Vs[i] = X_i
#                 vs[i] = v_i

#         # EM iterations with early stopping
#         best_step_cost = float("inf")
#         no_improve = 0
#         for step in range(steps):
#             # ---------- E-step ----------
#             dists_to_flats = torch.empty((n, k), device=DEVICE, dtype=P.dtype)
#             for flat_idx in range(k):
#                 dists = computeDistanceToSubspace(P, Vs[flat_idx], vs[flat_idx], jitter=jitter)
#                 dists_to_flats[:, flat_idx] = OBJECTIVE_LOSS(dists)

#             cluster_indices = torch.argmin(dists_to_flats, dim=1)

#             # Cluster sizes (for logging)
#             cluster_sizes = torch.bincount(cluster_indices, minlength=k).cpu().tolist()

#             # ---------- M-step ----------
#             unique_clusters = torch.unique(cluster_indices)
#             for flat_idx in unique_clusters:
#                 mask = (cluster_indices == flat_idx)
#                 idx_points = mask.nonzero(as_tuple=True)[0]
#                 if idx_points.numel() > 0:
#                     X_new, v_new, _ = computeSuboptimalSubspace(P[idx_points], w[idx_points], j)
#                     Vs[flat_idx] = X_new
#                     vs[flat_idx] = v_new

#             # ---------- Logging: cost after this EM step ----------
#             step_cost, _ = computeCost(P, w, Vs, vs, jitter=jitter)
#             step_cost_val = float(step_cost.item())
#             print(
#                 f"[init {init + 1}][step {step + 1}] "
#                 f"cost = {step_cost_val:.6f}, "
#                 f"cluster_sizes = {cluster_sizes}"
#             )
#             # ---------- Early stopping ----------
#             if best_step_cost == float("inf"):
#                 best_step_cost = step_cost_val
#                 no_improve = 0
#             else:
#                 # Check for improvement: cost must decrease
#                 if step_cost_val < best_step_cost:
#                     # relative improvement
#                     denom = max(best_step_cost, 1e-12)
#                     rel_impr = (best_step_cost - step_cost_val) / denom
#                     if rel_impr > min_delta:
#                         best_step_cost = step_cost_val
#                         no_improve = 0
#                     else:
#                         # Improved but below threshold - still count as no improvement
#                         no_improve += 1
#                 else:
#                     # No improvement (cost stayed same or increased)
#                     no_improve += 1
                
#                 if no_improve >= patience:
#                     print(f"[init {init + 1}] early stopping at step {step + 1} "
#                           f"(no improvement for {patience} steps, best cost: {best_step_cost:.6f}).")
#                     break

#         # Evaluate total cost for this initialization
#         total_cost, _ = computeCost(P, w, Vs, vs, jitter=jitter)
#         cost_val = float(total_cost.item())
#         print(f"[init {init + 1}] final cost = {cost_val:.6f}")

#         if cost_val < best_cost:
#             best_cost = cost_val
#             best_Vs = Vs.clone()
#             best_vs = vs.clone()

#     elapsed_total = time.time() - start_time
#     return best_Vs, best_vs, elapsed_total


import time
import torch

# ==============================
# Configuration parameters
# ==============================
# LAMBDA: set dynamically by em_parse.EmParser._configure_randomness()
# DEVICE: determined at runtime based on CUDA availability

LAMBDA = 1.0  # Default; overridden by caller
Z = 2.0
NUM_INIT_FOR_EM = 10
STEPS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# M-estimator losses (vectorized)
# ==============================

def lp_loss(x: torch.Tensor) -> torch.Tensor:
    # |x|^Z / Z
    return (x.abs() ** Z) / Z


def huber_loss(x: torch.Tensor, delta: float = LAMBDA) -> torch.Tensor:
    abs_x = x.abs()
    quad = 0.5 * x ** 2
    lin = delta * (abs_x - 0.5 * delta)
    return torch.where(abs_x <= delta, quad, lin)


def cauchy_loss(x: torch.Tensor) -> torch.Tensor:
    return (LAMBDA ** 2 / 2.0) * torch.log1p((x ** 2) / (LAMBDA ** 2))


def geman_McClure_loss(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2) / (2.0 * (1.0 + x ** 2))


def welsch_loss(x: torch.Tensor) -> torch.Tensor:
    return (LAMBDA ** 2 / 2.0) * (1.0 - torch.exp(-(x ** 2) / (LAMBDA ** 2)))


def tukey_loss(x: torch.Tensor) -> torch.Tensor:
    abs_x = x.abs()
    r2 = (x ** 2) / (LAMBDA ** 2)
    inner = 1.0 - r2
    val = (LAMBDA ** 2 / 6.0) * (1.0 - inner ** 3)
    const = (LAMBDA ** 2 / 6.0)
    return torch.where(abs_x <= LAMBDA, val, torch.full_like(x, const))


M_ESTIMATOR_FUNCS = {
    "lp": lp_loss,
    "huber": huber_loss,
    "cauchy": cauchy_loss,
    "geman_McClure": geman_McClure_loss,
    "welsch": welsch_loss,
    "tukey": tukey_loss,
}

# Default
OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS["lp"]

# ==============================
# Geometry helpers
# ==============================

def compute_nullspace(X: torch.Tensor, rtol: float = 1e-6) -> torch.Tensor:
    """
    Compute orthonormal basis for null space of X, matching scipy.linalg.null_space.

    X: (J, d)
    Returns: Null: (d, m) where columns form an orthonormal basis of null(X).
    """
    # SVD
    U, S, Vh = torch.linalg.svd(X, full_matrices=True)
    # Vh: (d, d), rows are right singular vectors^T
    V = Vh.transpose(-2, -1)  # (d, d)

    if S.numel() == 0:
        # degenerate case: X is all zeros
        return V  # full basis

    tol = rtol * S.max()
    rank = int((S > tol).sum().item())

    # Nullspace = last d - rank columns of V
    if rank >= V.size(1):
        # full rank: nullspace is {0}
        return V[:, 0:0]  # (d, 0)

    Null = V[:, rank:]   # (d, d-rank)
    return Null


def computeDistanceToSubspace(
    P: torch.Tensor,
    X: torch.Tensor,
    v: torch.Tensor = None,
) -> torch.Tensor:
    """
    Faithful GPU version of the CPU function computeDistanceToSubspace.

    P: (n, d)
    X: (J, d) basis matrix
    v: (d,) translation vector

    Returns: (n,) distances
    """
    # Ensure 2D
    if P.dim() == 1:
        P = P.unsqueeze(0)

    # Center by v if provided
    if v is None:
        Y = P
    else:
        Y = P - v.unsqueeze(0)

    # Nullspace of X (columns basis, like scipy.null_space)
    Null = compute_nullspace(X)

    if Null.size(1) == 0:
        # Subspace spans full space -> distance zero
        return torch.zeros(P.size(0), device=P.device, dtype=P.dtype)

    # Project onto nullspace: (n,d) @ (d,m) -> (n,m)
    proj = Y @ Null
    dists = torch.linalg.norm(proj, dim=1)  # (n,)
    return dists


# ==============================
# Cost function
# ==============================

def computeCost(
    P: torch.Tensor,
    w: torch.Tensor,
    X: torch.Tensor,
    v: torch.Tensor = None,
    show_indices: bool = False,
):
    """
    Faithful GPU version of CPU computeCost.

    P: (n, d)
    w: (n,)
    X:
      - (J, d): single flat
      - (k, J, d): multiple flats
    v:
      - (d,) or (k, d)
    """
    global OBJECTIVE_LOSS

    P = P.to(DEVICE)
    w = w.view(-1).to(DEVICE)
    X = X.to(DEVICE)
    if v is not None:
        v = v.to(DEVICE)

    n = P.size(0)

    if X.dim() == 2:
        # Single flat
        dists = computeDistanceToSubspace(P, X, v)
        losses = OBJECTIVE_LOSS(dists)
        cost_per_point = w * losses
        total_cost = cost_per_point.sum()
        return total_cost, cost_per_point

    # Multiple flats
    k = X.size(0)
    temp_cost_per_point = torch.empty((n, k), device=DEVICE, dtype=P.dtype)

    for i in range(k):
        v_i = v[i] if v is not None else None
        dists = computeDistanceToSubspace(P, X[i], v_i)
        losses = OBJECTIVE_LOSS(dists)
        temp_cost_per_point[:, i] = w * losses

    cost_per_point, indices = temp_cost_per_point.min(dim=1)
    total_cost = cost_per_point.sum()

    if not show_indices:
        return total_cost, cost_per_point
    else:
        return total_cost, cost_per_point, indices


# ==============================
# Suboptimal subspace (weighted mean, *unweighted* SVD)
# ==============================

def computeSuboptimalSubspace(P: torch.Tensor, w: torch.Tensor, J: int):
    """
    GPU version of CPU computeSuboptimalSubspace.

    P: (n, d)
    w: (n,)
    J: target dimension

    Returns: X: (J, d), v: (d,), elapsed: float
    """
    start_time = time.time()

    P = P.to(DEVICE)
    w = w.view(-1).to(DEVICE)

    # Weighted mean (same as CPU)
    w_norm = w / w.sum()
    v = (P * w_norm.unsqueeze(1)).sum(dim=0)  # (d,)

    centered = P - v

    # Unweighted SVD on centered data (CPU code does np.linalg.svd(P - v, ...))
    # full_matrices=False to get (min(n,d)) singular vectors
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    # First J right singular vectors (rows of Vh)
    X = Vh[:J, :]  # (J, d)

    elapsed = time.time() - start_time
    return X, v, elapsed


# ==============================
# EM-like algorithm
# ==============================

def EMLikeAlg(
    P: torch.Tensor,
    w: torch.Tensor,
    j: int,
    k: int,
    steps: int = STEPS,
    objective: str = "lp",
    jitter: float = 1e-6,   # kept for API compatibility, unused
    num_inits: int = NUM_INIT_FOR_EM,
    patience: int = 5,
    min_delta: float = 0.0,
):
    """
    GPU version of Maalouf's EM-like projective clustering.

    IMPORTANT: Returns (Vs_best, vs_best, elapsed_total) — elapsed_total is time in seconds.
    """
    global OBJECTIVE_LOSS

    start_time = time.time()

    P = P.to(DEVICE)
    w = w.view(-1).to(DEVICE)

    n, d = P.shape

    # Select objective
    OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS.get(objective, lp_loss)

    best_Vs = None
    best_vs = None
    optimal_cost = float("inf")

    # No gradients are required for this EM-like procedure; disable autograd for speed.
    with torch.no_grad():
        for init in range(num_inits):
            # ----- initialization -----
            # random centers (vs) like CPU version, though CPU code overwrites vs via computeSuboptimalSubspace
            vs = P[torch.randperm(n, device=DEVICE)[:k]]  # (k, d)

            Vs = torch.empty((k, j, d), device=DEVICE, dtype=P.dtype)

            # random partition of indices, as in CPU code
            idxs = torch.randperm(n, device=DEVICE)
            idx_splits = idxs.chunk(k)

            for i in range(k):
                cluster_idx = idx_splits[i]
                if cluster_idx.numel() == 0:
                    # fallback: global subspace
                    X_i, v_i, _ = computeSuboptimalSubspace(P, w, j)
                else:
                    X_i, v_i, _ = computeSuboptimalSubspace(P[cluster_idx], w[cluster_idx], j)
                Vs[i] = X_i
                vs[i] = v_i

            # ----- EM iterations -----
            best_step_cost = float("inf")
            no_improve = 0

            for step in range(steps):
                # E-step + step cost in one pass for all flats
                step_cost, cost_per_point, cluster_indices = computeCost(
                    P, w, Vs, vs, show_indices=True
                )
                step_cost_val = float(step_cost.item())
                cluster_sizes = torch.bincount(cluster_indices, minlength=k).cpu().tolist()

                # M-step: update each flat using assigned points
                unique_idxs = torch.unique(cluster_indices)
                for idx in unique_idxs:
                    mask = (cluster_indices == idx)
                    cluster_points_indices = mask.nonzero(as_tuple=True)[0]
                    if cluster_points_indices.numel() > 0:
                        X_new, v_new, _ = computeSuboptimalSubspace(
                            P[cluster_points_indices], w[cluster_points_indices], j
                        )
                        Vs[idx] = X_new
                        vs[idx] = v_new

                print(
                    f"[init {init + 1}][step {step + 1}] "
                    f"cost = {step_cost_val:.6f}, "
                    f"cluster_sizes = {cluster_sizes}"
                )

                if best_step_cost == float("inf"):
                    best_step_cost = step_cost_val
                    no_improve = 0
                else:
                    if step_cost_val < best_step_cost:
                        denom = max(best_step_cost, 1e-12)
                        rel_impr = (best_step_cost - step_cost_val) / denom
                        if rel_impr > min_delta:
                            best_step_cost = step_cost_val
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1

                    if no_improve >= patience:
                        print(
                            f"[init {init + 1}] early stopping at step {step + 1} "
                            f"(no improvement for {patience} steps, best cost: {best_step_cost:.6f})."
                        )
                        break

            # Final cost for this initialization (already computed as step_cost_val on last step)
            current_cost_val = step_cost_val

            print(f"[init {init + 1}] final cost = {current_cost_val:.6f}")

            if current_cost_val < optimal_cost:
                optimal_cost = current_cost_val
                best_Vs = Vs.clone()
                best_vs = vs.clone()

    elapsed_total = time.time() - start_time
    return best_Vs, best_vs, elapsed_total

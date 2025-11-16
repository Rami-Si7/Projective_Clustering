"""
Original NumPy/SciPy implementation of the EM-like projective clustering
algorithm described by Maalouf et al.

This file operates purely on CPU and serves as a reference for the GPU
implementation in `ours/emlike_cuda.py`. It defines:
  * several robust loss functions via `M_ESTIMATOR_FUNCS`,
  * helpers for distances to affine subspaces and their costs, and
  * `EMLikeAlg`, the EM-like heuristic that optimizes a collection of
    j-dimensional affine subspaces (\"flats\").
"""

import numpy as np
from scipy.linalg import null_space
import copy
import time
# import getdim # file in directory to load BERT
LAMBDA = 1
Z=2
NUM_INIT_FOR_EM = 10
STEPS = 20
M_ESTIMATOR_FUNCS = {
    'lp': (lambda x: np.abs(x) ** Z / Z),
    'huber': (lambda x: x ** 2 / 2 if np.abs(x) <= LAMBDA else LAMBDA * (np.abs(x) - LAMBDA / 2)),
    'cauchy': (lambda x: LAMBDA ** 2 / 2 * np.log(1 + x ** 2 / LAMBDA ** 2)),
    'geman_McClure': (lambda x: x ** 2 / (2 * (1 + x ** 2))),
    'welsch': (lambda x: LAMBDA ** 2 / 2 * (1 - np.exp(-x ** 2 / LAMBDA ** 2))),
    'tukey': (lambda x: LAMBDA ** 2 / 6 * (1 - (1 - x ** 2 / LAMBDA ** 2) ** 3) if np.abs(x) <= LAMBDA
                else LAMBDA**2 / 6)
}
global OBJECTIVE_LOSS
OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS['lp']

def computeDistanceToSubspace(point, X, v=None):
    """
    This function is responsible for computing the distance between a point and a J dimensional affine subspace.

    :param point: A numpy array representing a .
    :param X: A numpy matrix representing a basis for a J dimensional subspace.
    :param v: A numpy array representing the translation of the subspace from the origin.
    :return: The distance between the point and the subspace which is spanned by X and translated from the origin by v.
    """
    if point.ndim > 1:
        return np.linalg.norm(np.dot(point - v[np.newaxis, :], null_space(X)), ord=2, axis=1)
    return np.linalg.norm(np.dot(point-v if v is not None else point, null_space(X)))


def computeCost(P, w, X, v=None, show_indices=False):
    """
    This function represents our cost function which is a generalization of k-means where the means are now J-flats.

    :param P: A weighed set, namely, a PointSet object.
    :param X: A numpy matrix of J x d which defines the basis of the subspace which we would like to compute the
              distance to.
    :param v: A numpy array of d entries which defines the translation of the J-dimensional subspace spanned by the
              rows of X.
    :return: The sum of weighted distances of each point to the affine J dimensional flat which is denoted by (X,v)
    """
    global OBJECTIVE_LOSS
    if X.ndim == 2:
        dist_per_point = OBJECTIVE_LOSS(computeDistanceToSubspace(P, X, v))
        cost_per_point = np.multiply(w, dist_per_point)
    else:
        temp_cost_per_point = np.empty((P.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            temp_cost_per_point[:, i] = \
                np.multiply(w, OBJECTIVE_LOSS(computeDistanceToSubspace(P, X[i, :, :], v[i, :])))

        cost_per_point = np.min(temp_cost_per_point, 1)
        indices = np.argmin(temp_cost_per_point, 1)
    if not show_indices:
        return np.sum(cost_per_point), cost_per_point
    else:
        return np.sum(cost_per_point), cost_per_point, indices



def computeSuboptimalSubspace(P,w, J):
    """
    This function computes a suboptimal subspace in case of having the generalized K-means objective function.

    :param P: A weighted set, namely, an object of PointSet.
    :return: A tuple of a basis of J dimensional spanning subspace, namely, X and a translation vector denoted by v.
    """

    start_time = time.time()
    v = np.average(P, axis=0, weights=w)  # weighted mean of the point
    _, _, V = np.linalg.svd(P-v, full_matrices=False)  # computing the spanning subspace

    return V[:J, :], v, time.time() - start_time


def EMLikeAlg(P, w, j, k, steps, patience=5, min_delta=0.0):
    """
    The function at hand, is an EM-like algorithm which is heuristic in nature. It finds a suboptimal solution for the
    (K,J)-projective clustering problem with respect to a user chosen

    :param P: A weighted set, namely, a PointSet object
    :param j: An integer denoting the desired dimension of each flat (affine subspace)
    :param k: An integer denoting the number of j-flats
    :param steps: An integer denoting the max number of EM steps
    :return: A list of k j-flats which locally optimize the cost function
    """
    start_time = time.time()
    n,d = P.shape
    max_norm = np.max(np.linalg.norm(P, axis=1))
    min_vs = None
    min_Vs = None
    optimal_cost = np.inf
    for iter in range(NUM_INIT_FOR_EM):  # run EM for 10 random initializations
        # np.random.seed()
        vs = P[np.random.choice(np.arange(n), size=k, replace=False), :]
        Vs = np.empty((k, j, d))
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        idxs = np.array_split(idxs, k)
        for i in range(k):  # initialize k random orthogonal matrices
            Vs[i, :, :], vs[i, :], _ = computeSuboptimalSubspace(P[idxs[i],:],w[idxs[i]], j)
            # Vs[i, :, :] = ortho_group.rvs(dim=P.d)[:j, :]

        best_step_cost = float("inf")
        no_improve = 0
        for i in range(steps):  # find best k j-flats which can attain local optimum
            dists = np.empty((n, k))  # distance of point to each one of the k j-flats
            for l in range(k):
                _, dists[:, l] = computeCost(P, w, Vs[l, :, :], vs[l, :])

            cluster_indices = np.argmin(dists, 1)  # determine for each point, the closest flat to it
            unique_idxs, counts = np.unique(cluster_indices, return_counts=True)
            cluster_sizes = dict(zip(unique_idxs, counts))
            
            # Create a dictionary for all possible cluster indices (0 to k-1)
            full_cluster_sizes = {cluster_idx: 0 for cluster_idx in range(k)}
            full_cluster_sizes.update(cluster_sizes)

            for idx in unique_idxs:  # recompute better flats with respect to the updated cluster matching
                cluster_points_indices = np.where(cluster_indices == idx)[0]
                if len(cluster_points_indices) > 0:
                    Vs[idx, :, :], vs[idx, :], _ = \
                        computeSuboptimalSubspace(P[cluster_points_indices, :], w[cluster_points_indices], j)
            
            step_cost = computeCost(P, w, Vs, vs)[0]
            print(
                f"[init {iter + 1}][step {i + 1}] "
                f"cost = {step_cost:.6f}, "
                f"cluster_sizes = {list(full_cluster_sizes.values())}"
            )

            if best_step_cost == float("inf"):
                best_step_cost = step_cost
                no_improve = 0
            else:
                if step_cost < best_step_cost:
                    denom = max(best_step_cost, 1e-12)
                    rel_impr = (best_step_cost - step_cost) / denom
                    if rel_impr > min_delta:
                        best_step_cost = step_cost
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            if no_improve >= patience:
                print(f"[init {iter + 1}] early stopping at step {i + 1} "
                      f"(no improvement for {patience} steps, best cost: {best_step_cost:.6f}).")
                break

        current_cost = computeCost(P, w, Vs, vs)[0]
        if current_cost < optimal_cost:
            min_Vs = copy.deepcopy(Vs)
            min_vs = copy.deepcopy(vs)
            optimal_cost = current_cost
        print(f"[init {iter + 1}] final cost = {current_cost:.6f}")
        
    return min_Vs, min_vs, time.time()-start_time



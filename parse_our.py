#!/usr/bin/env python3
import argparse
import json
import pickle
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color
from sklearn.decomposition import PCA
import tqdm

# =============== Optional (GPU) ===============
import torch

# ---------------- Utils: device/dtype ----------------
def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)

def pick_dtype(choice: str):
    return torch.float32 if choice == "float32" else torch.float64

# =====================================================
#                  Image helpers
# =====================================================
def _gather_images(root: Path, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()])

def visualize(output_dir: Path):
    """Save grayscale previews for arrs.npy components: n_comps x 3 x H x W → vis/{i}-{j}.png"""
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    array = np.load(output_dir / "arrs.npy")  # (n_comps, 3, H, W)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            img = (array[i, j] * 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(vis_dir / f"{i}-{j}.png")

def _load_img_ycbcr(path: Path, target_wh=None):
    im = Image.open(path).convert("RGB")
    if target_wh is not None:
        im = im.resize(tuple(target_wh), Image.Resampling.LANCZOS)
    arr_u8 = np.asarray(im, dtype=np.uint8)
    ycbcr = color.rgb2ycbcr(arr_u8).astype(np.float32)  # (H,W,3) in 0..255
    return ycbcr

# =====================================================
#            Split builder (random, deterministic)
# =====================================================
def prepare_imgs(args, output_dir: Path):
    """
    Create train_imgs/, val_imgs/, test_imgs/ with resized copies.

    When sampling from --source (default):
      - Deterministically shuffle all images by --seed.
      - Use up to args.n_samples for TRAIN+VAL (VAL = val_frac of that).
      - TEST is taken from the remaining pool (disjoint), capped by args.n_test.

    When explicit folders are used:
      - --train_from fills train   (shuffled; capped by n_samples if provided)
      - --val_from   fills val     (shuffled)
      - --test_from  fills test    (shuffled; capped by n_test if provided)
    """
    rng = np.random.default_rng(args.seed)

    W, H = args.img_size
    train_dir = output_dir / "train_imgs"
    val_dir   = output_dir / "val_imgs"
    test_dir  = output_dir / "test_imgs"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    written_train, written_val, written_test = [], [], []

    # ---------- TRAIN / VAL ----------
    if args.train_from is not None:
        train_src = _gather_images(args.train_from)
        if len(train_src) == 0:
            raise FileNotFoundError(f"No images found under --train_from: {args.train_from}")
        train_src = [train_src[i] for i in rng.permutation(len(train_src))]
        if args.n_samples is not None and args.n_samples > 0:
            train_src = train_src[:min(args.n_samples, len(train_src))]
        print(f"[train_from] Using {len(train_src)} train images")

        if args.val_from is not None:
            val_src = _gather_images(args.val_from)
            val_src = [val_src[i] for i in rng.permutation(len(val_src))]
            print(f"[val_from] Using {len(val_src)} val images")
        else:
            val_src = []
            print("[info] No --val_from provided; VAL set is empty when using --train_from.")

        counter = 0
        for p in tqdm.tqdm(train_src, desc="Prepare training images (train_from)"):
            img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            outp = train_dir / f"{counter:06d}.png"
            img.save(outp); written_train.append(str(outp)); counter += 1

        counter = 0
        for p in tqdm.tqdm(val_src, desc="Prepare validation images (val_from)"):
            img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            outp = val_dir / f"{counter:06d}.png"
            img.save(outp); written_val.append(str(outp)); counter += 1

    else:
        pool = _gather_images(args.source)
        if len(pool) == 0:
            raise FileNotFoundError(f"No images found under --source: {args.source}")
        pool = [pool[i] for i in rng.permutation(len(pool))]  # deterministic shuffle

        n_trainval_budget = min(args.n_samples, len(pool))
        n_val = int(round(n_trainval_budget * float(args.val_frac)))
        n_val = max(0, min(n_val, n_trainval_budget))
        n_train = n_trainval_budget - n_val

        train_src = pool[:n_train]
        val_src   = pool[n_train:n_trainval_budget]

        remaining = pool[n_trainval_budget:]
        n_test = min(args.n_test, len(remaining))
        test_src = remaining[:n_test]

        print(f"[source] TRAIN={len(train_src)}  VAL={len(val_src)}  TEST={len(test_src)} "
              f"(from {len(pool)} shuffled; seed={args.seed})")

        counter = 0
        for p in tqdm.tqdm(train_src, desc="Prepare training images (source/random)"):
            img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            outp = train_dir / f"{counter:06d}.png"
            img.save(outp); written_train.append(str(outp)); counter += 1

        counter = 0
        for p in tqdm.tqdm(val_src, desc="Prepare validation images (source/random)"):
            img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            outp = val_dir / f"{counter:06d}.png"
            img.save(outp); written_val.append(str(outp)); counter += 1

        counter = 0
        for p in tqdm.tqdm(test_src, desc="Prepare testing images (source/random)"):
            img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            outp = test_dir / f"{counter:06d}.png"
            img.save(outp); written_test.append(str(outp)); counter += 1

    # ---------- TEST (explicit) ----------
    if args.train_from is not None and args.test_from is not None:
        rng = np.random.default_rng(args.seed)  # re-init OK
        test_src = _gather_images(args.test_from)
        test_src = [test_src[i] for i in rng.permutation(len(test_src))]
        if args.n_test is not None and args.n_test > 0:
            test_src = test_src[:min(args.n_test, len(test_src))]
        print(f"[test_from] Using {len(test_src)} test images")
        counter = 0
        for p in tqdm.tqdm(test_src, desc="Prepare testing images (test_from)"):
            img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            outp = test_dir / f"{counter:06d}.png"
            img.save(outp); written_test.append(str(outp)); counter += 1
    elif args.train_from is not None and args.test_from is None:
        print("[info] --train_from provided without --test_from; leaving test_imgs empty.")

    with open(output_dir / "splits.json", "w") as f:
        json.dump({"train": written_train, "val": written_val, "test": written_test}, f, indent=2)

# =====================================================
#      Feature builders (TRAIN/VAL) & Global PCA
# =====================================================
def _features_from_dir(img_dir: Path, img_size):
    img_list = sorted([f for f in img_dir.glob("*.png")])
    N = len(img_list)
    concat_list = []
    ch_lists = {"Y": [], "Cb": [], "Cr": []}
    per_img_flat = []

    for i in tqdm.trange(N, desc=f"Loading {img_dir.name}"):
        ycbcr = _load_img_ycbcr(img_list[i], target_wh=img_size)  # 0..255
        per_img_flat.append(ycbcr.reshape(-1, 3))                  # (d,3)
        arr01 = ycbcr / 255.0
        concat_list.append(arr01.reshape(-1))                      # (3HW,)
        ch_lists["Y"].append(arr01[..., 0].reshape(-1))
        ch_lists["Cb"].append(arr01[..., 1].reshape(-1))
        ch_lists["Cr"].append(arr01[..., 2].reshape(-1))

    concat = np.stack(concat_list, axis=0) if concat_list else np.empty((0, 0), np.float32)
    chs = {k: (np.stack(v, axis=0) if v else np.empty((0, 0), np.float32)) for k, v in ch_lists.items()}
    per_img_flat = np.stack(per_img_flat, axis=0) if per_img_flat else np.empty((0, 0, 0), np.float32)
    return concat, chs, per_img_flat

def prepare_arrs(args, output_dir: Path):
    """
    Build features for train and val.
    Fit global PCA (per-channel) on TRAIN only and save arrs.npy for visualization.
    """
    train_dir = output_dir / "train_imgs"
    val_dir   = output_dir / "val_imgs"

    P_concat_tr, P_channels_tr, train_imgstack = _features_from_dir(train_dir, args.img_size)
    assert train_imgstack.size > 0, "No training images found in train_imgs/."

    train_arrs = train_imgstack.transpose(2, 0, 1)  # (3, N, d)
    pca_object, norm_infos, norm_comps = [], [], []
    for ch in tqdm.trange(3, desc="Fitting PCA comps (TRAIN)"):
        ch_arr = train_arrs[ch] / 255.0  # (N, d) in [0,1]
        pca = PCA(n_components=args.n_comps, whiten=False)
        pca.fit(ch_arr)
        comps = pca.components_.astype(np.float32)  # (n_comps, d)
        gmax, gmin = float(comps.max()), float(comps.min())
        den = (gmax - gmin) if gmax != gmin else 1.0
        norm_comp = (comps - gmin) / den
        pca_object.append(pca); norm_infos.append({"min": gmin, "max": gmax}); norm_comps.append(norm_comp)

    # Save arrs.npy and previews
    norm_comps = np.stack(norm_comps, axis=-1)     # (n_comps, d, 3)
    W, H = args.img_size
    norm_comps = norm_comps.transpose(0, 2, 1).reshape(args.n_comps, 3, H, W)
    np.save(output_dir / "arrs.npy", norm_comps)
    visualize(output_dir)

    P_concat_val, P_channels_val, _ = _features_from_dir(val_dir, args.img_size)

    return (pca_object, norm_infos,
            P_concat_tr, P_channels_tr,
            P_concat_val, P_channels_val)

# =====================================================
#            PCA pack helper for locals
# =====================================================
def pca_pack_fit(X, n_comps):
    if X.ndim != 2 or min(X.shape) < 2:
        # fallback to trivial pack
        return {"mean": np.zeros((X.shape[1],), np.float32) if X.size else np.zeros((0,), np.float32),
                "components": np.zeros((min(n_comps, X.shape[1] if X.size else 0), X.shape[1] if X.size else 0), np.float32),
                "explained_variance": np.zeros((min(n_comps, X.shape[1] if X.size else 0),), np.float32)}
    p = PCA(n_components=min(n_comps, max(1, min(X.shape)-1)), whiten=False)
    p.fit(X)
    return {
        "mean": p.mean_.astype(np.float32),
        "components": p.components_.astype(np.float32),
        "explained_variance": p.explained_variance_.astype(np.float32)
    }

# =====================================================
#                   EM-like algorithm
# =====================================================
LAMBDA = 1.0
Z = 2
NUM_INIT_FOR_EM = 10

def _lp(x: torch.Tensor): return (torch.abs(x) ** Z) / Z
def _huber(x: torch.Tensor):
    ax = torch.abs(x)
    return torch.where(ax <= LAMBDA, 0.5 * x * x, LAMBDA * (ax - 0.5 * LAMBDA))
def _cauchy(x: torch.Tensor):        return (LAMBDA**2 / 2.0) * torch.log1p((x * x) / (LAMBDA**2))
def _geman_mcclure(x: torch.Tensor): return (x * x) / (2.0 * (1.0 + x * x))
def _welsch(x: torch.Tensor):        return (LAMBDA**2 / 2.0) * (1.0 - torch.exp(-(x * x) / (LAMBDA**2)))
def _tukey(x: torch.Tensor):
    ax2 = (x * x) / (LAMBDA**2)
    inside = 1.0 - ax2
    val = (LAMBDA**2 / 6.0) * (1.0 - (inside ** 3))
    return torch.where(torch.abs(x) <= LAMBDA, val, torch.full_like(x, LAMBDA**2 / 6.0))

M_ESTIMATOR_FUNCS = {
    'lp': _lp, 'huber': _huber, 'cauchy': _cauchy, 'geman_McClure': _geman_mcclure,
    'welsch': _welsch, 'tukey': _tukey,
}
OBJECTIVE_LOSS = _lp  # overwritten by args.loss

@torch.no_grad()
def orthonormalize_rows_(Vs: torch.Tensor):
    K, J, d = Vs.shape
    for k in range(K):
        Q, _ = torch.linalg.qr(Vs[k].T, mode='reduced')  # (d,J)
        Vs[k].copy_(Q.T.contiguous())

@torch.no_grad()
def dists_to_flats_streaming_orthonorm(P: torch.Tensor, Vs: torch.Tensor, vs: torch.Tensor, batch_N: int = 2048) -> torch.Tensor:
    N, d = P.shape
    K, J, _ = Vs.shape
    out = torch.empty((N, K), dtype=P.dtype, device=P.device)
    for k in range(K):
        B = Vs[k]        # (J,d)
        v = vs[k]        # (d,)
        for i in range(0, N, batch_N):
            Pi = P[i:i+batch_N]
            xc = Pi - v
            s_center = torch.sum(xc * xc, dim=1)
            Y = xc @ B.T
            proj2 = torch.sum(Y * Y, dim=1)
            dist2 = torch.clamp(s_center - proj2, min=0.)
            out[i:i+batch_N, k] = torch.sqrt(dist2)
    return out

@torch.no_grad()
def dists_to_flats_batched_general(P: torch.Tensor, Vs: torch.Tensor, vs: torch.Tensor, jitter: float = 1e-8) -> torch.Tensor:
    N, d = P.shape
    K, J, _ = Vs.shape
    out = torch.empty((N, K), dtype=P.dtype, device=P.device)
    Ij = torch.eye(J, dtype=P.dtype, device=P.device)
    for k in range(K):
        B = Vs[k]
        v = vs[k]
        xc = (P - v)
        G = B @ B.t() + jitter * Ij
        L = torch.linalg.cholesky(G)
        Y = (B @ xc.T)
        Z = torch.cholesky_solve(Y, L)
        dist2 = torch.clamp(torch.sum(xc * xc, dim=1) - torch.sum(Y * Z, dim=0), min=0.)
        out[:, k] = dist2
    return out

@torch.no_grad()
def objective_on_set(P: torch.Tensor, w: torch.Tensor, Vs: torch.Tensor, vs: torch.Tensor,
                     use_orthonorm: bool = True, jitter: float = 1e-8) -> float:
    dmat = (dists_to_flats_streaming_orthonorm(P, Vs, vs)
            if use_orthonorm else
            dists_to_flats_batched_general(P, Vs, vs, jitter))
    losses = OBJECTIVE_LOSS(dmat)                                # (N,K)
    per_point = (losses * w.unsqueeze(1)).min(dim=1).values      # (N,)
    return float(per_point.sum().item())

@torch.no_grad()
def computeSuboptimalSubspace_torch(P: torch.Tensor, w: torch.Tensor, J: int):
    v = (P * w.unsqueeze(1)).sum(dim=0) / w.sum()
    Y = P - v.unsqueeze(0)
    U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
    V_rows = Vh[:J, :].contiguous()
    return V_rows, v

@torch.no_grad()
def EMLikeAlg_torch(
    P: torch.Tensor, w: torch.Tensor,
    j: int, k: int, steps: int,
    num_inits: int = NUM_INIT_FOR_EM,
    seed: int | None = None,
    verbose: bool = False,
    P_val: torch.Tensor | None = None,
    w_val: torch.Tensor | None = None,
    jitter: float = 1e-8,
    early_stop_patience: int = 1,
    tol_obj_abs: float = 1e-9,
    tol_obj_rel: float = 0.0,
    tol_subspace: float = 1e-9,
    use_orthonorm: bool = False,
    enforce_orthonorm_each_step: bool = True,
):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device, dtype = P.device, P.dtype
    w = w.to(device=device, dtype=dtype)
    if P_val is not None:
        P_val = P_val.to(device=device, dtype=dtype)
        w_val = (torch.ones(P_val.shape[0], dtype=dtype, device=device)
                 if w_val is None else w_val.to(device=device, dtype=dtype))

    N, d = int(P.shape[0]), int(P.shape[1])
    j, k, steps, num_inits = int(j), int(k), int(steps), int(num_inits)

    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)

    best_score = float("inf")
    best_Vs = best_vs = best_assign = None

    for it in range(num_inits):
        # init
        perm = torch.randperm(N, device=device)
        chunks = torch.chunk(perm, k)
        vs = torch.zeros((k, d), dtype=dtype, device=device)
        Vs = torch.zeros((k, j, d), dtype=dtype, device=device)
        for i in range(k):
            idx = chunks[i]
            Ji = int(min(j, max(1, idx.numel() - 1)))
            Vi, vi = computeSuboptimalSubspace_torch(P[idx], w[idx], Ji)
            Vs[i, :Ji, :] = Vi
            vs[i, :] = vi
        if enforce_orthonorm_each_step and use_orthonorm:
            orthonormalize_rows_(Vs)

        prev_assign = None
        prev_obj = None
        prev_Vs = None
        prev_vs = None
        stall_runs = 0

        for ss in range(steps):
            if use_orthonorm:
                dmat = dists_to_flats_streaming_orthonorm(P, Vs, vs)
            else:
                dmat = dists_to_flats_batched_general(P, Vs, vs, jitter)

            assign = torch.argmin(OBJECTIVE_LOSS(dmat), dim=1)

            # M-step
            for ci in torch.unique(assign):
                idx = torch.where(assign == ci)[0]
                if idx.numel() == 0:
                    continue
                Ji = int(min(j, max(1, idx.numel() - 1)))
                Vi, vi = computeSuboptimalSubspace_torch(P[idx], w[idx], Ji)
                Vs[ci].zero_()
                Vs[ci, :Ji, :] = Vi
                vs[ci, :] = vi
            if enforce_orthonorm_each_step and use_orthonorm:
                orthonormalize_rows_(Vs)

            train_obj = objective_on_set(P, w, Vs, vs, use_orthonorm=use_orthonorm, jitter=jitter)

            # delta checks
            if prev_Vs is None:
                dV = float("inf"); dv = float("inf")
            else:
                dV = torch.linalg.norm(Vs - prev_Vs).item()
                dv = torch.linalg.norm(vs - prev_vs).item()
            max_subspace_delta = max(dV, dv)

            # early stop
            stalled = False
            if prev_assign is not None and torch.equal(assign, prev_assign):
                stalled = True
            elif prev_obj is not None:
                tol = max(tol_obj_abs, tol_obj_rel * abs(prev_obj))
                if abs(train_obj - prev_obj) <= tol:
                    stalled = True
            if max_subspace_delta <= tol_subspace:
                stalled = True

            stall_runs = stall_runs + 1 if stalled else 0
            prev_assign = assign.clone()
            prev_obj = train_obj
            prev_Vs = Vs.clone()
            prev_vs = vs.clone()

            if stall_runs >= early_stop_patience:
                if verbose:
                    print(f"[init {it+1}] early stop at step {ss+1}")
                break

        # model selection
        score = (objective_on_set(P_val, w_val, Vs, vs, use_orthonorm=use_orthonorm, jitter=jitter)
                 if P_val is not None else
                 objective_on_set(P, w, Vs, vs, use_orthonorm=use_orthonorm, jitter=jitter))
        if verbose:
            print(f"[EM init {it+1}/{num_inits}] score={score:.6f} ({'val' if P_val is not None else 'train'})")

        if score < best_score:
            best_score = score
            best_Vs = Vs.clone()
            best_vs = vs.clone()
            dmat = (dists_to_flats_streaming_orthonorm(P, best_Vs, best_vs) if use_orthonorm
                    else dists_to_flats_batched_general(P, best_Vs, best_vs, jitter))
            best_assign = torch.argmin(OBJECTIVE_LOSS(dmat), dim=1).clone()

    return (best_Vs.detach().cpu().numpy(),
            best_vs.detach().cpu().numpy(),
            best_assign.detach().cpu().numpy())

# =====================================================
#            Metrics: EM vs Global PCA residuals
# =====================================================
@torch.no_grad()
def _pca_residuals(P: torch.Tensor, pca_pack: dict, batch: int = 2048) -> torch.Tensor:
    mu = torch.as_tensor(pca_pack["mean"], dtype=P.dtype, device=P.device)       # (d,)
    U  = torch.as_tensor(pca_pack["components"], dtype=P.dtype, device=P.device) # (j,d)
    Ut = U.transpose(0, 1)  # (d, j)
    N = P.shape[0]
    out = torch.empty((N,), dtype=P.dtype, device=P.device)
    for i in range(0, N, batch):
        Pi = P[i:i+batch]
        xc = Pi - mu
        coeffs = xc @ Ut
        proj   = coeffs @ U
        resid  = xc - proj
        out[i:i+batch] = torch.linalg.norm(resid, dim=1)
    return out

def compute_and_store_metrics(output_dir: Path, args, pca_object, ch_name, P_np, Vs_np, vs_np, use_orthonorm):
    device = pick_device(args.device)
    dtype  = pick_dtype(args.dtype)
    P = torch.from_numpy(P_np).to(device=device, dtype=dtype)
    Vs_t = torch.from_numpy(Vs_np).to(device=device, dtype=dtype)
    vs_t = torch.from_numpy(vs_np).to(device=device, dtype=dtype)

    # EM residuals
    if use_orthonorm:
        dmat = dists_to_flats_streaming_orthonorm(P, Vs_t, vs_t)
        em_resid = torch.min(dmat, dim=1).values
    else:
        dmat2 = dists_to_flats_batched_general(P, Vs_t, vs_t, jitter=args.jitter)
        em_resid = torch.min(torch.sqrt(dmat2), dim=1).values
    em_mean = float(torch.mean(em_resid).item())
    em_p95  = float(torch.quantile(em_resid, 0.95).item())

    # PCA (top-J) residuals
    j = min(args.J, pca_object.n_components_)
    pca_pack = {
        "mean": pca_object.mean_.astype(np.float32),
        "components": pca_object.components_[:j].astype(np.float32)
    }
    pca_resid = _pca_residuals(P, pca_pack)
    pca_mean = float(torch.mean(pca_resid).item())
    pca_p95  = float(torch.quantile(pca_resid, 0.95).item())

    # write/append file
    metrics_path = output_dir / "em_vs_pca_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics.setdefault(ch_name, {})
    metrics[ch_name] = {
        "EM":  {"mean_L2": em_mean,  "p95_L2": em_p95},
        "PCA": {"mean_L2": pca_mean, "p95_L2": pca_p95},
        "settings": {"K": args.K, "J": j, "loss": args.loss, "orthonorm": args.em_use_orthonorm}
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics][{ch_name}] EM mean={em_mean:.6f} p95={em_p95:.6f} | PCA mean={pca_mean:.6f} p95={pca_p95:.6f}")

# =====================================================
#                        Main
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse image set (Global PCA + optional EM local PCA; GPU EM + validation pick)")

    # Core args
    parser.add_argument("-s", "--source", required=True, type=Path,
                        help="Folder with images (recursively) used when explicit sources are not given")
    parser.add_argument("-c", "--n_comps", type=int, default=300)
    parser.add_argument("-n", "--n_samples", type=int, default=10000,
                        help="Budget for TRAIN+VAL when sampling from --source")
    parser.add_argument("-t", "--n_test", type=int, default=100)
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 512], metavar=('width','height'))
    parser.add_argument("--seed", type=int, default=42)

    # Explicit sources
    parser.add_argument("--train_from", type=Path, default=None)
    parser.add_argument("--val_from",   type=Path, default=None)
    parser.add_argument("--test_from",  type=Path, default=None)
    parser.add_argument("--val_frac", type=float, default=0.0,
                        help="Fraction of TRAIN budget reserved for validation (only when sampling from --source).")

    # EM controls
    parser.add_argument("--em_mode", type=str, choices=["none", "concat", "per-channel"], default="none")
    parser.add_argument("--K", type=int, default=4, help="Number of j-flats for EM")
    parser.add_argument("--J", type=int, default=32, help="Flat dimension for EM (per flat)")
    parser.add_argument("--steps", type=int, default=20, help="EM iterations per init")
    parser.add_argument("--loss", type=str, choices=list(M_ESTIMATOR_FUNCS.keys()), default="lp")
    parser.add_argument("--em_use_orthonorm", action="store_true", help="Use orthonormal-row kernels in EM distance")
    parser.add_argument("--emit_metrics", action="store_true", help="Write EM vs PCA residual metrics JSON")

    # GPU / numerics
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])
    parser.add_argument("--jitter", type=float, default=1e-8, help="Jitter for Gram in general distance")

    args = parser.parse_args()

    device = pick_device(args.device)
    dtype  = pick_dtype(args.dtype)

    # Output folder
    random_string = str(uuid.uuid4())[:6]
    output_dir = args.source.parent / f"{random_string}-{args.n_comps}-{args.J}-{args.K}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build train/val/test resized sets (random, reproducible)
    prepare_imgs(args, output_dir)

    # 2) Fit global PCA on TRAIN and build TRAIN/VAL features
    (pca_object, norm_infos,
     P_concat_tr, P_channels_tr,
     P_concat_val, P_channels_val) = prepare_arrs(args, output_dir)

    # 3) Save global PCA artifacts
    with open(output_dir / "pca_object.pkl", "wb") as f:
        pickle.dump(pca_object, f)
    with open(output_dir / "norm_infos.pkl", "wb") as f:
        pickle.dump(norm_infos, f)
    globals_pack = {
        "mode": "per-channel",
        "pca": {
            "Y": {
                "mean": pca_object[0].mean_.astype(np.float32),
                "components": pca_object[0].components_.astype(np.float32),
                "explained_variance": pca_object[0].explained_variance_.astype(np.float32),
            },
            "Cb": {
                "mean": pca_object[1].mean_.astype(np.float32),
                "components": pca_object[1].components_.astype(np.float32),
                "explained_variance": pca_object[1].explained_variance_.astype(np.float32),
            },
            "Cr": {
                "mean": pca_object[2].mean_.astype(np.float32),
                "components": pca_object[2].components_.astype(np.float32),
                "explained_variance": pca_object[2].explained_variance_.astype(np.float32),
            },
        }
    }
    with open(output_dir / "pca_globals.pkl", "wb") as f:
        pickle.dump(globals_pack, f)

    with open(output_dir / "shapes.json", "w") as f:
        json.dump({"width": args.img_size[0], "height": args.img_size[1],
                   "color": "ycbcr", "mode": args.em_mode}, f, indent=2)

    if args.em_mode == "none":
        print("[info] EM disabled (--em_mode none). Finished.")
        raise SystemExit(0)

    # ----- set loss -----

    OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS[args.loss]

    # 4) EM + local PCA (validation pick using VAL if available)
    if args.em_mode == "concat":
        # TRAIN
        P_tr = torch.as_tensor(P_concat_tr, dtype=dtype, device=device)
        w_tr = torch.ones(P_tr.shape[0], dtype=dtype, device=device)
        # VAL (optional)
        if P_concat_val.size:
            P_val = torch.as_tensor(P_concat_val, dtype=dtype, device=device)
            w_val = torch.ones(P_val.shape[0], dtype=dtype, device=device)
        else:
            P_val, w_val = None, None

        Vs, vs, assign = EMLikeAlg_torch(
            P_tr, w_tr, j=args.J, k=args.K, steps=args.steps,
            num_inits=NUM_INIT_FOR_EM, seed=args.seed, verbose=True,
            P_val=P_val, w_val=w_val, jitter=args.jitter,
            use_orthonorm=args.em_use_orthonorm
        )

        # Local PCA per cluster (TRAIN only)
        locals_pack = {"mode": "concat", "clusters": []}
        for k_idx in range(args.K):
            idx = np.where(assign == k_idx)[0]
            if len(idx) < 2:
                idx = np.arange(min(2, P_tr.shape[0]))
            pack = pca_pack_fit(P_concat_tr[idx], args.n_comps)
            locals_pack["clusters"].append(pack)

        em_art = {"mode": "concat", "color": "ycbcr", "Vs": Vs, "vs": vs, "assign": assign}

        # Optional metrics vs global PCA (use Y-channel PCA as reference for J)
        if args.emit_metrics:
            compute_and_store_metrics(output_dir, args, pca_object[0], "Concat(3ch)", P_concat_tr, Vs, vs, args.em_use_orthonorm)

    else:  # per-channel
        Vs, vs, assign = {}, {}, {}
        locals_pack = {"mode": "per-channel", "clusters": {"Y": [], "Cb": [], "Cr": []}}

        for ch_name, ch_idx in (("Y", 0), ("Cb", 1), ("Cr", 2)):
            Pc_tr_np = P_channels_tr[ch_name]
            P_tr = torch.as_tensor(Pc_tr_np, dtype=dtype, device=device)
            w_tr = torch.ones(P_tr.shape[0], dtype=dtype, device=device)

            if P_channels_val[ch_name].size:
                P_val = torch.as_tensor(P_channels_val[ch_name], dtype=dtype, device=device)
                w_val = torch.ones(P_val.shape[0], dtype=dtype, device=device)
            else:
                P_val, w_val = None, None

            Vc, vc, ac = EMLikeAlg_torch(
                P_tr, w_tr, j=args.J, k=args.K, steps=args.steps,
                num_inits=NUM_INIT_FOR_EM, seed=args.seed, verbose=True,
                P_val=P_val, w_val=w_val, jitter=args.jitter,
                use_orthonorm=args.em_use_orthonorm
            )
            Vs[ch_name], vs[ch_name], assign[ch_name] = Vc, vc, ac

            # Local PCA per (channel, cluster)
            for k_idx in range(args.K):
                idx = np.where(assign[ch_name] == k_idx)[0]
                if len(idx) < 2:
                    idx = np.arange(min(2, Pc_tr_np.shape[0]))
                pack = pca_pack_fit(Pc_tr_np[idx], args.n_comps)
                locals_pack["clusters"][ch_name].append(pack)

            if args.emit_metrics:
                compute_and_store_metrics(output_dir, args, pca_object[ch_idx], ch_name, Pc_tr_np, Vc, vc, args.em_use_orthonorm)

        em_art = {"mode": "per-channel", "color": "ycbcr", "Vs": Vs, "vs": vs, "assign": assign}

    # Save EM/local artifacts
    with open(output_dir / "em_artifacts.pkl", "wb") as f:
        pickle.dump(em_art, f)
    with open(output_dir / "pca_locals.pkl", "wb") as f:
        pickle.dump(locals_pack, f)

    print(f"[OK] Saved EM + local PCA artifacts to {output_dir}")
    print(f"  em_mode={args.em_mode} K={args.K} J={args.J} steps={args.steps} "
          f"loss={args.loss} n_comps={args.n_comps} device={device} dtype={dtype}")

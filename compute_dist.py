"""
Compute distances of TEST images to:
  - GLOBAL (per-channel) PCA subspace  [unchanged]
  - LOCAL FLAT subspace from EM (not the cluster PCA)

Examples:
  python eval_global_vs_local_gpu.py --art_dir /path/to/output_dir --img_dir /path/to/images \
      --width 512 --height 512 --color ycbcr --out_png scatter.png --save_csv distances.csv \
      --device auto --dtype float32
"""

import argparse, json, pickle, csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt
import torch

# ------------- CLI utils -------------
def list_images(root: Path, exts=(".png",".jpg",".jpeg",".bmp",".webp")):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()])

def load_image_to_array(path: Path, target_wh=None, color_space="ycbcr"):
    img = Image.open(path).convert("RGB")
    if target_wh is not None:
        img = img.resize(target_wh, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if color_space.lower() == "ycbcr":
        arr = color.rgb2ycbcr(arr)
    elif color_space.lower() == "rgb":
        pass
    else:
        raise ValueError("color_space must be 'rgb' or 'ycbcr'")
    return arr  # (H,W,3) in [0,1]

def flatten_concat(arr):   # numpy -> numpy
    return arr.reshape(-1)

def flatten_channels(arr): # numpy -> dict of numpy
    H,W,C = arr.shape
    return {
        "Y":  arr[...,0].reshape(-1),
        "Cb": arr[...,1].reshape(-1),
        "Cr": arr[...,2].reshape(-1),
        # compatibility aliases
        "ch0": arr[...,0].reshape(-1),
        "ch1": arr[...,1].reshape(-1),
        "ch2": arr[...,2].reshape(-1),
    }

def has_concat_global(glob_pack):
    p = glob_pack.get("pca", None)
    if p is None:
        return False
    return isinstance(p, dict) and ("mean" in p and "components" in p)

def get_channel_pack(glob_pack, ch_key):
    p = glob_pack["pca"]
    if ch_key in p:
        return p[ch_key]
    mapping = {"Y":"ch0", "Cb":"ch1", "Cr":"ch2"}
    return p[mapping[ch_key]]

# ------------- Torch helpers -------------
def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def pick_dtype(dtype_arg: str):
    return torch.float64 if dtype_arg == "float64" else torch.float32

@torch.no_grad()
def dist_to_pca_subspace_torch(x: torch.Tensor, pca_pack: dict) -> torch.Tensor:
    """
    Residual L2 distance to PCA affine subspace:
      distance = || (I - U^T U)(x - mu) ||_2
    U are PCA components as (r, d) (orthonormal rows up to numeric precision).
    Returns scalar tensor on same device/dtype as x.
    """
    mu = torch.as_tensor(pca_pack["mean"], dtype=x.dtype, device=x.device)
    U  = torch.as_tensor(pca_pack["components"], dtype=x.dtype, device=x.device)  # (r,d)
    xc = x - mu
    proj = U.transpose(0,1).matmul(U.matmul(xc))  # U^T(U xc)
    resid = xc - proj
    return torch.linalg.vector_norm(resid)

@torch.no_grad()
def dist_to_affine_flat_torch(
    x: torch.Tensor,    # (d,) vector
    X: np.ndarray,      # (J,d) basis rows (numpy)
    v: np.ndarray,      # (d,) translation (numpy)
    jitter: float = 1e-8,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None
) -> torch.Tensor:
    """
    L2 distance to affine flat { v + span(rows(X)) } using Gram/Cholesky.
    Equivalent to ||x - (v + P span)||_2.
    """
    # inherit device/dtype from x if not provided
    if device is None: device = x.device
    if dtype  is None: dtype  = x.dtype

    B  = torch.as_tensor(X, dtype=dtype, device=device)   # (J,d)
    vv = torch.as_tensor(v, dtype=dtype, device=device)   # (d,)
    xx = x.to(dtype).to(device)
    xc = xx - vv

    G = B @ B.t()
    if jitter > 0:
        G = G + jitter * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    L = torch.linalg.cholesky(G)
    y = B @ xc
    z = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
    dist2 = torch.clamp(torch.dot(xc, xc) - torch.dot(y, z), min=0)
    return torch.sqrt(dist2)

# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art_dir", type=str, required=True,
                    help="dir with em_artifacts.pkl, pca_globals.pkl, pca_locals.pkl, shapes.json (+ optional splits.json)")
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--color", type=str, choices=["rgb","ycbcr"], default=None,
                    help="optional override; else read shapes.json")
    ap.add_argument("--max_tests", type=int, default=100000)
    ap.add_argument("--out_png", type=str, default="scatter_global_vs_local.png")
    ap.add_argument("--save_csv", type=str, default=None, help="optional CSV path to save per-image distances")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])
    ap.add_argument("--jitter", type=float, default=1e-8, help="diagonal jitter for Gram matrix in flat distance")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype  = pick_dtype(args.dtype)

    art_dir = Path(args.art_dir)
    with open(art_dir / "em_artifacts.pkl", "rb") as f:
        em = pickle.load(f)
    with open(art_dir / "pca_globals.pkl", "rb") as f:
        glob = pickle.load(f)
    with open(art_dir / "pca_locals.pkl", "rb") as f:
        locl = pickle.load(f)  # not used for local distance anymore, but kept for compatibility

    # shapes + test list
    with open(art_dir / "shapes.json", "r") as f:
        sh = json.load(f)
    W = args.width  if args.width  is not None else sh.get("width", 256)
    H = args.height if args.height is not None else sh.get("height", 256)
    color_space = args.color if args.color else sh.get("color", "ycbcr")

    # Prefer explicit test split if present
    if (art_dir / "splits.json").exists():
        with open(art_dir / "splits.json", "r") as f:
            splits = json.load(f)
        test_paths = [Path(p) for p in splits.get("train", [])]
        if not test_paths:
            test_paths = list_images(Path(args.img_dir))
    else:
        test_paths = list_images(Path(args.img_dir))

    mode = em["mode"]  # "concat" or "per-channel"
    globals_are_concat = has_concat_global(glob)

    xs_local, ys_global, names = [], [], []
    used = 0

    for p in test_paths:
        if used >= args.max_tests:
            break

        # --- load and vectorize on CPU -> then move to torch for math ---
        arr = load_image_to_array(p, target_wh=(W, H), color_space=color_space)
        flat_np = flatten_concat(arr)
        chs_np  = flatten_channels(arr)

        x_flat = torch.as_tensor(flat_np, dtype=dtype, device=device)
        xY  = torch.as_tensor(chs_np["Y"],  dtype=dtype, device=device)
        xCb = torch.as_tensor(chs_np["Cb"], dtype=dtype, device=device)
        xCr = torch.as_tensor(chs_np["Cr"], dtype=dtype, device=device)

        # ----- GLOBAL distance (unchanged) -----
        if globals_are_concat:
            d_global = dist_to_pca_subspace_torch(x_flat, glob["pca"])
        else:
            d2 = torch.zeros((), dtype=dtype, device=device)
            for ch, xch in zip(("Y","Cb","Cr"), (xY, xCb, xCr)):
                pack = get_channel_pack(glob, ch)
                d2 = d2 + dist_to_pca_subspace_torch(xch, pack).pow(2)
            d_global = torch.sqrt(d2)

        # ----- LOCAL distance (NOW: distance to EM FLAT) -----
        if mode == "concat":
            K = em["Vs"].shape[0]
            best_d = None
            for k in range(K):
                d = dist_to_affine_flat_torch(
                    x_flat, em["Vs"][k], em["vs"][k],
                    jitter=args.jitter, dtype=dtype, device=device
                )
                best_d = d if (best_d is None or d < best_d) else best_d
            d_local = best_d  # distance to nearest flat in concat space
        else:
            # per-channel: choose nearest flat per channel and combine ||·|| as sqrt(sum_c d_c^2)
            d2_local = torch.zeros((), dtype=dtype, device=device)
            for ch, xch in zip(("Y","Cb","Cr"), (xY, xCb, xCr)):
                K = em["Vs"][ch].shape[0]
                best_d = None
                for k in range(K):
                    d = dist_to_affine_flat_torch(
                        xch, em["Vs"][ch][k], em["vs"][ch][k],
                        jitter=args.jitter, dtype=dtype, device=device
                    )
                    best_d = d if (best_d is None or d < best_d) else best_d
                d2_local = d2_local + best_d.pow(2)
            d_local = torch.sqrt(d2_local)

        ys_global.append(float(d_global.detach().cpu()))
        xs_local.append(float(d_local.detach().cpu()))
        names.append(str(p))
        used += 1

    xs_local = np.array(xs_local, dtype=np.float64)
    ys_global = np.array(ys_global, dtype=np.float64)

    # Scatter plot: x = local-flat, y = global-PCA, plus y=x
    plt.figure(figsize=(6,6), dpi=150)
    plt.scatter(xs_local, ys_global, s=8, alpha=0.6)
    if xs_local.size and ys_global.size:
        lim = float(np.percentile(np.concatenate([xs_local, ys_global]), 99.5))
        lim = max(lim, 1e-6)
    else:
        lim = 1.0
    plt.plot([0, lim], [0, lim], linewidth=1.0)  # y=x
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel("Distance to LOCAL FLAT (EM)")
    plt.ylabel("Distance to GLOBAL PCA")
    plt.title(f"Global PCA vs Local Flat distances (EM={mode}, {color_space.upper()})")
    plt.tight_layout()
    plt.savefig(args.out_png)
    print(f"[OK] Saved scatter to {args.out_png}")

    # quick stats
    if len(xs_local) > 0:
        better_local = int(np.sum(ys_global > xs_local))
        print(f"Points above y=x (local closer): {better_local}/{len(xs_local)} ({100.0*better_local/len(xs_local):.1f}%)")

    # optional CSV
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            # keep column names but clarify local is flat distance
            w.writerow(["image_path", "dist_local_flat", "dist_global_pca"])
            for name, dx, dy in zip(names, xs_local, ys_global):
                w.writerow([name, f"{dx:.6f}", f"{dy:.6f}"])
        print(f"[OK] Wrote distances CSV to {args.save_csv}")

if __name__ == "__main__":
    main()


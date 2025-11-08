import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color
import torch
import matplotlib.pyplot as plt

# Import geometry helper + device from EM implementation
from emlike_cuda import computeDistanceToSubspace, DEVICE


def load_eval_images(art_dir: Path, split: str):
    """
    Load evaluation images from art_dir according to chosen split.

    Args:
        art_dir: directory containing train_imgs/ and/or test_imgs/.
        split: 'test', 'train', or 'auto'.

    Returns:
        eval_arrs: ndarray of shape (3, N, d) in [0,1],
                   where channels are (Y, Cb, Cr).

    Behavior:
        - 'test':  use art_dir/test_imgs/*.png (error if missing/empty).
        - 'train': use art_dir/train_imgs/*.png (error if missing/empty).
        - 'auto':  prefer test_imgs if non-empty; else fallback to train_imgs.
    """
    test_dir = art_dir / "test_imgs"
    train_dir = art_dir / "train_imgs"

    # Decide which directory to use
    if split == "test":
        if not test_dir.exists():
            raise FileNotFoundError("Requested split 'test' but test_imgs/ does not exist.")
        img_list = sorted([p for p in test_dir.glob("*.png") if p.is_file()])
        if len(img_list) == 0:
            raise FileNotFoundError("Requested split 'test' but test_imgs/ is empty.")
        print(f"[eval] Using {len(img_list)} images from test_imgs/ (split=test).")

    elif split == "train":
        if not train_dir.exists():
            raise FileNotFoundError("Requested split 'train' but train_imgs/ does not exist.")
        img_list = sorted([p for p in train_dir.glob("*.png") if p.is_file()])
        if len(img_list) == 0:
            raise FileNotFoundError("Requested split 'train' but train_imgs/ is empty.")
        print(f"[eval] Using {len(img_list)} images from train_imgs/ (split=train).")

    else:  # 'auto'
        img_list = []
        if test_dir.exists():
            img_list = sorted([p for p in test_dir.glob("*.png") if p.is_file()])
        if len(img_list) > 0:
            print(f"[eval] Using {len(img_list)} images from test_imgs/ (split=auto).")
        else:
            if not train_dir.exists():
                raise FileNotFoundError("No test_imgs/ or train_imgs/ found in art_dir.")
            img_list = sorted([p for p in train_dir.glob("*.png") if p.is_file()])
            if len(img_list) == 0:
                raise FileNotFoundError("train_imgs/ is empty; nothing to evaluate.")
            print(f"[eval] Using {len(img_list)} images from train_imgs/ (split=auto fallback).")

    # Build per-channel arrays
    per_ch = [[], [], []]  # Y, Cb, Cr

    for p in img_list:
        im = Image.open(p).convert("RGB")
        arr = np.asarray(im)
        # Match parse.py: YCbCr in 0..255 then scale to [0,1]
        ycbcr = color.rgb2ycbcr(arr).astype(np.float32)    # (H, W, 3)
        flat = ycbcr.reshape(-1, 3) / 255.0               # (d, 3)

        for ch in range(3):
            per_ch[ch].append(flat[:, ch])

    ch_arrays = []
    for ch in range(3):
        ch_arr = np.stack(per_ch[ch], axis=0)  # (N, d)
        ch_arrays.append(ch_arr)

    eval_arrs = np.stack(ch_arrays, axis=0)    # (3, N, d)
    return eval_arrs


def load_artifacts(art_dir: Path):
    """
    Load PCA and EM artifacts from art_dir.

    Requires:
      - pca_object.pkl: list of 3 sklearn PCA objects (Y, Cb, Cr).
      - em_artifacts.pkl: dict with:
            {
              "k": int,
              "j": int,
              "steps": int,
              "channels": [
                  {"Vs": (k,j,d), "vs": (k,d), "elapsed": float},
                  ... (3 entries)
              ]
            }
    """
    pca_path = art_dir / "pca_object.pkl"
    em_path = art_dir / "em_artifacts.pkl"

    if not pca_path.exists():
        raise FileNotFoundError(f"Missing {pca_path}")
    if not em_path.exists():
        raise FileNotFoundError(f"Missing {em_path} (run parse.py with EM first).")

    with open(pca_path, "rb") as f:
        pca_object = pickle.load(f)

    with open(em_path, "rb") as f:
        em_artifacts = pickle.load(f)

    return pca_object, em_artifacts


def compute_distances(eval_arrs: np.ndarray, pca_object, em_artifacts):
    """
    Compute per-image distances to:
      - Global PCA subspace (sum over channels).
      - EM subspaces (sum over channels, min over k flats per channel).

    Uses the same computeDistanceToSubspace as emlike_cuda for consistency.

    Args:
        eval_arrs: (3, N, d) in [0,1].
        pca_object: list of 3 PCA objects.
        em_artifacts: EM results as saved by parse.py.

    Returns:
        dist_pca_total: (N,) numpy array.
        dist_em_total:  (N,) numpy array.
    """
    assert eval_arrs.ndim == 3 and eval_arrs.shape[0] == 3, \
        "eval_arrs must have shape (3, N, d)."

    _, N, d = eval_arrs.shape

    assert len(pca_object) == 3, "Expected 3 PCA objects (Y, Cb, Cr)."
    assert "channels" in em_artifacts and len(em_artifacts["channels"]) == 3, \
        "EM artifacts must contain 3 channel entries."

    dist_pca_all = []
    dist_em_all = []

    for ch in range(3):
        print(f"[dist] Processing channel {ch} ...")

        # ----- Data for this channel -----
        P_ch_np = eval_arrs[ch]  # (N, d)
        P_ch = torch.from_numpy(P_ch_np.astype(np.float32)).to(DEVICE)

        # ----- PCA subspace -----
        pca = pca_object[ch]
        X_pca = torch.from_numpy(pca.components_.astype(np.float32)).to(DEVICE)  # (J, d)
        v_pca = torch.from_numpy(pca.mean_.astype(np.float32)).to(DEVICE)        # (d,)

        d_pca = computeDistanceToSubspace(P_ch, X_pca, v_pca)  # (N,)
        dist_pca_all.append(d_pca)

        # ----- EM subspaces -----
        ch_em = em_artifacts["channels"][ch]
        Vs_np = ch_em["Vs"]  # (k, j, d)
        vs_np = ch_em["vs"]  # (k, d)

        Vs = torch.from_numpy(Vs_np.astype(np.float32)).to(DEVICE)
        vs = torch.from_numpy(vs_np.astype(np.float32)).to(DEVICE)

        k, j, d_em = Vs.shape
        assert d_em == d, "Dimension mismatch between EM bases and eval vectors."

        dists_to_flats = []
        for flat_idx in range(k):
            d_flat = computeDistanceToSubspace(P_ch, Vs[flat_idx], vs[flat_idx])  # (N,)
            dists_to_flats.append(d_flat.unsqueeze(1))

        dists_to_flats = torch.cat(dists_to_flats, dim=1)  # (N, k)
        d_em, _ = dists_to_flats.min(dim=1)                # (N,)

        dist_em_all.append(d_em)

    # Sum across channels -> one scalar per image
    dist_pca_total = torch.stack(dist_pca_all, dim=0).sum(dim=0)  # (N,)
    dist_em_total = torch.stack(dist_em_all, dim=0).sum(dim=0)    # (N,)

    return dist_pca_total.cpu().numpy(), dist_em_total.cpu().numpy()


def plot_comparison(dist_pca: np.ndarray, dist_em: np.ndarray, output_path: Path, split: str):
    """
    Scatter plot:

      x = distance to EM subspaces
      y = distance to global PCA subspace

    Points:
      - BELOW y=x: closer to GLOBAL PCA.
      - ABOVE y=x: closer to EM j-flats.
    """
    assert dist_pca.shape == dist_em.shape
    N = dist_pca.shape[0]
    print(f"[plot] Creating scatter for {N} images (split={split}).")

    x = dist_em
    y = dist_pca

    max_val = float(max(x.max(), y.max()))
    min_val = float(min(x.min(), y.min()))
    pad = 0.02 * (max_val - min_val if max_val > min_val else 1.0)
    lo = max(0.0, min_val - pad)
    hi = max_val + pad

    plt.figure()
    plt.scatter(x, y, alpha=0.5)
    xs = np.linspace(lo, hi, 200)
    plt.plot(xs, xs)  # y = x

    plt.xlabel("Distance to EM subspaces")
    plt.ylabel("Distance to global PCA subspace")
    plt.title(f"Global PCA vs EM Subspaces ({split} split)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"[plot] Saved comparison plot to: {output_path}")


def main():
    """
    Compare per-image distances to PCA vs EM subspaces:

      - Load artifacts from --art_dir.
      - Select split via --split {test,train,auto}.
      - Compute distances.
      - Plot scatter with y=x.
    """
    parser = argparse.ArgumentParser(
        description="Compare per-image distances to Global PCA vs EM subspaces."
    )
    parser.add_argument(
        "--art_dir",
        type=Path,
        required=True,
        help="Directory with pca_object.pkl, em_artifacts.pkl, and train_imgs/test_imgs."
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "train", "auto"],
        default="auto",
        help="Which split to evaluate on: 'test', 'train', or 'auto' "
             "(auto: prefer test if available, else train)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for scatter plot (default: art_dir/dist_em_vs_pca_<split>.png)."
    )

    args = parser.parse_args()
    art_dir: Path = args.art_dir
    split: str = args.split

    if args.out is None:
        out_path = art_dir / f"dist_em_vs_pca_{split}.png"
    else:
        out_path = args.out

    # 1) Load artifacts
    pca_object, em_artifacts = load_artifacts(art_dir)

    # 2) Load evaluation data for requested split
    eval_arrs = load_eval_images(art_dir, split=split)

    # 3) Compute distances
    dist_pca, dist_em = compute_distances(eval_arrs, pca_object, em_artifacts)

    # 4) Plot comparison
    plot_comparison(dist_pca, dist_em, out_path, split)


if __name__ == "__main__":
    main()

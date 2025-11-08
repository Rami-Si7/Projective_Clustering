import numpy as np
from sklearn.decomposition import PCA
from skimage import color
from pathlib import Path
from PIL import Image
import pickle
import argparse
import tqdm
import uuid
import torch



""
"Example for run command"

"python Thesis/ours/em_parse.py -s ImageNet/images -c 50 -n 1000 -t 100 --img_size 128 128 --em_j 50 --em_k 3 --em_steps 20"

""


# Import EM-like CUDA implementation
from emlike_cuda import EMLikeAlg


def visualize(output_dir: Path):
    """
    Visualize normalized components saved in arrs.npy as grayscale images
    under output_dir/vis for quick inspection.
    """
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    array = np.load(output_dir / "arrs.npy")
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            img = (array[i, j] * 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(vis_dir / f"{i}-{j}.png")


def _gather_images(root: Path, exts=(".png",)):
    """
    Recursively collect all image files with given extensions under `root`.
    """
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()])


def prepare_imgs(args, output_dir: Path):
    """
    Create output_dir/{train_imgs,test_imgs} with resized copies.

    Behavior:
      - If --train_from is given:
          TRAIN from that folder (as-is, recursive). Optionally truncated by --n_samples.
      - Else:
          TRAIN sampled RANDOMLY (no stride-2) from --source (non-recursive, *.png).

      - If --test_from is given:
          TEST from that folder (as-is, recursive). Optionally truncated by --n_test.
      - Else if --train_from is None:
          TEST sampled RANDOMLY from remaining images of --source
          (disjoint from train set).
      - Else (--train_from set, no --test_from):
          TEST left empty (allowed).
    """
    W, H = args.img_size
    train_dir = output_dir / "train_imgs"
    test_dir = output_dir / "test_imgs"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    img_list_source = None
    perm_indices = None
    n_train = 0

    # ---------- TRAIN ----------
    if args.train_from is not None:
        # Explicit training source (unchanged semantics)
        train_src = _gather_images(args.train_from)
        if args.n_samples is not None and args.n_samples > 0:
            train_src = train_src[:min(args.n_samples, len(train_src))]
        print(f"[train_from] Using {len(train_src)} images from: {args.train_from}")
        counter = 0
        for p in tqdm.tqdm(train_src, desc="Prepare training images (train_from)"):
            img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            img.save(train_dir / f"{counter:06d}.png")
            counter += 1
    else:
        # Sample TRAIN from --source using RANDOM indices (no stride-2)
        img_list_source = sorted([f for f in args.source.glob("*.png") if f.is_file()])
        if len(img_list_source) == 0:
            raise RuntimeError(f"No PNG images found in source: {args.source}")

        max_train = args.n_samples if args.n_samples is not None else len(img_list_source)
        n_train = min(max_train, len(img_list_source))

        perm_indices = np.random.permutation(len(img_list_source))
        train_indices = perm_indices[:n_train]

        print(f"[source] Using {n_train} random images for TRAIN from: {args.source}")
        counter = 0
        for idx in tqdm.tqdm(train_indices, desc="Prepare training images (source, random)"):
            p = img_list_source[idx]
            img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            img.save(train_dir / f"{counter:06d}.png")
            counter += 1

    # ---------- TEST ----------
    if args.test_from is not None:
        # Explicit testing source (unchanged semantics)
        test_src = _gather_images(args.test_from)
        if args.n_test is not None and args.n_test > 0:
            test_src = test_src[:min(args.n_test, len(test_src))]
        print(f"[test_from] Using {len(test_src)} images from: {args.test_from}")
        counter = 0
        for p in tqdm.tqdm(test_src, desc="Prepare testing images (test_from)"):
            img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
            img.save(test_dir / f"{counter:06d}.png")
            counter += 1

    else:
        # No explicit test_from
        if args.train_from is None:
            # TRAIN from source, so TEST from remaining source images (random, disjoint)
            assert img_list_source is not None and perm_indices is not None
            remaining = perm_indices[n_train:]
            max_test = args.n_test if args.n_test is not None else len(remaining)
            n_test = min(max_test, len(remaining))
            test_indices = remaining[:n_test]

            print(f"[source] Using {n_test} random images for TEST from remaining source images.")
            counter = 0
            for idx in tqdm.tqdm(test_indices, desc="Prepare testing images (source, random)"):
                p = img_list_source[idx]
                img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
                img.save(test_dir / f"{counter:06d}.png")
                counter += 1
        else:
            # train_from provided, no test_from -> leave test empty
            print("[info] --train_from provided without --test_from; leaving test_imgs empty.")


def prepare_arrs(args, output_dir: Path):
    """
    From output_dir/train_imgs:
      - Load all images.
      - Convert to YCbCr in [0,255].
      - Build array train_arrs with shape (3, N, d).
      - Fit PCA per channel.
      - Save normalized PCA components as arrs.npy for visualization.

    Returns:
        pca_object: list of 3 PCA objects (Y, Cb, Cr).
        norm_infos: list of 3 dicts with {min, max} for PCA components.
        train_arrs: ndarray of shape (3, N, d) with raw channel data.
    """
    train_dir = output_dir / "train_imgs"
    img_list = sorted([f for f in train_dir.glob("*.png") if f.is_file()])
    n_samples = len(img_list)
    assert n_samples > 0, "No training images found in train_imgs/."

    # Build (3, N, d)
    train_imgs = []
    for i in tqdm.trange(n_samples, desc="Loading images"):
        im = Image.open(img_list[i]).convert("RGB")
        ycbcr = color.rgb2ycbcr(np.asarray(im)).astype(np.float32)  # (H,W,3) in 0..255
        flat = ycbcr.reshape(-1, 3)                                 # (d,3)
        train_imgs.append(flat)

    train_arrs = np.stack(train_imgs)           # (N, d, 3)
    train_arrs = train_arrs.transpose(2, 0, 1)  # (3, N, d)

    pca_object = []
    norm_infos = []
    norm_comps = []

    for ch in tqdm.trange(3, desc="Fitting PCA comps"):
        ch_arr = train_arrs[ch] / 255.0            # (N, d) in [0,1]
        pca = PCA(n_components=args.n_comps, whiten=False)
        pca.fit(ch_arr)

        comps = pca.components_.astype(np.float32)  # (n_comps, d)
        gmax = float(comps.max())
        gmin = float(comps.min())
        den = (gmax - gmin) if gmax != gmin else 1.0
        norm_comp = (comps - gmin) / den           # normalized for visualization

        pca_object.append(pca)
        norm_infos.append({"min": gmin, "max": gmax})
        norm_comps.append(norm_comp)

    # Shape: (n_comps, d, 3) -> (n_comps, 3, H, W)
    norm_comps = np.stack(norm_comps, axis=-1)     # (n_comps, d, 3)
    W, H = args.img_size
    norm_comps = norm_comps.transpose(0, 2, 1)     # (n_comps, 3, d)
    norm_comps = norm_comps.reshape(args.n_comps, 3, H, W)

    np.save(output_dir / "arrs.npy", norm_comps)
    visualize(output_dir)

    return pca_object, norm_infos, train_arrs


def run_em_per_channel(train_arrs: np.ndarray, args, output_dir: Path):
    """
    Run EM-like (K,J)-projective clustering per channel (Y, Cb, Cr) using CUDA-PyTorch
    implementation from emlike_cuda.EMLikeAlg, and save EM-related artifacts.

    Uses:
        P for each channel: (N, d) in [0,1]
        w: ones (N,)

    Saves:
        - em_artifacts.pkl:
            {
              "k": int,
              "j": int,
              "steps": int,
              "channels": [
                {"Vs": (k,j,d), "vs": (k,d), "elapsed": float},
                ... for 3 channels ...
              ]
            }
        - em_norm_infos.pkl:
            list of 3 dicts {min, max} for EM bases (per channel).
        - em_arrs.npy:
            normalized EM bases as (k*j, 3, H, W) for visualization.
    """
    assert train_arrs.ndim == 3 and train_arrs.shape[0] == 3, \
        "train_arrs must have shape (3, N, d)."

    _, N, d = train_arrs.shape
    W, H = args.img_size

    em_artifacts = {
        "k": args.em_k,
        "j": args.em_j,
        "steps": args.em_steps,
        "channels": []
    }
    em_norm_infos = []
    em_norm_comps_channels = []

    for ch in range(3):
        print(f"[EM] Running EM-like clustering for channel {ch} ...")
        ch_arr = (train_arrs[ch] / 255.0).astype(np.float32)  # (N, d)
        P = torch.from_numpy(ch_arr)                          # CPU tensor; EMLikeAlg will move to CUDA if available
        w = torch.ones(P.size(0), dtype=torch.float32)

        Vs, vs, elapsed = EMLikeAlg(
            P,
            w,
            j=args.em_j,
            k=args.em_k,
            steps=args.em_steps,
        )

        Vs_np = Vs.cpu().numpy()   # (k, j, d)
        vs_np = vs.cpu().numpy()   # (k, d)

        em_artifacts["channels"].append({
            "Vs": Vs_np,
            "vs": vs_np,
            "elapsed": float(elapsed),
        })

        # Flatten all basis vectors across all flats for visualization
        comps = Vs_np.reshape(-1, Vs_np.shape[-1]).astype(np.float32)  # (k*j, d)
        gmax = float(comps.max())
        gmin = float(comps.min())
        den = (gmax - gmin) if gmax != gmin else 1.0
        norm = (comps - gmin) / den

        em_norm_infos.append({"min": gmin, "max": gmax})
        em_norm_comps_channels.append(norm)  # each (k*j, d)

    # Stack normalized EM bases across channels into (k*j, d, 3)
    em_norm_comps_channels = [np.asarray(x) for x in em_norm_comps_channels]
    em_norm_comps = np.stack(em_norm_comps_channels, axis=-1)   # (k*j, d, 3)

    # Reshape to (k*j, 3, H, W) for visualization like PCA arrs.npy
    em_norm_comps = em_norm_comps.transpose(0, 2, 1)            # (k*j, 3, d)
    em_norm_comps = em_norm_comps.reshape(em_norm_comps.shape[0], 3, H, W)

    # Save EM artifacts
    with open(output_dir / "em_artifacts.pkl", "wb") as f:
        pickle.dump(em_artifacts, f)

    with open(output_dir / "em_norm_infos.pkl", "wb") as f:
        pickle.dump(em_norm_infos, f)

    np.save(output_dir / "em_arrs.npy", em_norm_comps)

    print("[EM] Saved EM artifacts: em_artifacts.pkl, em_norm_infos.pkl, em_arrs.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse image set (Global PCA + EM per channel)")

    # Original args
    parser.add_argument("-s", "--source", required=True, type=Path,help="Folder with *.png (used when --train_from/--test_from not given)")
    parser.add_argument("-c", "--n_comps", type=int, default=300)
    parser.add_argument("-n", "--n_samples", type=int, default=10000)
    parser.add_argument("-t", "--n_test", type=int, default=100)
    parser.add_argument( "--img_size", type=int, nargs=2, default=[512, 512], metavar=("width", "height"), help="Target image size (e.g., 512 512)"
    )

    # Explicit training/testing sources (unchanged semantics)
    parser.add_argument(
        "--train_from", type=Path, default=None,
        help="If set, take TRAIN images from this folder (recursively). "
             "Resized copies will be written to output_dir/train_imgs. "
             "If set, overrides sampling from --source."
    )
    parser.add_argument(
        "--test_from", type=Path, default=None,
        help="Optional: take TEST images from this folder (recursively). "
             "If not set and --train_from is given, test set will be empty."
    )

    # EM-like algorithm configuration
    parser.add_argument(
        "--em_j", type=int, default=5,
        help="Dimension j of each affine subspace for EM-like clustering."
    )
    parser.add_argument(
        "--em_k", type=int, default=9,
        help="Number k of j-flats for EM-like clustering."
    )
    parser.add_argument(
        "--em_steps", type=int, default=20,
        help="Number of EM steps per initialization."
    )

    args = parser.parse_args()

    # Output folder naming stays similar
    random_string = str(uuid.uuid4())[:6] + "_" + args.em_j + "_"  + args.em_k
    output_dir = args.source.parent / f"{random_string}-{args.n_comps}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build train_imgs/ and test_imgs/
    prepare_imgs(args, output_dir)

    # 2) Fit global PCA from train_imgs/
    pca_object, norm_infos, train_arrs = prepare_arrs(args, output_dir)

    # 3) Save PCA artifacts
    with open(output_dir / "pca_object.pkl", "wb") as f:
        pickle.dump(pca_object, f)
    with open(output_dir / "norm_infos.pkl", "wb") as f:
        pickle.dump(norm_infos, f)

    print(f"[done] Global PCA saved to: {output_dir}")

    # 4) Run EM-like (K,J)-projective clustering per channel & save artifacts
    run_em_per_channel(train_arrs, args, output_dir)

    print(f"[done] EM per-channel artifacts saved to: {output_dir}")

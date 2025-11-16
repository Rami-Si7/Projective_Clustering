"""
Distance comparison between global PCA and EM-learned subspaces.

Given an artifacts directory produced by `EmParser`, this module:
  * loads the saved PCA object and EM per-channel flats,
  * loads either train or test images (in YCbCr, per pixel),
  * computes, for each image, the total distance of its pixels to:
      1. the global PCA subspace, and
      2. the union of EM flats,
  * saves a scatter plot comparing these distances.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color
import torch
import matplotlib.pyplot as plt

from ours.emlike_cuda import computeDistanceToSubspace, DEVICE

# File extensions considered as images.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


class DistanceComparison:
    """
    High-level runner for the PCA vs EM distance comparison.

    All configuration comes from a flat dict, usually loaded from YAML.
    """

    def __init__(self, cfg):
        if cfg is None:
            raise ValueError("DistanceComparison requires configuration data.")
        cfg_dict = dict(cfg)
        if "art_dir" not in cfg_dict or cfg_dict["art_dir"] is None:
            raise ValueError("compute_dist requires 'art_dir'")

        # Where PCA and EM artifacts (and images) live.
        self.art_dir = Path(cfg_dict["art_dir"]).resolve()

        # Which split(s) to evaluate.
        self.split = str(cfg_dict.get("split", "auto")).lower()
        if self.split not in {"train", "test", "auto", "both"}:
            raise ValueError(f"Unsupported split: {self.split}")
        self._resolved_splits = (
            ["test", "train"] if self.split in {"auto", "both"} else [self.split]
        )

        # Optional explicit output path when evaluating a single split.
        out_value = cfg_dict.get("out")
        self.out_path = Path(out_value).resolve() if out_value else None

        # Cleaned configuration snapshot for logging.
        self.cfg = {
            "art_dir": self.art_dir,
            "split": self.split,
            "resolved_splits": list(self._resolved_splits),
            "out": self.out_path,
        }

    def run(self):
        """
        Main entrypoint: run the configured comparisons and return list of plots.
        """
        pca_object, em_artifacts = self._load_artifacts(self.art_dir)

        splits_to_run = list(self._resolved_splits)
        multi_split = len(splits_to_run) > 1
        if multi_split and self.out_path is not None:
            print(
                "[compute_dist] Ignoring custom 'out' because multiple splits were requested."
            )

        saved_plots = []
        channel_names = ["Y", "Cb", "Cr"]
        for split in splits_to_run:
            print(f"\n[compute_dist] Processing split: {split}")
            try:
                eval_arrs = self._load_eval_images(self.art_dir, split=split)
                dist_pca_per_ch, dist_em_per_ch = self._compute_distances(
                    eval_arrs, pca_object, em_artifacts
                )
                # Generate one plot per channel instead of summing them
                for ch in range(3):
                    ch_name = channel_names[ch] if ch < len(channel_names) else f"ch{ch}"
                    if multi_split:
                        out_path = self.art_dir / f"dist_em_vs_pca_{split}_ch{ch_name}.png"
                    else:
                        # If a custom --out was provided, append channel name before suffix
                        if self.out_path is not None:
                            stem = self.out_path.stem
                            suffix = self.out_path.suffix or ".png"
                            out_path = self.out_path.with_name(f"{stem}_{ch_name}{suffix}")
                        else:
                            out_path = self.art_dir / f"dist_em_vs_pca_{split}_ch{ch_name}.png"

                    self._plot_comparison(
                        dist_pca_per_ch[ch],
                        dist_em_per_ch[ch],
                        out_path,
                        split,
                        channel_name=ch_name,
                    )
                    saved_plots.append(out_path)
            except FileNotFoundError as e:
                print(f"[compute_dist] Skipping {split}: {e}")

        return saved_plots

    @staticmethod
    def _load_eval_images(art_dir: Path, split: str):
        def list_images(directory: Path):
            if not directory.exists():
                return []
            return sorted(
                [
                    p
                    for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ]
            )

        # Directories with resized training and test images prepared by EmParser.
        test_dir = art_dir / "test_imgs"
        train_dir = art_dir / "train_imgs"

        if split == "test":
            img_list = list_images(test_dir)
            if len(img_list) == 0:
                raise FileNotFoundError(
                    "Requested split 'test' but test_imgs/ is missing or empty."
                )
            print(f"[eval] Using {len(img_list)} images from test_imgs/ (split=test).")

        elif split == "train":
            img_list = list_images(train_dir)
            if len(img_list) == 0:
                raise FileNotFoundError(
                    "Requested split 'train' but train_imgs/ is missing or empty."
                )
            print(
                f"[eval] Using {len(img_list)} images from train_imgs/ (split=train)."
            )

        else:
            raise ValueError(f"Unsupported split passed to _load_eval_images: {split}")

        per_ch = [[], [], []]
        for p in img_list:
            im = Image.open(p).convert("RGB")
            arr = np.asarray(im)
            ycbcr = color.rgb2ycbcr(arr).astype(np.float32)
            flat = ycbcr.reshape(-1, 3) / 255.0

            for ch in range(3):
                per_ch[ch].append(flat[:, ch])

        ch_arrays = []
        for ch in range(3):
            ch_arr = np.stack(per_ch[ch], axis=0)
            ch_arrays.append(ch_arr)

        eval_arrs = np.stack(ch_arrays, axis=0)
        return eval_arrs

    @staticmethod
    def _load_artifacts(art_dir: Path):
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

    @staticmethod
    def _compute_distances(eval_arrs: np.ndarray, pca_object, em_artifacts):
        assert (
            eval_arrs.ndim == 3 and eval_arrs.shape[0] == 3
        ), "eval_arrs must have shape (3, N, d)."
        _, N, d = eval_arrs.shape

        assert len(pca_object) == 3, "Expected 3 PCA objects (Y, Cb, Cr)."
        assert (
            "channels" in em_artifacts and len(em_artifacts["channels"]) == 3
        ), "EM artifacts must contain 3 channel entries."

        dist_pca_all = []
        dist_em_all = []

        for ch in range(3):
            print(f"[dist] Processing channel {ch} ...")

            P_ch_np = eval_arrs[ch]
            P_ch = torch.from_numpy(P_ch_np.astype(np.float32)).to(DEVICE)

            pca = pca_object[ch]
            X_pca = torch.from_numpy(pca.components_.astype(np.float32)).to(DEVICE)
            v_pca = torch.from_numpy(pca.mean_.astype(np.float32)).to(DEVICE)

            d_pca = computeDistanceToSubspace(P_ch, X_pca, v_pca)
            dist_pca_all.append(d_pca)

            ch_em = em_artifacts["channels"][ch]
            Vs_np = ch_em["Vs"]
            vs_np = ch_em["vs"]

            Vs = torch.from_numpy(Vs_np.astype(np.float32)).to(DEVICE)
            vs = torch.from_numpy(vs_np.astype(np.float32)).to(DEVICE)

            k, _, d_em = Vs.shape
            assert d_em == d, "Dimension mismatch between EM bases and eval vectors."

            dists_to_flats = []
            for flat_idx in range(k):
                d_flat = computeDistanceToSubspace(P_ch, Vs[flat_idx], vs[flat_idx])
                dists_to_flats.append(d_flat.unsqueeze(1))

            dists_to_flats = torch.cat(dists_to_flats, dim=1)
            d_em, _ = dists_to_flats.min(dim=1)

            dist_em_all.append(d_em)

        # Shape: (3, N) – one distance per channel per image
        dist_pca_per_channel = torch.stack(dist_pca_all, dim=0)
        dist_em_per_channel = torch.stack(dist_em_all, dim=0)

        return dist_pca_per_channel.cpu().numpy(), dist_em_per_channel.cpu().numpy()

    @staticmethod
    def _plot_comparison(
        dist_pca: np.ndarray,
        dist_em: np.ndarray,
        output_path: Path,
        split: str,
        channel_name: str | None = None,
    ):
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
        plt.plot(xs, xs)

        plt.xlabel("Distance to EM subspaces")
        plt.ylabel("Distance to global PCA subspace")
        if channel_name is not None:
            plt.title(f"Global PCA vs EM Subspaces ({split} split, channel={channel_name})")
        else:
            plt.title(f"Global PCA vs EM Subspaces ({split} split)")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close()

        print(f"[plot] Saved comparison plot to: {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare per-image distances to Global PCA vs EM subspaces."
    )
    parser.add_argument(
        "--art_dir",
        type=Path,
        required=True,
        help="Directory with pca_object.pkl, em_artifacts.pkl, and train_imgs/test_imgs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "train", "auto", "both"],
        default="auto",
        help="Which split(s) to evaluate: 'train', 'test', 'both', or 'auto' (same as both).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for scatter plot (default: art_dir/dist_em_vs_pca_<split>.png). "
        "Ignored when multiple splits are requested.",
    )
    return parser


def main():
    parser = argparse.ArgumentParser(description="Compare EM vs PCA distances")
    parser.add_argument(
        "--art_dir",
        type=Path,
        required=True,
        help="Directory with pca_object.pkl, em_artifacts.pkl, and train_imgs/test_imgs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "train", "auto", "both"],
        default="auto",
        help="Which split(s) to evaluate: 'train', 'test', 'both', or 'auto' (same as both).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path for scatter plot (ignored when multiple splits run).",
    )
    args = parser.parse_args()

    cfg = vars(args)
    runner = DistanceComparison(cfg)
    runner.run()


if __name__ == "__main__":
    main()

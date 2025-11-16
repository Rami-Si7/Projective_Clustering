"""
High-level image parsing and EM-like clustering pipeline (GPU-centric).

This module does the following, end-to-end:
  1. Gathers training and test images from directories on disk.
  2. Resizes images and converts them to YCbCr, then fits a global PCA basis
     per color channel.
  3. Runs an EM-like algorithm (see `ours.emlike_cuda`) on per-channel pixels
     to learn a collection of j-dimensional flats.
  4. Saves PCA and EM artifacts (and small visualizations) under an output
     directory that encodes the main hyperparameters.

The whole workflow is wrapped by `EmParser.run()`, which is invoked from
`run.py` using configuration loaded from a YAML file.
"""

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

# Import EM-like CUDA implementation
import random
from ours import emlike_cuda as emc
from ours.emlike_cuda import EMLikeAlg

# File extensions considered as images.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def visualize(output_dir: Path):
    """
    Save PCA components (from arrs.npy) as grayscale images under output_dir/vis.

    This is mostly for quick qualitative inspection of the learned basis.
    """
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    array = np.load(output_dir / "arrs.npy")
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            img = (array[i, j] * 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(vis_dir / f"{i}-{j}.png")


def _gather_images(root: Path, recursive: bool = True, exts=None):
    """
    Collect image files with given extensions under `root`.

    If `recursive` is True we descend into subdirectories, otherwise we only
    consider direct children.
    """
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    exts = tuple(exts or IMAGE_EXTENSIONS)
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        [p for p in iterator if p.is_file() and p.suffix.lower() in exts]
    )


class EmParser:
    """
    Main driver for the EM parsing pipeline.

    The constructor takes a flat config dictionary (typically loaded from YAML),
    normalizes types/paths, and stores a cleaned version on `self.cfg`.
    """

    def __init__(self, cfg):
        if cfg is None:
            raise ValueError("EmParser requires configuration data.")

        cfg_dict = dict(cfg)

        # If a generic `test_data` directory is given and no explicit `test_from`
        # is specified, use it as the test source.
        if cfg_dict.get("test_data") is not None and cfg_dict.get("test_from") is None:
            cfg_dict["test_from"] = cfg_dict["test_data"]

        # Normalize `img_size` into a (W, H) tuple of ints.
        img_size = cfg_dict.get("img_size") or (128, 128)
        if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            self.img_size = (int(img_size[0]), int(img_size[1]))
        else:
            size = int(img_size)
            self.img_size = (size, size)

        def _to_path(value):
            """Convert a config value to an absolute `Path`, or None."""
            return Path(value).resolve() if value is not None else None

        self.source = _to_path(cfg_dict.get("source"))
        self.train_from = _to_path(cfg_dict.get("train_from"))
        self.test_from = _to_path(cfg_dict.get("test_from"))
        self.test_data = _to_path(cfg_dict.get("test_data"))
        self.output_dir = _to_path(cfg_dict.get("output_dir"))

        self.n_comps = int(cfg_dict.get("n_comps", 100))
        self.n_samples = (
            None if cfg_dict.get("n_samples") is None else int(cfg_dict["n_samples"])
        )
        self.n_test = None if cfg_dict.get("n_test") is None else int(cfg_dict["n_test"])
        self.em_j = int(cfg_dict.get("em_j", 100))
        self.em_k = int(cfg_dict.get("em_k", 5))
        self.em_steps = int(cfg_dict.get("em_steps", 20))
        self.em_num_inits = int(cfg_dict.get("em_num_inits", 10))
        self.em_patience = int(cfg_dict.get("em_patience", 5))
        self.em_objective = str(cfg_dict.get("em_objective", "lp"))
        self.em_jitter = float(cfg_dict.get("em_jitter", 1e-6))
        self.em_min_delta = float(cfg_dict.get("em_min_delta", 0.0))
        self.em_lambda = float(cfg_dict.get("em_lambda", 1.0))
        self.seed = None if cfg_dict.get("seed") is None else int(cfg_dict["seed"])

        # Store a cleaned snapshot of the resolved configuration for logging.
        self.cfg = {
            "source": self.source,
            "train_from": self.train_from,
            "test_from": self.test_from,
            "test_data": self.test_data,
            "img_size": list(self.img_size),
            "n_comps": self.n_comps,
            "n_samples": self.n_samples,
            "n_test": self.n_test,
            "em_j": self.em_j,
            "em_k": self.em_k,
            "em_steps": self.em_steps,
            "em_num_inits": self.em_num_inits,
            "em_patience": self.em_patience,
            "em_objective": self.em_objective,
            "em_jitter": self.em_jitter,
            "em_min_delta": self.em_min_delta,
            "em_lambda": self.em_lambda,
            "seed": self.seed,
            "output_dir": self.output_dir,
        }

        self._output_dir = None

    def ensure_output_dir(self) -> Path:
        """
        Create (or return) the artifacts directory without running the full pipeline.

        Useful for logging configuration files before long-running steps begin.
        """
        return self._prepare_output_dir()

    def run(self) -> Path:
        """
        Execute the full pipeline and return the artifacts directory.
        """
        self._configure_randomness()
        output_dir = self._prepare_output_dir()
        self._prepare_imgs(output_dir)
        pca_object, norm_infos, train_arrs = self._prepare_arrs(output_dir)
        self._save_pca_artifacts(output_dir, pca_object, norm_infos)
        self._run_em_per_channel(train_arrs, output_dir)
        print(f"[done] EM per-channel artifacts saved to: {output_dir}")
        return output_dir

    # ----- internal helpers -----
    def _configure_randomness(self) -> None:
        """
        Seed all relevant RNGs and propagate `em_lambda` into the EM module.
        """
        seed = self.seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        # Configure robust-loss scale parameter used inside `emlike_cuda`.
        emc.LAMBDA = self.em_lambda

    def _prepare_output_dir(self) -> Path:
        """
        Create (or reuse) the directory where all artifacts will be stored.

        The directory name encodes `em_j`, `em_k`, and `n_comps` so that
        multiple runs with different hyperparameters do not collide.
        """
        if self._output_dir is not None:
            return self._output_dir
        base_output_dir = self.output_dir or (Path.cwd() / "outputs")
        random_string = (
            str(uuid.uuid4())[:6]
            + "_"
            + str(self.em_j)
            + "_"
            + str(self.em_k)
        )
        output_dir = base_output_dir / f"{random_string}-{self.n_comps}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir
        return output_dir

    def _prepare_imgs(self, output_dir: Path) -> None:
        """
        Populate `train_imgs/` and `test_imgs/` under `output_dir`.

        Training and testing images are taken either from:
          - explicit folders (`train_from`, `test_from`), or
          - sampled randomly from `source` if explicit folders are not given.
        """
        W, H = self.img_size
        train_dir = output_dir / "train_imgs"
        test_dir = output_dir / "test_imgs"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        img_list_source = None
        perm_indices = None
        n_train = 0

        if self.train_from is not None:
            train_src = _gather_images(self.train_from)
            if self.n_samples is not None and self.n_samples > 0:
                train_src = train_src[: min(self.n_samples, len(train_src))]
            print(f"[train_from] Using {len(train_src)} images from: {self.train_from}")
            counter = 0
            for p in tqdm.tqdm(train_src, desc="Prepare training images (train_from)"):
                img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
                img.save(train_dir / f"{counter:06d}.png")
                counter += 1
        else:
            img_list_source = _gather_images(self.source, recursive=False)
            if len(img_list_source) == 0:
                raise RuntimeError(f"No supported images found in source: {self.source}")

            max_train = self.n_samples if self.n_samples is not None else len(img_list_source)
            n_train = min(max_train, len(img_list_source))

            perm_indices = np.random.permutation(len(img_list_source))
            train_indices = perm_indices[:n_train]

            print(f"[source] Using {n_train} random images for TRAIN from: {self.source}")
            counter = 0
            for idx in tqdm.tqdm(
                train_indices, desc="Prepare training images (source, random)"
            ):
                p = img_list_source[idx]
                img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
                img.save(train_dir / f"{counter:06d}.png")
                counter += 1

        if self.test_from is not None:
            test_src = _gather_images(self.test_from)
            if self.n_test is not None and self.n_test > 0:
                test_src = test_src[: min(self.n_test, len(test_src))]
            print(f"[test_from] Using {len(test_src)} images from: {self.test_from}")
            counter = 0
            for p in tqdm.tqdm(test_src, desc="Prepare testing images (test_from)"):
                img = Image.open(p).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
                img.save(test_dir / f"{counter:06d}.png")
                counter += 1
        else:
            if self.train_from is None:
                assert img_list_source is not None and perm_indices is not None
                remaining = perm_indices[n_train:]
                max_test = self.n_test if self.n_test is not None else len(remaining)
                n_test = min(max_test, len(remaining))
                test_indices = remaining[:n_test]

                print(
                    f"[source] Using {n_test} random images for TEST from remaining source images."
                )
                counter = 0
                for idx in tqdm.tqdm(
                    test_indices, desc="Prepare testing images (source, random)"
                ):
                    p = img_list_source[idx]
                    img = (
                        Image.open(p)
                        .convert("RGB")
                        .resize((W, H), Image.Resampling.LANCZOS)
                    )
                    img.save(test_dir / f"{counter:06d}.png")
                    counter += 1
            else:
                print(
                    "[info] --train_from provided without --test_from; leaving test_imgs empty."
                )

    def _prepare_arrs(self, output_dir: Path):
        """
        Load training images, convert to YCbCr, and fit PCA per channel.

        Returns
        -------
        pca_object : list
            Fitted `sklearn` PCA objects, one per channel.
        norm_infos : list
            Per-channel min/max information used to normalize components.
        train_arrs : ndarray
            Raw per-channel pixel arrays with shape (3, N, d).
        """
        train_dir = output_dir / "train_imgs"
        img_list = _gather_images(train_dir, recursive=False)
        n_samples = len(img_list)
        assert n_samples > 0, "No training images found in train_imgs/."

        train_imgs = []
        for i in tqdm.trange(n_samples, desc="Loading images"):
            im = Image.open(img_list[i]).convert("RGB")
            ycbcr = color.rgb2ycbcr(np.asarray(im)).astype(np.float32)
            flat = ycbcr.reshape(-1, 3)
            train_imgs.append(flat)

        train_arrs = np.stack(train_imgs)
        train_arrs = train_arrs.transpose(2, 0, 1)

        pca_object = []
        norm_infos = []
        norm_comps = []

        for ch in tqdm.trange(3, desc="Fitting PCA comps"):
            ch_arr = train_arrs[ch] / 255.0
            pca = PCA(n_components=self.n_comps, whiten=False)
            pca.fit(ch_arr)

            comps = pca.components_.astype(np.float32)
            gmax = float(comps.max())
            gmin = float(comps.min())
            den = (gmax - gmin) if gmax != gmin else 1.0
            norm_comp = (comps - gmin) / den

            pca_object.append(pca)
            norm_infos.append({"min": gmin, "max": gmax})
            norm_comps.append(norm_comp)

        norm_comps = np.stack(norm_comps, axis=-1)
        W, H = self.img_size
        norm_comps = norm_comps.transpose(0, 2, 1)
        norm_comps = norm_comps.reshape(self.n_comps, 3, H, W)

        np.save(output_dir / "arrs.npy", norm_comps)
        visualize(output_dir)

        return pca_object, norm_infos, train_arrs

    def _save_pca_artifacts(
        self, output_dir: Path, pca_object, norm_infos
    ) -> None:
        """
        Persist PCA objects and normalization metadata to disk.
        """
        with open(output_dir / "pca_object.pkl", "wb") as f:
            pickle.dump(pca_object, f)
        with open(output_dir / "norm_infos.pkl", "wb") as f:
            pickle.dump(norm_infos, f)
        print(f"[done] Global PCA saved to: {output_dir}")

    def _run_em_per_channel(self, train_arrs: np.ndarray, output_dir: Path) -> None:
        """
        Run the EM-like algorithm independently on each Y/Cb/Cr channel.

        For each channel we:
          - convert pixel intensities to [0, 1],
          - run `EMLikeAlg` to obtain k j-flats,
          - normalize the learned bases for visualization,
          - save both raw and normalized EM artifacts.
        """
        assert train_arrs.ndim == 3 and train_arrs.shape[0] == 3, \
            "train_arrs must have shape (3, N, d)."

        _, _, _ = train_arrs.shape
        W, H = self.img_size

        em_artifacts = {
            "k": self.em_k,
            "j": self.em_j,
            "steps": self.em_steps,
            "channels": [],
        }
        em_norm_infos = []
        em_norm_comps_channels = []

        for ch in range(3):
            print(f"[EM] Running EM-like clustering for channel {ch} ...")
            ch_arr = (train_arrs[ch] / 255.0).astype(np.float32)
            P = torch.from_numpy(ch_arr)
            w = torch.ones(P.size(0), dtype=torch.float32)

            Vs, vs, elapsed = EMLikeAlg(
                P,
                w,
                j=self.em_j,
                k=self.em_k,
                steps=self.em_steps,
                objective=self.em_objective,
                jitter=self.em_jitter,
                num_inits=self.em_num_inits,
                patience=self.em_patience,
                min_delta=self.em_min_delta,
            )

            Vs_np = Vs.cpu().numpy()
            vs_np = vs.cpu().numpy()

            em_artifacts["channels"].append(
                {"Vs": Vs_np, "vs": vs_np, "elapsed": float(elapsed)}
            )

            comps = Vs_np.reshape(-1, Vs_np.shape[-1]).astype(np.float32)
            gmax = float(comps.max())
            gmin = float(comps.min())
            den = (gmax - gmin) if gmax != gmin else 1.0
            norm = (comps - gmin) / den

            em_norm_infos.append({"min": gmin, "max": gmax})
            em_norm_comps_channels.append(norm)

        em_norm_comps_channels = [np.asarray(x) for x in em_norm_comps_channels]
        em_norm_comps = np.stack(em_norm_comps_channels, axis=-1)
        em_norm_comps = em_norm_comps.transpose(0, 2, 1)
        em_norm_comps = em_norm_comps.reshape(em_norm_comps.shape[0], 3, H, W)

        with open(output_dir / "em_artifacts.pkl", "wb") as f:
            pickle.dump(em_artifacts, f)
        with open(output_dir / "em_norm_infos.pkl", "wb") as f:
            pickle.dump(em_norm_infos, f)
        np.save(output_dir / "em_arrs.npy", em_norm_comps)

        print("[EM] Saved EM artifacts: em_artifacts.pkl, em_norm_infos.pkl, em_arrs.npy")


def main():
    parser = argparse.ArgumentParser(description="Parse dataset with EM workflow")
    parser.add_argument(
        "-s",
        "--source",
        type=Path,
        default=None,
        help="Folder with *.png (used when --train_from/--test_from not given)",
    )
    parser.add_argument("-c", "--n_comps", type=int, default=100)
    parser.add_argument("-n", "--n_samples", type=int, default=2500)
    parser.add_argument("-t", "--n_test", type=int, default=100)
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[128, 128],
        metavar=("width", "height"),
        help="Target image size",
    )
    parser.add_argument(
        "--train_from",
        type=Path,
        default=None,
        help="Take TRAIN images from this folder (recursively). Overrides --source sampling.",
    )
    parser.add_argument(
        "--test_from",
        type=Path,
        default=None,
        help="Optional TEST image folder (recursively).",
    )
    parser.add_argument(
        "--test_data",
        type=Path,
        default=None,
        help="Optional dataset for evaluation separate from the training source.",
    )
    parser.add_argument("--em_j", type=int, default=100)
    parser.add_argument("--em_k", type=int, default=5)
    parser.add_argument("--em_steps", type=int, default=20)
    parser.add_argument(
        "--em_objective",
        type=str,
        default="lp",
        choices=["lp", "huber", "cauchy", "geman_McClure", "welsch", "tukey"],
    )
    parser.add_argument("--em_num_inits", type=int, default=10)
    parser.add_argument("--em_jitter", type=float, default=1e-6)
    parser.add_argument("--em_patience", type=int, default=5)
    parser.add_argument("--em_min_delta", type=float, default=0.0)
    parser.add_argument("--em_lambda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Base output directory (default: ./outputs).",
    )
    args = parser.parse_args()

    # When used as a script, we build a config dict directly from CLI args.
    cfg = vars(args)
    parser_obj = EmParser(cfg)
    parser_obj.run()


if __name__ == "__main__":
    main()

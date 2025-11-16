## Projective_Clustering

This repository implements an **EM-like projective clustering algorithm** and tooling around it:

- `em_parse` step: parses an image dataset, builds a global PCA basis per color channel, and runs an EM-like algorithm that learns a set of low-dimensional affine subspaces ("flats").
- `compute_dist` step: given the saved artifacts, compares per-image distances to the global PCA subspace vs the learned EM subspaces.

The modern entrypoint is the orchestrator script `run.py`, which wires these pieces together and tracks runs.

### Repository layout

- `run.py` – high‑level runner that reads a YAML config, runs selected steps (`em_parse`, `compute_dist`), and writes run metadata directly next to the produced artifacts (optionally also under a user-provided `--run-dir`).
- `configs/` – example YAML configs:
  - `default.yaml` – main default configuration (all parameters filled in).
  - `example.yaml` – small example overriding a few parameters for a custom experiment.
- `ours/` – newer, GPU‑aware implementation:
  - `em_parse.py` – end‑to‑end pipeline to prepare images, fit global PCA, and run EM‑like clustering per channel.
  - `compute_dist.py` – distance comparison between PCA and EM subspaces, with plotting.
  - `emlike_cuda.py` – CUDA / PyTorch implementation of the EM‑like algorithm and related geometry helpers.
  - `utils.py` – small helpers for reading/writing YAML.
- `base/` – original CPU reference implementation:
  - `EmLike.py` – NumPy/SciPy version of the EM‑like algorithm.
  - `parse_base.py` – older parsing/PCA script for images.

### Quick start

1. **Prepare your data**

   Point `configs/default.yaml` (or a copy of it) to your image directory:

   ```yaml
   em_parse:
     source: /absolute/path/to/your/images
   ```

   You can also use `train_from` / `test_from` to specify explicit train/test folders.

2. **Run the default pipeline**

   From the repository root:

   ```bash
   python run.py --config configs/default.yaml
   ```

   This will:

   - write the config, per-step settings, and a summary directly into the artifact directory (and optionally into `--run-dir` if you ask for one),
   - write EM/PCA artifacts under an automatically named directory in `outputs/`,
   - optionally run `compute_dist` and save comparison plots into the artifacts directory.

3. **Custom experiments**

   Copy the default config and tweak what you need:

   ```bash
   cp configs/default.yaml configs/my_experiment.yaml
   # edit configs/my_experiment.yaml
   python run.py --config configs/my_experiment.yaml
   ```

   You can also look at `configs/example.yaml` for a minimal override‑style config.

#### Config inheritance

Config files can declare an `extends` list to reuse a base configuration and
override only the values you care about:

```yaml
extends:
  - configs/default.yaml

name: my_custom_run
em_parse:
  n_samples: 1000
  em_k: 8
```

The orchestrator resolves the chain of base configs before running, so the
stored `run_config.yaml` always reflects the full, merged configuration.

### Core concepts (very short)

- **Global PCA**: For each Y/Cb/Cr channel, we fit a PCA basis over pixels from the training images.
- **EM‑like flats**: The EM‑like algorithm learns `k` affine `j`‑dimensional flats per channel that approximately minimize a robust loss over distances from pixels to their closest flat.
- **Distance comparison**: `compute_dist` evaluates, per image, how far its pixels lie from:
  1. the global PCA subspace, and
  2. the union of EM flats,
  then plots these distances against each other.

The code is heavily commented so that an outside reader can follow the flow module‑by‑module. If you want a place to start reading, open `run.py`, then follow into `ours/em_parse.py`, `ours/emlike_cuda.py`, and `ours/compute_dist.py`.



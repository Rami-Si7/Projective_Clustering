#!/usr/bin/env python3
"""
Sweep over EM subspace dimension j while reusing a single PCA.

Usage:
    python sweep_j.py --config configs/example.yaml --j-min 250 --step 1
"""

import argparse
from pathlib import Path
import shutil
import csv

import matplotlib.pyplot as plt

# <<< NEW: import dump_yaml as well >>>
from ours.utils import load_config, dump_yaml
from ours.em_parse import EmParser
from ours.compute_dist import DistanceComparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep EM dimension j (subspace size) using a single PCA."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration (same as used for run.py).",
    )
    parser.add_argument(
        "--j-min",
        type=int,
        default=1,
        help="Minimum j (inclusive) to sweep down to. Default: 1",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step by which to decrement j. Default: 1",
    )
    return parser.parse_args()


# <<< NEW: same cleaner as in run.py, to make YAML-serializable dicts >>>
def _clean_config(data):
    """
    Convert nested configuration data into something YAML/JSON serializable.

    - `Path` objects become strings.
    - Lists/tuples are processed recursively.
    """
    if isinstance(data, dict):
        return {k: _clean_config(v) for k, v in data.items()}
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, (list, tuple)):
        return [_clean_config(v) for v in data]
    return data


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load configuration (respects `extends` etc. via load_config).
    # ------------------------------------------------------------------
    config_path = args.config
    config_data = load_config(config_path)

    em_cfg_raw = dict(config_data.get("em_parse", {}))
    dist_cfg_raw = dict(config_data.get("compute_dist", {}))

    if not em_cfg_raw:
        raise RuntimeError("Config has no 'em_parse' section.")

    # Base j from YAML (starting point for the sweep).
    base_j = int(em_cfg_raw.get("em_j", 100))
    j_min = int(args.j_min)
    step = int(abs(args.step))

    if j_min > base_j:
        raise ValueError(f"j_min ({j_min}) must be <= base_j ({base_j})")

    print(f"[sweep_j] Config: {config_path}")
    print(f"[sweep_j] Sweeping j from {base_j} down to {j_min} (step={step})")

    # ------------------------------------------------------------------
    # 2. One-time PCA + image prep using EmParser internals.
    #    We DO NOT call EmParser.run(), to avoid running EM at this stage.
    # ------------------------------------------------------------------
    em_parser = EmParser(em_cfg_raw)

    # Configure seeds & LAMBDA once
    em_parser._configure_randomness()

    # Prepare artifacts directory (encodes base j, k, n_comps in its name)
    output_dir = em_parser.ensure_output_dir()
    print(f"[sweep_j] Artifacts base directory: {output_dir}")

    # <<< NEW: save top-level config + sweep-specific config into output_dir >>>
    # (similar in spirit to run.py)
    run_config_path = output_dir / "run_config.yaml"
    dump_yaml(run_config_path, _clean_config(config_data))
    print(f"[sweep_j] Saved run_config.yaml to: {run_config_path}")

    sweep_cfg = {
        "config_file": str(config_path.resolve()),
        "base_j": base_j,
        "j_min": j_min,
        "step": step,
        "em_parse": _clean_config(em_cfg_raw),
        "compute_dist": _clean_config(dist_cfg_raw),
    }
    sweep_cfg_path = output_dir / "sweep_j_config.yaml"
    dump_yaml(sweep_cfg_path, sweep_cfg)
    print(f"[sweep_j] Saved sweep_j_config.yaml to: {sweep_cfg_path}")
    # <<< NEW END >>>

    # Prepare train/test images under output_dir/train_imgs and output_dir/test_imgs
    em_parser._prepare_imgs(output_dir)

    # Compute PCA once and save pca_object.pkl, arrs.npy, vis, etc.
    pca_object, norm_infos, train_arrs = em_parser._prepare_arrs(output_dir)
    em_parser._save_pca_artifacts(output_dir, pca_object, norm_infos)
    print("[sweep_j] PCA computed once and saved.")

    # ------------------------------------------------------------------
    # 3. Determine which splits to use, based on compute_dist.split
    # ------------------------------------------------------------------
    split_cfg = str(dist_cfg_raw.get("split", "auto")).lower()
    if split_cfg in {"auto", "both"}:
        splits_to_run = ["train", "test"]
    else:
        splits_to_run = [split_cfg]

    print(f"[sweep_j] Will run distance comparison for splits: {splits_to_run}")

    # ------------------------------------------------------------------
    # 4. Prepare CSV + containers for plotting delta vs j.
    # ------------------------------------------------------------------
    metrics_csv_path = output_dir / "sweep_j_metrics.csv"
    print(f"[sweep_j] Metrics CSV will be written to: {metrics_csv_path}")

    # For plotting: mean delta (PCA - EM) per (split, j) for OVERALL distances.
    deltas_overall_for_plot = {split: [] for split in splits_to_run}
    # For plotting per-channel deltas: dict[split][channel] -> list of (j, delta)
    channel_names = ["Y", "Cb", "Cr"]
    deltas_channels_for_plot = {
        split: {ch: [] for ch in channel_names} for split in splits_to_run
    }

    with metrics_csv_path.open("w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            ["j", "split", "scope", "mean_pca", "mean_em", "mean_delta"]
        )

        # --------------------------------------------------------------
        # 5. Sweep j: for each j, run EM (per-channel) and DistanceComparison.
        # --------------------------------------------------------------
        for j in range(base_j, j_min - 1, -step):
            print("\n" + "=" * 80)
            print(f"[sweep_j] Running EM and distance comparison for j = {j}")
            print("=" * 80)

            # Update parser's j and re-run EM only (no PCA)
            em_parser.em_j = j
            em_parser._run_em_per_channel(train_arrs, output_dir)
            print(f"[sweep_j] EM finished for j={j}. Artifacts written to {output_dir}")

            # For convenience, keep a j-specific copy of EM artifacts
            base_em_file = output_dir / "em_artifacts.pkl"
            if base_em_file.exists():
                j_specific_em = output_dir / f"em_artifacts_j{j}.pkl"
                shutil.copy2(base_em_file, j_specific_em)
                print(f"[sweep_j] Saved j-specific EM artifacts: {j_specific_em}")

            # Run DistanceComparison for each split separately so
            # we can control output filenames (and avoid overwriting).
            for split in splits_to_run:
                dist_cfg = dict(dist_cfg_raw)
                dist_cfg["art_dir"] = str(output_dir)
                dist_cfg["split"] = split

                # j-specific base filename; DistanceComparison will derive
                # per-channel names from this when split is a single value.
                out_name = f"dist_j{j}_{split}.png"
                dist_cfg["out"] = output_dir / out_name

                print(
                    f"[sweep_j]  -> DistanceComparison for split='{split}', out='{out_name}'"
                )
                dist_runner = DistanceComparison(dist_cfg)
                plots = dist_runner.run()
                for p in plots:
                    print(f"[sweep_j]     saved plot: {p}")

                # --- Collect metrics from DistanceComparison ---
                metrics = getattr(dist_runner, "last_metrics", None)
                if metrics is None:
                    print(
                        "[sweep_j] WARNING: DistanceComparison did not expose "
                        "last_metrics; skipping CSV rows for this run."
                    )
                    continue

                # metrics[split] holds summary for this split
                split_metrics = metrics.get(split)
                if split_metrics is None:
                    print(
                        f"[sweep_j] WARNING: No metrics recorded for split '{split}'."
                    )
                    continue

                # Overall row for this (j, split)
                overall = split_metrics.get("overall", {})
                mean_pca = overall.get("mean_pca")
                mean_em = overall.get("mean_em")
                mean_delta = overall.get("mean_delta")

                if (
                    mean_pca is not None
                    and mean_em is not None
                    and mean_delta is not None
                ):
                    writer.writerow(
                        [
                            j,
                            split,
                            "overall",
                            mean_pca,
                            mean_em,
                            mean_delta,
                        ]
                    )
                    # Keep for plotting overall
                    deltas_overall_for_plot.setdefault(split, []).append(
                        (j, mean_delta)
                    )

                # Per-channel rows: Y, Cb, Cr
                channels = split_metrics.get("channels", {})
                for ch_name in channel_names:
                    m = channels.get(ch_name)
                    if m is None:
                        continue
                    writer.writerow(
                        [
                            j,
                            split,
                            ch_name,
                            m.get("mean_pca"),
                            m.get("mean_em"),
                            m.get("mean_delta"),
                        ]
                    )
                    # Keep per-channel deltas for plotting
                    ch_delta = m.get("mean_delta")
                    if ch_delta is not None:
                        deltas_channels_for_plot[split][ch_name].append(
                            (j, ch_delta)
                        )

    # ------------------------------------------------------------------
    # 6. Plot delta vs j for each split (OVERALL distances).
    # ------------------------------------------------------------------
    delta_plot_path = output_dir / "sweep_j_delta_overall.png"
    print(f"[sweep_j] Plotting mean delta (PCA - EM) vs j to: {delta_plot_path}")

    plt.figure()
    for split, pairs in deltas_overall_for_plot.items():
        if not pairs:
            continue
        # Sort by j ascending for a nice left-to-right curve.
        pairs_sorted = sorted(pairs, key=lambda t: t[0])
        js = [p[0] for p in pairs_sorted]
        deltas = [p[1] for p in pairs_sorted]
        plt.plot(js, deltas, marker="o", label=f"{split} (overall)")

    plt.xlabel("EM subspace dimension j")
    plt.ylabel("mean(distance_PCA - distance_EM)")
    plt.title("Delta of distances (PCA - EM) vs j (overall)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(delta_plot_path, bbox_inches="tight", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # 7. Plot per-channel delta vs j for each split.
    # ------------------------------------------------------------------
    for split in splits_to_run:
        split_pairs = deltas_channels_for_plot.get(split, {})
        if not split_pairs:
            continue

        ch_plot_path = output_dir / f"sweep_j_delta_channels_{split}.png"
        print(
            f"[sweep_j] Plotting per-channel mean delta (PCA - EM) vs j for split='{split}' "
            f"to: {ch_plot_path}"
        )

        plt.figure()
        for ch_name in channel_names:
            pairs = split_pairs.get(ch_name, [])
            if not pairs:
                continue
            pairs_sorted = sorted(pairs, key=lambda t: t[0])
            js = [p[0] for p in pairs_sorted]
            deltas = [p[1] for p in pairs_sorted]
            plt.plot(js, deltas, marker="o", label=f"{ch_name}")

        plt.xlabel("EM subspace dimension j")
        plt.ylabel("mean(distance_PCA - distance_EM)")
        plt.title(f"Delta of distances per channel vs j ({split} split)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(ch_plot_path, bbox_inches="tight", dpi=200)
        plt.close()

    print("\n[sweep_j] Done sweeping j.")
    print(f"[sweep_j] Metrics CSV: {metrics_csv_path}")
    print(f"[sweep_j] Overall delta plot:  {delta_plot_path}")
    for split in splits_to_run:
        print(
            f"[sweep_j] Channel delta plot for {split}: "
            f"{output_dir / f'sweep_j_delta_channels_{split}.png'}"
        )


if __name__ == "__main__":
    main()

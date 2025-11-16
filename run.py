"""
High-level experiment runner.

This script is the main entrypoint for the modern pipeline. It:
  1. Loads a YAML config (see files under `configs/`).
  2. Determines which steps to run (EM parsing and/or distance comparison).
  3. Creates a timestamped run directory under `runs/`.
  4. Dispatches to `EmParser` and `DistanceComparison` with the corresponding
     configuration sections.

All step-specific logic lives in `ours.em_parse` and `ours.compute_dist`.
"""

import argparse
import datetime
from pathlib import Path

from ours.compute_dist import DistanceComparison
from ours.em_parse import EmParser
from ours.utils import dump_yaml, load_config

# Steps that this orchestrator knows how to run.
ALLOWED_STEPS = ["em_parse", "compute_dist"]


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


def _dump_to_dirs(directories: set[Path], filename: str, data) -> None:
    """
    Write the same YAML payload into every directory in `directories`.
    """
    for directory in directories:
        dump_yaml(directory / filename, data)


def _track_metadata_dir(
    metadata_dirs: set[Path], directory: Path, config_data: dict
) -> None:
    """
    Remember a directory for metadata and immediately save the top-level config there.
    """
    resolved = directory.resolve()
    if resolved in metadata_dirs:
        return
    metadata_dirs.add(resolved)
    _dump_to_dirs({resolved}, "run_config.yaml", config_data)


def main() -> None:
    """
    Parse CLI arguments, load the config, and run the requested steps.
    """
    parser = argparse.ArgumentParser(description="Run experiments with YAML configs.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration describing the experiments to run.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALLOWED_STEPS,
        help="Subset of steps to execute (default: order from config).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Where to write metadata for this run (default: ./runs).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional label for the run (falls back to config name or config file stem).",
    )
    args = parser.parse_args()

    # Load high-level configuration.
    config_path = args.config
    config_data = load_config(config_path)

    # Human-readable label for this run.
    run_label = args.name or config_data.get("name") or config_path.stem

    # Determine which steps to execute:
    #   1. CLI wins if provided.
    #   2. Otherwise use the `steps` list from the config.
    #   3. Fallback: run all known steps in the default order.
    steps_from_config = config_data.get("steps")
    if args.steps:
        steps_order = args.steps
    elif steps_from_config:
        steps_order = list(steps_from_config)
    else:
        steps_order = ALLOWED_STEPS[:]

    invalid = [step for step in steps_order if step not in ALLOWED_STEPS]
    if invalid:
        raise ValueError(f"Unknown steps requested: {invalid}")

    run_dir: Path | None = None
    if args.run_dir:
        run_root = Path(args.run_dir).resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = run_root / f"{run_label}-{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate a small summary dictionary as we go.
    run_summary = {
        "name": run_label,
        "config_file": str(config_path.resolve()),
        "steps": steps_order,
        "results": {},
    }

    # Remember the most recent EM artifacts directory so compute_dist can use it
    # if the config does not explicitly specify an `art_dir`.
    previous_art_dir: Path | None = None
    metadata_dirs: set[Path] = set()
    if run_dir:
        _track_metadata_dir(metadata_dirs, run_dir, config_data)

    for step in steps_order:
        if step == "em_parse":
            # Run the EM parsing pipeline on images.
            em_raw = dict(config_data.get("em_parse", {}))
            em_parser = EmParser(em_raw)
            output_dir = em_parser.ensure_output_dir()
            _track_metadata_dir(metadata_dirs, output_dir, config_data)
            output_dir = em_parser.run()

            # Persist the exact resolved configuration used by EmParser.
            clean_em_cfg = _clean_config(em_parser.cfg)
            em_targets = {output_dir}
            if run_dir:
                em_targets.add(run_dir)
            _dump_to_dirs(em_targets, "em_parse_config.yaml", clean_em_cfg)

            previous_art_dir = output_dir
            run_summary["results"]["em_parse"] = {
                "output_dir": str(output_dir),
            }

        elif step == "compute_dist":
            # Compare distances to PCA vs EM subspaces.
            dist_raw = dict(config_data.get("compute_dist", {}))

            # If `art_dir` is not given, reuse the artifacts directory from the
            # preceding `em_parse` step.
            if dist_raw.get("art_dir") is None:
                if previous_art_dir is None:
                    raise RuntimeError(
                        "compute_dist requires an art_dir when em_parse has not run."
                    )
                dist_raw["art_dir"] = str(previous_art_dir)

            dist_runner = DistanceComparison(dist_raw)
            art_dir = Path(dist_runner.cfg["art_dir"])
            _track_metadata_dir(metadata_dirs, art_dir, config_data)
            plots = dist_runner.run()

            clean_dist_cfg = _clean_config(dist_runner.cfg)
            dist_targets = {art_dir}
            if run_dir:
                dist_targets.add(run_dir)
            _dump_to_dirs(dist_targets, "compute_dist_config.yaml", clean_dist_cfg)

            run_summary["results"]["compute_dist"] = {
                "art_dir": str(art_dir),
                "plots": [str(p) for p in plots],
            }

    # Persist a compact summary of the whole run.
    if metadata_dirs:
        _dump_to_dirs(metadata_dirs, "run_summary.yaml", run_summary)

    if metadata_dirs:
        for target in sorted(metadata_dirs):
            print(f"[run] Stored run metadata at: {target}")
    for step_name, info in run_summary["results"].items():
        print(f"[run] {step_name}: {info}")


if __name__ == "__main__":
    main()


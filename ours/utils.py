from __future__ import annotations
"""
Small YAML helper utilities.

These helpers keep all YAML I/O in one place, so other modules can assume
they are working with plain Python dicts and not worry about file handling.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Set

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return a dictionary.

    The function:
      - returns an empty dict when the file is empty,
      - enforces that the top-level YAML object is a mapping.
    """
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(data).__name__}")
    return data


def dump_yaml(path: Path, data: Any) -> None:
    """
    Write a Python object to YAML.

    The parent directory is created if needed; keys are kept in insertion order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries without mutating the inputs.

    - Nested mappings are merged depth-first.
    - Non-mapping values from `overrides` replace those in `base`.
    """
    merged = dict(base)
    for key, value in overrides.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(base_value, value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load a YAML config file, resolving any `extends` references recursively.

    Files can declare:

    ```
    extends:
      - ../configs/default.yaml
    ```

    Relative paths are resolved against the directory of the file that declares
    them. Later entries override earlier ones, and the file's own contents
    override everything they extend.
    """
    return _load_config_recursive(Path(path).resolve(), seen=set())


def _load_config_recursive(path: Path, seen: Set[Path]) -> Dict[str, Any]:
    if path in seen:
        raise ValueError(f"Detected cyclic config inheritance at {path}")

    seen.add(path)
    try:
        data = load_yaml(path)
        extends = data.pop("extends", None)
        if not extends:
            return data

        if isinstance(extends, (str, Path)):
            extends_list: Iterable[Any] = [extends]
        else:
            extends_list = list(extends)

        merged: Dict[str, Any] = {}
        for entry in extends_list:
            base_path = Path(entry)
            if not base_path.is_absolute():
                base_path = (path.parent / base_path).resolve()
            merged = _deep_merge_dicts(merged, _load_config_recursive(base_path, seen))

        merged = _deep_merge_dicts(merged, data)
        return merged
    finally:
        seen.remove(path)


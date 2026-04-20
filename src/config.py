from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.is_absolute():
        p = project_root() / p
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Invalid config")
    cfg["_config_path"] = str(p)
    cfg["_project_root"] = str(project_root())
    return cfg


def resolve_path(path_value: str | Path, base: str | Path | None = None) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)

    if base is None:
        base = project_root()
    b = Path(base)
    return str((b / p).resolve())

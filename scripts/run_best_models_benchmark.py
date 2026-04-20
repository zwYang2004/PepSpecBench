import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _symlink_or_copy(src: Path, dst: Path) -> None:
    _ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(str(src), str(dst))
    except Exception:
        import shutil

        shutil.copy2(str(src), str(dst))


def _stage_checkpoint(model_name: str, ckpt_path: str, model_dir: Path) -> None:
    src = Path(str(ckpt_path)).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(str(src))

    model_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "prosit":
        _symlink_or_copy(src, model_dir / "prosit.pt")
    elif model_name == "prosit_transformer":
        _symlink_or_copy(src, model_dir / "prosit_transformer.pt")
    elif model_name == "unispec":
        _symlink_or_copy(src, model_dir / "unispec.pt")
    elif model_name == "alphapeptdeep":
        _symlink_or_copy(src, model_dir / "alphapeptdeep.pt")
    elif model_name == "predfull_torch":
        _symlink_or_copy(src, model_dir / "predfull_torch.pt")
    elif model_name == "fastspel":
        ckpt_dir = model_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _symlink_or_copy(src, ckpt_dir / "X.npz")
    else:
        raise ValueError(f"Unknown model for staging: {model_name}")


def _select_checkpoint(model_spec: Dict[str, Any], eval_dataset: str) -> Optional[str]:
    ckpt = model_spec.get("checkpoint")
    if isinstance(ckpt, str) and ckpt.strip():
        return str(ckpt)

    ckpts = model_spec.get("checkpoints")
    if not isinstance(ckpts, dict):
        return None

    for key in (eval_dataset, "default"):
        v = ckpts.get(key)
        if isinstance(v, str) and v.strip():
            return str(v)

    return None


def _default_ood_source_for_model(model_name: str) -> str:
    if model_name in ("predfull_torch", "unispec"):
        return "prospect_unimod135"
    return "prospect_unimod135_234d"


def _resolve_ood_base_dir(ood_dir_arg: str) -> Path:
    if isinstance(ood_dir_arg, str) and ood_dir_arg.strip():
        return Path(ood_dir_arg).expanduser()
    env_dir = os.environ.get("PEPSPECBENCH_OOD_DIR", "")
    if env_dir.strip():
        return Path(env_dir).expanduser()
    return Path("data/ood_reference")


def _load_base_config(config_path: str) -> Dict[str, Any]:
    from scripts.run_benchmark import load_config

    return load_config(config_path)


def _build_constraints(model_cfg: Dict[str, Any]) -> Any:
    from src.constraints import DataConstraints

    constraints_cfg = model_cfg.get("constraints", {}) if isinstance(model_cfg, dict) else {}
    return DataConstraints(
        min_len=int(constraints_cfg.get("min_len", 6)),
        max_len=int(constraints_cfg.get("max_len", 40)),
        max_charge=int(constraints_cfg.get("max_charge", 6)),
        allowed_unimod_ids=tuple(constraints_cfg.get("allowed_unimod_ids", [1, 4, 35])),
    )


def _get_runner(model_name: str, constraints: Any) -> Any:
    from src.models.registry import MODEL_REGISTRY

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not registered")
    return MODEL_REGISTRY[model_name](constraints=constraints)


def _get_dataset_subset_paths(config: Dict[str, Any], dataset_key: str) -> Dict[str, str]:
    ds = config.get("datasets", {}).get(dataset_key, {})
    test = ds.get("test", {}) if isinstance(ds, dict) else {}
    subsets = test.get("subsets", {}) if isinstance(test, dict) else {}
    out: Dict[str, str] = {}
    for k, v in subsets.items():
        if isinstance(v, dict) and v.get("path"):
            out[str(k)] = str(v["path"])
    if not out and isinstance(test, dict) and test.get("path"):
        out["all"] = str(test["path"])
    return out


def _read_parquet_head(parquet_path: str, *, max_rows: int, columns: List[str]) -> "Any":
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = Path(str(parquet_path))
    if p.is_dir():
        files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix == ".parquet"])
        if not files:
            raise FileNotFoundError(str(p))
        p = files[0]

    pf = pq.ParquetFile(str(p))
    batches = []
    remaining = int(max_rows)
    for batch in pf.iter_batches(batch_size=min(8192, remaining), columns=columns):
        batches.append(batch)
        remaining -= int(len(batch))
        if remaining <= 0:
            break
    if not batches:
        import pandas as pd

        return pd.DataFrame()
    table = pa.Table.from_batches(batches)
    return table.to_pandas()


def _prepare_ood_for_ion_models(src_parquet: str, out_parquet: str, *, max_samples: int) -> str:
    import numpy as np
    import pandas as pd

    from scripts.data_process.add_level1_labels import compute_level1_for_row

    cols = ["sequence", "charge", "nce", "mz", "intensity"]
    df = _read_parquet_head(src_parquet, max_rows=int(max_samples), columns=cols)
    if df is None or len(df) == 0:
        raise ValueError(f"Empty OOD parquet: {src_parquet}")

    df = df.copy().reset_index(drop=True)
    df["collision_energy"] = df["nce"].astype(float)

    labels = []
    for i in range(len(df)):
        row = df.iloc[i]
        seq = str(row.get("sequence", ""))
        if not seq:
            labels.append(np.full((234,), -1.0, dtype=np.float32))
            continue
        labels.append(compute_level1_for_row(row, "sequence", "charge", "mz", "intensity"))

    df["intensities_raw"] = labels

    outp = Path(out_parquet)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)
    return str(outp)


def _prepare_ood_for_predfull(src_parquet: str, out_parquet: str, *, max_samples: int) -> str:
    df = _read_parquet_head(
        src_parquet,
        max_rows=int(max_samples),
        columns=["sequence", "charge", "nce", "mz", "intensity", "pepmass"],
    )
    if df is None or len(df) == 0:
        raise ValueError(f"Empty OOD parquet: {src_parquet}")

    df = df.copy().reset_index(drop=True)
    df["modified_sequence"] = df["sequence"].astype(str)
    df["precursor_charge"] = df["charge"].astype(int)
    df["collision_energy"] = df["nce"].astype(float)
    df["intensities"] = df["intensity"]
    df["precursor_mz"] = df.get("pepmass", 0.0)

    keep = ["modified_sequence", "precursor_charge", "collision_energy", "precursor_mz", "mz", "intensities"]
    df = df[keep]

    outp = Path(out_parquet)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)
    return str(outp)


def _ood_reference_sources(base: Path) -> List[Tuple[str, str, str]]:
    return [
        ("ood_homo_sapiens", str(base / "homo_sapiens_with_seq.parquet"), "core"),
        ("ood_coli", str(base / "coli_with_seq.parquet"), "core"),
        ("ood_c_elegans", str(base / "c_elegans_with_seq.parquet"), "core"),
        ("ood_a_lia", str(base / "a_lia_with_seq.parquet"), "core"),
        ("ood_ptm_ecoli", str(base / "PTM" / "Ecoli.parquet"), "ptm"),
        ("ood_ptm_hela_trypsin", str(base / "PTM" / "HeLa_trypsin.parquet"), "ptm"),
        ("ood_ptm_yeast", str(base / "PTM" / "Yeast.parquet"), "ptm"),
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-config", default="configs/benchmark_unimod135.yaml")
    p.add_argument("--best-models", default="configs/best_models.yaml")
    p.add_argument("--datasets", default="prospect_unimod135,massive_kb_unimod135,ood_reference")
    p.add_argument("--models", default="all")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--run-tag", default="best_models_eval")
    p.add_argument("--run-dir", default="")
    p.add_argument("--continue-run", action="store_true")
    p.add_argument("--export-per-sample", action="store_true")
    p.add_argument("--export-max-samples", type=int, default=20000)
    p.add_argument("--ood-dir", default="")
    p.add_argument("--ood-max-samples", type=int, default=50000)
    p.add_argument("--prepare-ood", action="store_true")
    args = p.parse_args()

    base_cfg = _load_base_config(str(args.benchmark_config))
    best_cfg = _read_yaml(str(args.best_models))

    models_cfg = best_cfg.get("best_models", {})
    if not isinstance(models_cfg, dict):
        raise ValueError("best_models.yaml must contain key: best_models")

    model_names = sorted(models_cfg.keys())
    if args.models != "all":
        requested = [x.strip() for x in str(args.models).split(",") if x.strip()]
        for m in requested:
            if m not in model_names:
                raise ValueError(f"Model '{m}' not found in best_models.yaml")
        model_names = requested

    output_root = Path(str(args.output_dir)).expanduser().resolve()
    if args.run_dir:
        run_dir = Path(str(args.run_dir)).expanduser().resolve()
        if run_dir.exists() and not bool(args.continue_run):
            raise ValueError(f"--run-dir exists: {run_dir}; use --continue-run")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / f"eval_{args.run_tag}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

    base_cfg.setdefault("protocol", {}).setdefault("analysis", {})
    base_cfg["protocol"]["analysis"]["export_per_sample"] = bool(args.export_per_sample)
    base_cfg["protocol"]["analysis"]["export_max_samples"] = int(args.export_max_samples)

    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]

    ood_base_dir = _resolve_ood_base_dir(str(args.ood_dir))
    ood_cache_dir = run_dir / "ood_prepared"

    results: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "ts": float(time.time()),
        "datasets": datasets,
        "models": model_names,
        "results": [],
    }

    for model_name in model_names:
        model_spec = models_cfg.get(model_name, {})
        if not isinstance(model_spec, dict):
            continue

        model_cfg = base_cfg.get("models", {}).get(model_name, {}) if isinstance(base_cfg.get("models", {}), dict) else {}
        if model_name in ("prosit", "prosit_transformer", "alphapeptdeep", "fastspel", "unispec", "predfull_torch"):
            model_cfg = dict(model_cfg)
            model_cfg["eval_max_samples"] = None

        constraints = _build_constraints(model_cfg)
        runner = _get_runner(model_name, constraints)

        for ds in datasets:
            if ds in ("prospect_unimod135", "massive_kb_unimod135"):
                ckpt_path = _select_checkpoint(model_spec, ds)
                if not ckpt_path:
                    results["results"].append(
                        {
                            "model": model_name,
                            "dataset": ds,
                            "subset": "*",
                            "status": "skipped",
                            "reason": f"No checkpoint configured for dataset '{ds}'",
                        }
                    )
                    continue

                model_dir = run_dir / model_name / ds
                if bool(args.continue_run) and (model_dir / "_staged.ok").exists():
                    pass
                else:
                    _stage_checkpoint(str(model_name), str(ckpt_path), model_dir)
                    (model_dir / "_staged.ok").write_text("ok")

                subset_paths = _get_dataset_subset_paths(base_cfg, ds)
                schema = "massive_kb" if ds.startswith("massive") else "prospect"
                cfg = dict(base_cfg)
                cfg["__dataset_schema"] = schema
                cfg.setdefault("models", {})
                cfg["models"] = dict(cfg.get("models", {}))
                cfg["models"][model_name] = dict(cfg["models"].get(model_name, {}))
                if bool(args.export_per_sample):
                    cfg.setdefault("protocol", {})
                    cfg["protocol"] = dict(cfg.get("protocol", {}))
                    analysis = dict(cfg.get("protocol", {}).get("analysis", {}) or {})
                    analysis["export_per_sample"] = True
                    analysis["export_max_samples"] = int(args.export_max_samples)
                    cfg["protocol"]["analysis"] = analysis

                for subset_name, parquet_path in subset_paths.items():
                    try:
                        eval_res = runner.predict(parquet_path=str(parquet_path), model_dir=str(model_dir), config=cfg)
                        results["results"].append(
                            {
                                "model": model_name,
                                "dataset": ds,
                                "subset": subset_name,
                                "parquet": str(parquet_path),
                                "output_dir": str(model_dir),
                                "checkpoint": str(ckpt_path),
                                "result": eval_res,
                                "status": "success",
                            }
                        )
                    except Exception as e:
                        results["results"].append(
                            {
                                "model": model_name,
                                "dataset": ds,
                                "subset": subset_name,
                                "parquet": str(parquet_path),
                                "output_dir": str(model_dir),
                                "checkpoint": str(ckpt_path),
                                "error": str(e),
                                "status": "failed",
                            }
                        )

            elif ds == "ood_reference":
                ood_source = str(model_spec.get("ood_source") or _default_ood_source_for_model(model_name))
                ckpt_path = _select_checkpoint(model_spec, ood_source)
                if not ckpt_path:
                    results["results"].append(
                        {
                            "model": model_name,
                            "dataset": "ood_reference",
                            "subset": "*",
                            "status": "skipped",
                            "reason": f"No checkpoint configured for ood_source '{ood_source}'",
                        }
                    )
                    continue

                model_dir = run_dir / model_name / f"ood_reference_from_{ood_source}"
                if bool(args.continue_run) and (model_dir / "_staged.ok").exists():
                    pass
                else:
                    _stage_checkpoint(str(model_name), str(ckpt_path), model_dir)
                    (model_dir / "_staged.ok").write_text("ok")

                schema = "prospect"
                cfg = dict(base_cfg)
                cfg["__dataset_schema"] = schema

                for tag, src, group in _ood_reference_sources(ood_base_dir):
                    if not Path(src).exists():
                        continue

                    parquet_path_use = str(src)

                    if model_name == "predfull_torch":
                        if bool(args.prepare_ood):
                            parquet_path_use = _prepare_ood_for_predfull(
                                src,
                                str(ood_cache_dir / f"{tag}_predfull_n{int(args.ood_max_samples)}.parquet"),
                                max_samples=int(args.ood_max_samples),
                            )
                    elif model_name == "unispec":
                        results["results"].append(
                            {
                                "model": model_name,
                                "dataset": "ood_reference",
                                "subset": tag,
                                "parquet": str(src),
                                "output_dir": str(model_dir),
                                "status": "skipped",
                                "checkpoint": str(ckpt_path),
                                "reason": "The OOD reference set only provides peak lists; UniSpecRunner expects fixed-length full-spectrum label vectors used during training",
                            }
                        )
                        continue
                    else:
                        if bool(args.prepare_ood):
                            parquet_path_use = _prepare_ood_for_ion_models(
                                src,
                                str(ood_cache_dir / f"{tag}_234d_n{int(args.ood_max_samples)}.parquet"),
                                max_samples=int(args.ood_max_samples),
                            )
                        else:
                            results["results"].append(
                                {
                                    "model": model_name,
                                    "dataset": "ood_reference",
                                    "subset": tag,
                                    "parquet": str(src),
                                    "output_dir": str(model_dir),
                                    "status": "skipped",
                                    "checkpoint": str(ckpt_path),
                                    "reason": "This OOD reference set needs intensities_raw(234d). Re-run with --prepare-ood",
                                }
                            )
                            continue

                    try:
                        eval_res = runner.predict(parquet_path=str(parquet_path_use), model_dir=str(model_dir), config=cfg)
                        results["results"].append(
                            {
                                "model": model_name,
                                "dataset": "ood_reference",
                                "subset": tag,
                                "parquet": str(parquet_path_use),
                                "output_dir": str(model_dir),
                                "group": group,
                                "checkpoint": str(ckpt_path),
                                "result": eval_res,
                                "status": "success",
                            }
                        )
                    except Exception as e:
                        results["results"].append(
                            {
                                "model": model_name,
                                "dataset": "ood_reference",
                                "subset": tag,
                                "parquet": str(parquet_path_use),
                                "output_dir": str(model_dir),
                                "group": group,
                                "checkpoint": str(ckpt_path),
                                "error": str(e),
                                "status": "failed",
                            }
                        )

            else:
                raise ValueError(f"Unknown dataset: {ds}")

    with open(run_dir / "best_models_benchmark_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

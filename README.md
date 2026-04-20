# PepSpecBench

Unified benchmark for peptide MS/MS spectrum prediction.

## Introduction
This repository provides a minimal, reproducible code release for PepSpecBench:
- unified in-domain benchmark entrypoint,
- cross-species OOD evaluation,
- metadata sensitivity analysis (NCE/charge),
- figure/table regeneration scripts used by the manuscript.

Official reporting uses the shared canonical `level1` metrics only.

## Data Sources
- PROSPECT: https://github.com/wilhelm-lab/PROSPECT
- MassIVE-KB (Zenodo release): https://zenodo.org/records/14967861

## Repository Structure
- `src/`: benchmark core framework
- `configs/`: benchmark configuration files
- `scripts/data_process/`: public-data reconstruction pipeline
- `scripts/`: benchmark and OOD entrypoints
- `scripts/analysis/`: analysis and plotting scripts
- `requirements.txt`

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
1. Reconstruct benchmark-ready data from public sources:
```bash
python scripts/data_process/preprocess_parquet.py
python scripts/data_process/process_prospect_data.py
python scripts/data_process/process_massivekb_data.py
python scripts/data_process/sample_smart_splits.py
```

2. Run unified benchmark:
```bash
python scripts/run_benchmark.py --config configs/benchmark_mini.yaml
```

3. Run OOD evaluation:
```bash
python scripts/eval_ood.py
bash scripts/run_ood_eval_all.sh
```

For OOD evaluation, provide your local OOD dataset directory explicitly instead of relying on a repository-specific path:
```bash
PEPSPECBENCH_OOD_DIR=/path/to/ood_reference bash scripts/run_ood_eval_all.sh
python scripts/run_best_models_benchmark.py --datasets ood_reference --ood-dir /path/to/ood_reference --prepare-ood
```

4. Run model-sensitivity analysis:
```bash
python scripts/analysis/execute_model_sensitivity.py --help
python scripts/analysis/build_model_sensitivity_tables.py --help
```

5. Regenerate key figures:
```bash
python scripts/analysis/plot_dataset_schematic.py
python scripts/analysis/plot_benchmark_comparison.py
python scripts/analysis/plot_properties_grid_4x3_top4_sa_pcc.py
python scripts/analysis/plot_model_sensitivity_nce_gradient.py --help
python scripts/analysis/plot_charge_mode_collapse.py --help
```

## Add a New Model (Minimal Example)
1. Add a new runner by copying an existing template:
```bash
cp src/models/runners/prosit_runner.py src/models/runners/my_model_runner.py
```

2. Implement your runner to satisfy `BenchmarkModel` API in `src/models/base_model.py`:
- `get_data_constraints()`
- `fit(train_path, val_path, output_dir, config)`
- `predict(parquet_path, model_dir, config)`

3. Register the model key in `src/models/registry.py`.
```python
from src.models.runners.my_model_runner import MyModelRunner
MODEL_REGISTRY["my_model"] = MyModelRunner
```

4. Add model config (example) to your benchmark yaml:
```yaml
models:
  my_model:
    active: true
    type: deep_gpu
    batch_size: 64
    base_lr: 1.0e-4
    max_epochs: 5
```

5. Run compatibility check, then run full benchmark entrypoint:
```bash
python scripts/run_benchmark.py --config configs/benchmark_mini.yaml --dataset prospect_mini --models my_model --check-only
python scripts/run_benchmark.py --config configs/benchmark_mini.yaml --dataset prospect_mini --models my_model
```

## Reproducibility Notes
- This package intentionally excludes raw outputs, logs, checkpoints, and archived temporary scripts.
- Third-party baselines are not vendored; see `THIRD_PARTY.md` for upstream sources and license reminders.

## Citation
If you use PepSpecBench, please cite the associated manuscript (BibTeX will be added in camera-ready release).

## License
This project is licensed under the MIT License. See `LICENSE`.

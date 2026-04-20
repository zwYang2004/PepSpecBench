import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _iter_parquet_files(path: str) -> List[Path]:
    p = Path(path)
    if p.is_dir():
        files = sorted([f for f in p.rglob("*.parquet") if f.is_file()])
        if not files:
            raise FileNotFoundError(f"No parquet files found under: {p}")
        return files
    if not p.exists():
        raise FileNotFoundError(f"Parquet path not found: {p}")
    return [p]


def _ion_name(pos_1idx: int, ion_type: str, frag_charge: int) -> str:
    base = f"{ion_type}{int(pos_1idx)}"
    if int(frag_charge) > 1:
        base = f"{base}^{int(frag_charge)}"
    return base


def _ion_order(max_len: int = 40, max_frag_charge: int = 3) -> List[str]:
    ions: List[str] = []
    for pos in range(1, int(max_len)):
        for ion_type in ("y", "b"):
            for z in range(1, int(max_frag_charge) + 1):
                ions.append(_ion_name(pos, ion_type, z))
    return ions


def _load_label_matrix(values: Iterable[object], dim: int) -> Tuple[np.ndarray, int]:
    rows: List[np.ndarray] = []
    bad = 0
    for v in values:
        try:
            y = np.asarray(v, dtype=np.float32).reshape(-1)
        except Exception:
            bad += 1
            continue
        if y.size != int(dim):
            bad += 1
            continue
        rows.append(y)
    if not rows:
        return np.zeros((0, int(dim)), dtype=np.float32), int(bad)
    return np.stack(rows, axis=0), int(bad)


def build_ion_stats(
    parquet_path: str,
    *,
    out_dir: str,
    label_column: str = "intensities_raw",
    max_len: int = 40,
    max_frag_charge: int = 3,
) -> None:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    ions = _ion_order(max_len=max_len, max_frag_charge=max_frag_charge)
    dim = len(ions)

    occurs = np.zeros((dim,), dtype=np.int64)
    sum_int = np.zeros((dim,), dtype=np.float64)
    cnt_int = np.zeros((dim,), dtype=np.int64)

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("pyarrow is required to build UniSpec dictionaries") from e

    files = _iter_parquet_files(parquet_path)

    bad_rows = 0
    total_rows = 0

    for fp in files:
        pf = pq.ParquetFile(str(fp))
        for rg in range(int(pf.num_row_groups)):
            table = pf.read_row_group(rg, columns=[label_column])
            df = table.to_pandas()
            if label_column not in df.columns:
                raise KeyError(f"Label column not found: {label_column}")
            mat, bad = _load_label_matrix(df[label_column].values, dim)
            bad_rows += int(bad)
            if mat.size == 0:
                continue
            total_rows += int(mat.shape[0])

            pos = mat > 0
            occurs += pos.sum(axis=0).astype(np.int64)
            sum_int += np.where(pos, mat, 0.0).sum(axis=0).astype(np.float64)
            cnt_int += pos.sum(axis=0).astype(np.int64)

    mean_int = np.zeros((dim,), dtype=np.float64)
    nz = cnt_int > 0
    mean_int[nz] = sum_int[nz] / cnt_int[nz]

    ion_stats_path = out_p / "ion_stats_train.txt"
    with open(ion_stats_path, "w") as f:
        for ion, occ, mi in zip(ions, occurs.tolist(), mean_int.tolist()):
            f.write(f"{ion:>20} {int(occ)} {float(mi):.4f}\n")

    criteria_path = out_p / "criteria.txt"
    with open(criteria_path, "w") as f:
        f.write("occurs>=0\n")

    mod_path = out_p / "modifications.txt"
    with open(mod_path, "w") as f:
        f.write("Acetyl\n")
        f.write("Carbamidomethyl\n")
        f.write("Oxidation\n")

    summary_path = out_p / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"parquet={parquet_path}\n")
        f.write(f"label_column={label_column}\n")
        f.write(f"max_len={int(max_len)}\n")
        f.write(f"max_frag_charge={int(max_frag_charge)}\n")
        f.write(f"dim={int(dim)}\n")
        f.write(f"total_rows={int(total_rows)}\n")
        f.write(f"bad_rows={int(bad_rows)}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label_column", default="intensities_raw")
    ap.add_argument("--max_len", type=int, default=40)
    ap.add_argument("--max_frag_charge", type=int, default=3)
    args = ap.parse_args()

    build_ion_stats(
        args.train_parquet,
        out_dir=args.out_dir,
        label_column=str(args.label_column),
        max_len=int(args.max_len),
        max_frag_charge=int(args.max_frag_charge),
    )


if __name__ == "__main__":
    main()

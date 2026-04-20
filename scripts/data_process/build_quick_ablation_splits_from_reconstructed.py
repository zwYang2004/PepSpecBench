#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "reconstructed"
IN_DIR = Path(os.environ.get("MS2B_ABLATION_IN_DIR", DATA_ROOT / "prospect" / "234d"))
OUT_RANDOM = Path(os.environ.get("MS2B_ABLATION_RANDOM_OUT", DATA_ROOT / "prospect_quick_random_234d"))
OUT_SEQUENCE = Path(os.environ.get("MS2B_ABLATION_SEQUENCE_OUT", DATA_ROOT / "prospect_quick_sequence_234d"))


def _bucket_from_text(s: str) -> int:
    h = hashlib.md5(s.encode('utf-8')).hexdigest()
    return int(h[:8], 16) % 100


def _target_split(bucket: int) -> str:
    if bucket < 80:
        return 'train'
    if bucket < 90:
        return 'val'
    return 'test'


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _split_mode(mode: str, out_dir: Path) -> Dict[str, int]:
    _ensure_dir(out_dir)
    writers: Dict[str, pq.ParquetWriter] = {}
    counts = {'train': 0, 'val': 0, 'test': 0}

    for src in ['train', 'val', 'test']:
        src_path = IN_DIR / f'{src}.parquet'
        pf = pq.ParquetFile(src_path)

        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            n = table.num_rows
            mods = table['modified_sequence'].to_pylist()

            if mode == 'sequence':
                keys = [str(m) for m in mods]
            elif mode == 'random':
                # Use stable row-wise key to emulate random assignment after mixing.
                # sample_key is already deterministic and present in reconstructed parquet.
                if 'sample_key' in table.schema.names:
                    sk = table['sample_key'].to_pylist()
                    keys = [f'{int(k)}|{i}' for i, k in enumerate(sk)]
                else:
                    keys = [f'{str(m)}|{i}|42' for i, m in enumerate(mods)]
            else:
                raise ValueError(mode)

            masks = {'train': [False] * n, 'val': [False] * n, 'test': [False] * n}
            for i, key in enumerate(keys):
                b = _bucket_from_text(key)
                sp = _target_split(b)
                masks[sp][i] = True

            for sp in ['train', 'val', 'test']:
                mask_arr = pa.array(masks[sp], type=pa.bool_())
                subt = table.filter(mask_arr)
                if subt.num_rows == 0:
                    continue
                if sp not in writers:
                    writers[sp] = pq.ParquetWriter(out_dir / f'{sp}.parquet', subt.schema, compression='zstd')
                writers[sp].write_table(subt)
                counts[sp] += subt.num_rows

    for w in writers.values():
        w.close()
    return counts


def main() -> None:
    c_rand = _split_mode('random', OUT_RANDOM)
    c_seq = _split_mode('sequence', OUT_SEQUENCE)

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / 'quick_ablation_split_counts.txt'
    report.write_text(
        'Quick ablation split from reconstructed prospect/234d (mixed train+val+test)\n'
        f'random counts: {c_rand}\n'
        f'sequence counts: {c_seq}\n'
        f'random dir: {OUT_RANDOM}\n'
        f'sequence dir: {OUT_SEQUENCE}\n'
    )
    print('random', c_rand)
    print('sequence', c_seq)
    print('report', report)


if __name__ == '__main__':
    main()

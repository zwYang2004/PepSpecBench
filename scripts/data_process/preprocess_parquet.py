import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_UNIMOD_RE = re.compile(r"unimod:(\d+)", re.IGNORECASE)


def _parse_allowed_unimods(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    out = sorted(set(out))
    return tuple(out) if out else None


def _build_allowed_ptm_token_regex(allowed_unimods: Tuple[int, ...]) -> str:
    allowed = set(int(x) for x in allowed_unimods)
    unimod_alt = "|".join(str(x) for x in sorted(allowed))

    allowed_parts = [rf"[^\]]*unimod:(?:{unimod_alt})[^\]]*"]
    if 35 in allowed:
        allowed_parts.append(r"[^\]]*oxidation[^\]]*")
    if 4 in allowed:
        allowed_parts.append(r"[^\]]*carbamidomethyl[^\]]*")
    if 1 in allowed:
        allowed_parts.append(r"[^\]]*acetyl[^\]]*")
    inner = "|".join(allowed_parts)

    return rf"(?i)\[(?:{inner})\]"


def _detect_columns(schema_names: Iterable[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    names = set(schema_names)
    charge_col = None
    len_col = None
    seq_col = None

    if "precursor_charge" in names:
        charge_col = "precursor_charge"
    elif "charge" in names:
        charge_col = "charge"

    if "peptide_length" in names:
        len_col = "peptide_length"

    if "modified_sequence" in names:
        seq_col = "modified_sequence"
    elif "sequence" in names:
        seq_col = "sequence"

    return charge_col, len_col, seq_col


def _add_value_counts(dst: Dict[str, int], arr: pa.Array) -> None:
    vc = pc.value_counts(arr)
    for row in vc.to_pylist():
        k = row["values"]
        if k is None:
            continue
        dst[str(k)] = int(dst.get(str(k), 0) + int(row["counts"]))


def _normalize_mod_token(token: str) -> str:
    if not isinstance(token, str):
        return ""
    t = token.strip()
    m = _UNIMOD_RE.search(t)
    if m:
        return f"UNIMOD:{m.group(1)}"
    return t


def _compute_ptm_counts(
    parquet_path: str,
    seq_col: str,
    batch_size: int,
    max_rows: Optional[int],
) -> Counter:
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = dataset.scanner(columns=[seq_col], batch_size=int(batch_size))

    counter: Counter = Counter()
    seen = 0
    for batch in scanner.to_batches():
        seqs = batch.column(0).to_pylist()
        for s in seqs:
            if not isinstance(s, str) or "[" not in s:
                continue
            for tok in _BRACKET_RE.findall(s):
                nt = _normalize_mod_token(tok)
                if nt:
                    counter[nt] += 1
        seen += len(seqs)
        if max_rows is not None and seen >= int(max_rows):
            break
    return counter


def filter_parquet(
    input_path: str,
    output_path: str,
    filtered_output_path: Optional[str],
    min_charge: int,
    max_charge: int,
    max_len: Optional[int],
    allowed_unimods: Optional[Tuple[int, ...]],
    batch_size: int,
    progress_every: int,
) -> Dict[str, object]:
    dataset = ds.dataset(input_path, format="parquet")
    schema = dataset.schema

    charge_col, len_col, seq_col = _detect_columns(schema.names)
    if charge_col is None:
        raise KeyError("Charge column not found. Expected 'precursor_charge' or 'charge'.")

    len_mode = "peptide_length"
    compute_len_from_sequence = False
    if max_len is not None and len_col is None:
        if seq_col == "sequence":
            len_mode = "computed_from_sequence"
            compute_len_from_sequence = True
        else:
            raise KeyError("Length column not found. Expected 'peptide_length'.")

    allowed_ptm_token_regex = None
    if allowed_unimods is not None:
        if seq_col is None:
            raise KeyError("Sequence column not found. Expected 'modified_sequence' or 'sequence'.")
        allowed_ptm_token_regex = _build_allowed_ptm_token_regex(allowed_unimods)

    scan_cols = schema.names
    scanner = dataset.scanner(columns=scan_cols, batch_size=int(batch_size))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filt_writer = None
    if filtered_output_path is not None:
        fp = Path(filtered_output_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        filt_writer = pq.ParquetWriter(str(fp), schema)

    kept_writer = pq.ParquetWriter(str(out_path), schema)

    total_rows = 0
    kept_rows = 0
    filtered_rows = 0
    filtered_by_charge = 0
    filtered_by_len = 0
    filtered_by_ptm = 0
    max_len_observed = 0

    charge_dist_all: Dict[str, int] = {}
    charge_dist_kept: Dict[str, int] = {}

    t0 = time.time()
    batch_idx = 0

    try:
        for batch in scanner.to_batches():
            batch_idx += 1
            total_rows += int(batch.num_rows)

            c = batch.column(batch.schema.get_field_index(charge_col))
            _add_value_counts(charge_dist_all, c)

            charge_ok = pc.and_(pc.greater_equal(c, int(min_charge)), pc.less_equal(c, int(max_charge)))
            mask = charge_ok

            if max_len is not None:
                if compute_len_from_sequence:
                    seq_arr = batch.column(batch.schema.get_field_index("sequence"))
                    l = pc.utf8_length(pc.cast(seq_arr, pa.string()))
                else:
                    l = batch.column(batch.schema.get_field_index(len_col))
                ml = pc.max(l).as_py()
                if ml is not None and int(ml) > max_len_observed:
                    max_len_observed = int(ml)

                len_ok = pc.less_equal(l, int(max_len))
                mask = pc.and_(mask, len_ok)
                filtered_by_len += int(pc.sum(pc.cast(pc.and_(pc.invert(len_ok), pc.is_valid(l)), pa.int64())).as_py() or 0)

            filtered_by_charge += int(pc.sum(pc.cast(pc.and_(pc.invert(charge_ok), pc.is_valid(c)), pa.int64())).as_py() or 0)

            if allowed_ptm_token_regex is not None:
                mask_before_ptm = mask
                s = batch.column(batch.schema.get_field_index(seq_col))
                s = pc.cast(s, pa.string())
                s_stripped = pc.replace_substring_regex(s, allowed_ptm_token_regex, "")
                disallowed = pc.match_substring_regex(s_stripped, r"\[[^\]]+\]")
                disallowed = pc.fill_null(disallowed, False)
                ptm_ok = pc.invert(disallowed)
                mask = pc.and_(mask, ptm_ok)
                removed_by_ptm = pc.and_(disallowed, mask_before_ptm)
                filtered_by_ptm += int(pc.sum(pc.cast(removed_by_ptm, pa.int64())).as_py() or 0)

            t = pa.Table.from_batches([batch], schema=schema)
            t_kept = t.filter(mask)
            t_filt = t.filter(pc.invert(mask))

            if t_kept.num_rows > 0:
                kept_writer.write_table(t_kept)
                kept_rows += int(t_kept.num_rows)

                c_kept = pc.filter(c, mask)
                _add_value_counts(charge_dist_kept, c_kept)

            if t_filt.num_rows > 0:
                filtered_rows += int(t_filt.num_rows)
                if filt_writer is not None:
                    filt_writer.write_table(t_filt)

            if progress_every and (batch_idx % int(progress_every) == 0):
                dt = time.time() - t0
                rps = (total_rows / dt) if dt > 0 else 0.0
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "batches": int(batch_idx),
                            "total_rows": int(total_rows),
                            "kept_rows": int(kept_rows),
                            "filtered_rows": int(filtered_rows),
                            "seconds": float(round(dt, 3)),
                            "rows_per_sec": float(round(rps, 3)),
                            "input_path": str(input_path),
                            "output_path": str(output_path),
                        }
                    ),
                    flush=True,
                )

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "filtered_output_path": str(filtered_output_path) if filtered_output_path is not None else None,
            "charge_col": charge_col,
            "len_col": len_col,
            "seq_col": seq_col,
            "len_mode": len_mode,
            "min_charge": int(min_charge),
            "max_charge": int(max_charge),
            "max_len": int(max_len) if max_len is not None else None,
            "allowed_unimods": list(allowed_unimods) if allowed_unimods is not None else None,
            "total_rows": int(total_rows),
            "kept_rows": int(kept_rows),
            "filtered_rows": int(filtered_rows),
            "filtered_by_charge_count": int(filtered_by_charge),
            "filtered_by_len_count": int(filtered_by_len),
            "filtered_by_ptm_count": int(filtered_by_ptm),
            "max_len_observed": int(max_len_observed),
            "charge_distribution_all": charge_dist_all,
            "charge_distribution_kept": charge_dist_kept,
        }
    finally:
        kept_writer.close()
        if filt_writer is not None:
            filt_writer.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    p_filter = sub.add_parser("filter")
    p_filter.add_argument("--input", required=True)
    p_filter.add_argument("--output", required=True)
    p_filter.add_argument("--filtered-output", default=None)
    p_filter.add_argument("--min-charge", type=int, default=1)
    p_filter.add_argument("--max-charge", type=int, default=6)
    p_filter.add_argument("--max-len", type=int, default=None)
    p_filter.add_argument("--allowed-unimods", default=None)
    p_filter.add_argument("--batch-size", type=int, default=200_000)
    p_filter.add_argument("--progress-every", type=int, default=0)
    p_filter.add_argument("--summary-json", default=None)
    p_filter.add_argument("--ptm-stats", choices=["none", "full"], default="none")
    p_filter.add_argument("--ptm-max-rows", type=int, default=None)
    p_filter.add_argument("--ptm-batch-size", type=int, default=500_000)

    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.command == "filter":
        allowed_unimods = _parse_allowed_unimods(args.allowed_unimods)
        stats = filter_parquet(
            input_path=args.input,
            output_path=args.output,
            filtered_output_path=args.filtered_output,
            min_charge=args.min_charge,
            max_charge=args.max_charge,
            max_len=args.max_len,
            allowed_unimods=allowed_unimods,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
        )

        if args.ptm_stats == "full":
            charge_col, len_col, seq_col = _detect_columns(ds.dataset(args.output, format="parquet").schema.names)
            if seq_col is not None:
                ptm_counts = _compute_ptm_counts(
                    parquet_path=args.output,
                    seq_col=seq_col,
                    batch_size=args.ptm_batch_size,
                    max_rows=args.ptm_max_rows,
                )
                stats["ptm_types"] = dict(ptm_counts)
                stats["ptm_type_count"] = int(len(ptm_counts))
            else:
                stats["ptm_types"] = {}
                stats["ptm_type_count"] = 0

        if args.summary_json is not None:
            out = Path(args.summary_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(stats, f, indent=2)
        else:
            print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

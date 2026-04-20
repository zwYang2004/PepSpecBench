import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_root_str = str(_root)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

from src.scripts.build_unispec_dict import build_ion_stats


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

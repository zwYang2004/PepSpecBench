from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Set, Tuple


@dataclass(frozen=True)
class DataConstraints:
    min_len: int = 6
    max_len: int = 40
    max_charge: int = 6
    allowed_unimod_ids: Tuple[int, ...] = (1, 4, 35)

    def allowed_unimod_set(self) -> Set[int]:
        return set(int(x) for x in self.allowed_unimod_ids)

    def merge(self, other: Optional["DataConstraints"]) -> "DataConstraints":
        if other is None:
            return self
        return DataConstraints(
            min_len=int(other.min_len if other.min_len is not None else self.min_len),
            max_len=int(other.max_len if other.max_len is not None else self.max_len),
            max_charge=int(other.max_charge if other.max_charge is not None else self.max_charge),
            allowed_unimod_ids=tuple(other.allowed_unimod_ids) if other.allowed_unimod_ids else self.allowed_unimod_ids,
        )


def normalize_allowed_unimod_ids(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted({int(v) for v in values}))

from __future__ import annotations

from typing import Dict, Type

from src.models.base_model import BenchmarkModel
from src.models.runners.fastspel_runner import FastSpelRunner
from src.models.runners.predfull_runner import PredFullRunner
from src.models.runners.prosit_runner import PrositRunner
from src.models.runners.prosit_transformer_runner import PrositTransformerRunner
from src.models.runners.unispec_runner import UniSpecRunner
from src.models.runners.alphapeptdeep_runner import AlphaPeptDeepRunner


MODEL_REGISTRY: Dict[str, Type[BenchmarkModel]] = {
    "fastspel": FastSpelRunner,
    "predfull_torch": PredFullRunner,
    "predfull_torch_234d": PredFullRunner,  # PredFull-234d: ion-tensor variant for OOD ablation
    "prosit": PrositRunner,
    "prosit_transformer": PrositTransformerRunner,
    "unispec": UniSpecRunner,
    "alphapeptdeep": AlphaPeptDeepRunner,
}

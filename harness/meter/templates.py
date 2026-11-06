"""Load and validate the architecture and serving templates.

Templates are the single source of truth for every forward calculation.
A template that is missing a field needed by meter.work raises at load
time, not at run time on a rented GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

_REQUIRED_COMMON = {
    "hf_id", "role", "family", "architecture", "attention", "n_layers",
    "hidden_size", "n_heads", "n_params_total_b", "n_params_active_b",
    "native_context", "weight_precisions", "kv_precisions", "gated",
}
_REQUIRED_GQA = {"n_kv_heads", "head_dim"}
_REQUIRED_MLA = {"mla"}
_REQUIRED_MOE = {"moe"}
_REQUIRED_MOE_SUB = {"n_experts", "top_k", "shared_experts", "routed_layers",
                     "expert_intermediate"}


@dataclass(frozen=True)
class ModelTemplate:
    key: str
    raw: dict[str, Any] = field(repr=False)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.raw[name]
        except KeyError as e:
            raise AttributeError(name) from e

    @property
    def is_moe(self) -> bool:
        return self.raw["architecture"] == "moe"

    @property
    def is_mla(self) -> bool:
        return self.raw["attention"] == "mla"

    @property
    def is_hybrid_local_global(self) -> bool:
        return self.raw["attention"] == "gqa_hybrid_local_global"


@dataclass(frozen=True)
class Catalogue:
    models: dict[str, ModelTemplate]
    bytes_per_param: dict[str, float]
    bytes_per_kv_element: dict[str, float]
    accelerators: dict[str, dict[str, Any]]
    platforms: dict[str, dict[str, Any]]
    serving: dict[str, Any]

    def model(self, key: str) -> ModelTemplate:
        return self.models[key]

    def calibration_models(self) -> list[ModelTemplate]:
        return [m for m in self.models.values() if m.role == "calibration"]

    def withheld_models(self) -> list[ModelTemplate]:
        return [m for m in self.models.values() if m.role == "withheld"]


def _validate_model(key: str, raw: dict[str, Any]) -> None:
    missing = _REQUIRED_COMMON - raw.keys()
    if missing:
        raise ValueError(f"model {key}: missing {sorted(missing)}")
    att = raw["attention"]
    if att in ("gqa", "gqa_hybrid_local_global"):
        missing = _REQUIRED_GQA - raw.keys()
        if missing:
            raise ValueError(f"model {key}: GQA needs {sorted(missing)}")
        if att == "gqa_hybrid_local_global":
            for k in ("sliding_window", "local_global_pattern"):
                if k not in raw:
                    raise ValueError(f"model {key}: hybrid attention needs {k}")
    elif att == "mla":
        if "mla" not in raw or not {"kv_lora_rank", "qk_rope_head_dim"} <= raw["mla"].keys():
            raise ValueError(f"model {key}: MLA needs mla.kv_lora_rank and mla.qk_rope_head_dim")
    else:
        raise ValueError(f"model {key}: unknown attention {att!r}")
    if raw["architecture"] == "moe":
        if "moe" not in raw:
            raise ValueError(f"model {key}: MoE needs a moe block")
        missing = _REQUIRED_MOE_SUB - raw["moe"].keys()
        if missing:
            raise ValueError(f"model {key}: moe block missing {sorted(missing)}")
    elif raw["architecture"] != "dense":
        raise ValueError(f"model {key}: unknown architecture {raw['architecture']!r}")
    if raw["role"] not in ("calibration", "withheld"):
        raise ValueError(f"model {key}: role must be calibration or withheld")


def load_catalogue(config_dir: Path = CONFIG_DIR) -> Catalogue:
    models_raw = yaml.safe_load((config_dir / "models.yaml").read_text())
    serving_raw = yaml.safe_load((config_dir / "serving.yaml").read_text())

    models: dict[str, ModelTemplate] = {}
    for key, raw in models_raw["models"].items():
        _validate_model(key, raw)
        models[key] = ModelTemplate(key=key, raw=raw)

    return Catalogue(
        models=models,
        bytes_per_param=models_raw["bytes_per_param"],
        bytes_per_kv_element=models_raw["bytes_per_kv_element"],
        accelerators=serving_raw["accelerators"],
        platforms=serving_raw["platforms"],
        serving=serving_raw,
    )

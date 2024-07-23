from dataclasses import dataclass
import dataclasses
from typing import Any, Dict, Optional

import lass.data
import lass.log_handling


@dataclass
class LogInfo:
    output_dir: str
    log_group: str
    use_wandb: bool
    model_alias: Optional[str]


@dataclass
class HyperParams:
    batch_size: int
    gradient_accumulation_steps: int
    n_epochs: int
    warmup_steps: int
    learning_rate: float
    extra: Dict[str, Any]

    def reduce_mem(self, max_batch_size: int) -> "HyperParams":
        assert self.batch_size % max_batch_size == 0
        ratio = self.batch_size // max_batch_size
        if ratio < 1:
            return self

        return dataclasses.replace(
            self,
            batch_size=max_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps * ratio,
        )

    def with_fields(self, **kwargs) -> "HyperParams":
        for k, v in kwargs.items():
            assert hasattr(self, k), f"Field {k} not found in HyperParams"
            if v is None:
                kwargs[k] = getattr(self, k)
        return dataclasses.replace(self, **kwargs)


@dataclass
class Config:
    seed: int
    is_test_run: bool
    data_spec: lass.log_handling.LogLoaderArgs
    model: str
    split_type: lass.data.splitting.SplitType
    test_fraction: float
    include_model_in_input: bool
    include_n_targets_in_input: bool
    filter_bad_tasks: bool
    hypers: HyperParams
    log_info: LogInfo
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        data_spec = lass.log_handling.LogLoaderArgs(**d.pop("data_spec"))
        hypers = HyperParams(**d.pop("hypers"))
        log_info = LogInfo(**d.pop("log_info"))
        return Config(data_spec=data_spec, hypers=hypers, log_info=log_info, **d)


HYPER_DEFAULT = HyperParams(
    batch_size=32,
    gradient_accumulation_steps=1,
    n_epochs=6,
    warmup_steps=3000,
    learning_rate=2e-5,
    extra={},
)

HYPER_SMALL_DATA = HyperParams(
    batch_size=32,
    gradient_accumulation_steps=1,
    n_epochs=18,
    warmup_steps=50,
    learning_rate=1e-5,
    extra={},
)

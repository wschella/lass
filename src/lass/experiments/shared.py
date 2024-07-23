import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import torch

import lass.test
import lass.train
from lass.config import Config


def assert_GPU_available():
    assert torch.cuda.is_available(), "No CUDA available"
    assert torch.cuda.device_count() > 0, "No CUDA device available"


def load_config(model_dir: Path) -> Config:
    """
    Load the config file that was used to train a model.

    Args:
        model_dir: Path to the directory containing the model.
            Will look something like "./artifact/assessors/q1indistribution/deberta-base_20210901120000/checkpoint-4000"
    """
    with open(model_dir.parent / "config.json", "r") as f:
        return Config.from_dict(json.load(f))


def make_model_id(config: Config) -> Tuple[str, str]:
    """
    Create a model id based on the config.

    Args:
        config: The config object.
    """
    model_id, _ = lass.train.make_model_id(config)
    model_id_timed = f"{model_id}-{datetime.now().strftime('%m%d%H%M')}"
    return model_id, model_id_timed


def save_config(config: Config, dir: Union[Path, str]):
    """
    Save the config to the output directory.
    """
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    # config.log_info = dataclasses.replace(config.log_info, output_dir=str(output_dir))
    with open(dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)


def dump_results(results: lass.test.TestResults, dir: Union[Path, str]):
    dir = Path(dir)
    dumpable = lambda r: {"logits": r.logits.tolist(), "labels": r.labels.tolist()}
    with open(dir / "metrics.json", "w") as f:
        json.dump(results.metrics, f, indent=2)
    with open(dir / "predictions.json", "w") as f:
        json.dump(dumpable(results), f)


def dump_results_per_task(
    results: dict[str, lass.test.TestResults], dir: Union[Path, str]
):
    dir = Path(dir)
    dumpable = lambda r: {"logits": r.logits.tolist(), "labels": r.labels.tolist()}
    with open(dir / "metrics_per_task.json", "w") as f:
        metrics = {k: v.metrics for k, v in results.items()}
        json.dump(metrics, f, indent=2)
    with open(dir / "predictions_per_task.json", "w") as f:
        predictions = {k: dumpable(v) for k, v in results.items()}
        json.dump(predictions, f)


def dump_as_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def earliest_checkpoint(dir: Path) -> Path:
    """
    Find the earliest checkpoint in a directory.
    Usually we have two saved checkpoints, the best so far, and the latest.
    We usually want to load the best so far, i.e. the earliest.
    """
    step = min(
        int(p.name.split("-")[1])  # looks like checkpoint-4000
        for p in dir.glob("checkpoint-*")
    )
    return dir / f"checkpoint-{step}"


def get_alias(model_name: str) -> str:
    """
    Get the alias for a model name.
    """
    #
    # models=("albert/albert-base-v2" "google-bert/bert-base-cased" "FacebookAI/roberta-base" "microsoft/deberta-v3-base" "openai-community/gpt2"
    mapping = {
        "albert/albert-base-v2": "albert-base",
        "google-bert/bert-base-cased": "bert-base",
        "FacebookAI/roberta-base": "roberta-base",
        "microsoft/deberta-v3-small": "deberta-small",
        "microsoft/deberta-v3-base": "deberta-base",
        "microsoft/deberta-v3-large": "deberta-large",
        "openai-community/gpt2": "gpt2",
    }
    if model_name not in mapping:
        raise ValueError(f"Model alias not found for {model_name}.")
    return mapping[model_name]

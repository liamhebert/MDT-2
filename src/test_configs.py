"""
These tests validate whether components can be loaded in using Hydra.
"""

from pathlib import Path

import hydra
from hydra import compose
from hydra import initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import open_dict
import pytest
import rootutils
from glob import glob
import os


def global_prepare(
    base_yaml: str = "train.yaml", experiment_override: str | None = None
) -> DictConfig:
    with initialize(version_base=None, config_path="configs"):
        if experiment_override:
            overrides = ["experiment=" + experiment_override]
        else:
            overrides = []
        cfg = compose(
            config_name=base_yaml,
            return_hydra_config=True,
            overrides=overrides,
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(
                rootutils.find_root(indicator=".project-root")
            )
            cfg.trainer.max_epochs = 1
            cfg.trainer.strategy = "auto"
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1

            cfg.dataset.num_workers = 0
            cfg.dataset.pin_memory = False
            cfg.modality_encoder.text_model_name = "bert-base-uncased"
            cfg.modality_encoder.vision_model_name = (
                "google/vit-base-patch16-224"
            )

            cfg.model.scheduler.dataset_steps = 100

            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


def add_tmp_paths(cfg: DictConfig, tmp_path: Path) -> DictConfig:
    """Augments a given test_config with temporary paths for data, logs, and
    output.

    Args:
        cfg (DictConfig): The test configuration.
        tmp_path (Path): The temporary path to store test files.
    """
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    return cfg


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    Returns:
        A DictConfig object containing a default Hydra configuration for
    training.
    """
    return global_prepare(base_yaml="train.yaml")


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for
    evaluation.

    Returns:
        A DictConfig containing a default Hydra configuration for
    evaluation.
    """
    return global_prepare(base_yaml="eval.yaml")


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which
    accepts a temporary logging path `tmp_path` for generating a temporary
    logging path.

    This is called by each test which uses the `cfg_train` arg. Each test
    generates its own temporary logging path.

    Args:
        cfg_train_global: The input DictConfig object to be modified.
        tmp_path: The temporary logging path.

    Returns:
        A DictConfig with updated output and log directories corresponding to
        `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    return add_tmp_paths(cfg, tmp_path)


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which
    accepts a temporary logging path `tmp_path` for generating a temporary
    logging path.

    This is called by each test which uses the `cfg_eval` arg. Each test
    generates its own temporary logging path.

    Args:
        cfg_eval_global: The input DictConfig object to be modified.
        tmp_path: The temporary logging path.

    Returns:
        A DictConfig with updated output and log directories corresponding to
        `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    return add_tmp_paths(cfg, tmp_path)


experiment_yamls = [
    os.path.basename(x).removesuffix(".yaml")
    for x in glob("configs/experiment/*.yaml")
]


@pytest.mark.parametrize("experiment", experiment_yamls)
def test_train_config(tmp_path: Path, experiment: str) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest
    fixture.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    cfg = global_prepare(base_yaml="train.yaml", experiment_override=experiment)
    cfg = add_tmp_paths(cfg, tmp_path)

    assert cfg
    assert cfg.dataset
    assert cfg.model
    assert cfg.trainer

    HydraConfig().set_config(cfg)

    hydra.utils.instantiate(cfg.dataset)
    hydra.utils.instantiate(cfg.model)
    hydra.utils.instantiate(cfg.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest
    fixture.

    Args:
        cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.dataset
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.dataset)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_test_config(cfg_test: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest
    fixture.

    Args:
        cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_test
    assert cfg_test.dataset
    assert cfg_test.model
    assert cfg_test.trainer

    HydraConfig().set_config(cfg_test)

    hydra.utils.instantiate(cfg_test.dataset)
    hydra.utils.instantiate(cfg_test.model)
    hydra.utils.instantiate(cfg_test.trainer)

"""
These tests validate whether components can be loaded in using Hydra.
"""

from pathlib import Path
from typing import Generator

import hydra
from hydra import compose
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import open_dict
import pytest
import rootutils


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    Returns:
        A DictConfig object containing a default Hydra configuration for
    training.
    """
    with initialize(version_base="1.3", config_path="configs"):
        cfg = compose(
            config_name="train.yaml", return_hydra_config=True, overrides=[]
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(
                rootutils.find_root(indicator=".project-root")
            )
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.task.num_workers = 0
            cfg.task.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for
    evaluation.

    Returns:
        A DictConfig containing a default Hydra configuration for
    evaluation.
    """
    with initialize(version_base="1.3", config_path="configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=["ckpt_path=."],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(
                rootutils.find_root(indicator=".project-root")
            )
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.task.num_workers = 0
            cfg.task.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(
    cfg_train_global: DictConfig, tmp_path: Path
) -> Generator[DictConfig, None, None]:
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

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(
    cfg_eval_global: DictConfig, tmp_path: Path
) -> Generator[DictConfig, None, None]:
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

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest
    fixture.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.task
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.task)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest
    fixture.

    Args:
        cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.task
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.task)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_test_config(cfg_test: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest
    fixture.

    Args:
        cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_test
    assert cfg_test.task
    assert cfg_test.model
    assert cfg_test.trainer

    HydraConfig().set_config(cfg_test)

    hydra.utils.instantiate(cfg_test.task)
    hydra.utils.instantiate(cfg_test.model)
    hydra.utils.instantiate(cfg_test.trainer)

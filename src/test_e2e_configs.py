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


@pytest.fixture(scope="session")
def cfg_test_e2e_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for running
    test functions.

    This differs from other fixtures in that the model is dramatically smaller
    and designed for quick testing.

    Returns:
        A DictConfig containing a default Hydra configuration for
    testing.
    """
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(
            config_name="test_fixture_e2e_node.yaml",
            return_hydra_config=True,
            overrides=["ckpt_path=."],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(
                rootutils.find_root(indicator=".project-root")
            )
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_test_e2e(
    cfg_test_e2e_global: DictConfig, tmp_path: Path
) -> Generator[DictConfig, None, None]:
    """A pytest fixture built on top of the `cfg_test_e2e_global()` fixture, which
    accepts a temporary logging path `tmp_path` for generating a temporary
    logging path.

    This is called by each test which uses the `cfg_test` arg. Each test
    generates its own temporary logging path.

    Args:
        cfg_test_global: The input DictConfig object to be modified.
        tmp_path: The temporary logging path.

    Returns:
        A DictConfig with updated output and log directories corresponding to
        `tmp_path`.
    """
    cfg = cfg_test_e2e_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


def test_end_to_end(cfg_test_e2e: DictConfig) -> None:

    HydraConfig().set_config(cfg_test_e2e)

    data = hydra.utils.instantiate(cfg_test_e2e.dataset)
    model = hydra.utils.instantiate(cfg_test_e2e.model)
    trainer = hydra.utils.instantiate(cfg_test_e2e.trainer)

    trainer.fit(model=model, datamodule=data, ckpt_path=None)

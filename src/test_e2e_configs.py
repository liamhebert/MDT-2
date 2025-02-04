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


def process_global_configs(path: str):
    """Setting up hydra configs with global settings, such as the root_dir and
    disabling unnecessary prints and logging.

    Args:
        path (str): The path to the test config yaml file.

    Returns:
        A DictConfig configured with project root and other global settings.
    """
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(
            config_name=path,
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


def process_test_configs(cfg: DictConfig, tmp_path: Path):
    """
    Augments a given test_config with temporary paths for data, logs, and output.

    Args:
        cfg (DictConfig): The test configuration.
        tmp_path (Path): The temporary path to store test files.

    Returns:
        A DictConfig with temporary paths for data, logs, and output.
    """
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.data_dir = str(tmp_path)

    return cfg


@pytest.mark.parametrize(
    "yaml",
    ["test_fixture_e2e_node.yaml", "test_fixture_e2e_contrast.yaml"],
)
def test_end_to_end(yaml: str, tmp_path: Path) -> None:
    """Tests whether a given yaml file can be instantiated and run.

    Args:
        yamls (str): The path to the test yaml file.
        tmp_path (Path): A temporary folder created by pytest to store files.
    """
    cfg = process_global_configs(yaml)
    cfg = process_test_configs(cfg, tmp_path)

    HydraConfig().set_config(cfg)

    data = hydra.utils.instantiate(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(model=model, datamodule=data, ckpt_path=None)

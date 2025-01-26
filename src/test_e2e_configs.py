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


def process_global(path: str):
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


def process_local(cfg: DictConfig, tmp_path: Path):
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.data_dir = str(tmp_path)

    return cfg


@pytest.mark.parametrize(
    "yamls",
    ["test_fixture_e2e_node.yaml", "test_fixture_e2e_contrast.yaml"],
)
def test_end_to_end(yamls: str, tmp_path: Path) -> None:
    cfg = process_global(yamls)
    cfg = process_local(cfg, tmp_path)

    HydraConfig().set_config(cfg)

    data = hydra.utils.instantiate(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(model=model, datamodule=data, ckpt_path=None)

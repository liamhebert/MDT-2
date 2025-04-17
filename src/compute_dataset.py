"""Utility script to preprocess the dataset according to a given experiment.

Launch using python compute_dataset.py experiment=<experiment_name> env=<env>.
"""

import hydra
from omegaconf import DictConfig
from utils import RankedLogger
from utils import rich_utils

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point for dataset processing."""
    paths = cfg.paths

    log.info("Will write using following config:")
    rich_utils.print_config_tree(paths, resolve=True, save_to_file=False)

    log.info(f"Instantiating datamodule <{cfg.dataset._target_}>")
    cfg.dataset.dataset.force_reload = True
    cfg.dataset.dataset.skip_invalid_label_graphs = True
    datamodule = hydra.utils.instantiate(cfg.dataset)

    log.info("Starting preprocessing...")
    datamodule.prepare_data()

    log.info("All done!")


if __name__ == "__main__":
    main()

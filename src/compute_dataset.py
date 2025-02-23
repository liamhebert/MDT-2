import hydra
from omegaconf import DictConfig
from utils import RankedLogger
from utils import rich_utils

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    paths = cfg.paths

    log.info("Will write using following config:")
    rich_utils.print_config_tree(paths, resolve=True, save_to_file=False)

    log.info(f"Instantiating datamodule <{cfg.dataset._target_}>")
    cfg.dataset.dataset.force_reload = True
    datamodule = hydra.utils.instantiate(cfg.dataset)

    log.info("Starting preprocessing...")
    datamodule.prepare_data()

    log.info("All done!")


if __name__ == "__main__":
    main()

"""
Trainer class and parameters.
"""

from dataclasses import dataclass
from datetime import datetime
import getpass

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import wandb


@dataclass
class TrainerParameters:
    """
    Dataclass containing hyperparameters for the dataset.
    """

    MAX_EPOCHS: int


def get_trainer(params: TrainerParameters, experiment_name: str) -> Trainer:
    """Get a PyTorch Lightning Trainer configured with the given parameters.

    Args:
        params: The TrainerParameters to use.
        experiment_name: The name of the experiment to use for logging.

    Returns:
        A PyTorch Lightning Trainer configured to log to Weights and Bias.
    """
    # Get the current date and time
    now = datetime.now()

    # Format it into the desired format: YYYY-MM-DD-HH:MM:SS
    formatted_date = now.strftime("%Y-%m-%d-%H:%M:%S")
    user = getpass.getuser()

    # init trainer
    trainer = Trainer(
        max_epochs=params.MAX_EPOCHS,
        logger=wandb.WandbLogger(
            project="PersonalIOL",
            name=f"{experiment_name}-{user}-{formatted_date}",
        ),
        callbacks=[ModelCheckpoint(monitor="val/loss", mode="min")],
        precision="bf16-mixed",
    )

    return trainer

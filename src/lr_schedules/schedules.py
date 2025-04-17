from torch import optim
from abc import ABC, abstractmethod
from torch.optim import lr_scheduler


class LearningRateSchedulePrototype(ABC):
    """
    A prototype for creating learning rate schedules.
    An example learning rate schedule would be:
    ```
    {
        "scheduler": self.scheduler,
        # "monitor": "val/loss",
        "interval": "step",
        "frequency": 1,
        "name": "lr_scheduler",
    }
    ```
    """

    schedule: lr_scheduler.LRScheduler

    @abstractmethod
    def create_schedule(self, optimizer: optim.Optimizer) -> dict: ...

    def get_last_lr(self) -> float:
        """
        Returns the last learning rate.
        """
        return self.schedule.get_last_lr()[0]


class PretrainSchedule(LearningRateSchedulePrototype):
    """
    A learning rate schedule for pretraining.
    """

    def __init__(
        self,
        max_epochs: int,
        dataset_steps: int,
        warmup_percentage: float = 0.05,
        start_lr: float = 1e-6,
    ):
        """
        Initializes the learning rate schedule for pretraining.

        Args:
            warmup: Percentage of warmup steps.
            max_epochs: Total number of epochs
            dataset_steps: Total number of steps in the dataset
        """
        self.warmup_percentage = warmup_percentage
        self.max_epochs = max_epochs
        self.dataset_steps = dataset_steps
        self.start_lr = start_lr

    def create_schedule(self, optimizer: optim.Optimizer) -> dict:
        """
        Creates a learning rate schedule for pretraining.
        """

        total_steps = self.max_epochs * self.dataset_steps
        warmup = int(self.warmup_percentage * total_steps)

        self.schedule = lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[
                lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.start_lr,
                    end_factor=1.0,
                    total_iters=warmup,
                ),
                lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_steps - warmup, eta_min=0.0
                ),
            ],
            milestones=[warmup],
        )
        return {
            "scheduler": self.schedule,
            "interval": "step",
            "frequency": 1,
            "name": "lr_scheduler",
        }


class FineTuneSchedule(LearningRateSchedulePrototype):
    """
    A learning rate schedule for fine-tuning.
    """

    def __init__(self, max_epochs: int, dataset_steps: int):
        """
        Initializes the learning rate schedule for fine-tuning.

        Args:
            max_epochs: Total number of epochs
            dataset_steps: Total number of steps in the dataset
        """
        self.max_epochs = max_epochs
        self.dataset_steps = dataset_steps

    def create_schedule(self, optimizer: optim.Optimizer) -> dict:
        """
        Creates a learning rate schedule for fine-tuning.
        """

        total_steps = self.max_epochs * self.dataset_steps

        self.schedule = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=total_steps, eta_min=0.0
        )
        return {
            "scheduler": self.schedule,
            "interval": "step",
            "frequency": 1,
            "name": "lr_scheduler",
        }

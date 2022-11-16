from pytorch_lightning.callbacks import Callback
from typing import List
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
import pytorch_lightning as pl
import torch


@CALLBACK_REGISTRY
class UpdateBatchSizeDataLoader(Callback):
    """Updates the batch size of the dataloader based on the current depth of the model.

    Parameters
    ----------
    batch_sizes : List[int]
        A list of batch sizes for each depth of the model.
    """

    def __init__(self, batch_sizes: List[int]) -> None:
        super().__init__()
        self.batch_sizes = batch_sizes

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Updates the batch size of the dataloader based on the current depth of the model.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer object.
        pl_module : pl.LightningModule
            The model object.
        """
        current_depth = pl_module.current_depth
        trainer.datamodule.batch_size = self.batch_sizes[current_depth]


@CALLBACK_REGISTRY
class UpdateMixingDepth(Callback):
    """Updates the mixing parameter and the depth of the model based on the current
    epoch/batch of the model.

    Parameters
    ----------
    epochs_for_each_depth : List[int]
        A list of epochs for each depth of the model.
    fade_for_each_depth : List[int]
        A list of fade points for each depth of the model.
    """

    def __init__(
        self, epochs_for_each_depth: List[int], fade_for_each_depth: List[int]
    ) -> None:

        super().__init__()
        self.epochs_for_each_depth = epochs_for_each_depth
        self.fade_for_each_depth = fade_for_each_depth

        self.n_epochs_current_depth = 0
        self.step_current_depth = 0
        self.last_epoch = 0

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Updates the depth of the model based on the current epoch of the training.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer object.
        pl_module : pl.LightningModule
            The model object.
        """
        current_depth = pl_module.current_depth
        current_epoch = pl_module.current_epoch
        epochs = self.epochs_for_each_depth[current_depth]

        if self.n_epochs_current_depth >= epochs:
            # set next depth
            self.n_epochs_current_depth = 0
            self.step_current_depth = 0
            current_depth += 1
            pl_module.depth = current_depth

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Updates the mixing parameter (alpha) of the model based on the current depth
        of the model.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer object.
        pl_module : pl.LightningModule
            The model object.
        batch : torch.Tensor
            The batch of data.
        batch_idx : int
            The index of the batch.
        """
        current_depth = pl_module.current_depth
        current_epoch = pl_module.current_epoch
        fade = self.fade_for_each_depth[current_depth]
        epochs = self.epochs_for_each_depth[current_depth]

        if current_epoch > self.last_epoch:
            self.last_epoch = current_epoch
            self.n_epochs_current_depth += 1

        fade = self.fade_for_each_depth[current_depth]
        epochs = self.epochs_for_each_depth[current_depth]
        total_batches = len(trainer.train_dataloader)
        fade_point = int((fade / 100) * epochs * total_batches)

        if self.step_current_depth <= fade_point:
            alpha = self.step_current_depth / fade_point
        else:
            alpha = 1.0
        self.step_current_depth += 1

        pl_module.alpha = alpha

import torch
import torch.nn.functional as F
import torch.utils.data as tud
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from typing import List
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY


@CALLBACK_REGISTRY
class UpdateBatchSizeDataLoader(Callback):
    def __init__(self, batch_sizes: List[int]):
        super().__init__()
        self.batch_sizes = batch_sizes

    def on_train_epoch_start(self, trainer, pl_module):
        current_depth = pl_module.current_depth
        trainer.datamodule.set_batch_size(self.batch_sizes[current_depth])


@CALLBACK_REGISTRY
class UpdateMixingDepth(Callback):
    def __init__(
        self, epochs_for_each_depth: List[int], fade_for_each_depth: List[int]
    ) -> None:
        super().__init__()
        self.epochs_for_each_depth = epochs_for_each_depth
        self.fade_for_each_depth = fade_for_each_depth

        self.n_epochs_current_depth = 0
        self.step_current_depth = 0
        self.last_epoch = 0

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        current_depth = pl_module.current_depth
        current_epoch = pl_module.current_epoch
        epochs = self.epochs_for_each_depth[current_depth]

        if self.n_epochs_current_depth >= epochs:
            # set next depth
            self.n_epochs_current_depth = 0
            self.step_current_depth = 0
            current_depth += 1
            pl_module.set_depth(current_depth)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0):
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

        pl_module.set_alpha(alpha)

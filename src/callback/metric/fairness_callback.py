from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .base_callback import BaseMetricCallback


class FairnessCallback(BaseMetricCallback):
    def _align_pred_label_list(
        self,
        label_names: list[str],
        pred: torch.Tensor,
        label: torch.Tensor,
        group: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
        assert (
            pred.shape[-1] == label.shape[-1]
        ), "Make sure that prediction, label, label_names are aligned"

        aligned_list: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for i in range(len(label_names)):
            aligned_list.append((label_names[i], pred[:, i], label[:, i], group))

        return aligned_list

    def _update_batch_end(
        self,
        prefix: str,
        metric_dict: nn.ModuleDict,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
    ) -> None:
        aligned_list = self._align_pred_label_list(
            pl_module.label_names, outputs["pred"], outputs["label"], outputs["group"]
        )

        for label_name, pred_item, label_item, group_item in aligned_list:
            metric_dict[f"{prefix}/{self._logging_name}/{label_name}"].update(
                pred_item, label_item, group_item
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        self._update_batch_end("train", self._train_metric_dict, pl_module, outputs)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        self._update_batch_end("valid", self._valid_metric_dict, pl_module, outputs)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        self._update_batch_end("test", self._test_metric_dict, pl_module, outputs)

    def _log_metric(
        self, pl_module: pl.LightningModule, metric_dict: dict[str]
    ) -> None:
        for label_name, metric in metric_dict.items():
            pl_module.log(label_name, metric, on_step=False, on_epoch=True)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._log_metric(pl_module, self._train_metric_dict)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._log_metric(pl_module, self._valid_metric_dict)

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._log_metric(pl_module, self._test_metric_dict)

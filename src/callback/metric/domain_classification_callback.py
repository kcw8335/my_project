from abc import ABCMeta
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback


class DomainClassificationCallback(Callback):
    def __init__(self, metric: ABCMeta):
        self._metric = metric
        self._logging_name = metric._get_name().replace("Metric", "", -1).lower()

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        label_names = pl_module.label_names

        self._train_metric_dict: nn.ModuleDict[str, ABCMeta] = nn.ModuleDict()
        self._valid_metric_dict: nn.ModuleDict[str, ABCMeta] = nn.ModuleDict()
        self._test_metric_dict: nn.ModuleDict[str, ABCMeta] = nn.ModuleDict()

        for label in label_names:
            label_renamed = self.drop_number_from_label(label)
            self._train_metric_dict[f"train/{self._logging_name}/{label_renamed}"] = self._metric.clone()
            self._valid_metric_dict[f"valid/{self._logging_name}/{label_renamed}"] = self._metric.clone()
            self._test_metric_dict[f"test/{self._logging_name}/{label_renamed}"] = self._metric.clone()

        pl_module.register_module(f"train/{self._logging_name}", self._train_metric_dict)
        pl_module.register_module(f"valid/{self._logging_name}", self._valid_metric_dict)
        pl_module.register_module(f"test/{self._logging_name}", self._test_metric_dict)

    def drop_number_from_label(self, label: str):
        return label.split("(")[0]

    def _align_pred_label_list(
        self, label_names: list[str], pred: torch.Tensor, label: torch.Tensor
    ) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        assert pred.shape[-1] == label.shape[-1], "Make sure that prediction, label, label_names are aligned"

        aligned_list: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for i in range(len(label_names)):
            aligned_list.append((self.drop_number_from_label(label_names[i]), pred[:, i], label[:, i]))

        return aligned_list

    def _update_batch_end(
        self,
        prefix: str,
        metric_dict: nn.ModuleDict,
        pl_module: pl.LightningModule,
        outputs: dict[str, torch.Tensor],
    ) -> None:
        aligned_list = self._align_pred_label_list(pl_module.label_names, outputs["out"], outputs["label"])

        for label_name, pred_item, label_item in aligned_list:
            metric_dict[f"{prefix}/{self._logging_name}/{label_name}"].update(pred_item, label_item)

    # NOTE: caution; not compatible with DataParallel mode !!
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

    def _log_metric(self, pl_module: pl.LightningModule, metric_dict: dict[str]) -> None:
        for label_name, metric in metric_dict.items():
            pl_module.log(label_name, metric, on_step=False, on_epoch=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_metric(pl_module, self._train_metric_dict)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_metric(pl_module, self._valid_metric_dict)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_metric(pl_module, self._test_metric_dict)

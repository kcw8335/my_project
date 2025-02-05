import torch
import torch.nn as nn
from itertools import chain

from typing import Any, Iterable
from pytorch_lightning import LightningModule


class ChalearnLightningModule(LightningModule):
    def __init__(
        self,
        multimodal_model: nn.Module,
        base_loss_func: nn.Module,
        domain_loss_func: nn.Module,
        dann_gamma: float,
        optim: torch.optim,
        lr_regressor: float,
        lr_classifier: float,
        lr_scheduler: torch.optim,
        label_names: list,
    ) -> None:
        super().__init__()
        self._multimodal_model = multimodal_model
        self._base_loss_func = base_loss_func
        self._domain_loss_func = domain_loss_func
        self._dann_gamma = dann_gamma
        self._optim = optim
        self._lr_regressor = lr_regressor
        self._lr_classifier = lr_classifier
        self._lr_scheduler = lr_scheduler
        self.label_names = label_names
        self.automatic_optimization = False

    def _parse_batch(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = batch[0]
        video = inputs.get("video", None)
        audio = inputs.get("audio", None)
        text = inputs.get("text", None)

        label = batch[1]["target_data"]
        group = batch[1]["gender_label"]

        return video, audio, text, label, group

    def _concat_net_params(self, *net: nn.Module) -> Iterable[nn.Parameter]:
        return chain(*map(lambda x: x.parameters(), net))

    def configure_optimizers(self):
        model_params = self._concat_net_params(
            self._multimodal_model._encoder, self._multimodal_model._regressor
        )
        optim_en_re = self._optim(model_params, self._lr_regressor)

        optim_en_cl = None
        if self._dann_gamma == 0.0:
            optim_en_cl = self._optim(
                self._multimodal_model._classifier.parameters(), self._lr_classifier
            )
        else:
            model_params = self._concat_net_params(
                self._multimodal_model._encoder, self._multimodal_model._classifier
            )
            optim_en_cl = self._optim(model_params, self._lr_classifier)
        return optim_en_re, optim_en_cl

    def training_step(self, batch, batch_idx) -> dict[str, Any]:
        optim_en_re, optim_en_cl = self.optimizers()

        video, audio, text, label, group = self._parse_batch(batch)
        feature, pred = self._multimodal_model(video, audio, text)
        loss_dict = self._base_loss_func(feature, pred, label, group)

        optim_en_re.zero_grad()
        self.manual_backward(loss_dict["total_loss"])
        optim_en_re.step()

        if self._dann_gamma != 0.0:
            p = self.current_epoch / self.trainer.max_epochs
            lambd = (2.0 / (1.0 + torch.exp(-1 * torch.Tensor([p])))) - 1
            lambd = lambd.numpy()[0]
            self.log(
                "train/lambda",
                value=lambd,
                on_step=False,
                on_epoch=True,
                batch_size=video.shape[0],
            )
            self._multimodal_model.update_lambd(lambd)

        group_pred = self._multimodal_model.domain_classification(video, audio, text)
        domain_loss = self._domain_loss_func(group_pred, group)

        optim_en_cl.zero_grad()
        self.manual_backward(domain_loss)
        optim_en_cl.step()

        loss_dict["domain_loss"] = domain_loss
        for key, value in loss_dict.items():
            self.log(
                f"train/{key}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=video.shape[0],
            )

        return {
            "pred": pred.detach(),
            "label": label.detach(),
            "group": group.detach(),
        }

    def validation_step(self, batch, batch_idx) -> dict[str, Any]:
        video, audio, text, label, group = self._parse_batch(batch)
        feature, pred = self._multimodal_model(video, audio, text)
        loss_dict = self._base_loss_func(feature, pred, label, group)

        group_pred = self._multimodal_model.domain_classification(video, audio, text)
        domain_loss = self._domain_loss_func(group_pred, group)

        loss_dict["domain_loss"] = domain_loss
        for key, value in loss_dict.items():
            self.log(
                f"valid/{key}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=video.shape[0],
            )

        return {
            "pred": pred.detach(),
            "label": label.detach(),
            "group": group.detach(),
        }

    def test_step(self, batch, batch_idx) -> None:
        video, audio, text, label, gender = self._parse_batch(batch)
        _, pred = self._multimodal_model(video, audio, text)

        return {
            "pred": pred.detach(),
            "label": label.detach(),
            "group": gender.detach(),
        }

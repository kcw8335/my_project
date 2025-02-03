import torch
import torch.nn as nn
from .binary_l2_loss import BinaryL2Loss
from .binary_mmd_loss import BinaryMMDLoss
from .binary_wd_loss import BinaryWDLoss


class FairnessLoss(nn.Module):
    def __init__(
        self,
        base_loss_function: nn.Module,
        base_loss_weight: float,
        l2_loss_weight: float,
        mmd_loss_weight: float,
        wd_loss_weight: float,
    ) -> dict:
        super().__init__()
        self._base_loss_weight = base_loss_weight
        self._base_loss_function = base_loss_function

        self._l2_loss_weight = l2_loss_weight
        self._l2_loss_function = BinaryL2Loss()

        self._mmd_loss_weight = mmd_loss_weight
        self._mmd_loss_function = BinaryMMDLoss()

        self._wd_loss_weight = wd_loss_weight
        self._wd_loss_function = BinaryWDLoss()

    def forward(
        self,
        batch_feature: torch.Tensor,
        batch_pred: torch.Tensor,
        batch_target: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> dict:
        base_loss = self._base_loss_function(batch_pred, batch_target)
        l2_loss = self._l2_loss_function(batch_feature, batch_group)
        mmd_loss = self._mmd_loss_function(batch_pred, batch_group)
        wd_loss = self._wd_loss_function(batch_pred, batch_group)

        total_loss = (
            self._base_loss_weight * base_loss
            + (self._l2_loss_weight * l2_loss)
            + (self._mmd_loss_weight * mmd_loss)
            + (self._wd_loss_weight * wd_loss)
        )

        return {
            "total_loss": total_loss,
            "base_loss": base_loss,
            "l2_loss": l2_loss,
            "mmd_loss": mmd_loss,
            "wd_loss": wd_loss,
        }

import torch

from .base_metric import BaseMetric


class DemographicParity(BaseMetric):
    """statistical demographic disparity"""

    def __init__(self, tau: float = 0.5) -> None:
        BaseMetric.__init__(self)
        self.tau = tau

    def _get_group_parity(
        self,
        target: torch.Tensor,
        pred: torch.Tensor,
        group: torch.Tensor,
        group_idx: int,
    ):
        group_target, group_pred = self._get_group_tensor(
            target, pred, group, group_idx
        )
        group_confusion = self._cal_confusion(group_target, group_pred)

        tot_data = (
            group_confusion["fp"]
            + group_confusion["tp"]
            + group_confusion["fn"]
            + group_confusion["tn"]
        )
        group_parity = (group_confusion["fp"] + group_confusion["tp"]) / (
            tot_data + 1e-10
        )
        return group_parity

    def _cal_difference(
        self, target: torch.Tensor, pred: torch.Tensor, group: torch.Tensor, tau: float
    ) -> torch.Tensor:
        target_binary = self._make_binary(target, tau)
        pred_binary = self._make_binary(pred, tau)

        major_parity = self._get_group_parity(target_binary, pred_binary, group, 0)
        minor_parity = self._get_group_parity(target_binary, pred_binary, group, 1)

        difference = torch.abs(major_parity - minor_parity)
        return difference

    def compute(self) -> torch.Tensor:
        target, pred, group = self._concat_states()
        difference = self._cal_difference(target, pred, group, self.tau)

        return difference

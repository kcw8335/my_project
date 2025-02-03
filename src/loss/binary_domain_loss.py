import torch
import torch.nn as nn


class DomainLoss(nn.Module):
    def __init__(self, domain_loss_weight: float):
        super().__init__()
        self._domain_loss_weight = domain_loss_weight
        self._domain_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        batch_group_pred: torch.Tensor,
        batch_group_label: torch.Tensor,
    ) -> torch.Tensor:
        domain_loss = self._domain_loss_fn(batch_group_pred, batch_group_label)
        domain_loss = self._domain_loss_weight * domain_loss
        return domain_loss

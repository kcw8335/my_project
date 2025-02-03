import torch
from .equal_opertunity import EqualOpportunity


class SPEO(EqualOpportunity):
    """string pairwise equal opportunity"""

    def __init__(self, interval: float = 0.02) -> None:
        EqualOpportunity.__init__(self)
        self.tau = torch.arange(0, 1, interval)

    def compute(self) -> torch.Tensor:
        target, pred, group = self._concat_states()
        differences = []
        for tau in self.tau:
            difference = self._cal_difference(target, pred, group, tau)
            differences.append(difference)

        eopdd = torch.Tensor(differences).mean()

        return eopdd

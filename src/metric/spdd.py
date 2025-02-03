import torch
from .demographic_parity import DemographicParity


class SPDD(DemographicParity):
    """string pairwise demographic disparity"""

    def __init__(self, interval: float = 0.02) -> None:
        DemographicParity.__init__(self)
        self.tau = torch.arange(0, 1, interval)

    def compute(self) -> torch.Tensor:
        target, pred, group = self._concat_states()
        differences = []
        for tau in self.tau:
            difference = self._cal_difference(target, pred, group, tau)
            differences.append(difference)

        spdd = torch.Tensor(differences).mean()

        return spdd

import torch
import torch.nn as nn
from network.layers import GradReverse


class ChalearnMultiModalModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        regressor: nn.Module,
        classifier: nn.Module,
    ):
        super().__init__()
        self._encoder = encoder
        self._regressor = regressor
        self._classifier = classifier

    def update_lambd(self, lambd):
        self.lambd = lambd
        GradReverse.lambd = self.lambd

    def forward(self, video: torch.Tensor, audio: torch.Tensor, text: torch.Tensor):
        multimodal_feature = self._encoder(video, audio, text)
        out_regressor = self._regressor(multimodal_feature)
        return multimodal_feature, out_regressor

    def domain_classification(self, video, audio, text):
        multimodal_feature = self._encoder(video, audio, text)
        out_classifier = GradReverse.apply(multimodal_feature)
        out_classifier = self._classifier(out_classifier)
        return out_classifier

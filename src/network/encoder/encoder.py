import torch
import torch.nn as nn

from network.layers import ConvBlock2D


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        video_channels: list,
        video_input_size: int,
        video_hidden_size: int,
        video_num_layers: int,
        audio_input_size: int = 68,
        audio_hidden_size: int = 32,
        audio_num_layers: int = 1,
        text_input_size: int = 300,
        text_hidden_size: int = 64,
    ):
        super().__init__()

        self.video_conv = self._make_conv_layers(video_channels)
        self.video_lstm = nn.LSTM(
            video_input_size, video_hidden_size, video_num_layers, batch_first=True
        )
        self.audio_fc = nn.Linear(audio_input_size, audio_hidden_size)
        self.audio_LSTM = nn.LSTM(
            audio_hidden_size, audio_hidden_size, audio_num_layers, batch_first=True
        )
        self.text_fc = nn.Linear(text_input_size, text_hidden_size)

    def _make_conv_layers(self, channels: list):
        layers = []
        for channel in channels:
            layers.append(ConvBlock2D(channel[0], channel[1]))
        return nn.Sequential(*layers)

    def _merge_multimodal_features(
        self, video: torch.Tensor, audio: torch.Tensor, text: torch.Tensor
    ) -> torch.Tensor:
        video = video.flatten(start_dim=1)
        audio = audio.flatten(start_dim=1)
        text = text.flatten(start_dim=1)
        out = torch.cat([video, audio, text], dim=1)
        return out

    def _get_video_feature(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.size()
        video = video.reshape(B * T, C, H, W)
        video_feature = self.video_conv(video)
        video_feature = video_feature.reshape(B, T, -1)
        video_out, _ = self.video_lstm(video_feature)
        return video_out[:, -1]

    def _get_audio_feature(self, audio: torch.Tensor) -> torch.Tensor:
        audio_feature = self.audio_fc(audio)
        audio_out, _ = self.audio_LSTM(audio_feature)
        return audio_out[:, -1]

    def _get_text_feature(self, text: torch.Tensor) -> torch.Tensor:
        text_out = self.text_fc(text[:, 0])
        return text_out

    def forward(
        self, video: torch.Tensor, audio: torch.Tensor, text: torch.Tensor
    ) -> torch.Tensor:
        video_out = self._get_video_feature(video)
        audio_out = self._get_audio_feature(audio)
        text_out = self._get_text_feature(text)

        multimodal_feature = self._merge_multimodal_features(
            video_out, audio_out, text_out
        )
        return multimodal_feature

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.fetcher import video_fetcher, audio_fetcher


class ChalearnFirstImpressionMultiModalDataset(Dataset):
    def __init__(
        self,
        video_dir: str,
        audio_dir: str,
        text_dir: str,
        label_dir: str,
        target_list: list,
        frame_count: int = 30,
        seed=0,
        video_transform=None,
        target_transform=None,
    ) -> None:
        super().__init__()

        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.frame_count = frame_count
        self.target_list = target_list

        self.seed = seed
        self.annotation_df = self._set_annotation_df(label_dir)
        self.video_transform = video_transform
        self.target_transform = target_transform

    def _set_annotation_df(self, label_dir: str) -> pd.DataFrame:
        return pd.read_csv(label_dir)

    def __len__(self) -> int:
        return len(self.annotation_df)

    def _get_target_and_meta_data(self, index: int) -> dict:
        target_data = []
        for target in self.target_list:
            target_data.append(self.annotation_df.iloc[index][target])
        target_data = torch.tensor(target_data).float()

        data_item = self.annotation_df.iloc[index]
        return {
            "video_name": data_item["video_name"],
            "youtube_id": data_item["youtube_id"],
            "ethnicity": data_item["ethnicity"],  # (Asian, Caucasian, African-American)
            "ethnicity_label": data_item["ethnicity_label"],  # (0, 1, 2)
            "gender": data_item["gender"],  # (M, F)
            "gender_label": data_item["gender_label"],  # (0, 1)
            "target_data": target_data,
        }

    def _get_video(self, index: int) -> torch.Tensor:
        file_name = self.annotation_df.iloc[index]["video_name"]
        video_path = os.path.join(self.video_dir, file_name)
        video_tensor = video_fetcher(video_path, self.frame_count, self.seed)
        return video_tensor

    def _get_audio(self, index: int) -> torch.Tensor:
        file_name = self.annotation_df.iloc[index]["video_name"]
        audio_path = os.path.join(self.audio_dir, file_name + ".wav_st.csv")
        audio_tensor = audio_fetcher(audio_path, self.frame_count)
        return audio_tensor

    def _get_text(self, index: int) -> torch.Tensor:
        file_name = self.annotation_df.iloc[index]["video_name"]
        text_path = os.path.join(self.text_dir, file_name + ".npy")
        text_tensor = torch.from_numpy(np.load(text_path))
        return text_tensor

    def __getitem__(self, index):
        target_and_meta_data = self._get_target_and_meta_data(index)

        video_tensor = self._get_video(index)
        if self.video_transform is not None:
            video_tensor = self.video_transform(video_tensor)
        audio_tensor = self._get_audio(index)
        text_tensor = self._get_text(index)

        input_data = {
            "video": video_tensor.float(),
            "audio": audio_tensor.float(),
            "text": text_tensor.float(),
        }
        return input_data, target_and_meta_data

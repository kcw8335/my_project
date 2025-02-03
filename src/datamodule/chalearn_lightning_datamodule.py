from typing import Optional

from beartype import beartype
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataset import ChalearnFirstImpressionMultiModalDataset
from dataset.tranform import ChalearnVideoTransform


class ChalearnLightningDataModule(LightningDataModule):
    @beartype
    def __init__(
        self,
        video_dir: str,
        audio_dir: str,
        text_dir: str,
        label_dir: dict,
        target_list: list,
        frame_count: int,
        seed: int,
        num_workers: int,
        batch_size: int,
    ) -> None:
        super().__init__()

        self._video_dir = video_dir
        self._audio_dir = audio_dir
        self._text_dir = text_dir
        self._label_dir = label_dir
        self._target_list = target_list
        self._frame_count = frame_count
        self._seed = seed
        self._num_workers = num_workers
        self._batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        video_transform = ChalearnVideoTransform()
        self._dataset_train = ChalearnFirstImpressionMultiModalDataset(
            self._video_dir,
            self._audio_dir,
            self._text_dir,
            self._label_dir["train"],
            self._target_list,
            self._frame_count,
            self._seed,
            video_transform=video_transform,
        )
        self._dataset_valid = ChalearnFirstImpressionMultiModalDataset(
            self._video_dir,
            self._audio_dir,
            self._text_dir,
            self._label_dir["valid"],
            self._target_list,
            self._frame_count,
            self._seed,
            video_transform=video_transform,
        )
        self._dataset_test = ChalearnFirstImpressionMultiModalDataset(
            self._video_dir,
            self._audio_dir,
            self._text_dir,
            self._label_dir["test"],
            self._target_list,
            self._frame_count,
            self._seed,
            video_transform=video_transform,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset_train,
            shuffle=True,
            drop_last=True,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset_valid,
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset_test,
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers,
            batch_size=self._batch_size,
            persistent_workers=True,
        )
